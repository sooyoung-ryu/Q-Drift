# -*- coding: utf-8 -*-
"""Nunchaku backend utilities."""

import torch

from ..tinychat.utils import convert_to_tinychat_w4x16y16_linear_weight
from ..utils import MmaWeightPackerBase, ceil_divide, fp_quantize, pad

__all__ = [
    "convert_to_nunchaku_w4x4y16_linear_weight",
    "convert_to_nunchaku_w8x8y16_linear_weight",
    "convert_to_nunchaku_w4x16_linear_weight",
]


class NunchakuWeightPacker(MmaWeightPackerBase):
    def __init__(self, bits: int, warp_n: int = 128):
        super().__init__(bits=bits, warp_n=warp_n)
        self.num_k_unrolls = 2

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        assert weight.dtype == torch.int32, f"quantized weight should be torch.int32, but got {weight.dtype}."
        n, k = weight.shape
        assert n % self.mem_n == 0, f"output channel size ({n}) should be divisible by mem_n ({self.mem_n})."
        # currently, Nunchaku did not check the boundry of unrolled `k` dimension
        assert k % (self.mem_k * self.num_k_unrolls) == 0, (
            f"input channel size ({k}) should be divisible by "
            f"mem_k ({self.mem_k}) * num_k_unrolls ({self.num_k_unrolls})."
        )
        n_tiles, k_tiles = n // self.mem_n, k // self.mem_k
        weight = weight.reshape(
            n_tiles,
            self.num_n_packs,  # 8 when warp_n = 128
            self.n_pack_size,  # always 2 in nunchaku
            self.num_n_lanes,  # constant 8
            self.reg_n,  # constant 1
            k_tiles,
            self.num_k_packs,  # 1
            self.k_pack_size,  # always 2 in nunchaku
            self.num_k_lanes,  # constant 4
            self.reg_k,  # always 8 = 32 bits / 4 bits
        )
        # (n_tiles, num_n_packs, n_pack_size, num_n_lanes, reg_n, k_tiles, num_k_packs, k_pack_size, num_k_lanes, reg_k)
        # =>
        # (n_tiles, k_tiles, num_k_packs, num_n_packs, num_n_lanes, num_k_lanes, n_pack_size, k_pack_size, reg_n, reg_k)
        weight = weight.permute(0, 5, 6, 1, 3, 8, 2, 7, 4, 9).contiguous()
        assert weight.shape[4:-2] == (8, 4, 2, 2)
        if self.bits == 4:
            weight = weight.bitwise_and_(0xF)
            shift = torch.arange(0, 32, 4, dtype=torch.int32, device=weight.device)
            weight = weight.bitwise_left_shift_(shift)
            weight = weight.sum(dim=-1, dtype=torch.int32)
        elif self.bits == 8:
            weight = weight.bitwise_and_(0xFF)
            shift = torch.arange(0, 32, 8, dtype=torch.int32, device=weight.device)
            weight = weight.bitwise_left_shift_(shift)
            weight = weight.sum(dim=-1, dtype=torch.int32)
        else:
            raise NotImplementedError(f"weight bits {self.bits} is not supported.")
        return weight.view(dtype=torch.int8).view(n, -1)  # assume little-endian

    def pack_scale(self, scale: torch.Tensor, group_size: int) -> torch.Tensor:
        if self.check_if_micro_scale(group_size=group_size):
            return self.pack_micro_scale(scale, group_size=group_size)
        # note: refer to https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-16864-c
        assert scale.dtype in (torch.float16, torch.bfloat16), "currently nunchaku only supports fp16 and bf16."
        n = scale.shape[0]
        # nunchaku load scales all in one access
        # for `[warp_n, warp_k]` weights, we load `[warp_n, warp_k / group_size]` scales
        # scale loading is parallelized in `n` dimension, that is,
        #     `num_s_lanes` in a warp load `num_s_packs` of `s_pack_size` elements, in total `warp_s` elements
        # each element in `n` dimension is 16 bit as it contains 1 fp16
        # min `s_pack_size` set to 2 element, since each lane at least holds 2 accumulator results in `n` dimension
        # max `s_pack_size` set to 128b/16b = 8 elements
        # for `warp_n = 8`, we have
        #     `s_pack_size = 2`, `num_s_lanes = 4`,  `num_s_packs = 1`
        # for `warp_n = 128`, we have
        #     `s_pack_size = 4`, `num_s_lanes = 32`, `num_s_packs = 1`
        # for `warp_n = 512`, we have
        #     `s_pack_size = 8`, `num_s_lanes = 32`, `num_s_packs = 2`
        s_pack_size = min(max(self.warp_n // self.num_lanes, 2), 8)
        num_s_lanes = min(self.num_lanes, self.warp_n // s_pack_size)
        num_s_packs = self.warp_n // (s_pack_size * num_s_lanes)
        warp_s = num_s_packs * num_s_lanes * s_pack_size
        assert warp_s == self.warp_n, "warp_n for scales should be equal to warp_n for weights."
        # `num_n_lanes = 8 (constant)` generates 8 elements consecutive in `n` dimension
        # however, they are held by 4 lanes, each lane holds 2 elements in `n` dimension
        # thus, we start from first 4 lanes, assign 2 elements to each lane, until all 8 elements are assigned
        #       we then repeat the process for the same 4 lanes, until each lane holds `s_pack_size` elements
        #       finally, we move to next 4 lanes, and repeat the process until all `num_s_lanes` lanes are assigned
        #       the process is repeated for `num_s_packs` times
        # here is an example for `warp_n = 128, s_pack_size = 4, num_s_lanes = 32, num_s_packs = 1`
        # wscales store order:
        #  0   1   8   9   <-- load by lane 0, broadcast to lane {0, 4, 8, ..., 28} (8x)
        #  2   3   10  11  <-- load by lane 1, broadcast to lane {1, 5, 9, ..., 29} (8x)
        #  4   5   12  13  <-- load by lane 2, broadcast to lane {2, 6, 10, ..., 30} (8x)
        #  6   7   14  15  <-- load by lane 3, broadcast to lane {3, 7, 11, ..., 31} (8x)
        #  16  17  24  25  <-- load by lane 4, broadcast to lane {0, 4, 8, ..., 28} (8x)
        #  ...
        #  22  23  30  31  <-- load by lane 7, broadcast to lane {3, 7, 11, ..., 31} (8x)
        #  ... ...
        #  112 113 120 121 <-- load by lane 28, broadcast to lane {0, 4, 8, ..., 28} (8x)
        #  ...
        #  118 119 126 127 <-- load by lane 31, broadcast to lane {3, 7, 11, ..., 31} (8x)
        scale = scale.reshape(n // warp_s, num_s_packs, num_s_lanes // 4, s_pack_size // 2, 4, 2, -1)
        scale = scale.permute(0, 6, 1, 2, 4, 3, 5).contiguous()
        return scale.view(-1) if group_size == -1 else scale.view(-1, n)  # the shape is just used for validation

    def pack_micro_scale(self, scale: torch.Tensor, group_size: int) -> torch.Tensor:
        assert scale.dtype in (torch.float16, torch.bfloat16), "currently nunchaku only supports fp16 and bf16."
        assert scale.max() <= 448, "scale should be less than 448."
        assert scale.min() >= -448, "scale should be greater than -448."
        assert group_size == 16, "currently only support group size 16."
        assert self.insn_k == 64, "insn_k should be 64."
        scale = scale.to(dtype=torch.float8_e4m3fn)
        n = scale.shape[0]
        assert self.warp_n >= 32, "currently only support warp_n >= 32."
        # for `[warp_n, warp_k]` weights, we load `[warp_n, warp_k / group_size]` scales
        # scale loading is parallelized in `n` dimension, that is,
        #     `num_s_lanes` in a warp load `num_s_packs` of `s_pack_size` elements, in total `warp_s` elements
        # each element in `n` dimension is 32 bit as it contains 4 fp8 in `k` dimension
        # min `s_pack_size` set to 1 element
        # max `s_pack_size` set to 128b/32b = 4 elements
        # for `warp_n = 128`, we have
        #     `s_pack_size = 4`, `num_s_lanes = 32`, `num_s_packs = 1`
        # for `warp_n = 512`, we have
        #     `s_pack_size = 8`, `num_s_lanes = 32`, `num_s_packs = 2`
        s_pack_size = min(max(self.warp_n // self.num_lanes, 1), 4)
        num_s_lanes = 4 * 8  # 32 lanes is divided into 4 pieces, each piece has 8 lanes at a stride of 4
        num_s_packs = ceil_divide(self.warp_n, s_pack_size * num_s_lanes)
        warp_s = num_s_packs * num_s_lanes * s_pack_size
        assert warp_s == self.warp_n, "warp_n for scales should be equal to warp_n for weights."
        # note: refer to https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-scaling-thread-id-b-selection
        # we start from first 8 lines at a stride of 4, assign 1 element to each lane, until all 8 elements are assigned
        #    we then move to next 8 lines at a stride of 4, and repeat the process until all 32 lanes are assigned
        # here is an example for `warp_n = 128, s_pack_size = 4, num_s_lanes = 32, num_s_packs = 1`
        # wscales store order:
        #  0   32  64  96   <-- load by lane 0
        #  8   40  72  104  <-- load by lane 1
        #  16  48  80  112  <-- load by lane 2
        #  24  56  88  120  <-- load by lane 3
        #  1   33  65  97   <-- load by lane 4
        #  ...
        #  25  57  81  113  <-- load by lane 7
        #  ...
        #  7   39  71  103  <-- load by lane 28
        #  ...
        #  31  63  95  127  <-- load by lane 31
        scale = scale.view(n // warp_s, num_s_packs, s_pack_size, 4, 8, -1, self.insn_k // group_size)
        scale = scale.permute(0, 5, 1, 4, 3, 2, 6).contiguous()
        return scale.view(-1, n)  # the shape is just used for validation

    def pack_lowrank_weight(self, weight: torch.Tensor, down: bool) -> torch.Tensor:
        """Pack Low-Rank Weight.

        Args:
            weight (`torch.Tensor`):
                low-rank weight tensor.
            down (`bool`):
                whether the weight is for down projection in low-rank branch.
        """
        assert weight.dtype in (torch.float16, torch.bfloat16), f"Unsupported weight dtype {weight.dtype}."
        reg_n, reg_k = 1, 2  # reg_n is always 1, reg_k is 32 bits // 16 bits = 2
        pack_n = self.n_pack_size * self.num_n_lanes * reg_n
        pack_k = self.k_pack_size * self.num_k_lanes * reg_k
        weight = pad(weight, divisor=(pack_n, pack_k), dim=(0, 1))
        if down:
            r, c = weight.shape
            r_packs, c_packs = r // pack_n, c // pack_k
            weight = weight.view(r_packs, pack_n, c_packs, pack_k).permute(2, 0, 1, 3)
        else:
            c, r = weight.shape
            c_packs, r_packs = c // pack_n, r // pack_k
            weight = weight.view(c_packs, pack_n, r_packs, pack_k).permute(0, 2, 1, 3)
        weight = weight.reshape(
            c_packs, r_packs, self.n_pack_size, self.num_n_lanes, reg_n, self.k_pack_size, self.num_k_lanes, reg_k
        )
        # (c_packs, r_packs, n_pack_size, num_n_lanes, reg_n, k_pack_size, num_k_lanes, reg_k)
        # =>
        # (c_packs, r_packs, num_n_lanes, num_k_lanes, n_pack_size, k_pack_size, reg_n, reg_k)
        weight = weight.permute(0, 1, 3, 6, 2, 5, 4, 7).contiguous()
        return weight.view(c, r)

    def unpack_lowrank_weight(self, weight: torch.Tensor, down: bool) -> torch.Tensor:
        """Unpack Low-Rank Weight.

        Args:
            weight (`torch.Tensor`):
                low-rank weight tensor.
            down (`bool`):
                whether the weight is for down projection in low-rank branch.
        """
        c, r = weight.shape
        assert weight.dtype in (torch.float16, torch.bfloat16), f"Unsupported weight dtype {weight.dtype}."
        reg_n, reg_k = 1, 2  # reg_n is always 1, reg_k is 32 bits // 16 bits = 2
        pack_n = self.n_pack_size * self.num_n_lanes * reg_n
        pack_k = self.k_pack_size * self.num_k_lanes * reg_k
        if down:
            r_packs, c_packs = r // pack_n, c // pack_k
        else:
            c_packs, r_packs = c // pack_n, r // pack_k
        weight = weight.view(
            c_packs, r_packs, self.num_n_lanes, self.num_k_lanes, self.n_pack_size, self.k_pack_size, reg_n, reg_k
        )
        # (c_packs, r_packs, num_n_lanes, num_k_lanes, n_pack_size, k_pack_size, reg_n, reg_k)
        # =>
        # (c_packs, r_packs, n_pack_size, num_n_lanes, reg_n, k_pack_size, num_k_lanes, reg_k)
        weight = weight.permute(0, 1, 4, 2, 6, 5, 3, 7).contiguous()
        weight = weight.view(c_packs, r_packs, pack_n, pack_k)
        if down:
            weight = weight.permute(1, 2, 0, 3).contiguous().view(r, c)
        else:
            weight = weight.permute(0, 2, 1, 3).contiguous().view(c, r)
        return weight

    def check_if_micro_scale(self, group_size: int) -> bool:
        return self.insn_k == group_size * 4

    def pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        assert weight.ndim == 2, "weight tensor should be 2D."
        return pad(weight, divisor=(self.mem_n, self.mem_k * self.num_k_unrolls), dim=(0, 1))

    def pad_scale(self, scale: torch.Tensor, group_size: int) -> torch.Tensor:
        if group_size > 0 and scale.numel() > scale.shape[0]:
            scale = scale.view(scale.shape[0], 1, -1, 1)
            if self.check_if_micro_scale(group_size=group_size):
                scale = pad(scale, divisor=(self.warp_n, self.insn_k // group_size), dim=(0, 2), fill_value=1)
            else:
                scale = pad(scale, divisor=(self.warp_n, self.num_k_unrolls), dim=(0, 2), fill_value=1)
        else:
            scale = pad(scale, divisor=self.warp_n, dim=0, fill_value=1)
        return scale

    def pad_lowrank_weight(self, weight: torch.Tensor, down: bool) -> torch.Tensor:
        assert weight.ndim == 2, "weight tensor should be 2D."
        return pad(weight, divisor=self.warp_n, dim=1 if down else 0)


def convert_to_nunchaku_w4x4y16_linear_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    smooth: torch.Tensor | None = None,
    lora: tuple[torch.Tensor, torch.Tensor] | None = None,
    float_point: bool = False,
    subscale: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor] | None,
    torch.Tensor | None,
]:
    assert weight.ndim == 2, "weight tensor should be 2D."
    device, dtype = weight.device, weight.dtype
    assert dtype in (torch.float16, torch.bfloat16), "currently nunchaku only supports fp16 and bf16."
    assert scale is not None, "scale tensor is required for quantization."

    oc, ic = weight.shape
    if scale.numel() == 1:
        scale = scale.view(-1).expand(oc).reshape(oc, 1, 1, 1)
        per_tensor_scale = True
    else:
        per_tensor_scale = False
    assert scale.ndim == 4, "scale tensor should be 4D."
    assert scale.shape[1] == scale.shape[3] == 1
    assert scale.shape[0] == oc
    ng, gs = scale.shape[2], ic // scale.shape[2]
    assert ic == gs * ng, "input channel size should be equal to group size times number of groups."
    if subscale is not None:
        assert subscale.ndim == 4, "subscale tensor should be 4D."
        assert subscale.shape[1] == subscale.shape[3] == 1
        assert subscale.shape[0] == oc
        nsg, sgs = subscale.shape[2], ic // subscale.shape[2]
        assert ic == sgs * nsg, "input channel size should be equal to subgroup size times number of subgroups."
        assert gs > sgs and gs % sgs == 0, "group size should be divisible by subgroup size."
    else:
        nsg, sgs = ng, gs
    # region quantize and pack weight tensor
    weight = weight.to(dtype=torch.float32).view(oc, 1, ng, gs).div_(scale.to(dtype=torch.float32, device=device))
    if subscale is not None:
        weight = weight.view(oc, 1, nsg, sgs).div_(subscale.to(dtype=torch.float32, device=device))
    weight = weight.view(oc, ic)
    if float_point:
        weight = fp_quantize(weight)
        assert weight.min() >= 0 and weight.max() <= 15, "quantized weight should be in [0, 15]."
    else:
        weight = weight.round_()
        assert weight.min() >= -8 and weight.max() <= 7, "quantized weight should be in [-8, 7]."
    # endregion
    bias = torch.zeros([oc, 1], dtype=dtype, device=device) if bias is None else bias.view(-1, 1)
    smooth = torch.ones([ic, 1], dtype=dtype, device=device) if smooth is None else smooth.view(-1, 1)

    packer = NunchakuWeightPacker(bits=4)
    weight = packer.pad_weight(weight.to(dtype=torch.int32))
    scale = packer.pad_scale(scale.to(dtype=dtype), group_size=gs)
    if subscale is not None:
        subscale = packer.pad_scale(subscale.to(dtype=dtype), group_size=sgs)
    bias = packer.pad_scale(bias.to(dtype=dtype), group_size=-1)
    smooth = packer.pad_scale(smooth.to(dtype=dtype), group_size=-1)

    weight = packer.pack_weight(weight)
    scale = packer.pack_scale(scale, group_size=gs if gs < ic else -1)
    if subscale is not None:
        subscale = packer.pack_scale(subscale, group_size=sgs if sgs < ic else -1)
    bias = packer.pack_scale(bias, group_size=-1)
    smooth = packer.pack_scale(smooth, group_size=-1)
    if lora is not None:
        lora_down = packer.pack_lowrank_weight(packer.pad_lowrank_weight(lora[0], down=True), down=True)
        lora_up = packer.pack_lowrank_weight(packer.pad_lowrank_weight(lora[1], down=False), down=False)
        lora = (lora_down, lora_up)
    if per_tensor_scale:
        scale = scale.view(-1)[0].view([1])
    return weight, scale, bias, smooth, lora, subscale


def convert_to_nunchaku_w8x8y16_linear_weight(
    weight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert weight.ndim == 2, "weight tensor should be 2D."
    device, dtype = weight.device, weight.dtype
    assert dtype in (torch.float16, torch.bfloat16), "currently nunchaku only supports fp16 and bf16."
    assert scale is not None, "scale tensor is required for quantization."
    oc, ic = weight.shape
    if scale.numel() == 1:
        scale = scale.view(-1).expand(oc)
    scale = scale.reshape(oc, 1)
    weight = weight.to(dtype=torch.float32)
    weight = weight.div_(scale.to(dtype=torch.float32, device=device)).round_().to(torch.int32).view(oc, ic)
    assert weight.min() >= -128 and weight.max() <= 127, "quantized weight should be in [-128, 127]."
    # endregion
    bias = torch.zeros([oc, 1], dtype=dtype, device=device) if bias is None else bias.view(-1, 1)
    packer = NunchakuWeightPacker(bits=8)
    weight = packer.pack_weight(packer.pad_weight(weight))
    scale = packer.pack_scale(packer.pad_scale(scale.to(dtype=dtype), group_size=-1), group_size=-1)
    bias = packer.pack_scale(packer.pad_scale(bias.to(dtype=dtype), group_size=-1), group_size=-1).view(-1)
    return weight, scale, bias


def convert_to_nunchaku_w4x16_linear_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    adanorm_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    oc, ic = weight.shape
    assert scale.ndim == 4, "scale tensor should be 4D."
    assert scale.shape[0] == oc
    assert scale.shape[1] == scale.shape[3] == 1
    ng = scale.shape[2]
    if bias is None:
        bias = torch.zeros([oc], dtype=weight.dtype, device=weight.device)
    assert oc % adanorm_splits == 0, "output channel size should be divisible by splits."
    if adanorm_splits > 1:
        weight = weight.view(adanorm_splits, oc // adanorm_splits, ic).transpose(0, 1).reshape(oc, ic)
        scale = scale.view(adanorm_splits, oc // adanorm_splits, ng).transpose(0, 1).reshape(oc, 1, ng, 1)
        bias = bias.reshape(adanorm_splits, oc // adanorm_splits).transpose(0, 1)
        delta = [0] * adanorm_splits
        delta[1] = delta[-2] = 1
        bias = bias.add_(torch.tensor(delta, dtype=bias.dtype, device=bias.device))
        bias = bias.reshape(oc)
    weight, scale, zero = convert_to_tinychat_w4x16y16_linear_weight(
        weight=weight, scale=scale, zero=torch.full_like(scale, 7) if zero is None else zero, zero_pre_scaled=True
    )
    weight = weight.view(torch.int32)
    return weight, scale, zero, bias
