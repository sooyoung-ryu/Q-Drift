# -*- coding: utf-8 -*-
"""Backend utilities."""

import typing as tp

import safetensors
import torch

__all__ = ["ceil_divide", "pad", "fp_quantize", "MmaWeightPackerBase"]


def ceil_divide(x: int, divisor: int) -> int:
    """Ceiling division.

    Args:
        x (`int`):
            dividend.
        divisor (`int`):
            divisor.

    Returns:
        `int`:
            ceiling division result.
    """
    return (x + divisor - 1) // divisor


def pad(
    tensor: tp.Optional[torch.Tensor],
    divisor: int | tp.Sequence[int],
    dim: int | tp.Sequence[int],
    fill_value: float | int = 0,
) -> torch.Tensor:
    if isinstance(divisor, int):
        if divisor <= 1:
            return tensor
    elif all(d <= 1 for d in divisor):
        return tensor
    if tensor is None:
        return None
    shape = list(tensor.shape)
    if isinstance(dim, int):
        assert isinstance(divisor, int)
        shape[dim] = ceil_divide(shape[dim], divisor) * divisor
    else:
        if isinstance(divisor, int):
            divisor = [divisor] * len(dim)
        for d, div in zip(dim, divisor, strict=True):
            shape[d] = ceil_divide(shape[d], div) * div
    result = torch.full(shape, fill_value, dtype=tensor.dtype, device=tensor.device)
    result[[slice(0, extent) for extent in tensor.shape]] = tensor
    return result


def load_state_dict_in_safetensors(
    path: str, device: str | torch.device = "cpu", filter_prefix: str = ""
) -> dict[str, torch.Tensor]:
    """Load state dict in SafeTensors.

    Args:
        path (`str`):
            file path.
        device (`str` | `torch.device`, optional, defaults to `"cpu"`):
            device.
        filter_prefix (`str`, optional, defaults to `""`):
            filter prefix.

    Returns:
        `dict`:
            loaded SafeTensors.
    """
    state_dict = {}
    with safetensors.safe_open(path, framework="pt", device=device) as f:
        for k in f.keys():
            if filter_prefix and not k.startswith(filter_prefix):
                continue
            state_dict[k.removeprefix(filter_prefix)] = f.get_tensor(k)
    return state_dict


def fp_quantize(x: torch.Tensor, codebook: torch.Tensor | None = None) -> torch.Tensor:
    if codebook is None:
        codebook = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=x.dtype,
            device=x.device,
        )
    return (x.unsqueeze(-1) - codebook.unsqueeze(0)).abs().argmin(dim=-1)


class MmaWeightPackerBase:
    def __init__(self, bits: int, warp_n: int, comp_n: int = None, comp_k: int = None):
        self.bits = bits
        assert self.bits in (1, 4, 8, 16, 32), "weight bits should be 1, 4, 8, 16, or 32."

        # region compute tile size
        self.comp_n = comp_n if comp_n is not None else 16
        """smallest tile size in `n` dimension for MMA computation."""
        self.comp_k = comp_k if comp_k is not None else 256 // self.bits
        """smallest tile size in `k` dimension for MMA computation."""
        # the smallest MMA computation may contain several MMA instructions
        self.insn_n = 8  # mma instruction tile size in `n` dimension
        """tile size in `n` dimension for MMA instruction."""
        self.insn_k = self.comp_k
        """tile size in `k` dimension for MMA instruction."""
        assert self.insn_k * self.bits in (128, 256), (
            f"insn_k ({self.insn_k}) * bits ({self.bits}) should be 128 or 256."
        )
        assert self.comp_n % self.insn_n == 0, f"comp_n ({self.comp_n}) should be divisible by insn_n ({self.insn_n})."
        self.num_lanes = 32
        """there are 32 lanes (or threds) in a warp."""
        self.num_k_lanes = 4
        self.num_n_lanes = 8
        assert warp_n >= self.comp_n and warp_n % self.comp_n == 0, (
            f"warp_n ({warp_n}) should be divisible by comp_n({self.comp_n})."
        )
        self.warp_n = warp_n
        # endregion
        # region memory
        self.reg_k = 32 // self.bits
        """number of elements in a register in `k` dimension."""
        self.reg_n = 1
        """number of elements in a register in `n` dimension (always 1)."""
        self.k_pack_size = self.comp_k // (self.num_k_lanes * self.reg_k)
        """number of elements in a pack in `k` dimension."""
        self.n_pack_size = self.comp_n // (self.num_n_lanes * self.reg_n)
        """number of elements in a pack in `n` dimension."""
        self.pack_size = self.k_pack_size * self.n_pack_size
        """number of elements in a pack accessed by a lane at a time."""
        assert 1 <= self.pack_size <= 4, "pack size should be less than or equal to 4."
        assert self.k_pack_size * self.num_k_lanes * self.reg_k == self.comp_k
        assert self.n_pack_size * self.num_n_lanes * self.reg_n == self.comp_n
        self.mem_k = self.comp_k
        """the tile size in `k` dimension for one tensor memory access."""
        self.mem_n = warp_n
        """the tile size in `n` dimension for one tensor memory access."""
        self.num_k_packs = self.mem_k // (self.k_pack_size * self.num_k_lanes * self.reg_k)
        """number of packs in `k` dimension for one tensor memory access."""
        self.num_n_packs = self.mem_n // (self.n_pack_size * self.num_n_lanes * self.reg_n)
        """number of packs in `n` dimension for one tensor memory access."""
        # endregion

    def get_view_shape(self, n: int, k: int) -> tuple[int, int, int, int, int, int, int, int, int, int]:
        assert n % self.mem_n == 0, "output channel size should be divisible by mem_n."
        assert k % self.mem_k == 0, "input channel size should be divisible by mem_k."
        return (
            n // self.mem_n,
            self.num_n_packs,
            self.n_pack_size,
            self.num_n_lanes,
            self.reg_n,
            k // self.mem_k,
            self.num_k_packs,
            self.k_pack_size,
            self.num_k_lanes,
            self.reg_k,
        )
