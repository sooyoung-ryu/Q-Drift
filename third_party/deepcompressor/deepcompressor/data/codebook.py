# -*- coding: utf-8 -*-
"""Codebook for quantization."""

from dataclasses import dataclass

import torch

from deepcompressor.csrc.load import _C

__all__ = ["Codebook"]


@dataclass
class Codebook:
    """A codebook for quantization.

    Attributes:
        size (`int`):
            Number of values in the codebook.
        bits (`int`):
            Number of bits for the binary code.
        values (`torch.FloatTensor`):
            A value book in ascending order.
        codes (`torch.ByteTensor`):
            A binary book containing the binary representation of the value.
    """

    size: int
    bits: int
    values: torch.Tensor
    codes: torch.Tensor

    def __post_init__(self):
        assert self.size <= self.values.numel(), "Codebook size is larger than the values size"
        assert self.values.shape == self.codes.shape, "Values and Codes must have the same shape"

    def round(self, tensor: torch.Tensor) -> torch.Tensor:
        """Round the tensor to the nearest value in the codebook.

        Args:
            tensor (`torch.Tensor`):
                A tensor to round.

        Returns:
            `torch.Tensor`:
                A rounded tensor.
        """
        dtype = tensor.dtype
        tensor = tensor.to(self.values.dtype).contiguous()
        return _C.round_to_nearest_in_codebook_cuda(tensor, self.values).to(dtype=dtype)

    def to(self, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> "Codebook":
        """Move the codebook to the specified device and dtype.

        Args:
            device (`torch.device`):
                Device to move the codebook.
            dtype (`torch.dtype`):
                Dtype to move the codebook.

        Returns:
            `Codebook`:
                A codebook.
        """
        device = device if device is not None else self.values.device
        dtype = dtype if dtype is not None else self.values.dtype
        return Codebook(
            size=self.size,
            bits=self.bits,
            values=self.values.to(device=device, dtype=dtype),
            codes=self.codes.to(device=device),
        )

    @staticmethod
    def construct(
        maps: list[tuple[float, int]],
        *,
        bits: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "Codebook":
        """Create a map of values to a code of `code_bits` bits.

        Args:
            maps (`list[tuple[float, int]]`):
                A list of tuples of (value, binary code).
            bits (`int`):
                Number of bits for the binary code.
            device (`torch.device` or str, *optional*, defaults to `"cpu"`):
                Device to put the codebook and binarybook on.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Dtype of the codebook.

        Returns:
            `Codebook`:
                A codebook.
        """
        if bits > 8:
            raise NotImplementedError("Codebook with more than 8 bits is not supported")
        assert len(maps) <= 2**bits, "Too many (value, code) maps for the code bits"
        size = len(maps)
        maps.sort(key=lambda x: x[0])
        values = torch.tensor([v[0] for v in maps], device=device, dtype=dtype)
        codes = torch.tensor(
            [v[1] for v in maps],
            dtype=torch.uint8 if bits <= 8 else (torch.int16 if bits < 16 else torch.int32),
            device=device,
        )
        return Codebook(size=size, bits=bits, values=values, codes=codes)

    @staticmethod
    def build_for_float_point(
        *,
        total_bits: int,
        exponent_bits: int,
        signed: bool = True,
        has_subnormal: bool = True,
        has_inf: bool = False,
        has_nan: bool = False,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "Codebook":
        """Create a map of floating point values to a code of `code_bits` bits.

        Args:
            total_bits (`int`):
                Number of bits for the floating point value.
            exponent_bits (`int`):
                Number of bits for the exponent.
            signed (`bool`, *optional*, defaults to `True`):
                Whether to use signed code.
            has_inf (`bool`, *optional*, defaults to `False`):
                Whether to include infinity.
            has_nan (`bool`, *optional*, defaults to `False`):
                Whether to include NaN.
            device (`torch.device` or str, *optional*, defaults to `"cpu"`):
                Device to put the codebook on.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Dtype of the codebook.

        Returns:
            `list[Codebook]`:
                A list of codebooks.
        """
        mantissa_bits = total_bits - exponent_bits - int(signed)
        assert exponent_bits > 0, "Exponent bits must be positive"
        assert mantissa_bits >= 0, "Mantissa bits must be non-negative"
        has_nan = has_inf or has_nan

        sign_mask = 1 << (total_bits - 1)
        if mantissa_bits > 0:
            end_evalue = 2**exponent_bits - int(has_inf)
        else:
            end_evalue = 2**exponent_bits - int(has_nan)
        end_mvalue = 2**mantissa_bits
        bias = 2 ** (exponent_bits - 1) - 1
        maps, code = [], 0
        for evalue in range(end_evalue):
            for mvalue in range(end_mvalue):
                if evalue == 0 and has_subnormal:
                    value = (mvalue / end_mvalue) * (2 ** (1 - bias))
                else:
                    value = (1 + mvalue / end_mvalue) * (2 ** (evalue - bias))
                maps.append((value, code))
                if signed:
                    maps.append((-value, code | sign_mask))
                code += 1
        if mantissa_bits > 0 and not has_inf and has_nan:
            maps = maps[: -(1 + int(signed))]
        return Codebook.construct(maps, bits=total_bits, device=device, dtype=dtype)

    @staticmethod
    def build_for_integer(
        *,
        total_bits: int,
        signed: bool = True,
        magnitude: bool = False,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "Codebook":
        """Create a map of integer values to a code of `code_bits` bits.

        Args:
            total_bits (`int`):
                Number of bits for the integer value.
            signed (`bool`, *optional*, defaults to `True`):
                Whether to use signed code.
            magnitude (`bool`, *optional*, defaults to `False`):
                Whether to use magnitude-based integer.
            device (`torch.device` or `str`, *optional*, defaults to `"cpu"`):
                Device to put the codebook on.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Dtype of the codebook.

        Returns:
            `list[Codebook]`:
                A list of codebooks.
        """
        if signed:
            end_value = 2 ** (total_bits - 1)
            min_value = -end_value + int(magnitude)
        else:
            end_value = 2**total_bits
            min_value = 0
        maps = []
        for value in range(min_value, end_value):
            if value >= 0:
                code = value
            elif magnitude:
                code = end_value - value
            else:
                code = end_value + end_value + value
            maps.append((value, code))
        return Codebook.construct(maps, bits=total_bits, device=device, dtype=dtype)
