# -*- coding: utf-8 -*-
"""Quantization data type."""

import typing as tp

import torch

from .codebook import Codebook

__all__ = ["QuantDataType", "QDType"]


class QuantDataType:
    """Quantization data type."""

    _registered: tp.ClassVar[dict[str, "QuantDataType"]] = {}

    def __init__(
        self,
        total_bits: int,
        *,
        signed: bool = True,
        exponent_bits: int = 0,
        has_subnormal: bool = True,
        has_nan: bool = False,
        has_inf: bool = False,
        magnitude: bool = False,
        codebook: Codebook | None = None,
        codebook_name: str = "",
    ):
        """Initialize the quantization data type.

        Args:
            total_bits (`int`):
                Total number of bits. Must be greater than 0.
            signed (`bool`, *optional*, defaults to `True`):
                Whether the data type is signed.
            exponent_bits (`int`, *optional*, defaults to `0`):
                Number of bits for the exponent.
            has_subnormal (`bool`, *optional*, defaults to `True`):
                Whether the data type has subnormal.
            has_nan (`bool`, *optional*, defaults to `False`):
                Whether the data type has NaN if it is float-point.
            has_inf (`bool`, *optional*, defaults to `False`):
                Whether the data type has Inf if it is float-point.
            magnitude (`bool`, *optional*, defaults to `False`):
                Whether the data type is magnitude-based if it is integer.
            codebook (`Codebook` or `None`, *optional*, defaults to `None`):
                Codebook for the data type.
            codebook_name (`str`, *optional*, defaults to `""`):
                Name of the codebook. Must be specified if `codebook` is not `None`.
        """
        self.__signed = signed
        # region set bit widths
        self.__total_bits = total_bits
        self.__exponent_bits = exponent_bits
        assert self.__total_bits > 0, "Total bits must be greater than 0."
        assert self.__exponent_bits >= 0, "Exponent bits must be non-negative."
        self.__mantissa_bits = self.__total_bits - self.__exponent_bits - int(self.__signed)
        # endregion
        # region set data type properties
        if self.__exponent_bits > 0:
            # for floating-point data type
            self.__has_subnormal = has_subnormal
            self.__has_inf = has_inf
            self.__has_nan = has_inf or has_nan
            self.__magnitude = True
            if self.__mantissa_bits == 0:
                assert not self.__has_inf, "Inf is not supported for exponent-only floating-point data type."
                if self.__exponent_bits == 1:
                    assert not self.__has_nan, "NaN is not supported for 1-bit exponent-only floating-point data type."
        else:
            # for integer data type
            self.__has_subnormal = False
            self.__has_inf = False
            self.__has_nan = False
            self.__magnitude = magnitude
        # endregion
        # region set codebook
        if codebook is not None:
            assert self.is_float_point, "Codebook is only supported for floating-point data type."
            self.__codebook = codebook
            assert codebook_name, "Codebook name must be specified."
            self.__codebook_name = codebook_name
            assert self.max_value >= 0, "Max value must be non-negative."
            self.__name = self.__codebook_name
            if self.__name not in QuantDataType._registered:
                QuantDataType._registered[self.__name] = self
            else:
                _registered = QuantDataType._registered[self.__name]
                assert _registered.total_bits == self.total_bits, "Total bits must be the same as the registered one."
                assert _registered.exponent_bits == self.exponent_bits, (
                    "Exponent bits must be the same as the registered one."
                )
                assert _registered.signed == self.signed, "Signed must be the same as the registered one."
                assert _registered.has_subnormal == self.has_subnormal, (
                    "Subnormal must be the same as the registered one."
                )
                assert _registered.has_inf == self.has_inf, "Inf must be the same as the registered one."
                assert _registered.has_nan == self.has_nan, "NaN must be the same as the registered one."
                assert _registered.magnitude == self.magnitude, "Magnitude must be the same as the registered one."
                assert _registered.__codebook is not None, "Codebook must be the same as the registered one."
                assert torch.allclose(_registered.__codebook.values, self.__codebook.values), (
                    "Codebook values must be the same as the registered one."
                )
        else:
            self.__codebook = None
            self.__codebook_name = ""
            self.__name = self._build_default_name()
            if self.__name not in QuantDataType._registered:
                QuantDataType._registered[self.__name] = self
        # endregion
        # region set codebooks
        self.__codebooks: dict[tuple[torch.device, torch.dtype], Codebook] = {}
        # endregion

    # region properties
    @property
    def name(self) -> str:
        """Name of the data type."""
        return self.__name

    @property
    def codebook_name(self) -> str:
        """Name of the codebook."""
        return self.__codebook_name

    @property
    def signed(self) -> bool:
        """Whether the data type is signed."""
        return self.__signed

    @property
    def unsigned(self) -> bool:
        """Whether the data type is unsigned."""
        return not self.__signed

    @property
    def total_bits(self) -> int:
        """Total number of bits."""
        return self.__total_bits

    @property
    def exponent_bits(self) -> int:
        """Number of bits for the exponent."""
        return self.__exponent_bits

    @property
    def mantissa_bits(self) -> int:
        """Number of bits for the mantissa."""
        return self.__mantissa_bits

    @property
    def has_subnormal(self) -> bool:
        """Whether the data type has subnormal."""
        return self.__has_subnormal

    @property
    def has_inf(self) -> bool:
        """Whether the data type has Inf."""
        return self.__has_inf

    @property
    def has_nan(self) -> bool:
        """Whether the data type has NaN."""
        return self.__has_nan

    @property
    def magnitude(self) -> bool:
        """Whether the data type is magnitude-based."""
        return self.__magnitude

    @property
    def is_float_point(self) -> bool:
        """Whether the data type is floating-point."""
        return self.exponent_bits > 0

    @property
    def is_integer(self) -> bool:
        """Whether the data type is integer."""
        return self.exponent_bits == 0

    @property
    def is_exponent(self) -> bool:
        """Whether the data type is exponent-only floating-point."""
        return self.exponent_bits > 0 and self.mantissa_bits == 0 and not self.has_subnormal

    @property
    def exponent_mask(self) -> int:
        """Bit mask for the exponent."""
        return ((1 << self.exponent_bits) - 1) << self.mantissa_bits

    @property
    def mantissa_mask(self) -> int:
        """Bit mask for the mantissa."""
        return (1 << self.mantissa_bits) - 1

    @property
    def _end_mantissa(self) -> int:
        return 2**self.mantissa_bits

    @property
    def _end_exponent(self) -> int:
        if self.mantissa_bits > 0:
            return 2**self.exponent_bits - int(self.has_inf)
        else:
            return 2**self.exponent_bits - int(self.has_nan)

    @property
    def exponent_bias(self) -> int:
        """Exponent bias."""
        if self.is_float_point:
            return 2 ** (self.exponent_bits - 1) - 1
        else:
            return 0

    @property
    def max_exponent_value(self) -> int:
        """Maximum exponent value."""
        if self.is_float_point:
            return self._end_exponent - 1 - self.exponent_bias
        else:
            return self.total_bits - 1 - int(self.signed)

    @property
    def min_exponent_value(self) -> int:
        """Minimum exponent value."""
        if self.is_float_point:
            return int(self.has_subnormal) - self.exponent_bias
        else:
            return 0

    @property
    def max_positive_normal_value(self) -> float:
        """Maximum positive normal value."""
        if self.is_float_point:
            if self.mantissa_bits > 0 and not self.has_inf and self.has_nan:
                base_value = 2 - 2 / self._end_mantissa
            else:
                base_value = 2 - 1 / self._end_mantissa
            return base_value * 2**self.max_exponent_value
        else:
            return self._end_mantissa - 1

    @property
    def min_positive_normal_value(self) -> float:
        """Minimum positive normal value."""
        return 2**self.min_exponent_value

    @property
    def max_positive_subnormal(self) -> float:
        """Maximum positive subnormal value."""
        if self.is_float_point and self.has_subnormal and self.mantissa_bits > 0:
            b = 1 - 1 / self._end_mantissa
            e = 1 - self.exponent_bias
            return b * 2**e
        else:
            return 0

    @property
    def min_positive_subnormal(self) -> float:
        """Minimum non-negative subnormal value."""
        if self.is_float_point and self.has_subnormal and self.mantissa_bits > 0:
            b = 1 / self._end_mantissa
            e = 1 - self.exponent_bias
            return b * 2**e
        else:
            return 0

    @property
    def max_value(self) -> float:
        """Maximum value."""
        return self.max_positive_normal_value if self.__codebook is None else self.__codebook.values[-1].item()

    @property
    def min_value(self) -> float:
        """Minimum value."""
        if self.__codebook is not None:
            return self.__codebook.values[0].item()
        if self.signed:
            if self.magnitude:
                return -self.max_value
            else:
                return -self.max_value - 1
        else:
            return 0

    # endregion

    def to_unsigned(self) -> "QuantDataType":
        """Get an unsigned version of the data type.

        Returns:
            `QuantDataType`:
                The unsigned version of the data type.
        """
        return QuantDataType.from_str("u" + self.name[1:])

    def get_codebook(self, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> Codebook:
        """Get a get_codebook of `code_bits` bits for the quantization.

        Args:
            device (`torch.device` or `str`, *optional*, defaults to `"cpu"`):
                Device to create the codebook on.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type to create the codebook with.

        Returns:
            `Codebook`:
                Codebook with the specified device and dtype.
        """
        device = torch.device("cpu") if device is None else torch.device(device)
        key = (device, dtype)
        if key not in self.__codebooks:
            if self.__codebook is not None:
                self.__codebooks[key] = self.__codebook.to(device=device, dtype=dtype)
            else:
                self.__codebook = self._build_codebook(device=device, dtype=dtype)
                self.__codebooks[key] = self.__codebook
        return self.__codebooks[key]

    def round(self, tensor: torch.Tensor) -> torch.Tensor:
        """Round the tensor to the nearest quantized value.

        Args:
            tensor (`torch.Tensor`):
                Tensor to round.

        Returns:
            `torch.Tensor`:
                Rounded tensor.
        """
        if self.is_integer:
            return tensor.round()
        else:
            return self.get_codebook(device=tensor.device).round(tensor)

    @classmethod
    def from_str(cls, s: str, /) -> "QuantDataType":
        """Create a QuantDataType from a string."""
        if s not in cls._registered:
            cls._registered[s] = cls._default_from_str(s)
        return cls._registered[s]

    def _build_codebook(self, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> Codebook:
        if self.is_float_point:
            return Codebook.build_for_float_point(
                total_bits=self.total_bits,
                exponent_bits=self.exponent_bits,
                signed=self.signed,
                has_subnormal=self.has_subnormal,
                has_inf=self.has_inf,
                has_nan=self.has_nan,
                device=device,
                dtype=dtype,
            )
        else:
            return Codebook.build_for_integer(
                total_bits=self.total_bits, signed=self.signed, magnitude=self.magnitude, device=device, dtype=dtype
            )

    def _build_default_name(self) -> str:
        s = "s" if self.signed else "u"
        if self.is_float_point:
            if self.has_subnormal or self.mantissa_bits > 0:
                s += "fp" if self.has_subnormal else "fn"
                s += f"{self.total_bits}_e{self.exponent_bits}m{self.mantissa_bits}"
                s += "_inf" if self.has_inf else ("_nan" if self.has_nan else "_all")
            else:
                assert not self.has_subnormal, "Subnormal is not supported for exponent-only floating-point data type."
                assert not self.has_inf, "Inf is not supported for exponent-only floating-point data type."
                s += f"exp{self.exponent_bits}"
                s += "_nan" if self.has_nan else "_all"
        else:
            s += "mag" if self.magnitude else "int"
            s += f"{self.total_bits}"
        return s

    @staticmethod
    def _default_from_str(s: str, /) -> "QuantDataType":
        s = s.strip().lower()
        signed = s[0] == "s"
        s = s[1:]
        if s.startswith("int"):
            return QuantDataType(int(s[3:]), signed=signed)
        elif s.startswith("mag"):
            return QuantDataType(int(s[3:]), signed=signed, magnitude=True)
        elif s.startswith("exp"):
            ss = s.split("_")
            total_bits = int(ss[0][3:])
            if len(ss) >= 2:
                has_nan = ss[1] == "nan"
            else:
                has_nan = False
            return QuantDataType(
                total_bits=total_bits,
                signed=signed,
                exponent_bits=total_bits - int(signed),
                has_subnormal=False,
                has_nan=has_nan,
            )
        elif s.startswith("f"):
            ss = s.split("_")
            has_subnormal = s[1] == "p"
            total_bits = int(ss[0][2:])
            exponent_bits = int(ss[1][1 : ss[1].find("m")])
            if len(ss) >= 3:
                has_inf = ss[2] == "inf"
                has_nan = has_inf or (ss[2] == "nan")
            else:
                has_inf, has_nan = False, False
            return QuantDataType(
                total_bits=total_bits,
                signed=signed,
                exponent_bits=exponent_bits,
                has_subnormal=has_subnormal,
                has_inf=has_inf,
                has_nan=has_nan,
            )
        else:
            raise ValueError(f"Unknown QuantDataType {s}")

    def __str__(self) -> str:
        return self.__name

    def __repr__(self) -> str:
        return self.__name

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, QuantDataType):
            return False
        return self.name == value.name

    def __hash__(self) -> int:
        return hash(self.name)


class _QDTypeMeta(type):
    def __getattr__(cls, __name: str) -> tp.Any:
        if __name.startswith("_"):
            return getattr(super(), __name)
        else:
            return QuantDataType.from_str(__name)


class QDType(metaclass=_QDTypeMeta):
    """QuantDataType class for easy access to QuantDataType by name."""

    pass
