# -*- coding: utf-8 -*-
"""Quantizatizer config."""

from dataclasses import dataclass, field

import torch
from omniconfig import configclass

from deepcompressor.calib.config import SkipBasedDynamicRangeCalibConfig, SkipBasedQuantLowRankCalibConfig
from deepcompressor.data.dtype import QuantDataType
from deepcompressor.quantizer.config import QuantizerConfig
from deepcompressor.quantizer.kernel import QuantGptqConfig
from deepcompressor.utils.config import EnableConfig, IncludeBasedConfig, SkipBasedConfig

__all__ = [
    "DiffusionQuantizerConfig",
    "DiffusionWeightQuantizerConfig",
    "DiffusionActivationQuantizerConfig",
    "DiffusionModuleQuantizerConfig",
]


@configclass
@dataclass
class DiffusionGPTQConfig(SkipBasedConfig, QuantGptqConfig):
    """Configuration for GPTQ quantization.

    Args:
        damp_percentage (`float`, *optional*, defaults to `0.01`):
            The percentage of damping.
        block_size (`int`, *optional*, defaults to `128`):
            The block size of the GPTQ quantization.
        num_inv_tries (`int`, *optional*, defaults to `200`):
            The number of tries for the inverse.
        hessian_block_size (`int`, *optional*, defaults to `-1`):
            The block size when calculing the Hessian.
        skips: list[str] = field(default_factory=list)
    """

    pass


@configclass
@dataclass
class DiffusionQuantizerConfig(QuantizerConfig):
    """Diffusion model quantizer configuration.

    Args:
        dtype (`QuantDataType` or `None`, *optional*, defaults to `None`):
            The quantization data type.
        zero_point (`ZeroPointDomain` or `None`, *optional*, defaults to `None`):
            The zero-point domain.
        group_shapes (`Sequence[Sequence[int]]`, *optional*, defaults to `((-1, -1, -1),)`):
            The shapes for per-group quantization.
        scale_dtypes (`Sequence[torch.dtype | QuantDataType | None]`, *optional*, defaults to `(None,)`):
            The quantization scale data type for per-group quantization.
        static (`bool`, *optional*, defaults to `False`):
            Whether to use static quantization.
        kernel_gptq (`DiffusionGPTQConfig` or `None`, *optional*, defaults to `None`):
            The gptq quantization configuration.
        low_rank (`SkipBasedQuantLowRankCalibConfig` or `None`, *optional*, defaults to `None`):
            The quantization low-rank branch calibration configuration.
        calib_range (`DynamicRangeCalibConfig` or `None`, *optional*, defaults to `None`):
            The quantizatizer dynamic range calibration configuration.
    """

    static: bool = False
    kernel_gptq: DiffusionGPTQConfig | None = None
    low_rank: SkipBasedQuantLowRankCalibConfig | None = None
    calib_range: SkipBasedDynamicRangeCalibConfig | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.quant_dtype is None:
            self.static = False
            self.kernel_gptq = None
            self.low_rank = None
            self.calib_range = None
        if self.kernel_gptq is not None and not self.kernel_gptq.is_enabled():
            self.kernel_gptq = None
        if self.static and self.calib_range is None:
            self.calib_range = SkipBasedDynamicRangeCalibConfig()
        if self.low_rank is not None and not self.low_rank.is_enabled():
            self.low_rank = None

    @property
    def enabled_gptq(self) -> bool:
        """Whether quantization kernel calibration is enabled."""
        return self.kernel_gptq is not None and self.kernel_gptq.is_enabled()

    @property
    def enabled_low_rank(self) -> bool:
        """Whether quantization SVD calibration is enabled."""
        return self.low_rank is not None and self.low_rank.is_enabled()

    @property
    def enabled_calib_range(self) -> bool:
        """Whether quantization dynamic range calibration is enabled."""
        return self.calib_range is not None

    def generate_calib_dirname(self) -> str:
        """Generate the name for quantization calibration.

        Returns:
            str: The name.
        """
        name = ""
        if self.static:
            name += ".static"
        if self.enabled_gptq:
            name += ".gptq"
        if self.enabled_low_rank:
            name += ".lowrank"
        if self.enabled_calib_range and (self.calib_range.needs_search or self.calib_range.ratio != 1):
            name += ".range"
        return name[1:] if name else ""


@configclass
@dataclass
class SkipBasedDiffusionQuantizerConfig(SkipBasedConfig, DiffusionQuantizerConfig):
    """Diffusion model quantizer configuration.

    Args:
        dtype (`QuantDataType` or `None`, *optional*, defaults to `None`):
            The quantization data type.
        zero_point (`ZeroPointDomain` or `None`, *optional*, defaults to `None`):
            The zero-point domain.
        group_shapes (`Sequence[Sequence[int]]`, *optional*, defaults to `((-1, -1, -1),)`):
            The shapes for per-group quantization.
        scale_dtypes (`Sequence[torch.dtype | QuantDataType | None]`, *optional*, defaults to `(None,)`):
            The quantization scale data type for per-group quantization.
        skips (`[str]`, *optional*, defaults to `[]`):
            The keys of the modules to skip.
        static (`bool`, *optional*, defaults to `False`):
            Whether to use static quantization.
        kernel_gptq (`DiffusionGPTQConfig` or `None`, *optional*, defaults to `None`):
            The gptq quantization configuration.
        low_rank (`SkipBasedQuantLowRankCalibConfig` or `None`, *optional*, defaults to `None`):
            The quantization low-rank branch calibration configuration.
        calib_range (`DynamicRangeCalibConfig` or `None`, *optional*, defaults to `None`):
            The quantizatizer dynamic range calibration configuration.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.quant_dtype is None:
            self.skips.clear()


@configclass
@dataclass
class DiffusionWeightQuantizerConfig(SkipBasedDiffusionQuantizerConfig):
    """Diffusion model weight quantizer configuration.

    Args:
        dtype (`QuantDataType` or `None`, *optional*, defaults to `None`):
            The quantization data type.
        zero_point (`ZeroPointDomain` or `None`, *optional*, defaults to `None`):
            The zero-point domain.
        group_shapes (`Sequence[Sequence[int]]`, *optional*, defaults to `((-1, -1, -1),)`):
            The shapes for per-group quantization.
        scale_dtypes (`Sequence[torch.dtype | QuantDataType | None]`, *optional*, defaults to `(None,)`):
            The quantization scale data type for per-group quantization.
        skips (`list[str]`, *optional*, defaults to `[]`):
            The keys of the modules to skip.
        low_rank (`SkipBasedQuantLowRankCalibConfig` or `None`, *optional*, defaults to `None`):
            The quantization low-rank branch calibration configuration.
        calib_range (`DynamicRangeCalibConfig` or `None`, *optional*, defaults to `None`):
            The quantizatizer dynamic range calibration configuration.
    """

    static: bool = field(init=False, default=True)

    @property
    def needs_calib_data(self) -> bool:
        return self.enabled_calib_range and self.calib_range.needs_search


@configclass
@dataclass
class DiffusionActivationQuantizerConfig(SkipBasedDiffusionQuantizerConfig):
    """Diffusion model activation quantizer configuration.

    Args:
        dtype (`QuantDataType` or `None`, *optional*, defaults to `None`):
            The quantization data type.
        zero_point (`ZeroPointDomain` or `None`, *optional*, defaults to `None`):
            The zero-point domain.
        group_shapes (`Sequence[Sequence[int]]`, *optional*, defaults to `((-1, -1, -1),)`):
            The shapes for per-group quantization.
        scale_dtypes (`Sequence[torch.dtype | QuantDataType | None]`, *optional*, defaults to `(None,)`):
            The quantization scale data type for per-group quantization.
        skips (`list[str]`, *optional*, defaults to `[]`):
            The keys of the modules to skip.
        static (`bool`, *optional*, defaults to `False`):
            Whether to use static quantization.
        calib_range (`DynamicRangeCalibConfig` or `None`, *optional*, defaults to `None`):
            The quantizatizer dynamic range calibration configuration.
        allow_unsigned (`bool`, *optional*, defaults to `False`):
            Whether to allow unsigned data type for activation quantization.
    """

    kernel_gptq: None = field(init=False, default=None)
    low_rank: None = field(init=False, default=None)
    allow_unsigned: bool = False

    @property
    def needs_calib_data(self) -> bool:
        return self.enabled_calib_range and (self.calib_range.needs_search or self.static)

    def generate_dirnames(
        self,
        *,
        prefix: str = "",
        shape: torch.Size | tuple[int, ...] = (1024, 1024, 16, 16),
        default_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> list[str]:
        """Get the directory names of the quantization configuration.

        Args:
            prefix (`str`, *optional*, defaults to `""`):
                The prefix for the directory names.
            shape (`torch.Size` or `tuple[int, ...]`, *optional*, defaults to `(1024, 1024, 16, 16)`):
                The shape of the tensor to be quantized.
            default_dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
                The dtype of the tensor to be quantized.

        Returns:
            `list[str]`:
                The directory names of the quantization configuration.
                    - The number of effective bits.
                    - The name of the quantization data type.
                    - The name of the group shapes.
                    - The name of the modules to skip.
        """
        names = super().generate_dirnames(prefix=prefix, shape=shape, default_dtype=default_dtype)
        if self.allow_unsigned:
            names[1] += ".u"
        return names

    def for_unsigned(self) -> "DiffusionActivationQuantizerConfig":
        """get the quantizer configuration for unsigned activations.

        Returns:
            `DiffusionActivationQuantizerConfig`:
                The quantizer configuration for unsigned activations.
        """
        if isinstance(self.dtype, QuantDataType) and self.allow_unsigned:
            return DiffusionActivationQuantizerConfig(
                dtype=self.dtype.to_unsigned(),
                zero_point=self.zero_point,
                group_shapes=self.group_shapes,
                scale_dtypes=self.scale_dtypes,
                skips=self.skips,
                static=self.static,
                calib_range=self.calib_range,
                allow_unsigned=self.allow_unsigned,
            )
        else:
            return self


@configclass
@dataclass
class DiffusionExtraWeightQuantizerConfig(IncludeBasedConfig, DiffusionQuantizerConfig):
    """Diffusion model extra weight quantizer configuration.

    Args:
        dtype (`QuantDataType` or `None`, *optional*, defaults to `None`):
            The quantization data type.
        zero_point (`ZeroPointDomain` or `None`, *optional*, defaults to `None`):
            The zero-point domain.
        group_shapes (`Sequence[Sequence[int]]`, *optional*, defaults to `((-1, -1, -1),)`):
            The shapes for per-group quantization.
        scale_dtypes (`Sequence[torch.dtype | QuantDataType | None]`, *optional*, defaults to `(None,)`):
            The quantization scale data type for per-group quantization.
        includes (`list[str]`, *optional*, defaults to `[]`):
            The keys of the modules to include.
        low_rank (`SkipBasedQuantLowRankCalibConfig` or `None`, *optional*, defaults to `None`):
            The quantization low-rank branch calibration configuration.
        calib_range (`DynamicRangeCalibConfig` or `None`, *optional*, defaults to `None`):
            The quantizatizer dynamic range calibration configuration.
    """

    static: bool = field(init=False, default=True)
    kernel_gptq: DiffusionGPTQConfig | None = field(init=False, default=None)
    low_rank: SkipBasedQuantLowRankCalibConfig | None = field(init=False, default=None)
    calib_range: SkipBasedDynamicRangeCalibConfig | None = field(init=False, default=None)

    @property
    def needs_calib_data(self) -> bool:
        return self.enabled_calib_range and self.calib_range.needs_search


@configclass
@dataclass(kw_only=True)
class DiffusionModuleQuantizerConfig(EnableConfig):
    """Diffusion model module quantizer configuration.

    Args:
        wgts (`DiffusionWeightQuantizerConfig`):
            The weight quantization configuration.
        ipts (`DiffusionActivationQuantizerConfig`):
            The input activation quantization configuration.
        opts (`DiffusionActivationQuantizerConfig`):
            The output activation quantization configuration.
    """

    wgts: DiffusionWeightQuantizerConfig
    ipts: DiffusionActivationQuantizerConfig
    opts: DiffusionActivationQuantizerConfig
    extra_wgts: DiffusionExtraWeightQuantizerConfig | None = None
    unsigned_ipts: DiffusionActivationQuantizerConfig = field(init=False)

    def is_enabled(self):
        return self.enabled_wgts or self.enabled_ipts or self.enabled_opts

    @property
    def enabled_wgts(self) -> bool:
        """Whether to enable weight quantization."""
        return self.wgts is not None and self.wgts.is_enabled()

    @property
    def enabled_ipts(self) -> bool:
        """Whether to enable activation quantization."""
        return self.ipts is not None and self.ipts.is_enabled()

    @property
    def enabled_opts(self) -> bool:
        """Whether to enable activation quantization."""
        return self.opts is not None and self.opts.is_enabled()

    @property
    def enabled_extra_wgts(self) -> bool:
        """Whether to enable extra weight quantization."""
        return self.extra_wgts is not None and self.extra_wgts.is_enabled()

    def __post_init__(self) -> None:
        if self.enabled_opts:
            raise NotImplementedError("Output activation quantization is not supported yet.")
        if self.wgts.is_enabled() and self.extra_wgts is not None:
            self.extra_wgts.includes = list(filter(lambda key: key not in self.wgts.skips, self.extra_wgts.includes))
            if self.extra_wgts.is_enabled():
                self.extra_wgts.kernel_gptq = self.wgts.kernel_gptq
                self.extra_wgts.low_rank = self.wgts.low_rank
                self.extra_wgts.calib_range = self.wgts.calib_range
            else:
                self.extra_wgts = None
        else:
            self.extra_wgts = None

    def generate_dirnames(
        self,
        *,
        prefix: str = "",
        shape: torch.Size | tuple[int, ...] = (1024, 1024, 16, 16),
        default_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> list[str]:
        """Get the directory names of the quantization configuration.

        Args:
            prefix (`str`, *optional*, defaults to `""`):
                The prefix for the directory names.
            shape (`torch.Size` or `tuple[int, ...]`, *optional*, defaults to `(1024, 1024, 16, 16)`):
                The shape of the tensor to be quantized.
            default_dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
                The dtype of the tensor to be quantized.

        Returns:
            `list[str]`:
                The directory names of the quantization configuration.
                    - The number of effective bits.
                    - The name of the quantization data type.
                    - The name of the group shapes.
                    - The name of the modules to skip.
        """
        wgts_names = self.wgts.generate_dirnames(prefix="w", shape=shape, default_dtype=default_dtype)
        ipts_names = self.ipts.generate_dirnames(prefix="x", shape=shape, default_dtype=default_dtype)
        opts_names = self.opts.generate_dirnames(prefix="y", shape=shape, default_dtype=default_dtype)
        names = [
            f"{wgts_name}-{ipts_name}-{opts_name}"
            for wgts_name, ipts_name, opts_name in zip(wgts_names, ipts_names, opts_names, strict=True)
        ]
        if self.extra_wgts is not None:
            extra_wgts_names = self.extra_wgts.generate_dirnames(prefix="w", shape=shape, default_dtype=default_dtype)
            names = [f"{name}-{extra_wgts_name}" for name, extra_wgts_name in zip(names, extra_wgts_names, strict=True)]
        if prefix:
            names = [f"{prefix}.[{name}]" for name in names]
        return names

    def generate_calib_dirname(self) -> str:
        """Generate the name for quantization calibration.

        Returns:
            `str`:
                The name.
        """
        name = ""
        if self.enabled_wgts:
            calib_name = self.wgts.generate_calib_dirname()
            if calib_name:
                name += f"-w.{calib_name}"
        if self.enabled_ipts:
            calib_name = self.ipts.generate_calib_dirname()
            if calib_name:
                name += f"-x.{calib_name}"
        if self.enabled_opts:
            calib_name = self.opts.generate_calib_dirname()
            if calib_name:
                name += f"-y.{calib_name}"
        return name[1:] if name else name
