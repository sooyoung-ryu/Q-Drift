# -*- coding: utf-8 -*-
"""Rotation Quantization module."""

import typing as tp

import torch
import torch.nn as nn

from ..utils.hooks import BaseInputPackager, IOHook
from ..utils.math import HadamardMatrix, hardmard_transform, random_hadamard_matrix

__all__ = [
    "rotate_in_channels",
    "rotate_out_channels",
    "hadamard_in_channels",
    "get_rotation_matrix",
    "transform_rms_norm_and_linear",
    "transform_layer_norm_to_rms_norm",
    "transform_norm_and_linear",
]


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)."""

    def __init__(self, hidden_size: int, eps=1e-6) -> None:
        """Initialize RMSNorm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm normalization to hidden states."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class HadamardTransformHook(IOHook):
    def __init__(
        self, rhs: torch.Tensor, lhs: torch.Tensor, lhs_k: int, scaled: bool = True, packager: BaseInputPackager = None
    ):
        super().__init__(pre=True, post=False, input_packager=packager, output_packager=None)
        self.rhs = rhs
        self.lhs = lhs
        self.lhs_k = lhs_k
        self.scaled = scaled

    def pre_forward(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, tp.Any]]:
        tensors = self.input_packager.unpack(module, input_args, input_kwargs)
        for k, x in tensors.items():
            tensors[k] = hardmard_transform(
                x, hadamard_rhs=self.rhs, hadamard_lhs=self.lhs, lhs_k=self.lhs_k, scaled=self.scaled
            )
        return self.input_packager.repack(tensors, module, input_args, input_kwargs)


def rotate_in_channels(weight: nn.Parameter, /, *, rotation: torch.Tensor) -> None:
    """Rotate the input channels of a weight matrix."""
    shape, dtype = weight.shape, weight.dtype
    weight.data = (
        torch.matmul(weight.data.view(-1, rotation.shape[0]).to(dtype=torch.float64), rotation.to(weight.device))
        .to(dtype=dtype)
        .view(shape)
    )


def rotate_out_channels(weight: nn.Parameter, /, *, rotation: torch.Tensor, bias: nn.Parameter | None = None) -> None:
    """Rotate the output channels of a weight matrix."""
    shape, dtype = weight.shape, weight.dtype
    out_channels, head_channels = shape[0], rotation.shape[0]
    num_heads = out_channels // head_channels
    weight.data = (
        torch.matmul(
            rotation.T.to(weight.device), weight.data.view(num_heads, head_channels, -1).to(dtype=torch.float64)
        )
        .to(dtype=dtype)
        .view(shape)
    )
    if bias is not None:
        bias.data = (
            torch.matmul(
                rotation.T.to(weight.device), bias.data.view(num_heads, head_channels, -1).to(dtype=torch.float64)
            )
            .to(dtype=dtype)
            .view(-1)
        )


def hadamard_in_channels(
    modules: tp.Iterable[nn.Module],
    packager: BaseInputPackager = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
):
    """Apply Hadamard quantization to the input channels of the modules."""
    for module in modules:
        if isinstance(module, nn.Linear):
            in_channels = module.in_features
            device, dtype = device or module.weight.device, dtype or module.weight.dtype
            rhs_double, lhs_double, k = HadamardMatrix.get(in_channels, scale=True, dtype=torch.float64)
            module.weight.data = hardmard_transform(
                module.weight.data.to(torch.float64), rhs_double.to(device), lhs_double.to(device), k, scaled=True
            ).to(device=device, dtype=module.weight.dtype)
            del rhs_double, lhs_double, k
            rhs, lhs, k = HadamardMatrix.get(in_channels, scale=True, dtype=dtype, device=device)
            HadamardTransformHook(rhs=rhs, lhs=lhs, lhs_k=k, packager=packager).register(module)
        else:
            raise NotImplementedError(f"Module {module} not supported!")


def get_rotation_matrix(num_channels: int, random: bool = True, compatible: bool = True) -> torch.Tensor:
    """Get a random rotation matrix for the given number of channels."""
    if random:
        return random_hadamard_matrix(num_channels)
    else:
        rhs, lhs, k = HadamardMatrix.get(num_channels, scale=False)
        rhs = rhs.to(dtype=torch.float64)
        if k == 1:
            rotation = rhs
        elif compatible:  # this is compatible with hadamard_transform
            rotation = torch.kron(lhs.T.contiguous().to(dtype=torch.float64), rhs)
        else:
            rotation = torch.kron(rhs, lhs.to(dtype=torch.float64))
        return rotation.mul_(1.0 / torch.tensor(num_channels, dtype=torch.float64).sqrt())


def transform_rms_norm_and_linear(norm: nn.LayerNorm | RMSNorm, next_modules: tp.Iterable[nn.Linear]) -> None:
    """Fuse the weight multiplication of rms norm into the next adjacent linear modules.

    Args:
        norm (`nn.LayerNorm` or `RMSNorm`):
            normalization module.
        next_modules (`Iterable[nn.Linear]`):
            modules after the normalization module.
    """
    ln_w = norm.weight.data.to(dtype=torch.float64)
    norm.weight.data = torch.ones_like(norm.weight.data)
    if hasattr(norm, "bias") and norm.bias is not None:
        ln_b = norm.bias.data.to(dtype=torch.float64)
        norm.bias = None
    else:
        ln_b = None
    for linear in next_modules:
        assert isinstance(linear, nn.Linear)
        dtype = linear.weight.dtype
        fc_w = linear.weight.data.to(dtype=torch.float64)
        ln_w = ln_w.to(fc_w.device)
        linear.weight.data = (fc_w * ln_w).to(dtype=dtype)
        if ln_b is not None:
            ln_b = ln_b.to(fc_w.device)
            if linear.bias is None:
                linear.bias = nn.Parameter(torch.zeros(linear.out_features, dtype=dtype, device=linear.weight.device))
            linear.bias.data = (linear.bias.data.to(dtype=torch.float64) + torch.matmul(fc_w, ln_b)).to(dtype=dtype)


def transform_layer_norm_to_rms_norm(
    parent: nn.Module,
    norm_name: str,
    prev_modules: tp.Iterable[nn.Linear],
    prev_out_channels_dims: int | tp.Iterable[int] = 0,
) -> None:
    """Transform LayerNorm to RMSNorm.

    Args:
        parent (`nn.Module`):
            Parent module that contains the normalization module.
        norm_name (`str`):
            Name of the normalization module in `parent`.
        prev_modules (`Iterable[nn.Linear]`):
            Previous adjacent linear modules.
        prev_out_channels_dims (`int` or `Iterable[int]`, *optional*, defaults to `0`):
            Output channels dimension of the previous modules' weights.
    """
    if "." in norm_name:
        norm_names = norm_name.split(".")
        for name in norm_names[:-1]:
            parent = getattr(parent, name)
        norm_name = norm_names[-1]
        del norm_names
    norm = getattr(parent, norm_name)
    assert isinstance(norm, nn.LayerNorm)
    assert len(norm.normalized_shape) == 1, f"LayerNorm's #dims must be 1, got {len(norm.normalized_shape)}"
    assert norm.bias is None, "LayerNorm's bias must be None, please call `transform_rms_norm_and_linear` in advance"
    # region move substract mean to the previous linear modules
    assert len(prev_modules) > 0, "No previous modules found"
    if isinstance(prev_out_channels_dims, int):
        prev_out_channels_dims = [prev_out_channels_dims] * len(prev_modules)
    for module, dim in zip(prev_modules, prev_out_channels_dims, strict=True):
        if isinstance(module, nn.LayerNorm):
            module.bias = None
        else:
            if isinstance(module, nn.Linear):
                assert dim == 0, "Linear module's output channels dimension is 0"
            elif isinstance(module, nn.Embedding):
                assert dim == 1, "Embedding module's output channels dimension is 1"
            dtype = module.weight.dtype
            w = module.weight.data.to(dtype=torch.float64)
            module.weight.data = w.sub_(w.mean(dim=dim, keepdim=True)).to(dtype=dtype)
            if hasattr(module, "bias") and module.bias is not None:
                b = module.bias.data.to(dtype=torch.float64)
                module.bias.data = b.sub_(b.mean()).to(dtype=dtype)
    # endregion
    # region replace LayerNorm with RMSNorm
    rms = RMSNorm(hidden_size=norm.normalized_shape[0], eps=norm.eps)
    rms.weight.data = norm.weight.data
    setattr(parent, norm_name, rms)
    # endregion


def transform_norm_and_linear(
    parent: nn.Module,
    norm_name: str,
    next_modules: tp.Iterable[nn.Linear],
    prev_modules: tp.Iterable[nn.Linear] | None = None,
    prev_out_channels_dims: int | tp.Iterable[int] = 0,
):
    """Transform the normalization module and the next adjacent linear modules.

    Args:
        parent (nn.Module): Parent module.
        norm_name (str): Name of the normalization module.
        next_modules (tp.Iterable[nn.Linear]): Next adjacent linear modules.
        prev_modules (tp.Iterable[nn.Linear]): Previous adjacent linear modules.
        prev_out_channels_dims (int | tp.Iterable[int], optional): Output channels dimension of the previous modules.
            Defaults to ``0``.
    """
    if "." in norm_name:
        norm_names = norm_name.split(".")
        for name in norm_names[:-1]:
            parent = getattr(parent, name)
        norm_name = norm_names[-1]
        del norm_names
    norm = getattr(parent, norm_name)
    transform_rms_norm_and_linear(norm, next_modules)
    if isinstance(norm, nn.LayerNorm):
        transform_layer_norm_to_rms_norm(parent, norm_name, prev_modules, prev_out_channels_dims)
