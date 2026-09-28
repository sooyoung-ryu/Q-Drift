# -*- coding: utf-8 -*-
"""Large Language Model Rotation module."""

import gc

import torch

from deepcompressor.calib.rotate import (
    get_rotation_matrix,
    hadamard_in_channels,
    rotate_in_channels,
    rotate_out_channels,
)
from deepcompressor.utils import tools

from ..nn.struct import DiffusionModelStruct
from .config import DiffusionQuantConfig

__all__ = ["rotate_diffusion"]


@torch.inference_mode()
def rotate_diffusion(  # noqa: C901
    model: DiffusionModelStruct, /, config: DiffusionQuantConfig
):
    """Rotate the weights of the diffusion model.

    Args:
        model (`PreTrainedModel` or `LlmStruct`):
            Model to be rotated.
        config (`QuantRotationConfig`):
            Rotation configuration.
    """
    if not isinstance(model, DiffusionModelStruct):
        model = DiffusionModelStruct.construct(model)
    assert isinstance(model, DiffusionModelStruct)
    devices: dict[str, torch.device] = {}
    dtypes: dict[str, torch.dtype] = {}
    linears: dict[str, torch.nn.Linear] = {}
    size: float = 0
    for n, m in model.module.named_modules():
        if isinstance(m, torch.nn.Linear):
            devices[n] = m.weight.device
            dtypes[n] = m.weight.dtype
            linears[n] = m
            size += m.weight.numel() / 1e9
    for linear in linears.values():
        linear.to(dtype=torch.float32, device="cpu" if size > 30 else None)

    logger = tools.logging.getLogger(f"{__name__}.Rotate")
    head_rotation = None
    for transformer_block in model.iter_transformer_block_structs():
        logger.debug(f"- Rotating {transformer_block.name}")
        tools.logging.Formatter.indent_inc()
        for attn in transformer_block.iter_attention_structs():
            if attn.qkv_proj_key in config.rotation.transforms:
                if attn.qkv_proj_key not in config.wgts.skips or attn.qkv_proj_key not in config.ipts.skips:
                    logger.debug(f"- Hadamard transform on {attn.name}.qkv_proj (in)")
                    hadamard_in_channels(
                        attn.qkv_proj, dtype=dtypes[attn.q_proj_name], device=devices[attn.q_proj_name]
                    )
            if not attn.is_self_attn() and attn.add_qkv_proj_key in config.rotation.transforms:
                if attn.add_qkv_proj_key not in config.wgts.skips or attn.add_qkv_proj_key not in config.ipts.skips:
                    logger.debug(f"- Hadamard transform on {attn.name}.add_qkv_proj (in)")
                    hadamard_in_channels(
                        attn.add_qkv_proj, dtype=dtypes[attn.add_k_proj_name], device=devices[attn.add_k_proj_name]
                    )
            if attn.out_proj_key in config.rotation.transforms or attn.add_out_proj_key in config.rotation.transforms:
                if (
                    attn.out_proj_key not in config.wgts.skips
                    or attn.out_proj_key not in config.ipts.skips
                    or attn.add_out_proj_key not in config.wgts.skips
                    or attn.add_out_proj_key not in config.ipts.skips
                ):
                    if head_rotation is None:
                        head_rotation = get_rotation_matrix(
                            attn.config.num_head_channels, random=config.rotation.random
                        )
                    if attn.v_proj is not None:
                        logger.debug(f"- Rotating {attn.v_proj_name} (out)")
                        rotate_out_channels(attn.v_proj.weight, rotation=head_rotation, bias=attn.v_proj.bias)
                    if attn.add_v_proj is not None:
                        logger.debug(f"- Rotating {attn.add_v_proj_name} (out)")
                        rotate_out_channels(attn.add_v_proj.weight, rotation=head_rotation, bias=attn.add_v_proj.bias)
                    if attn.o_proj is not None:
                        logger.debug(f"- Rotating {attn.o_proj_name} (in)")
                        rotate_in_channels(attn.o_proj.weight, rotation=head_rotation)
                    if attn.add_o_proj is not None:
                        logger.debug(f"- Rotating {attn.add_o_proj_name} (in)")
                        rotate_in_channels(attn.add_o_proj.weight, rotation=head_rotation)
            gc.collect()
            torch.cuda.empty_cache()
        ffn, add_ffn = transformer_block.ffn_struct, transformer_block.add_ffn_struct
        if ffn.up_proj_key in config.rotation.transforms:
            if ffn.up_proj_key not in config.wgts.skips or ffn.up_proj_key not in config.ipts.skips:
                logger.debug(f"- Hadamard transform on {ffn.up_proj_name} (in)")
                hadamard_in_channels(ffn.up_projs, dtype=dtypes[ffn.up_proj_name], device=devices[ffn.up_proj_name])
        if add_ffn is not None and add_ffn.up_proj_key in config.rotation.transforms:
            if add_ffn.up_proj_key not in config.wgts.skips or add_ffn.up_proj_key not in config.ipts.skips:
                logger.debug(f"- Hadamard transform on {add_ffn.up_proj_name} (in)")
                hadamard_in_channels(
                    add_ffn.up_projs, dtype=dtypes[add_ffn.up_proj_name], device=devices[add_ffn.up_proj_name]
                )
        if ffn.down_proj_key in config.rotation.transforms:
            if ffn.down_proj_key not in config.wgts.skips or ffn.down_proj_key not in config.ipts.skips:
                logger.debug(f"- Hadamard transform on {ffn.down_proj_name} (in)")
                hadamard_in_channels(
                    ffn.down_projs, dtype=dtypes[ffn.down_proj_name], device=devices[ffn.down_proj_name]
                )
        if add_ffn is not None and add_ffn.down_proj_key in config.rotation.transforms:
            if add_ffn.down_proj_key not in config.wgts.skips or add_ffn.down_proj_key not in config.ipts.skips:
                logger.debug(f"- Hadamard transform on {add_ffn.down_proj_name} (in)")
                hadamard_in_channels(
                    add_ffn.down_projs, dtype=dtypes[add_ffn.down_proj_name], device=devices[add_ffn.down_proj_name]
                )
        gc.collect()
        torch.cuda.empty_cache()
        tools.logging.Formatter.indent_dec()

    for n, m in linears.items():
        m.to(device=devices[n], dtype=dtypes[n])
