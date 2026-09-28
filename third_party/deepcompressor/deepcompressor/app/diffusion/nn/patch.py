import torch.nn as nn
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock

from deepcompressor.nn.patch.conv import ConcatConv2d, ShiftedConv2d
from deepcompressor.nn.patch.linear import ConcatLinear, ShiftedLinear
from deepcompressor.utils import patch, tools

from .attention import DiffusionAttentionProcessor
from .struct import DiffusionFeedForwardStruct, DiffusionModelStruct, DiffusionResnetStruct, UNetStruct

__all__ = [
    "replace_up_block_conv_with_concat_conv",
    "replace_fused_linear_with_concat_linear",
    "replace_attn_processor",
    "shift_input_activations",
    "patch_sdxl_text_time_embedding",
]


def replace_up_block_conv_with_concat_conv(model: nn.Module) -> None:
    """Replace up_block convolutions in UNet with ConcatConv."""
    model_struct = DiffusionModelStruct.construct(model)
    if not isinstance(model_struct, UNetStruct):
        return
    logger = tools.logging.getLogger(__name__)
    logger.info("Replacing up_block convolutions with ConcatConv.")
    tools.logging.Formatter.indent_inc()
    parents_map = patch.get_module_parents_map(model)
    for up_block in model_struct.up_block_structs:
        logger.info(f"+ Replacing convolutions in up_block {up_block.name}")
        tools.logging.Formatter.indent_inc()
        for resnet in up_block.resnet_structs:
            assert len(resnet.convs[0]) == 1
            conv, conv_name = resnet.convs[0][0], resnet.conv_names[0][0]
            logger.info(f"- Replacing {conv_name} in resnet {resnet.name}")
            tools.logging.Formatter.indent_inc()
            if resnet.idx == 0:
                if up_block.idx == 0:
                    prev_block = model_struct.mid_block_struct
                else:
                    prev_block = model_struct.up_block_structs[up_block.idx - 1]
                logger.info(f"+ using previous block {prev_block.name}")
                prev_channels = prev_block.resnet_structs[-1].convs[-1][-1].out_channels
            else:
                prev_channels = up_block.resnet_structs[resnet.idx - 1].convs[-1][-1].out_channels
            logger.info(f"+ conv_in_channels = {prev_channels}/{conv.in_channels}")
            logger.info(f"+ conv_out_channels = {conv.out_channels}")
            concat_conv = ConcatConv2d.from_conv2d(conv, [prev_channels])
            for parent_name, parent_module, child_name in parents_map[conv]:
                logger.info(f"+ replacing {child_name} in {parent_name}")
                setattr(parent_module, child_name, concat_conv)
            tools.logging.Formatter.indent_dec()
        tools.logging.Formatter.indent_dec()
    tools.logging.Formatter.indent_dec()


def replace_fused_linear_with_concat_linear(model: nn.Module) -> None:
    """Replace fused Linear in FluxSingleTransformerBlock with ConcatLinear."""
    logger = tools.logging.getLogger(__name__)
    logger.info("Replacing fused Linear with ConcatLinear.")
    tools.logging.Formatter.indent_inc()
    for name, module in model.named_modules():
        if isinstance(module, FluxSingleTransformerBlock):
            logger.info(f"+ Replacing fused Linear in {name} with ConcatLinear.")
            tools.logging.Formatter.indent_inc()
            logger.info(f"- in_features = {module.proj_out.out_features}/{module.proj_out.in_features}")
            logger.info(f"- out_features = {module.proj_out.out_features}")
            tools.logging.Formatter.indent_dec()
            module.proj_out = ConcatLinear.from_linear(module.proj_out, [module.proj_out.out_features])
    tools.logging.Formatter.indent_dec()


def shift_input_activations(model: nn.Module) -> None:
    """Shift input activations of convolutions and linear layers if their lowerbound is negative.

    Args:
        model (nn.Module): model to shift input activations.
    """
    logger = tools.logging.getLogger(__name__)
    model_struct = DiffusionModelStruct.construct(model)
    module_parents_map = patch.get_module_parents_map(model)
    logger.info("- Shifting input activations.")
    tools.logging.Formatter.indent_inc()
    for _, module_name, module, parent, field_name in model_struct.named_key_modules():
        lowerbound = None
        if isinstance(parent, DiffusionResnetStruct) and field_name.startswith("conv"):
            lowerbound = parent.config.intermediate_lowerbound
        elif isinstance(parent, DiffusionFeedForwardStruct) and field_name.startswith("down_proj"):
            lowerbound = parent.config.intermediate_lowerbound
        if lowerbound is not None and lowerbound < 0:
            shift = -lowerbound
            logger.info(f"+ Shifting input activations of {module_name} by {shift}")
            tools.logging.Formatter.indent_inc()
            if isinstance(module, nn.Linear):
                shifted = ShiftedLinear.from_linear(module, shift=shift)
                shifted.linear.unsigned = True
            elif isinstance(module, nn.Conv2d):
                shifted = ShiftedConv2d.from_conv2d(module, shift=shift)
                shifted.conv.unsigned = True
            else:
                raise NotImplementedError(f"Unsupported module type {type(module)}")
            for parent_name, parent_module, child_name in module_parents_map[module]:
                logger.info(f"+ Replacing {child_name} in {parent_name}")
                setattr(parent_module, child_name, shifted)
            tools.logging.Formatter.indent_dec()
    tools.logging.Formatter.indent_dec()


def replace_attn_processor(model: nn.Module) -> None:
    """Replace Attention processor with DiffusionAttentionProcessor."""
    logger = tools.logging.getLogger(__name__)
    logger.info("Replacing Attention processors.")
    tools.logging.Formatter.indent_inc()
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            logger.info(f"+ Replacing {name} processor with DiffusionAttentionProcessor.")
            module.set_processor(DiffusionAttentionProcessor(module.processor))
    tools.logging.Formatter.indent_dec()


def patch_sdxl_text_time_embedding(model: nn.Module) -> None:
    """Patch SDXL `addition_embed_type=text_time` to be robust to `add_time_proj` output dim mismatches.

    Some diffusers installations end up with a mismatch between `add_time_proj(time_ids)` output dim and
    `add_embedding.linear_1.in_features`, which causes a matmul shape error during UNet forward. This patch
    detects the mismatch and recomputes the expected sinusoidal time embedding using
    `diffusers.models.embeddings.get_timestep_embedding`.
    """
    import types

    import torch
    from diffusers.models.embeddings import get_timestep_embedding
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

    if not isinstance(model, UNet2DConditionModel):
        return
    if getattr(model.config, "addition_embed_type", None) != "text_time":
        return
    if not hasattr(model, "add_embedding") or not hasattr(model, "add_time_proj"):
        return
    if getattr(model, "_deepcompressor_sdxl_text_time_embedding_patched", False):
        return

    logger = tools.logging.getLogger(__name__)
    orig_get_aug_embed = model.get_aug_embed

    def _patched_get_aug_embed(self, emb: torch.Tensor, encoder_hidden_states: torch.Tensor, added_cond_kwargs: dict):
        if getattr(self.config, "addition_embed_type", None) != "text_time":
            return orig_get_aug_embed(emb, encoder_hidden_states, added_cond_kwargs)
        if not isinstance(added_cond_kwargs, dict) or "text_embeds" not in added_cond_kwargs or "time_ids" not in added_cond_kwargs:
            return orig_get_aug_embed(emb, encoder_hidden_states, added_cond_kwargs)

        text_embeds = added_cond_kwargs.get("text_embeds")
        time_ids = added_cond_kwargs.get("time_ids")
        if not isinstance(text_embeds, torch.Tensor) or not isinstance(time_ids, torch.Tensor):
            return orig_get_aug_embed(emb, encoder_hidden_states, added_cond_kwargs)

        # Be tolerant to cached tensors that carry an extra singleton dim.
        if text_embeds.ndim == 3 and text_embeds.shape[1] == 1:
            text_embeds = text_embeds[:, 0, :]
        if time_ids.ndim == 3 and time_ids.shape[1] == 1:
            time_ids = time_ids[:, 0, :]

        linear_1 = getattr(getattr(self, "add_embedding", None), "linear_1", None)
        expected_in = getattr(linear_1, "in_features", None)
        if expected_in is None or text_embeds.ndim != 2:
            return orig_get_aug_embed(emb, encoder_hidden_states, added_cond_kwargs)

        if time_ids.ndim != 2 or int(time_ids.shape[-1]) <= 0:
            return orig_get_aug_embed(emb, encoder_hidden_states, added_cond_kwargs)

        expected_time_total = int(expected_in) - int(text_embeds.shape[-1])
        if expected_time_total <= 0 or expected_time_total % int(time_ids.shape[-1]) != 0:
            return orig_get_aug_embed(emb, encoder_hidden_states, added_cond_kwargs)

        per_id_dim = expected_time_total // int(time_ids.shape[-1])
        flip_sin_to_cos = bool(getattr(self.config, "flip_sin_to_cos", False))
        downscale_freq_shift = float(getattr(self.config, "freq_shift", 1))

        bsz = int(text_embeds.shape[0])
        time_ids_flat = time_ids.to(device=emb.device, dtype=torch.float32).flatten()

        # First try the model's native projection. If it yields the wrong dim (observed in some envs),
        # fall back to an explicit sinusoidal embedding sized to match add_embedding.
        time_embeds = None
        got_in = None
        try:
            _time = self.add_time_proj(time_ids_flat).reshape((bsz, -1))
            got_in = int(text_embeds.shape[-1]) + int(_time.shape[-1])
            if int(_time.shape[-1]) == int(expected_time_total):
                time_embeds = _time.to(dtype=emb.dtype)
        except Exception:
            pass

        if time_embeds is None:
            logger.warning(
                "SDXL add_embedding input dim mismatch detected (got=%s expected=%s); "
                "recomputing time embedding with per_id_dim=%d.",
                got_in,
                int(expected_in),
                int(per_id_dim),
            )
            time_embeds = get_timestep_embedding(
                time_ids_flat,
                per_id_dim,
                flip_sin_to_cos=flip_sin_to_cos,
                downscale_freq_shift=downscale_freq_shift,
            ).reshape((bsz, -1))
            time_embeds = time_embeds.to(device=emb.device, dtype=emb.dtype)

        add_embeds = torch.concat(
            [text_embeds.to(device=emb.device, dtype=emb.dtype), time_embeds],
            dim=-1,
        )
        if int(add_embeds.shape[-1]) != int(expected_in):
            raise RuntimeError(
                "SDXL add_embeds dim mismatch after fallback: "
                f"add_embeds={tuple(add_embeds.shape)} expected_in={int(expected_in)} "
                f"text_embeds={tuple(text_embeds.shape)} time_ids={tuple(time_ids.shape)} "
                f"time_embeds={tuple(time_embeds.shape)}"
            )
        return self.add_embedding(add_embeds)

    model.get_aug_embed = types.MethodType(_patched_get_aug_embed, model)
    model._deepcompressor_sdxl_text_time_embedding_patched = True
