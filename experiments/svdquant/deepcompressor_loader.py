from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from diffusers import UNet2DConditionModel


def _maybe_add_deepcompressor_to_syspath() -> None:
    here = Path(__file__).resolve()
    deepcompressor_root = here.parents[2] / "third_party" / "deepcompressor"
    if deepcompressor_root.is_dir() and str(deepcompressor_root) not in sys.path:
        sys.path.insert(0, str(deepcompressor_root))


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _infer_sdxl_quant_config_filename(quant_model_dir: Path) -> str:
    name = quant_model_dir.name.lower()
    if "w2a4" in name:
        return "sdxl_w2a4.yaml"
    if "w3a4" in name:
        return "sdxl_w3a4.yaml"
    if "w3a3" in name:
        return "sdxl_w3a3.yaml"
    if "w4a4" in name:
        return "sdxl_w4a4.yaml"
    if "w6a6" in name:
        return "sdxl_w6a6.yaml"
    raise ValueError(f"Cannot infer DeepCompressor quant config from quant_model_dir name: {quant_model_dir.name}")


def _resolve_deepcompressor_config_paths(quant_model_dir: str) -> Tuple[Path, Path, Path, Path]:
    """
    Returns:
        examples_dir, default_cfg, model_cfg, quant_cfg
    """
    qdir = Path(quant_model_dir)
    cfg_json = qdir / "deepcompressor_config.json"
    meta = _read_json(cfg_json) if cfg_json.exists() else {}

    # Resolve config names recorded in the checkpoint against the configs shipped here.
    examples_dir = (Path(__file__).resolve().parents[2] / "third_party" / "deepcompressor" / "examples" / "diffusion").resolve()
    configs = examples_dir / "configs"
    default_cfg = configs / "__default__.yaml"
    model_name = Path(meta.get("model_config") or "sdxl.yaml").name
    model_cfg = configs / "paper_models" / model_name
    if not model_cfg.is_file():
        model_cfg = configs / "model" / model_name
    quant_name = Path(meta["quant_config"]).name if meta.get("quant_config") else _infer_sdxl_quant_config_filename(qdir)
    quant_cfg = configs / "svdquant" / quant_name
    for path in (default_cfg, model_cfg, quant_cfg):
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint configuration: {path}")
    return examples_dir, default_cfg, model_cfg, quant_cfg


def _parse_deepcompressor_run_config(
    *, examples_dir: Path, default_cfg: Path, model_cfg: Path, quant_cfg: Path
) -> Any:
    _maybe_add_deepcompressor_to_syspath()
    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig

    parser = DiffusionPtqRunConfig.get_parser()
    args = [str(default_cfg), str(model_cfg), str(quant_cfg)]

    try:
        config, _, unused_cfgs, unused_args, unknown_args = parser.parse_known_args(args)
        if len(unknown_args) != 0:
            raise ValueError(f"Unknown DeepCompressor args while parsing configs: {unknown_args}")
        if len(unused_cfgs) != 0:
            raise ValueError(f"Unused DeepCompressor configs while parsing configs: {unused_cfgs}")
        return config
    except Exception:
        cwd = os.getcwd()
        os.chdir(str(examples_dir))
        try:
            rel_args = [
                os.path.relpath(default_cfg, examples_dir),
                os.path.relpath(model_cfg, examples_dir),
                os.path.relpath(quant_cfg, examples_dir),
            ]
            config, _, unused_cfgs, unused_args, unknown_args = parser.parse_known_args(rel_args)
            if len(unknown_args) != 0:
                raise ValueError(f"Unknown DeepCompressor args while parsing configs: {unknown_args}")
            if len(unused_cfgs) != 0:
                raise ValueError(f"Unused DeepCompressor configs while parsing configs: {unused_cfgs}")
            return config
        finally:
            os.chdir(cwd)


def _load_unet_deepcompressor_checkpoint(
    *,
    base_model_id: str,
    quant_model_dir: str,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> UNet2DConditionModel:
    qdir = Path(quant_model_dir)
    model_pt = qdir / "model.pt"
    branch_pt = qdir / "branch.pt"
    smooth_pt = qdir / "smooth.pt"
    acts_pt = qdir / "acts.pt"

    if not model_pt.exists():
        raise FileNotFoundError(f"DeepCompressor checkpoint missing {model_pt}")

    examples_dir, default_cfg, model_cfg, quant_cfg = _resolve_deepcompressor_config_paths(quant_model_dir)
    config = _parse_deepcompressor_run_config(
        examples_dir=examples_dir, default_cfg=default_cfg, model_cfg=model_cfg, quant_cfg=quant_cfg
    )

    _maybe_add_deepcompressor_to_syspath()
    from deepcompressor.app.diffusion.nn.patch import (
        replace_fused_linear_with_concat_linear,
        replace_up_block_conv_with_concat_conv,
        shift_input_activations,
    )
    from deepcompressor.app.diffusion.quant import load_diffusion_weights_state_dict, quantize_diffusion_activations
    from deepcompressor.app.diffusion.quant.smooth import smooth_diffusion

    try:
        unet = UNet2DConditionModel.from_pretrained(
            base_model_id,
            subfolder="unet",
            torch_dtype=torch_dtype,
            variant="fp16",
        ).to(device)
    except Exception:
        # Some local diffusers folders do not use `*.fp16.safetensors` naming.
        unet = UNet2DConditionModel.from_pretrained(
            base_model_id,
            subfolder="unet",
            torch_dtype=torch_dtype,
        ).to(device)
    unet.eval()

    replace_fused_linear_with_concat_linear(unet)
    replace_up_block_conv_with_concat_conv(unet)
    if bool(getattr(config.pipeline, "shift_activations", False)):
        shift_input_activations(unet)

    if getattr(config.quant, "enabled_smooth", False):
        if not smooth_pt.exists():
            raise FileNotFoundError(
                f"DeepCompressor checkpoint missing {smooth_pt} (required because quant.enabled_smooth is true)"
            )
        smooth_cache = torch.load(smooth_pt, map_location="cpu")
        smooth_diffusion(unet, config.quant, smooth_cache=smooth_cache)

    if bool(getattr(config.quant, "enabled_wgts", False)) and bool(getattr(config.quant.wgts, "enabled_low_rank", False)):
        if not branch_pt.exists():
            raise FileNotFoundError(
                f"DeepCompressor checkpoint missing {branch_pt} (required because quant.wgts.enable_low_rank is true)"
            )
        branch_state_dict = torch.load(branch_pt, map_location="cpu")
    else:
        branch_state_dict = None

    state_dict = torch.load(model_pt, map_location="cpu")
    load_diffusion_weights_state_dict(unet, config.quant, state_dict=state_dict, branch_state_dict=branch_state_dict)

    if bool(getattr(config.quant, "enabled_ipts", False)) or bool(getattr(config.quant, "enabled_opts", False)):
        act_state_dict = torch.load(acts_pt, map_location="cpu") if acts_pt.exists() else None
        if getattr(config.quant, "needs_acts_quantizer_cache", False) and act_state_dict is None:
            raise FileNotFoundError(
                f"DeepCompressor checkpoint missing {acts_pt} (required because quant.needs_acts_quantizer_cache is true)"
            )
        quantize_diffusion_activations(
            unet,
            config.quant,
            quantizer_state_dict=act_state_dict,
            orig_state_dict=None,
        )

    unet.eval()
    return unet


def load_quant_unet(
    *,
    base_model_id: str,
    quant_model_dir: str,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> UNet2DConditionModel:
    """
    Load a quantized SDXL UNet for Q-Drift scripts.

    Supports two layouts:
      1) DeepCompressor PTQ checkpoint: contains `model.pt` (and optionally `smooth.pt`, `branch.pt`, `acts.pt`).
      2) Diffusers export: contains `unet/` (optionally `act_quantizer_state.pt` for activation hooks).
    """
    qdir = Path(quant_model_dir)
    if (qdir / "model.pt").exists():
        return _load_unet_deepcompressor_checkpoint(
            base_model_id=base_model_id, quant_model_dir=quant_model_dir, device=device, torch_dtype=torch_dtype
        )

    unet_dir = qdir / "unet"
    if not unet_dir.is_dir():
        raise FileNotFoundError(f"Expected either {qdir/'model.pt'} or {unet_dir} to exist.")

    unet = UNet2DConditionModel.from_pretrained(quant_model_dir, subfolder="unet", torch_dtype=torch_dtype).to(device)
    unet.eval()
    return unet


def _load_flux_transformer_deepcompressor_checkpoint(
    *,
    base_model_id: str,
    quant_model_dir: str,
    device: torch.device,
    torch_dtype: torch.dtype,
):
    from diffusers import FluxTransformer2DModel

    qdir = Path(quant_model_dir)
    model_pt = qdir / "model.pt"
    branch_pt = qdir / "branch.pt"
    smooth_pt = qdir / "smooth.pt"
    acts_pt = qdir / "acts.pt"

    if not model_pt.exists():
        raise FileNotFoundError(f"DeepCompressor checkpoint missing {model_pt}")

    examples_dir, default_cfg, model_cfg, quant_cfg = _resolve_deepcompressor_config_paths(quant_model_dir)
    config = _parse_deepcompressor_run_config(
        examples_dir=examples_dir, default_cfg=default_cfg, model_cfg=model_cfg, quant_cfg=quant_cfg
    )

    _maybe_add_deepcompressor_to_syspath()
    from deepcompressor.app.diffusion.nn.patch import (
        replace_fused_linear_with_concat_linear,
        replace_up_block_conv_with_concat_conv,
        shift_input_activations,
    )
    from deepcompressor.app.diffusion.quant import load_diffusion_weights_state_dict, quantize_diffusion_activations
    from deepcompressor.app.diffusion.quant.smooth import smooth_diffusion

    transformer = FluxTransformer2DModel.from_pretrained(
        base_model_id,
        subfolder="transformer",
        torch_dtype=torch_dtype,
    ).to(device)
    transformer.eval()

    replace_fused_linear_with_concat_linear(transformer)
    replace_up_block_conv_with_concat_conv(transformer)
    if bool(getattr(config.pipeline, "shift_activations", False)):
        shift_input_activations(transformer)

    if getattr(config.quant, "enabled_smooth", False):
        if not smooth_pt.exists():
            raise FileNotFoundError(
                f"DeepCompressor checkpoint missing {smooth_pt} (required because quant.enabled_smooth is true)"
            )
        smooth_cache = torch.load(smooth_pt, map_location="cpu")
        smooth_diffusion(transformer, config.quant, smooth_cache=smooth_cache)

    if bool(getattr(config.quant, "enabled_wgts", False)) and bool(getattr(config.quant.wgts, "enabled_low_rank", False)):
        if not branch_pt.exists():
            raise FileNotFoundError(
                f"DeepCompressor checkpoint missing {branch_pt} (required because quant.wgts.enable_low_rank is true)"
            )
        branch_state_dict = torch.load(branch_pt, map_location="cpu")
    else:
        branch_state_dict = None

    state_dict = torch.load(model_pt, map_location="cpu")
    load_diffusion_weights_state_dict(
        transformer, config.quant, state_dict=state_dict, branch_state_dict=branch_state_dict
    )

    if bool(getattr(config.quant, "enabled_ipts", False)) or bool(getattr(config.quant, "enabled_opts", False)):
        act_state_dict = torch.load(acts_pt, map_location="cpu") if acts_pt.exists() else None
        if getattr(config.quant, "needs_acts_quantizer_cache", False) and act_state_dict is None:
            raise FileNotFoundError(
                f"DeepCompressor checkpoint missing {acts_pt} (required because quant.needs_acts_quantizer_cache is true)"
            )
        quantize_diffusion_activations(
            transformer,
            config.quant,
            quantizer_state_dict=act_state_dict,
            orig_state_dict=None,
        )

    transformer.eval()
    return transformer


def load_quant_flux_transformer(
    *,
    base_model_id: str,
    quant_model_dir: str,
    device: torch.device,
    torch_dtype: torch.dtype,
):
    """
    Load a quantized FLUX Transformer for Q-Drift scripts.

    Supports two layouts:
      1) DeepCompressor PTQ checkpoint: contains `model.pt` (and optionally `smooth.pt`, `branch.pt`, `acts.pt`).
      2) Diffusers export: contains `transformer/`.
    """
    from diffusers import FluxTransformer2DModel

    qdir = Path(quant_model_dir)
    if (qdir / "model.pt").exists():
        return _load_flux_transformer_deepcompressor_checkpoint(
            base_model_id=base_model_id, quant_model_dir=quant_model_dir, device=device, torch_dtype=torch_dtype
        )

    transformer_dir = qdir / "transformer"
    if not transformer_dir.is_dir():
        raise FileNotFoundError(f"Expected either {qdir/'model.pt'} or {transformer_dir} to exist.")

    transformer = FluxTransformer2DModel.from_pretrained(
        quant_model_dir,
        subfolder="transformer",
        torch_dtype=torch_dtype,
    ).to(device)
    transformer.eval()
    return transformer
