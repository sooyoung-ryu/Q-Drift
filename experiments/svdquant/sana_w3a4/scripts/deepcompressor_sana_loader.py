from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from diffusers import SanaTransformer2DModel


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _maybe_add_deepcompressor_to_syspath() -> Path:
    deepcompressor_root = _repo_root() / "third_party" / "deepcompressor"
    if deepcompressor_root.is_dir() and str(deepcompressor_root) not in sys.path:
        sys.path.insert(0, str(deepcompressor_root))
    return deepcompressor_root


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _resolve_deepcompressor_config_paths(quant_model_dir: str) -> Tuple[Path, Path, Path]:
    """
    Returns:
        default_cfg, model_cfg, quant_cfg
    """
    qdir = Path(quant_model_dir)
    meta = _read_json(qdir / "deepcompressor_config.json")
    # Resolve config paths recorded in the checkpoint against the configs shipped here.
    configs = (_maybe_add_deepcompressor_to_syspath() / "examples" / "diffusion" / "configs").resolve()
    model_name = Path(meta.get("model_config") or 'sana-1.6b.yaml').name
    model_cfg = configs / "paper_models" / model_name
    if not model_cfg.is_file():
        model_cfg = configs / "model" / model_name
    paths = (configs / "__default__.yaml", model_cfg,
             configs / "svdquant" / Path(meta.get("quant_config") or 'sana_w3a4.yaml').name)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint configuration: {path}")
    return paths


def _parse_deepcompressor_run_config(*, default_cfg: Path, model_cfg: Path, quant_cfg: Path) -> Any:
    _maybe_add_deepcompressor_to_syspath()
    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig

    parser = DiffusionPtqRunConfig.get_parser()
    args = [str(default_cfg), str(model_cfg), str(quant_cfg)]
    old_cwd = os.getcwd()
    os.chdir(str(_repo_root()))
    try:
        config, _, unused_cfgs, unused_args, unknown_args = parser.parse_known_args(args)
    finally:
        os.chdir(old_cwd)
    if len(unknown_args) != 0:
        raise ValueError(f"Unknown DeepCompressor args while parsing configs: {unknown_args}")
    if len(unused_cfgs) != 0:
        raise ValueError(f"Unused DeepCompressor configs while parsing configs: {unused_cfgs}")
    if unused_args is not None and len(unused_args) != 0:
        raise ValueError(f"Unused DeepCompressor args while parsing configs: {unused_args}")
    return config


def apply_deepcompressor_patches(*, model: torch.nn.Module, shift_activations: bool) -> None:
    _maybe_add_deepcompressor_to_syspath()
    from deepcompressor.app.diffusion.nn.patch import (
        replace_fused_linear_with_concat_linear,
        replace_up_block_conv_with_concat_conv,
        shift_input_activations,
    )

    replace_fused_linear_with_concat_linear(model)
    replace_up_block_conv_with_concat_conv(model)
    if shift_activations:
        shift_input_activations(model)


def load_quant_transformer(
    *,
    base_model_id: str,
    quant_model_dir: str,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> SanaTransformer2DModel:
    """
    Load a DeepCompressor PTQ checkpoint (saved via `--save-model`) for Sana transformer.

    Expected files under `quant_model_dir`:
      - model.pt
      - wgts.pt
      - smooth.pt / branch.pt (optional, depending on quant config)
      - acts.pt (optional, depending on activation quant)
    """
    qdir = Path(quant_model_dir)
    model_pt = qdir / "model.pt"
    acts_pt = qdir / "acts.pt"
    smooth_pt = qdir / "smooth.pt"
    branch_pt = qdir / "branch.pt"

    if not model_pt.exists():
        raise FileNotFoundError(f"DeepCompressor checkpoint missing {model_pt}")

    default_cfg, model_cfg, quant_cfg = _resolve_deepcompressor_config_paths(quant_model_dir)
    config = _parse_deepcompressor_run_config(default_cfg=default_cfg, model_cfg=model_cfg, quant_cfg=quant_cfg)

    _maybe_add_deepcompressor_to_syspath()
    from deepcompressor.app.diffusion.quant import load_diffusion_weights_state_dict, quantize_diffusion_activations
    from deepcompressor.app.diffusion.quant.smooth import smooth_diffusion

    transformer = SanaTransformer2DModel.from_pretrained(
        base_model_id,
        subfolder="transformer",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    transformer.eval()

    apply_deepcompressor_patches(
        model=transformer, shift_activations=bool(getattr(config.pipeline, "shift_activations", False))
    )

    if bool(getattr(config.quant, "enabled_smooth", False)):
        if not smooth_pt.exists():
            raise FileNotFoundError(
                f"DeepCompressor checkpoint missing {smooth_pt} (required because quant.enable_smooth is true)"
            )
        smooth_cache = torch.load(smooth_pt, map_location="cpu")
        smooth_diffusion(transformer, config.quant, smooth_cache=smooth_cache)

    branch_state_dict = torch.load(branch_pt, map_location="cpu") if branch_pt.exists() else None
    state_dict = torch.load(model_pt, map_location="cpu")
    load_diffusion_weights_state_dict(transformer, config.quant, state_dict=state_dict, branch_state_dict=branch_state_dict)

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

