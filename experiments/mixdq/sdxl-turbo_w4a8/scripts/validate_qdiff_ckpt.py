"""
Sanity-check a MixDQ/qdiff checkpoint without running full evaluation.

This is mainly to answer: "Can we load a W4A8 model without calibration?"

What it does:
- Validates the checkpoint format (keys, delta_list/zero_point_list shapes).
- Optionally (if CUDA is available), builds the SDXL-Turbo UNet via qdiff, loads the
  checkpoint into QuantModel, applies bit-refactor, and reports bitwidth stats.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def _load_ckpt(path: str) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")
    return ckpt


def _quantizer_buffers(value: Any) -> Dict[str, Any] | None:
    """Return the buffer dict accepted by qdiff.load_quant_params, if present.

    The official MixDQ qdiff checkpoint uses the full format
    ckpt[module_name] = [buffers(OrderedDict), parameters(OrderedDict)].
    Some local artifacts use the buffer-only format
    ckpt[module_name] = {"delta_list": ..., "zero_point_list": ...}.
    """
    if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[0], dict):
        return value[0]
    if isinstance(value, dict):
        return value
    return None


def _is_full_quant_param_entry(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], dict)
        and isinstance(value[1], dict)
    )


def _infer_mp_nbits(ckpt: Dict[str, Any]) -> int | None:
    for v in ckpt.values():
        buffers = _quantizer_buffers(v)
        if buffers is not None and "delta_list" in buffers and torch.is_tensor(buffers["delta_list"]):
            if buffers["delta_list"].ndim >= 1:
                return int(buffers["delta_list"].shape[0])
    return None


def _count_key_suffixes(ckpt: Dict[str, Any]) -> Counter:
    c: Counter = Counter()
    for k in ckpt.keys():
        if k.endswith(".weight_quantizer"):
            c["weight_quantizer"] += 1
        elif k.endswith(".act_quantizer"):
            c["act_quantizer"] += 1
        elif ".weight_quantizer" in k:
            c["weight_quantizer_nested"] += 1
        elif ".act_quantizer" in k:
            c["act_quantizer_nested"] += 1
    return c


def _check_delta_shapes(ckpt: Dict[str, Any], mp_len: int | None) -> Tuple[int, int, int]:
    ok = 0
    empty = 0
    bad = 0
    for k, v in ckpt.items():
        buffers = _quantizer_buffers(v)
        if buffers is None:
            continue
        if "delta_list" not in buffers or "zero_point_list" not in buffers:
            continue
        d = buffers["delta_list"]
        z = buffers["zero_point_list"]
        if not (torch.is_tensor(d) and torch.is_tensor(z)):
            bad += 1
            continue
        if d.shape != z.shape:
            bad += 1
            continue
        if d.numel() == 0:
            empty += 1
            continue
        if mp_len is not None and d.ndim >= 1 and int(d.shape[0]) != mp_len:
            bad += 1
            continue
        ok += 1
    return ok, empty, bad


def _try_full_model_check(cfg_path: str, ckpt_path: str, w_bit: int, a_bit: int):
    if not torch.cuda.is_available():
        print("CUDA not available -> skipping full model build/load test.")
        return

    # Compatibility: some diffusers versions import `cached_download` from huggingface_hub.
    import huggingface_hub as _huggingface_hub
    if not hasattr(_huggingface_hub, "cached_download"):
        from huggingface_hub import hf_hub_download as _hf_hub_download

        _huggingface_hub.cached_download = _hf_hub_download  # type: ignore[attr-defined]

    from omegaconf import OmegaConf

    # Import qdiff without installing as a package.
    import sys

    quant_utils = Path(__file__).resolve().parents[4] / "third_party" / "mixdq" / "quant_utils"
    sys.path.insert(0, str(quant_utils))

    from qdiff.models.quant_model import QuantModel
    from qdiff.quantizer.base_quantizer import ActQuantizer, WeightQuantizer
    from qdiff.utils import get_model, load_quant_params

    cfg = OmegaConf.load(cfg_path)
    base_unet = get_model(
        cfg.model,
        fp16=True,
        return_pipe=False,
        convert_model_for_quant=True,
    )

    wq_params = cfg.quant.weight.quantizer
    aq_params = cfg.quant.activation.quantizer
    if cfg.get("mixed_precision", False):
        wq_params["mixed_precision"] = cfg.mixed_precision
        aq_params["mixed_precision"] = cfg.mixed_precision

    qnn = QuantModel(
        model=base_unet,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
        refactor_blocks=False,
    ).cuda().eval().half()
    qnn.set_quant_state(True, True)

    load_quant_params(qnn, ckpt_path, dtype=torch.float16)

    # Bit-refactor (power-of-two only in this codebase).
    if w_bit not in (2, 4, 8, 16):
        raise ValueError(f"w_bit must be in (2,4,8,16), got {w_bit}")
    if a_bit not in (2, 4, 8, 16):
        raise ValueError(f"a_bit must be in (2,4,8,16), got {a_bit}")
    qnn.set_layer_bit(model=qnn, n_bit=w_bit, quant_level="reset", bit_type="weight")
    qnn.set_layer_bit(model=qnn, n_bit=a_bit, quant_level="reset", bit_type="act")

    # Summarize bitwidth distribution.
    w_bits = []
    a_bits = []
    for m in qnn.model.modules():
        if isinstance(m, WeightQuantizer):
            w_bits.append(int(m.n_bits))
        elif isinstance(m, ActQuantizer):
            a_bits.append(int(m.n_bits))

    print("\n[Full Model Check]")
    print(f"- WeightQuantizers: {len(w_bits)}; bits: {Counter(w_bits)}")
    print(f"- ActQuantizers:    {len(a_bits)}; bits: {Counter(a_bits)}")

    # Coverage: how many QuantLayer quantizers are actually provided by the checkpoint.
    ckpt = _load_ckpt(ckpt_path)
    expected = []
    for m in qnn.model.modules():
        if hasattr(m, "weight_quantizer") and hasattr(m, "act_quantizer") and hasattr(m.weight_quantizer, "module_name"):
            expected.append(m.weight_quantizer.module_name)
            expected.append(m.act_quantizer.module_name)
    expected = list(dict.fromkeys(expected))
    provided = sum(1 for k in expected if k in ckpt)
    print(f"- QuantLayer quantizers covered by ckpt: {provided}/{len(expected)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to qdiff checkpoint (e.g., new_ckpt.pth)")
    ap.add_argument("--config", required=True, help="Path to qdiff config.yaml")
    ap.add_argument("--w_bit", type=int, default=4)
    ap.add_argument("--a_bit", type=int, default=8)
    ap.add_argument(
        "--try_build_model",
        action="store_true",
        help="If set (and CUDA is available), build SDXL-Turbo UNet and fully load ckpt to report bit stats.",
    )
    args = ap.parse_args()

    ckpt_path = str(Path(args.ckpt).expanduser())
    cfg_path = str(Path(args.config).expanduser())

    ckpt = _load_ckpt(ckpt_path)
    print("[Checkpoint]")
    print(f"- path: {ckpt_path}")
    print(f"- entries: {len(ckpt)}")

    mp_len = _infer_mp_nbits(ckpt)
    if mp_len is not None:
        print(f"- inferred mixed-precision list length: {mp_len} (e.g., [2,4,8] => 3)")

    suffix_counts = _count_key_suffixes(ckpt)
    print(f"- key type counts: {dict(suffix_counts)}")

    full_format = sum(1 for v in ckpt.values() if _is_full_quant_param_entry(v))
    buffer_only_format = sum(1 for v in ckpt.values() if isinstance(v, dict) and "delta_list" in v)
    print(f"- qdiff full-format entries: {full_format}")
    print(f"- qdiff buffer-only entries: {buffer_only_format}")

    ok, empty, bad = _check_delta_shapes(ckpt, mp_len)
    print(f"- quantizer buffers: ok={ok}, empty={empty}, bad={bad}")
    if ok == 0:
        raise RuntimeError(
            "No non-empty qdiff quantizer entries with delta_list/zero_point_list found in either "
            "full [buffers, parameters] format or buffer-only format; ckpt is not usable."
        )
    if bad > 0:
        raise RuntimeError("Some quantizer entries have malformed delta_list/zero_point_list shapes.")

    # Spot-check a few known keys.
    sample_keys = [
        "conv_in.weight_quantizer",
        "conv_in.act_quantizer",
        "time_embedding.linear_1.weight_quantizer",
        "time_embedding.linear_1.act_quantizer",
    ]
    missing = [k for k in sample_keys if k not in ckpt]
    print(f"- spotcheck missing keys: {missing}")

    if args.try_build_model:
        _try_full_model_check(cfg_path, ckpt_path, args.w_bit, args.a_bit)

    print("\nOK")


if __name__ == "__main__":
    main()
