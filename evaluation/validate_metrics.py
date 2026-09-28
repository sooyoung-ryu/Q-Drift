#!/usr/bin/env python3
"""Check that a metrics JSON written by evaluation/compute_metrics.py has every requested
metric (e.g. no ``fid: null``) for each model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

FP_KEY = "fp16"
QUANT_KEY = "quant_baseline"
QDRIFT_PREFIXES = (
    "quant_bias0.0_qdrift_scale1.0_scalar",
    "quant_bias_0.0_drift1.0_scalar",
)
REQUIRED_ALL = ("fid", "clip_score")
REQUIRED_NON_FP = ("psnr_vs_fp16", "lpips_vs_fp16", "ssim_vs_fp16")
COUNT_FIELDS_ALL = ("fid_num_images_used", "fid_num_images_total", "clip_score_num_pairs")
COUNT_FIELDS_NON_FP = ("similarity_num_pairs",)


class MetricsValidationError(ValueError):
    """Raised when a metrics JSON does not satisfy the release checks."""


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def discover_qdrift_key(models: Mapping[str, Any]) -> str:
    matches = [key for key in models if any(key.startswith(prefix) for prefix in QDRIFT_PREFIXES)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise MetricsValidationError(
            "Could not auto-detect scalar Q-Drift entry; pass --qdrift-key."
        )
    raise MetricsValidationError(
        f"Multiple scalar Q-Drift entries found {matches}; pass --qdrift-key."
    )


def require_model(models: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in models:
        raise MetricsValidationError(f"Missing required model entry: {key}")
    value = models[key]
    if not isinstance(value, Mapping):
        raise MetricsValidationError(f"Model entry {key} must be an object.")
    return value


def check_finite_metric(model_key: str, model: Mapping[str, Any], field: str) -> None:
    if field not in model:
        raise MetricsValidationError(f"{model_key}: missing metric {field}")
    if not is_finite_number(model[field]):
        raise MetricsValidationError(f"{model_key}: {field} must be finite, got {model[field]!r}")


def check_count(model_key: str, model: Mapping[str, Any], field: str, expected: int) -> None:
    if field not in model:
        raise MetricsValidationError(f"{model_key}: missing count {field}")
    value = model[field]
    if isinstance(value, bool) or int(value) != expected:
        raise MetricsValidationError(f"{model_key}: {field}={value!r}, expected {expected}")


def validate_metrics_obj(obj: Mapping[str, Any], num_samples: int, qdrift_key: Optional[str] = None) -> Dict[str, str]:
    if num_samples <= 0:
        raise MetricsValidationError("--num-samples must be positive.")
    models = obj.get("models")
    if not isinstance(models, Mapping):
        raise MetricsValidationError("Metrics JSON must contain a 'models' object.")

    qdrift = qdrift_key or discover_qdrift_key(models)
    required = {
        FP_KEY: require_model(models, FP_KEY),
        QUANT_KEY: require_model(models, QUANT_KEY),
        qdrift: require_model(models, qdrift),
    }

    for model_key, model in required.items():
        for field in REQUIRED_ALL:
            check_finite_metric(model_key, model, field)
        for field in COUNT_FIELDS_ALL:
            check_count(model_key, model, field, num_samples)
        if model_key != FP_KEY:
            for field in REQUIRED_NON_FP:
                check_finite_metric(model_key, model, field)
            for field in COUNT_FIELDS_NON_FP:
                check_count(model_key, model, field, num_samples)

    return {"fp_key": FP_KEY, "quant_key": QUANT_KEY, "qdrift_key": qdrift}


def validate_metrics_file(path: Path, num_samples: int, qdrift_key: Optional[str] = None) -> Dict[str, str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, Mapping):
        raise MetricsValidationError("Metrics JSON root must be an object.")
    return validate_metrics_obj(obj, num_samples=num_samples, qdrift_key=qdrift_key)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_json", type=Path)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--qdrift-key", help="override scalar Q-Drift model key auto-detection")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        keys = validate_metrics_file(args.metrics_json, args.num_samples, args.qdrift_key)
    except MetricsValidationError as exc:
        print(f"[invalid] {args.metrics_json}: {exc}")
        return 1
    print(
        f"[valid] {args.metrics_json}: fp={keys['fp_key']} "
        f"quant={keys['quant_key']} qdrift={keys['qdrift_key']} num_samples={args.num_samples}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
