#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from tqdm import tqdm


ArrayDict = Dict[int, np.ndarray]


def parse_indices_arg(text: str) -> List[int]:
    text = (text or "").strip()
    if not text:
        return []
    indices: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            indices.append(int(part))
    return indices


def _select_rows(x: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    if len(indices) == 0:
        raise ValueError("indices must be non-empty")
    idx = torch.as_tensor(list(indices), dtype=torch.long)
    return x.index_select(0, idx)


def _flatten_pair(fp16: torch.Tensor, quant: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if fp16.shape != quant.shape:
        raise ValueError(f"Shape mismatch: fp16={tuple(fp16.shape)} quant={tuple(quant.shape)}")
    quant_values = quant.detach().cpu().reshape(-1).to(dtype=torch.float64)
    err_values = (quant.detach().cpu() - fp16.detach().cpu()).reshape(-1).to(dtype=torch.float64)
    if not torch.isfinite(quant_values).all() or not torch.isfinite(err_values).all():
        raise ValueError("Input tensors contain non-finite values")
    return quant_values, err_values


def _fit_pooled_scalar_gaussian(fp16: torch.Tensor, quant: torch.Tensor, *, outlier_threshold: float) -> tuple[np.ndarray, np.ndarray, Dict[str, int | float]]:
    quant_values, err_values = _flatten_pair(fp16, quant)
    total_count = int(err_values.numel())
    if total_count == 0:
        raise ValueError("Cannot fit scalar Gaussian from empty tensors")

    mean_error = torch.mean(err_values)
    std_error = torch.std(err_values, unbiased=False)
    cutoff = float(outlier_threshold) * std_error
    if float(cutoff.item()) > 0.0:
        keep = torch.abs(err_values - mean_error) <= cutoff
        quant_values = quant_values[keep]
        err_values = err_values[keep]

    retained = int(err_values.numel())
    if retained < 2:
        raise ValueError(f"Need at least two retained values after trimming, got {retained}")

    mean_quant = torch.mean(quant_values)
    mean_error_after = torch.mean(err_values)
    denom = float(retained - 1)
    cov_qq = torch.sum((quant_values - mean_quant) ** 2) / denom
    cov_ee = torch.sum((err_values - mean_error_after) ** 2) / denom
    cov_qe = torch.sum((quant_values - mean_quant) * (err_values - mean_error_after)) / denom

    mu = np.array([[float(mean_quant.item())], [float(mean_error_after.item())]], dtype=np.float64)
    cov = np.array(
        [[[float(cov_qq.item())], [float(cov_qe.item())]], [[float(cov_qe.item())], [float(cov_ee.item())]]],
        dtype=np.float64,
    )
    stats: Dict[str, int | float] = {
        "total_values": total_count,
        "retained_values": retained,
        "trimmed_values": total_count - retained,
        "error_mean_before_trim": float(mean_error.item()),
        "error_std_before_trim": float(std_error.item()),
    }
    return mu, cov, stats


def materialize_calibration_from_indices(
    *,
    data_output_pairs_path: Path,
    output_dir: Path,
    indices: Sequence[int],
    outlier_threshold: float,
    overwrite: bool = False,
    metadata: Dict | None = None,
) -> Dict[str, str]:
    indices = [int(i) for i in indices]
    if not indices:
        raise ValueError("indices must be non-empty")
    if not data_output_pairs_path.exists():
        raise FileNotFoundError(f"Input not found: {data_output_pairs_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mu_path = output_dir / "mu_dict.npy"
    cov_path = output_dir / "cov_dict.npy"
    meta_path = output_dir / "calibration_subset.json"

    if (not overwrite) and mu_path.exists() and cov_path.exists():
        print(f"Skipping existing calibration under {output_dir}")
        return {"mu_dict": str(mu_path), "cov_dict": str(cov_path), "metadata": str(meta_path)}

    data = torch.load(str(data_output_pairs_path), map_location="cpu", mmap=True)
    fp16_dict = data["fp16_output"]
    quant_dict = data["quant_output"]
    timesteps = [int(t) for t in data["timesteps"]]
    total_n = int(data.get("num_samples", 0))
    if total_n <= 0:
        raise ValueError(f"Invalid num_samples in {data_output_pairs_path}: {total_n}")

    for idx in indices:
        if idx < 0 or idx >= total_n:
            raise ValueError(f"Index out of range: {idx} (valid: [0, {total_n - 1}])")

    mu_dict: ArrayDict = {}
    cov_dict: ArrayDict = {}
    counts: Dict[str, Dict[str, int | float]] = {}

    for timestep in tqdm(timesteps, desc=f"Materializing {output_dir.name}", leave=False):
        fp16 = _select_rows(fp16_dict[int(timestep)], indices)
        quant = _select_rows(quant_dict[int(timestep)], indices)
        mu, cov, stats = _fit_pooled_scalar_gaussian(
            fp16=fp16,
            quant=quant,
            outlier_threshold=float(outlier_threshold),
        )
        mu_dict[int(timestep)] = mu
        cov_dict[int(timestep)] = cov
        counts[str(int(timestep))] = stats

    np.save(str(mu_path), mu_dict)
    np.save(str(cov_path), cov_dict)

    meta = {
        "data_output_pairs_path": str(data_output_pairs_path),
        "indices": indices,
        "B": int(len(indices)),
        "num_samples_total": int(total_n),
        "timesteps": timesteps,
        "outlier_threshold": float(outlier_threshold),
        "format": "qdrift_scalar_gaussian_v1",
        "pooling": "samples, channels, and spatial positions are pooled before computing moments",
        "mu_shape": [2, 1],
        "cov_shape": [2, 2, 1],
        "counts": counts,
        "outputs": {"mu_dict": str(mu_path), "cov_dict": str(cov_path)},
    }
    if metadata:
        meta.update(metadata)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote: {mu_path}")
    print(f"Wrote: {cov_path}")
    print(f"Wrote: {meta_path}")
    return {"mu_dict": str(mu_path), "cov_dict": str(cov_path), "metadata": str(meta_path)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Create pooled scalar per-timestep mu_dict.npy and cov_dict.npy from a subset of SDXL W3A4 calibration samples."
        )
    )
    ap.add_argument(
        "--data_output_pairs_path",
        type=Path,
        required=True,
        help="Path to data_output_pairs.pth from collect_statistics.py",
    )
    ap.add_argument("--output_dir", type=Path, required=True, help="Output directory for mu_dict.npy and cov_dict.npy")
    ap.add_argument(
        "--indices",
        type=str,
        required=True,
        help="Comma-separated calibration indices (e.g. '0,12,99,101,203').",
    )
    ap.add_argument("--outlier_threshold", type=float, default=4.0)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = ap.parse_args()

    indices = parse_indices_arg(args.indices)
    if not indices:
        raise SystemExit("--indices must be non-empty")

    materialize_calibration_from_indices(
        data_output_pairs_path=args.data_output_pairs_path,
        output_dir=args.output_dir,
        indices=indices,
        outlier_threshold=float(args.outlier_threshold),
        overwrite=bool(args.overwrite),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
