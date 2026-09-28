"""
Build PTQD-style correction statistics from paired FP16/quantized SDXL UNet outputs.

The official PTQD implementation estimates, for each timestep,

    error = quant_output - fp_output ~= k_t * fp_output + bias_t + residual.

At sampling time it corrects the quantized denoiser output as

    fp_output_hat = (quant_output - bias_t) / (1 + k_t).

This script adapts that statistic collection to the `collect_statistics.py`
payload used by the SDXL W3A4 Q-Drift pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import torch


def _as_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 5 and x.shape[1] == 1:
        x = x.squeeze(1)
    if x.ndim != 4:
        raise ValueError(f"Expected [N,C,H,W] or [N,1,C,H,W], got {tuple(x.shape)}")
    return x.float()


def _linear_slope(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.flatten().double()
    y = y.flatten().double()
    x_mean = x.mean()
    y_mean = y.mean()
    denom = torch.sum((x - x_mean) ** 2).clamp_min(1e-20)
    slope = torch.sum((x - x_mean) * (y - y_mean)) / denom
    return float(slope.item())


def _sample_shard_path(shard_dir: str, sample_index: int) -> str:
    return os.path.join(shard_dir, f"sample_{sample_index:06d}.pth")


def _stream_ptqd_stats_from_shards(data: Dict, data_output_pairs_path: str) -> Tuple[Dict[int, float], Dict[int, np.ndarray], Dict[int, float]]:
    total_samples = int(data["num_samples"])
    timesteps = [int(t) for t in data["timesteps"]]
    shard_dir = data.get("sample_shards_dir", "sample_shards")
    if not os.path.isabs(shard_dir):
        shard_dir = os.path.join(os.path.dirname(data_output_pairs_path), shard_dir)

    first_shard = torch.load(_sample_shard_path(shard_dir, 0), map_location="cpu")
    first_t = timesteps[0]
    first_fp = _as_nchw(first_shard["fp16_output"][first_t])
    channels = int(first_fp.shape[1])

    accum = {
        t: {
            "n": 0,
            "sum_x": 0.0,
            "sum_y": 0.0,
            "sum_x2": 0.0,
            "sum_xy": 0.0,
            "bias_sum": torch.zeros(channels, dtype=torch.float64),
            "bias_count": 0,
        }
        for t in timesteps
    }

    for sample_index in range(total_samples):
        shard = torch.load(_sample_shard_path(shard_dir, sample_index), map_location="cpu")
        for timestep in timesteps:
            fp = _as_nchw(shard["fp16_output"][timestep])
            quant = _as_nchw(shard["quant_output"][timestep])
            error = quant - fp
            fp64 = fp.double()
            error64 = error.double()
            state = accum[timestep]
            state["n"] += fp64.numel()
            state["sum_x"] += float(fp64.sum().item())
            state["sum_y"] += float(error64.sum().item())
            state["sum_x2"] += float((fp64 * fp64).sum().item())
            state["sum_xy"] += float((fp64 * error64).sum().item())
            state["bias_sum"] += error64.sum(dim=(0, 2, 3)).cpu()
            state["bias_count"] += int(error64.shape[0] * error64.shape[2] * error64.shape[3])

    kt_dict: Dict[int, float] = {}
    bias_dict: Dict[int, np.ndarray] = {}
    for timestep in timesteps:
        state = accum[timestep]
        n = float(state["n"])
        denom = max(state["sum_x2"] - (state["sum_x"] * state["sum_x"] / n), 1e-20)
        numer = state["sum_xy"] - (state["sum_x"] * state["sum_y"] / n)
        kt_dict[timestep] = float(max(numer / denom, 0.0))
        bias_dict[timestep] = (state["bias_sum"] / state["bias_count"]).cpu().numpy().astype(np.float32)

    residual_accum = {t: {"n": 0, "sum": 0.0, "sum2": 0.0} for t in timesteps}
    for sample_index in range(total_samples):
        shard = torch.load(_sample_shard_path(shard_dir, sample_index), map_location="cpu")
        for timestep in timesteps:
            fp = _as_nchw(shard["fp16_output"][timestep])
            quant = _as_nchw(shard["quant_output"][timestep])
            residual = (quant - (1.0 + kt_dict[timestep]) * fp).double()
            state = residual_accum[timestep]
            state["n"] += residual.numel()
            state["sum"] += float(residual.sum().item())
            state["sum2"] += float((residual * residual).sum().item())

    sigmaq_dict: Dict[int, float] = {}
    for timestep in timesteps:
        state = residual_accum[timestep]
        n = float(state["n"])
        variance = max((state["sum2"] - (state["sum"] * state["sum"] / n)) / max(n - 1.0, 1.0), 0.0)
        sigmaq_dict[timestep] = float(variance ** 0.5)

    return kt_dict, bias_dict, sigmaq_dict


def compute_ptqd_stats(data_output_pairs_path: str) -> Tuple[Dict[int, float], Dict[int, np.ndarray], Dict[int, float]]:
    data = torch.load(data_output_pairs_path, map_location="cpu")

    if data.get("format") == "ptqd_sdxl_sample_shards_manifest_v1":
        return _stream_ptqd_stats_from_shards(data, data_output_pairs_path)

    if "fp16_output" in data:
        fp16_dict = data["fp16_output"]
        quant_dict = data["quant_output"]
    else:
        fp16_dict = data.get("fp16_eps", data.get("fp16_outputs", {}))
        quant_dict = data.get("quant_eps", data.get("quant_outputs", {}))

    timesteps = [int(t) for t in data["timesteps"]]

    kt_dict: Dict[int, float] = {}
    bias_dict: Dict[int, np.ndarray] = {}
    sigmaq_dict: Dict[int, float] = {}

    for timestep in timesteps:
        fp = _as_nchw(fp16_dict[timestep])
        quant = _as_nchw(quant_dict[timestep])
        error = quant - fp

        kt = max(_linear_slope(fp, error), 0.0)
        bias = error.mean(dim=(0, 2, 3)).cpu().numpy().astype(np.float32)
        residual = quant - (1.0 + kt) * fp

        kt_dict[timestep] = float(kt)
        bias_dict[timestep] = bias
        sigmaq_dict[timestep] = float(residual.std().item())

    return kt_dict, bias_dict, sigmaq_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PTQD correction statistics for SDXL W3A4")
    parser.add_argument("--data_output_pairs_path", required=True, help="Paired FP/quantized outputs (data_output_pairs.pth)")
    parser.add_argument("--output_dir", required=True, help="Directory to save ptqd_stats.npy")
    args = parser.parse_args()

    if not os.path.exists(args.data_output_pairs_path):
        raise FileNotFoundError(args.data_output_pairs_path)

    os.makedirs(args.output_dir, exist_ok=True)
    kt_dict, bias_dict, sigmaq_dict = compute_ptqd_stats(args.data_output_pairs_path)

    payload = {
        "format": "ptqd_stats_v1",
        "source": args.data_output_pairs_path,
        "kt_dict": kt_dict,
        "bias_dict": bias_dict,
        "sigmaq_dict": sigmaq_dict,
        "note": "Euler adaptation uses PTQD mean/correlation correction; DDIM variance fusion is not used.",
    }

    out_path = os.path.join(args.output_dir, "ptqd_stats.npy")
    np.save(out_path, payload, allow_pickle=True)

    summary = {
        "num_timesteps": len(kt_dict),
        "timesteps": sorted(kt_dict.keys()),
        "kt_mean": float(np.mean(list(kt_dict.values()))) if kt_dict else 0.0,
        "kt_max": float(np.max(list(kt_dict.values()))) if kt_dict else 0.0,
        "sigmaq_mean": float(np.mean(list(sigmaq_dict.values()))) if sigmaq_dict else 0.0,
    }
    summary_path = os.path.join(args.output_dir, "ptqd_stats_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved PTQD stats: {out_path}")
    print(f"Saved PTQD summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
