#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


_ROOT = Path(__file__).resolve().parents[1]

# Ensure we can import `schedulers/*` when executed from arbitrary working directories.
_PROJECT_ROOT = None
for p in Path(__file__).resolve().parents:
    if (p / "schedulers").is_dir():
        _PROJECT_ROOT = p
        break
if _PROJECT_ROOT is not None and str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

def _load_cov_dict(path: Path) -> Dict[int, np.ndarray]:
    obj = np.load(str(path), allow_pickle=True).item()
    if not isinstance(obj, dict) or not obj:
        raise ValueError(f"Expected a non-empty dict in {path}")
    return {int(k): np.asarray(v) for k, v in obj.items()}


def _v_sigma_from_cov(cov: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    cov = np.asarray(cov, dtype=np.float64)
    var_q = np.maximum(cov[0, 0, :], eps)
    var_e = np.maximum(cov[1, 1, :], eps)
    cov_qe = cov[0, 1, :]
    rho = cov_qe / (np.sqrt(var_q * var_e) + eps)
    rho = np.clip(rho, -0.9999, 0.9999)
    return var_e * (1.0 - rho * rho)


def _flatten_quant_error(fp16_outputs: torch.Tensor, quant_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if fp16_outputs.ndim == 5:
        fp16_outputs = fp16_outputs.squeeze(1)
        quant_outputs = quant_outputs.squeeze(1)
    if fp16_outputs.ndim != 4:
        raise ValueError(f"Expected [N,C,H,W], got {tuple(fp16_outputs.shape)}")

    fp16_outputs = fp16_outputs.to(dtype=torch.float32)
    quant_outputs = quant_outputs.to(dtype=torch.float32)

    _, c, _, _ = fp16_outputs.shape
    fp16_flat = fp16_outputs.permute(0, 2, 3, 1).reshape(-1, c)
    quant_flat = quant_outputs.permute(0, 2, 3, 1).reshape(-1, c)
    err_flat = quant_flat - fp16_flat
    return quant_flat, err_flat


def _correction_factor_from_flat(
    quant_flat: torch.Tensor,
    err_flat: torch.Tensor,
    *,
    sigma: float,
    dt: float,
    outlier_threshold: float,
    eps: float = 1e-12,
) -> float:
    if quant_flat.ndim != 2 or err_flat.ndim != 2 or quant_flat.shape != err_flat.shape:
        raise ValueError(
            f"Expected quant_flat/err_flat with same shape (M,C), got {tuple(quant_flat.shape)} vs {tuple(err_flat.shape)}"
        )

    quant_values = quant_flat.reshape(-1).to(dtype=torch.float64)
    err_values = err_flat.reshape(-1).to(dtype=torch.float64)

    mean_e = torch.mean(err_values)
    std_e = torch.std(err_values, unbiased=False)
    cutoff = float(outlier_threshold) * std_e
    if float(cutoff.item()) > 0.0:
        keep = torch.abs(err_values - mean_e) <= cutoff
        quant_values = quant_values[keep]
        err_values = err_values[keep]

    count = int(err_values.numel())
    if count < 2:
        raise ValueError(f"Need at least two retained values after trimming, got {count}")

    mean_q = torch.mean(quant_values)
    mean_e2 = torch.mean(err_values)
    denom = float(count - 1)
    var_q = torch.sum((quant_values - mean_q) ** 2) / denom
    var_e = torch.sum((err_values - mean_e2) ** 2) / denom
    cov_qe = torch.sum((quant_values - mean_q) * (err_values - mean_e2)) / denom

    var_q = torch.clamp(var_q, min=eps)
    var_e = torch.clamp(var_e, min=eps)
    rho = cov_qe / (torch.sqrt(var_q * var_e) + eps)
    rho = torch.clamp(rho, -0.9999, 0.9999)
    v_sigma = var_e * (1.0 - rho * rho)
    c = v_sigma * (abs(float(dt)) / (2.0 * float(sigma) + eps))
    return float(c.item())


@dataclass(frozen=True)
class SigmaStep:
    sigma: float
    dt: float


def _build_sigma_steps_for_timesteps(
    *,
    model_id: str,
    subfolder: str,
    num_inference_steps: int,
) -> Dict[int, SigmaStep]:
    from schedulers.euler_qdrift import EulerQDriftScheduler

    try:
        scheduler = EulerQDriftScheduler.from_pretrained(model_id, subfolder=subfolder)
    except Exception as e:
        print(
            "[warn] Failed to load scheduler config from "
            f"{model_id}/{subfolder}: {e}. Falling back to SDXL Euler defaults."
        )
        scheduler = EulerQDriftScheduler(
            beta_schedule="scaled_linear",
            timestep_spacing="leading",
            steps_offset=1,
            use_karras_sigmas=False,
        )
    scheduler.set_timesteps(int(num_inference_steps))

    timesteps = [int(t) for t in scheduler.timesteps.cpu().numpy().astype(int).tolist()]
    sigmas = scheduler.sigmas.to(dtype=torch.float64, device="cpu")
    if len(sigmas) != len(timesteps) + 1:
        raise RuntimeError(f"Expected len(sigmas)=len(timesteps)+1, got {len(sigmas)} vs {len(timesteps)}")

    out: Dict[int, SigmaStep] = {}
    for i, t in enumerate(timesteps):
        sigma_from = float(sigmas[i].item())
        sigma_to = float(sigmas[i + 1].item())
        dt = float(sigma_to - sigma_from)  # EulerQDrift uses dt = sigma_next - sigma_hat (sigma_hat=sigma when s_churn=0)
        out[int(t)] = SigmaStep(sigma=sigma_from, dt=dt)
    return out


def _correction_factor_scalar(*, v_sigma: np.ndarray, sigma: float, dt: float, eps: float = 1e-12) -> float:
    c = np.asarray(v_sigma, dtype=np.float64) * (abs(dt) / (2.0 * sigma + eps))
    return float(np.mean(c))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Per-timestep correction factor c_i over R random calibration subsets of size K, "
            "computed from saved FP/quantized output pairs and a reference cov_dict."
        )
    )
    ap.add_argument(
        "--data_output_pairs",
        type=Path,
        required=True,
        help="Path to data_output_pairs.pth (fp16_output/quant_output).",
    )
    ap.add_argument(
        "--oracle_cov_dict",
        type=Path,
        required=True,
        help="Path to reference cov_dict.npy.",
    )
    ap.add_argument("--num_inference_steps", type=int, default=30, help="Inference steps (SDXL default: 30).")
    ap.add_argument("--num_trials", type=int, default=200, help="Number of trials per calibration size.")
    ap.add_argument(
        "--calib_sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Calibration subset sizes K to evaluate (e.g., 50 10 5 1). Default: 1 (single-sample).",
    )
    ap.add_argument(
        "--nested_subsets",
        action="store_true",
        help="Use nested subsets per trial: sample max(K) indices then take prefixes for smaller K (default).",
    )
    ap.add_argument(
        "--independent_subsets",
        action="store_true",
        help="Use independent subsets per size (overrides --nested_subsets).",
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for selecting trial indices.")
    ap.add_argument("--max_samples", type=int, default=None, help="Restrict trial index sampling to the first N paired samples.")
    ap.add_argument("--outlier_threshold", type=float, default=4.0, help="Outlier threshold (sigma units).")
    ap.add_argument(
        "--progress_every",
        type=int,
        default=1,
        help="Print progress every N trials with elapsed time and ETA (default: 1).",
    )
    ap.add_argument(
        "--scheduler_model_id",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="HF model id used to load the scheduler config.",
    )
    ap.add_argument(
        "--scheduler_subfolder",
        type=str,
        default="scheduler",
        help="Subfolder for scheduler config (default: scheduler).",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=_ROOT,
        help="Output directory root (default: this folder).",
    )
    ap.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Output JSON path (default: <out_dir>/subset_trials/cfactor_ci*_trials.json).",
    )
    args = ap.parse_args()

    if not args.data_output_pairs.exists():
        raise FileNotFoundError(f"data_output_pairs not found: {args.data_output_pairs}")
    if not args.oracle_cov_dict.exists():
        raise FileNotFoundError(f"oracle_cov_dict not found: {args.oracle_cov_dict}")

    out_dir = args.out_dir.resolve()
    trials_dir = out_dir / "subset_trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    oracle_cov = _load_cov_dict(args.oracle_cov_dict)
    timesteps_desc = np.array(sorted(oracle_cov.keys(), reverse=True), dtype=np.int32)

    sigma_steps = _build_sigma_steps_for_timesteps(
        model_id=str(args.scheduler_model_id),
        subfolder=str(args.scheduler_subfolder),
        num_inference_steps=int(args.num_inference_steps),
    )

    missing = [int(t) for t in timesteps_desc.tolist() if int(t) not in sigma_steps]
    if missing:
        raise RuntimeError(
            "Reference cov_dict timesteps are missing from the scheduler mapping. "
            f"missing={missing} (num_inference_steps={args.num_inference_steps})"
        )

    # Oracle c_i (scalar per timestep, channel-averaged).
    oracle_c = np.zeros((timesteps_desc.size,), dtype=np.float64)
    for j, t in enumerate(timesteps_desc.tolist()):
        step = sigma_steps[int(t)]
        v = _v_sigma_from_cov(oracle_cov[int(t)])
        oracle_c[j] = _correction_factor_scalar(v_sigma=v, sigma=step.sigma, dt=step.dt)

    # Load saved pairs lazily via mmap.
    data = torch.load(str(args.data_output_pairs), map_location="cpu", mmap=True)
    fp16_dict = data["fp16_output"]
    quant_dict = data["quant_output"]
    total_n = int(data.get("num_samples", 0))
    if total_n <= 0:
        raise ValueError(f"Invalid num_samples in {args.data_output_pairs}: {total_n}")
    sampling_n = total_n
    if args.max_samples is not None:
        if int(args.max_samples) <= 0:
            raise ValueError("--max_samples must be positive when provided")
        sampling_n = min(total_n, int(args.max_samples))

    calib_sizes = sorted({int(k) for k in args.calib_sizes if int(k) > 0}, reverse=True)
    if not calib_sizes:
        raise ValueError("--calib_sizes must contain at least one positive integer")
    max_k = int(max(calib_sizes))
    if max_k > sampling_n:
        raise ValueError(f"max(--calib_sizes)={max_k} exceeds usable samples={sampling_n}")

    rng = np.random.default_rng(int(args.seed))
    r = int(args.num_trials)
    if r <= 0:
        raise ValueError("--num_trials must be > 0")

    nested = True
    if args.independent_subsets:
        nested = False
    elif args.nested_subsets:
        nested = True

    trials_c_by_size: Dict[int, np.ndarray] = {k: np.zeros((r, timesteps_desc.size), dtype=np.float64) for k in calib_sizes}
    trial_indices_by_size: Dict[int, List[List[int]]] = {k: [] for k in calib_sizes}

    timesteps_list = timesteps_desc.tolist()
    start_time = time.perf_counter()
    progress_every = max(int(args.progress_every), 1)
    for rid in range(r):
        if nested:
            base = rng.choice(sampling_n, size=max_k, replace=False).astype(int).tolist()
            base_arr = np.asarray(base, dtype=np.int64)
            base_idx = torch.as_tensor(base_arr, dtype=torch.long)

            for k in calib_sizes:
                idxs_k = base_arr[: int(k)]
                trial_indices_by_size[int(k)].append([int(x) for x in idxs_k.tolist()])

            for j, t in enumerate(timesteps_list):
                fp16_max = fp16_dict[int(t)][base_idx]
                quant_max = quant_dict[int(t)][base_idx]
                quant_flat_max, err_flat_max = _flatten_quant_error(fp16_max, quant_max)
                h = int(fp16_max.shape[-2])
                w = int(fp16_max.shape[-1])
                hw = h * w

                step = sigma_steps[int(t)]
                for k in calib_sizes:
                    m = int(k) * hw
                    trials_c_by_size[int(k)][rid, j] = _correction_factor_from_flat(
                        quant_flat_max[:m],
                        err_flat_max[:m],
                        sigma=step.sigma,
                        dt=step.dt,
                        outlier_threshold=float(args.outlier_threshold),
                    )
        else:
            for k in calib_sizes:
                idxs = rng.choice(sampling_n, size=int(k), replace=False).astype(np.int64)
                idxs_idx = torch.as_tensor(idxs, dtype=torch.long)
                trial_indices_by_size[int(k)].append([int(x) for x in idxs.tolist()])

                for j, t in enumerate(timesteps_list):
                    fp16 = fp16_dict[int(t)][idxs_idx]
                    quant = quant_dict[int(t)][idxs_idx]
                    quant_flat, err_flat = _flatten_quant_error(fp16, quant)
                    step = sigma_steps[int(t)]
                    trials_c_by_size[int(k)][rid, j] = _correction_factor_from_flat(
                        quant_flat,
                        err_flat,
                        sigma=step.sigma,
                        dt=step.dt,
                        outlier_threshold=float(args.outlier_threshold),
                    )

        if (rid + 1) % progress_every == 0 or (rid + 1) == r:
            elapsed = time.perf_counter() - start_time
            completed = rid + 1
            rate = completed / max(elapsed, 1e-9)
            eta = (r - completed) / max(rate, 1e-9)
            print(
                f"[trial] {completed}/{r} elapsed={elapsed/60.0:.2f}m eta={eta/60.0:.2f}m",
                flush=True,
            )

    envelopes_by_size = {
        int(k): {
            "min": np.min(trials_c_by_size[int(k)], axis=0),
            "max": np.max(trials_c_by_size[int(k)], axis=0),
            "median": np.median(trials_c_by_size[int(k)], axis=0),
        }
        for k in calib_sizes
    }

    trials_summary = {
        "experiment": "subset_trials_multisize" if calib_sizes != [1] else "expA_cfactor_ci",
        "model": "sdxl_w3a4",
        "num_inference_steps": int(args.num_inference_steps),
        "num_trials": int(args.num_trials),
        "seed": int(args.seed),
        "outlier_threshold": float(args.outlier_threshold),
        "nested_subsets": bool(nested),
        "calib_sizes": [int(k) for k in calib_sizes],
        "oracle_cov_dict": str(args.oracle_cov_dict),
        "data_output_pairs": str(args.data_output_pairs),
        "num_samples_total": int(total_n),
        "num_samples_used_for_trials": int(sampling_n),
        "timesteps_desc": timesteps_desc.astype(int).tolist(),
        "oracle_c": oracle_c.tolist(),
        "by_size": {
            str(int(k)): {
                "trial_indices": trial_indices_by_size[int(k)],
                "trials_c": trials_c_by_size[int(k)].tolist(),
                "envelope": {
                    "min": envelopes_by_size[int(k)]["min"].tolist(),
                    "max": envelopes_by_size[int(k)]["max"].tolist(),
                    "median": envelopes_by_size[int(k)]["median"].tolist(),
                },
            }
            for k in calib_sizes
        },
    }

    trials_path = trials_dir / ("cfactor_ci_trials.json" if calib_sizes == [1] else "cfactor_ci_subsample_trials.json")
    if args.output_json is not None:
        trials_path = args.output_json
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    trials_path.write_text(json.dumps(trials_summary, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {trials_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
