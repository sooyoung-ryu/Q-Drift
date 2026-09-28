#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from fit_subset_statistics import materialize_calibration_from_indices


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_unique(order: Iterable[int], used: set[int]) -> int:
    for rid in order:
        rid = int(rid)
        if rid not in used:
            used.add(rid)
            return rid
    rid = int(next(iter(order)))
    used.add(rid)
    return rid


def _ensure_size_payload(trials: Dict, calib_size: int) -> Tuple[List[List[int]], np.ndarray, np.ndarray]:
    by_size = trials.get("by_size")
    if not isinstance(by_size, dict):
        raise ValueError("Expected subset-trials JSON with a 'by_size' payload from the multisize experiment.")

    key = str(int(calib_size))
    if key not in by_size:
        available = sorted(by_size.keys(), key=lambda x: int(x))
        raise ValueError(f"Calibration size K={calib_size} not found in subset-trials JSON. Available sizes: {available}")

    payload = by_size[key]
    trial_indices = payload.get("trial_indices")
    trials_c = payload.get("trials_c")
    oracle_c = trials.get("oracle_c")
    if not isinstance(trial_indices, list) or not isinstance(trials_c, list) or not isinstance(oracle_c, list):
        raise ValueError("Malformed subset-trials JSON: missing trial_indices/trials_c/oracle_c.")

    trial_indices = [[int(x) for x in row] for row in trial_indices]
    trials_c_arr = np.asarray(trials_c, dtype=np.float64)
    oracle_c_arr = np.asarray(oracle_c, dtype=np.float64)

    if trials_c_arr.ndim != 2:
        raise ValueError(f"Expected trials_c to have shape [R, T], got {trials_c_arr.shape}")
    if trials_c_arr.shape[0] != len(trial_indices):
        raise ValueError(
            f"Mismatch between number of trials in indices ({len(trial_indices)}) and trials_c ({trials_c_arr.shape[0]})."
        )
    if trials_c_arr.shape[1] != oracle_c_arr.shape[0]:
        raise ValueError(
            f"Mismatch between trials_c timesteps ({trials_c_arr.shape[1]}) and oracle_c ({oracle_c_arr.shape[0]})."
        )
    return trial_indices, trials_c_arr, oracle_c_arr


def _build_trial_rows(
    *,
    trial_indices: List[List[int]],
    trials_c: np.ndarray,
    oracle_c: np.ndarray,
) -> List[Dict]:
    rows: List[Dict] = []
    for rid, indices in enumerate(trial_indices):
        delta = trials_c[rid] - oracle_c
        abs_delta = np.abs(delta)
        rows.append(
            {
                "trial_id": int(rid),
                "indices": [int(x) for x in indices],
                "sum_abs_delta_c": float(np.sum(abs_delta)),
                "sum_delta_c": float(np.sum(delta)),
                "mean_abs_delta_c": float(np.mean(abs_delta)),
                "max_abs_delta_c": float(np.max(abs_delta)),
            }
        )
    return rows


def _select_stress_trials(rows: List[Dict]) -> Dict[str, Dict]:
    sum_abs = np.asarray([row["sum_abs_delta_c"] for row in rows], dtype=np.float64)
    sum_signed = np.asarray([row["sum_delta_c"] for row in rows], dtype=np.float64)

    used: set[int] = set()
    keys_and_orders = [
        ("min_sum_abs_delta_c", np.argsort(sum_abs)),
        ("max_sum_abs_delta_c", np.argsort(-sum_abs)),
        ("max_sum_delta_c", np.argsort(-sum_signed)),
        ("min_sum_delta_c", np.argsort(sum_signed)),
    ]

    selected: Dict[str, Dict] = {}
    for key, order in keys_and_orders:
        rid = _pick_unique(order.tolist(), used)
        selected[key] = dict(rows[int(rid)])
    return selected


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Select the four K-sample stress-test subsets (min/max sum_i |Delta c_i|, min/max sum_i Delta c_i) "
            "from the subset trials and fit pooled scalar statistics for each."
        )
    )
    ap.add_argument("--trials_json", type=Path, required=True, help="Subset-trials JSON from sample_subset_trials.py.")
    ap.add_argument("--data_output_pairs", type=Path, required=True, help="Paired FP/quantized outputs of the reference pool.")
    ap.add_argument("--calib_size", type=int, default=10, help="Calibration subset size K.")
    ap.add_argument("--outlier_threshold", type=float, default=4.0)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    ap.add_argument("--out_dir", type=Path, required=True, help="Output directory (selected_trials.json, calib_*).")
    args = ap.parse_args()

    if not args.trials_json.exists():
        raise FileNotFoundError(f"subset-trials JSON not found: {args.trials_json}")
    if not args.data_output_pairs.exists():
        raise FileNotFoundError(f"data_output_pairs not found: {args.data_output_pairs}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    trials = _load_json(args.trials_json)
    trial_indices, trials_c, oracle_c = _ensure_size_payload(trials, int(args.calib_size))
    trial_rows = _build_trial_rows(trial_indices=trial_indices, trials_c=trials_c, oracle_c=oracle_c)
    selected = _select_stress_trials(trial_rows)

    variant_tags = {
        "min_sum_abs_delta_c": f"k{int(args.calib_size)}_min_sum_abs_delta_c",
        "max_sum_abs_delta_c": f"k{int(args.calib_size)}_max_sum_abs_delta_c",
        "max_sum_delta_c": f"k{int(args.calib_size)}_max_sum_delta_c",
        "min_sum_delta_c": f"k{int(args.calib_size)}_min_sum_delta_c",
    }

    selected_with_tags: Dict[str, Dict] = {}
    for key, row in selected.items():
        calib_dir = args.out_dir / f"calib_{variant_tags[key]}"
        row_out = dict(row)
        row_out["variant_tag"] = variant_tags[key]
        row_out["calibration_dir"] = str(calib_dir)

        outputs = materialize_calibration_from_indices(
            data_output_pairs_path=args.data_output_pairs,
            output_dir=calib_dir,
            indices=row["indices"],
            outlier_threshold=float(args.outlier_threshold),
            overwrite=bool(args.overwrite),
            metadata={
                "experiment": "k10_subset_statistics",
                "source_trials_json": str(args.trials_json),
                "source_trial_id": int(row["trial_id"]),
                "selection_key": key,
                "variant_tag": variant_tags[key],
            },
        )
        row_out["mu_dict"] = outputs["mu_dict"]
        row_out["cov_dict"] = outputs["cov_dict"]
        row_out["calibration_metadata"] = outputs["metadata"]
        selected_with_tags[key] = row_out

    selection_payload = {
        "experiment": "k10_selection",
        "model": "sdxl_w3a4",
        "calib_size": int(args.calib_size),
        "source_trials_json": str(args.trials_json),
        "source_data_output_pairs": str(args.data_output_pairs),
        "outlier_threshold": float(args.outlier_threshold),
        "timesteps_desc": [int(x) for x in trials["timesteps_desc"]],
        "oracle_c": [float(x) for x in trials["oracle_c"]],
        "num_trials": int(len(trial_rows)),
        "variant_tags": variant_tags,
        "trials": trial_rows,
        "selected": selected_with_tags,
    }
    selection_path = args.out_dir / "selected_trials.json"
    selection_path.write_text(json.dumps(selection_payload, indent=2) + "\n", encoding="utf-8")

    print(f"[saved] {selection_path}")
    print("[selected]")
    for key in ("min_sum_abs_delta_c", "max_sum_abs_delta_c", "max_sum_delta_c", "min_sum_delta_c"):
        row = selected_with_tags[key]
        idx_text = ",".join(str(i) for i in row["indices"])
        print(
            f"  - {key}: trial_id={row['trial_id']} indices=[{idx_text}] "
            f"(sum_abs_delta_c={row['sum_abs_delta_c']:.6g}, sum_delta_c={row['sum_delta_c']:.6g})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
