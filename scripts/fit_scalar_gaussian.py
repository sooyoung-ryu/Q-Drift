"""Fit scalar Q-Drift Gaussian calibration statistics from raw paired outputs.

This script estimates one pooled 2D Gaussian per timestep for
``[quant_output, quant_output - fp_output]``. All samples, channels, and spatial
positions are pooled before computing the mean and covariance.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch


ArrayDict = Dict[float | int, np.ndarray]
TensorDict = Dict[float | int, List[torch.Tensor]]


@dataclass
class ErrorStats:
    count: int
    mean: float
    std: float


@dataclass
class JointStats:
    count: int
    mu: np.ndarray
    cov: np.ndarray


@dataclass
class LoadedPairs:
    fp16_by_t: TensorDict
    quant_by_t: TensorDict
    timesteps: List[float | int]
    inferred_num_samples: int
    total_num_samples_available: int
    file_sample_counts: Dict[str, int]
    file_sample_counts_available: Dict[str, int]


def _torch_load(path: Path) -> Mapping:
    kwargs = {"map_location": "cpu", "weights_only": False}
    load_path = str(path)
    try:
        kwargs["mmap"] = True
        return torch.load(load_path, **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("mmap", None)
        try:
            return torch.load(load_path, **kwargs)
        except TypeError:
            kwargs.pop("weights_only", None)
            return torch.load(load_path, **kwargs)


def _load_pair_file(path: Path) -> Mapping:
    data = _torch_load(path)
    if "fp16_output" in data and "quant_output" in data:
        return data
    fp16_dict = data.get("fp16_eps", data.get("fp16_outputs"))
    quant_dict = data.get("quant_eps", data.get("quant_outputs"))
    if fp16_dict is None or quant_dict is None:
        raise KeyError(f"{path} does not contain fp16/quant output dictionaries")
    return {
        "fp16_output": fp16_dict,
        "quant_output": quant_dict,
        "timesteps": data.get("timesteps", sorted(fp16_dict.keys())),
        "num_samples": data.get("num_samples"),
    }


def _iter_flat_chunks(
    fp16: torch.Tensor,
    quant: torch.Tensor,
    *,
    chunk_elements: int,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    if fp16.shape != quant.shape:
        raise ValueError(f"Shape mismatch: fp16={tuple(fp16.shape)} quant={tuple(quant.shape)}")
    fp16_flat = fp16.detach().cpu().reshape(-1)
    quant_flat = quant.detach().cpu().reshape(-1)
    for start in range(0, fp16_flat.numel(), chunk_elements):
        stop = min(start + chunk_elements, fp16_flat.numel())
        fp_chunk = fp16_flat[start:stop].to(dtype=torch.float64)
        q_chunk = quant_flat[start:stop].to(dtype=torch.float64)
        if not torch.isfinite(fp_chunk).all() or not torch.isfinite(q_chunk).all():
            raise ValueError("Input tensors contain non-finite values")
        yield fp_chunk, q_chunk


def _accumulate_error_stats(
    fp16_tensors: Sequence[torch.Tensor],
    quant_tensors: Sequence[torch.Tensor],
    *,
    chunk_elements: int,
) -> ErrorStats:
    count = 0
    sum_error = 0.0
    sum_error2 = 0.0
    for fp16, quant in zip(fp16_tensors, quant_tensors):
        for fp_chunk, q_chunk in _iter_flat_chunks(fp16, quant, chunk_elements=chunk_elements):
            error = q_chunk - fp_chunk
            count += int(error.numel())
            sum_error += float(error.sum().item())
            sum_error2 += float((error * error).sum().item())
    if count == 0:
        raise ValueError("Cannot fit scalar Gaussian from empty tensors")
    mean = sum_error / count
    variance = max(sum_error2 / count - mean * mean, 0.0)
    return ErrorStats(count=count, mean=mean, std=float(np.sqrt(variance)))


def _accumulate_joint_stats(
    fp16_tensors: Sequence[torch.Tensor],
    quant_tensors: Sequence[torch.Tensor],
    *,
    error_stats: ErrorStats,
    outlier_threshold: float,
    chunk_elements: int,
) -> JointStats:
    count = 0
    sum_quant = 0.0
    sum_error = 0.0
    sum_quant2 = 0.0
    sum_error2 = 0.0
    sum_quant_error = 0.0
    cutoff = outlier_threshold * error_stats.std

    for fp16, quant in zip(fp16_tensors, quant_tensors):
        for fp_chunk, q_chunk in _iter_flat_chunks(fp16, quant, chunk_elements=chunk_elements):
            error = q_chunk - fp_chunk
            if cutoff > 0.0:
                keep = torch.abs(error - error_stats.mean) <= cutoff
                q_chunk = q_chunk[keep]
                error = error[keep]
            local_count = int(error.numel())
            if local_count == 0:
                continue
            count += local_count
            sum_quant += float(q_chunk.sum().item())
            sum_error += float(error.sum().item())
            sum_quant2 += float((q_chunk * q_chunk).sum().item())
            sum_error2 += float((error * error).sum().item())
            sum_quant_error += float((q_chunk * error).sum().item())

    if count < 2:
        raise ValueError(f"Need at least two retained values after trimming, got {count}")

    mu_quant = sum_quant / count
    mu_error = sum_error / count
    denom = count - 1
    cov_qq = (sum_quant2 - count * mu_quant * mu_quant) / denom
    cov_ee = (sum_error2 - count * mu_error * mu_error) / denom
    cov_qe = (sum_quant_error - count * mu_quant * mu_error) / denom
    mu = np.array([[mu_quant], [mu_error]], dtype=np.float64)
    cov = np.array([[[cov_qq], [cov_qe]], [[cov_qe], [cov_ee]]], dtype=np.float64)
    return JointStats(count=count, mu=mu, cov=cov)


def _append_tensor(store: MutableMapping[float | int, List[torch.Tensor]], timestep: float | int, tensor: torch.Tensor) -> None:
    store.setdefault(timestep, []).append(tensor)


def _validate_declared_timesteps(path: Path, data: Mapping) -> List[float | int]:
    fp16_dict = data["fp16_output"]
    quant_dict = data["quant_output"]
    declared = list(data.get("timesteps", sorted(fp16_dict.keys())))
    declared_set = set(declared)
    fp16_set = set(fp16_dict.keys())
    quant_set = set(quant_dict.keys())
    if fp16_set != quant_set or declared_set != fp16_set:
        raise ValueError(
            f"{path.name} has inconsistent timestep sets: "
            f"declared={len(declared_set)} fp16={len(fp16_set)} quant={len(quant_set)}"
        )
    return sorted(declared)


def _validate_tensor_pair(path: Path, timestep: float | int, fp16: torch.Tensor, quant: torch.Tensor) -> int:
    if not isinstance(fp16, torch.Tensor) or not isinstance(quant, torch.Tensor):
        raise TypeError(f"{path.name} timestep {timestep!r} values must be tensors")
    if fp16.shape != quant.shape:
        raise ValueError(f"{path.name} timestep {timestep!r} shape mismatch: {tuple(fp16.shape)} vs {tuple(quant.shape)}")
    if fp16.ndim == 0 or fp16.shape[0] <= 0:
        raise ValueError(f"{path.name} timestep {timestep!r} has invalid leading sample dimension {tuple(fp16.shape)}")
    return int(fp16.shape[0])


def _load_pairs(
    paths: Sequence[Path],
    *,
    expected_num_samples: int | None = None,
    max_samples: int | None = None,
) -> LoadedPairs:
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")
    fp16_by_t: TensorDict = {}
    quant_by_t: TensorDict = {}
    file_sample_counts: Dict[str, int] = {}
    file_sample_counts_available: Dict[str, int] = {}
    reference_timesteps: List[float | int] | None = None
    reference_trailing_shapes: Dict[float | int, Tuple[int, ...]] = {}
    remaining = max_samples
    name_counts: Dict[str, int] = {}
    for path in paths:
        name_counts[path.name] = name_counts.get(path.name, 0) + 1

    for path_index, path in enumerate(paths):
        metadata_key = path.name if name_counts[path.name] == 1 else f"{path_index:05d}_{path.name}"
        if remaining is not None and remaining <= 0:
            break
        data = _load_pair_file(path)
        fp16_dict = data["fp16_output"]
        quant_dict = data["quant_output"]
        timesteps = _validate_declared_timesteps(path, data)
        if reference_timesteps is None:
            reference_timesteps = timesteps
        elif timesteps != reference_timesteps:
            raise ValueError(f"{path.name} has timesteps inconsistent with earlier inputs")

        per_file_n: int | None = None
        for timestep in timesteps:
            fp16 = fp16_dict[timestep]
            quant = quant_dict[timestep]
            n = _validate_tensor_pair(path, timestep, fp16, quant)
            if per_file_n is None:
                per_file_n = n
            elif n != per_file_n:
                raise ValueError(f"{path.name} has inconsistent leading sample counts across timesteps")
            trailing_shape = tuple(fp16.shape[1:])
            previous_shape = reference_trailing_shapes.setdefault(timestep, trailing_shape)
            if trailing_shape != previous_shape:
                raise ValueError(
                    f"{path.name} timestep {timestep!r} trailing shape {trailing_shape} "
                    f"does not match previous shape {previous_shape}"
                )
        assert per_file_n is not None
        reported = data.get("num_samples")
        if reported is not None and int(reported) != per_file_n:
            raise ValueError(f"{path.name} reports num_samples={reported}, but tensors infer {per_file_n}")
        file_sample_counts_available[metadata_key] = per_file_n

        take_n = per_file_n if remaining is None else min(per_file_n, remaining)
        if take_n <= 0:
            continue
        for timestep in timesteps:
            _append_tensor(fp16_by_t, timestep, fp16_dict[timestep][:take_n])
            _append_tensor(quant_by_t, timestep, quant_dict[timestep][:take_n])
        file_sample_counts[metadata_key] = take_n
        if remaining is not None:
            remaining -= take_n

    timesteps = reference_timesteps or []
    if not timesteps:
        raise ValueError("No timesteps found in input data")
    inferred_num_samples = sum(file_sample_counts.values())
    total_num_samples_available = sum(file_sample_counts_available.values())
    if expected_num_samples is not None and inferred_num_samples != expected_num_samples:
        raise ValueError(f"Expected {expected_num_samples} samples, but input tensors infer {inferred_num_samples}")
    return LoadedPairs(
        fp16_by_t=fp16_by_t,
        quant_by_t=quant_by_t,
        timesteps=timesteps,
        inferred_num_samples=inferred_num_samples,
        total_num_samples_available=total_num_samples_available,
        file_sample_counts=file_sample_counts,
        file_sample_counts_available=file_sample_counts_available,
    )


def _fingerprint_paths(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(int(stat.st_mtime_ns)).encode("ascii"))
    return digest.hexdigest()


def _safe_source_paths(paths: Sequence[Path]) -> List[str]:
    return [path.name for path in paths]


def _check_duplicate_paths(paths: Sequence[Path]) -> List[Path]:
    resolved_paths = [Path(path).expanduser().resolve() for path in paths]
    seen: set[Path] = set()
    for path in resolved_paths:
        if path in seen:
            raise ValueError(f"Duplicate input path: {path}")
        if not path.exists():
            raise FileNotFoundError(path)
        seen.add(path)
    return resolved_paths


def _check_merged_plus_rank(paths: Sequence[Path]) -> None:
    by_parent: Dict[Path, set[str]] = {}
    for path in paths:
        by_parent.setdefault(path.parent, set()).add(path.name)
    for parent, names in by_parent.items():
        has_merged = "data_output_pairs.pth" in names
        has_rank_shard = any(name.startswith("data_output_pairs_rank") and name.endswith(".pth") for name in names)
        if has_merged and has_rank_shard:
            raise ValueError(
                "Input contains both merged data_output_pairs.pth and rank shards in "
                f"{parent}. Pass only one representation to avoid double-counting."
            )


def fit_scalar_gaussian_models(
    *,
    data_output_pairs_paths: Sequence[Path],
    output_dir: Path,
    outlier_threshold: float = 4.0,
    chunk_elements: int = 1_000_000,
    expected_num_samples: int | None = None,
    max_samples: int | None = None,
) -> Tuple[ArrayDict, ArrayDict, Mapping]:
    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    if outlier_threshold <= 0:
        raise ValueError("outlier_threshold must be positive")
    input_paths = _check_duplicate_paths(data_output_pairs_paths)
    _check_merged_plus_rank(input_paths)

    output_dir = Path(output_dir)
    mu_path = output_dir / "mu_dict.npy"
    cov_path = output_dir / "cov_dict.npy"
    metadata_path = output_dir / "scalar_gaussian_metadata.json"
    existing = [p for p in (mu_path, cov_path, metadata_path) if p.exists()]
    if existing:
        names = ", ".join(p.name for p in existing)
        raise FileExistsError(f"Refusing to overwrite existing calibration file(s): {names}")

    loaded = _load_pairs(input_paths, expected_num_samples=expected_num_samples, max_samples=max_samples)
    mu_dict: ArrayDict = {}
    cov_dict: ArrayDict = {}
    counts: Dict[str, Mapping[str, int | float]] = {}

    for timestep in loaded.timesteps:
        fp16_tensors = loaded.fp16_by_t[timestep]
        quant_tensors = loaded.quant_by_t[timestep]
        error_stats = _accumulate_error_stats(fp16_tensors, quant_tensors, chunk_elements=chunk_elements)
        joint_stats = _accumulate_joint_stats(
            fp16_tensors,
            quant_tensors,
            error_stats=error_stats,
            outlier_threshold=outlier_threshold,
            chunk_elements=chunk_elements,
        )
        mu_dict[timestep] = joint_stats.mu
        cov_dict[timestep] = joint_stats.cov
        counts[str(timestep)] = {
            "total_values": error_stats.count,
            "retained_values": joint_stats.count,
            "trimmed_values": error_stats.count - joint_stats.count,
            "error_mean_before_trim": error_stats.mean,
            "error_std_before_trim": error_stats.std,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(mu_path, mu_dict)
    np.save(cov_path, cov_dict)
    metadata = {
        "format": "qdrift_scalar_gaussian_v1",
        "statistic": "one 2D Gaussian per timestep for [quant_output, quant_output - fp_output]",
        "pooling": "samples, channels, and spatial positions are pooled before computing moments",
        "outlier_rule": f"global per-timestep |error - mean(error)| <= {outlier_threshold} * std(error)",
        "covariance": "unbiased covariance with denominator retained_values - 1",
        "mu_shape": [2, 1],
        "cov_shape": [2, 2, 1],
        "source_files": _safe_source_paths(input_paths),
        "source_fingerprint": _fingerprint_paths(input_paths),
        "num_input_files": len(input_paths),
        "num_samples_inferred": loaded.inferred_num_samples,
        "num_samples_available": loaded.total_num_samples_available,
        "num_samples_available_visited_inputs": loaded.total_num_samples_available,
        "expected_num_samples": expected_num_samples,
        "max_samples": max_samples,
        "file_sample_counts": loaded.file_sample_counts,
        "file_sample_counts_available": loaded.file_sample_counts_available,
        "timesteps": [str(t) for t in loaded.timesteps],
        "counts": counts,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    return mu_dict, cov_dict, metadata


def _resolve_input_paths(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    if args.data_output_pairs_path:
        paths.extend(Path(p) for p in args.data_output_pairs_path)
    if args.data_output_pairs_glob:
        for pattern in args.data_output_pairs_glob:
            paths.extend(Path(p) for p in sorted(glob.glob(pattern)))
    if not paths:
        raise ValueError("Provide at least one --data_output_pairs_path or --data_output_pairs_glob")
    unique = _check_duplicate_paths(paths)
    _check_merged_plus_rank(unique)
    return unique


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_output_pairs_path",
        action="append",
        help="Path to a data_output_pairs.pth file. May be passed multiple times.",
    )
    parser.add_argument(
        "--data_output_pairs_glob",
        action="append",
        help="Glob for data_output_pairs shards, e.g. 'calibration/data_output_pairs_rank*.pth'.",
    )
    parser.add_argument("--output_dir", required=True, help="New output directory for mu_dict.npy and cov_dict.npy.")
    parser.add_argument(
        "--expected_num_samples",
        type=int,
        default=None,
        help="Require the inferred leading-dimension sample count across all inputs to equal this value.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Use only the first N paired samples across sorted inputs before fitting.",
    )
    parser.add_argument(
        "--outlier_threshold",
        type=float,
        default=4.0,
        help="Global per-timestep error trimming threshold in standard deviations.",
    )
    parser.add_argument(
        "--chunk_elements",
        type=int,
        default=1_000_000,
        help="Number of flattened tensor values to process per CPU chunk.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    input_paths = _resolve_input_paths(args)
    mu_dict, _cov_dict, metadata = fit_scalar_gaussian_models(
        data_output_pairs_paths=input_paths,
        output_dir=Path(args.output_dir),
        outlier_threshold=args.outlier_threshold,
        chunk_elements=args.chunk_elements,
        expected_num_samples=args.expected_num_samples,
        max_samples=args.max_samples,
    )
    print(f"Saved scalar Gaussian calibration for {len(mu_dict)} timesteps to {args.output_dir}")
    print(f"Inferred samples: {metadata['num_samples_inferred']}")
    print(f"Source fingerprint: {metadata['source_fingerprint']}")


if __name__ == "__main__":
    main()
