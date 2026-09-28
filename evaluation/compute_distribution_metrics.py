#!/usr/bin/env python3
"""Compute KID and paired FID intervals from clean-fid feature caches."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


@dataclass(frozen=True)
class FeatureSet:
    features: np.ndarray
    names: List[str]


def iter_images(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            yield path


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.convert("RGB").load()
        return True
    except Exception:
        return False


def materialize_valid_symlinks(src: Path, dst: Path) -> Tuple[List[str], List[str]]:
    dst.mkdir(parents=True, exist_ok=True)
    names: List[str] = []
    invalid: List[str] = []
    for path in iter_images(src):
        if not is_valid_image(path):
            invalid.append(path.name)
            continue
        link = dst / path.name
        if not link.exists():
            try:
                os.symlink(str(path.resolve()), str(link))
            except FileExistsError:
                pass
        names.append(path.name)
    return names, invalid


def load_features(path: Path) -> FeatureSet:
    data = np.load(path, allow_pickle=False)
    if "features" not in data or "names" not in data:
        raise ValueError(f"{path} must contain 'features' and 'names' arrays.")
    features = data["features"].astype(np.float64, copy=False)
    names = [str(name) for name in data["names"]]
    if features.ndim != 2:
        raise ValueError(f"{path}: features must be a 2D array, got shape {features.shape}.")
    if features.shape[0] != len(names):
        raise ValueError(f"{path}: feature rows ({features.shape[0]}) != names ({len(names)}).")
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: image names must be unique.")
    if not np.isfinite(features).all():
        raise ValueError(f"{path}: features contain NaN or infinite values.")
    return FeatureSet(features=features, names=names)


def write_features(path: Path, feature_set: FeatureSet) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=feature_set.features.astype(np.float32, copy=False),
        names=np.asarray(feature_set.names),
    )


def resolve_feature_path(feature_root: Path, row: Mapping[str, object]) -> Path:
    if "features" not in row:
        raise KeyError(f"Missing 'features' for manifest row: {row}")
    path = Path(str(row["features"]))
    return path if path.is_absolute() else feature_root / path


def resolve_image_path(image_root: Path, row: Mapping[str, object]) -> Optional[Path]:
    image_dir = row.get("image_dir")
    if image_dir is None:
        return None
    path = Path(str(image_dir))
    return path if path.is_absolute() else image_root / path


def load_manifest(path: Path) -> Dict[str, object]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or "ref" not in obj or "rows" not in obj:
        raise ValueError("Manifest must be a JSON object with 'ref' and 'rows'.")
    if not isinstance(obj["rows"], list):
        raise ValueError("Manifest field 'rows' must be a list.")
    return obj


def polynomial_mmd2_unbiased(x: np.ndarray, y: np.ndarray) -> float:
    dim = x.shape[1]
    k_xx = (x @ x.T / dim + 1.0) ** 3
    k_yy = (y @ y.T / dim + 1.0) ** 3
    k_xy = (x @ y.T / dim + 1.0) ** 3
    n = x.shape[0]
    m = y.shape[0]
    sum_xx = (float(np.sum(k_xx)) - float(np.trace(k_xx))) / (n * (n - 1))
    sum_yy = (float(np.sum(k_yy)) - float(np.trace(k_yy))) / (m * (m - 1))
    sum_xy = float(np.sum(k_xy)) / (n * m)
    return float(sum_xx + sum_yy - 2.0 * sum_xy)


def feature_moments(features: torch.Tensor):
    mean = features.mean(dim=0)
    centered = features - mean
    return mean, centered.T @ centered / (len(features) - 1)


def covariance_sqrt(covariance: torch.Tensor):
    values, vectors = torch.linalg.eigh(covariance)
    return (vectors * values.clamp_min(0).sqrt().unsqueeze(0)) @ vectors.T


def fid_from_moments(mean_ref, cov_ref, sqrt_ref, mean_gen, cov_gen):
    product = sqrt_ref @ cov_gen @ sqrt_ref
    product = (product + product.T) * 0.5
    trace_sqrt = torch.linalg.eigvalsh(product).clamp_min(0).sqrt().sum()
    return ((mean_ref - mean_gen) ** 2).sum() + torch.trace(cov_ref) + torch.trace(cov_gen) - 2 * trace_sqrt


def frechet_distance_torch(feats1: np.ndarray, feats2: np.ndarray, device: str) -> float:
    x = torch.as_tensor(feats1, dtype=torch.float64, device=device)
    y = torch.as_tensor(feats2, dtype=torch.float64, device=device)
    mean_ref, cov_ref = feature_moments(x)
    mean_gen, cov_gen = feature_moments(y)
    value = fid_from_moments(mean_ref, cov_ref, covariance_sqrt(cov_ref), mean_gen, cov_gen)
    return float(value.item())


def kid_with_subsets(
    ref_feats: np.ndarray,
    gen_feats: np.ndarray,
    rng: np.random.Generator,
    n_subsets: int,
    subset_size: int,
) -> Dict[str, float]:
    size = min(subset_size, ref_feats.shape[0], gen_feats.shape[0])
    if size < 2:
        raise ValueError("KID requires at least two images in each set.")
    values = []
    for _ in tqdm(range(n_subsets), desc="KID subsets", leave=False):
        ref_idx = rng.choice(ref_feats.shape[0], size=size, replace=False)
        gen_idx = rng.choice(gen_feats.shape[0], size=size, replace=False)
        values.append(polynomial_mmd2_unbiased(ref_feats[ref_idx], gen_feats[gen_idx]))
    arr = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(arr, [2.5, 97.5])
    return {
        "kid": float(arr.mean()),
        "kid_x1000": float(arr.mean() * 1000.0),
        "kid_subset_p2p5_x1000": float(low * 1000.0),
        "kid_subset_p97p5_x1000": float(high * 1000.0),
        "kid_num_subsets": int(n_subsets),
        "kid_subset_size": int(size),
    }


def align_by_names(
    first: FeatureSet,
    second: FeatureSet,
    *,
    allow_intersection: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, object]]:
    first_names = set(first.names)
    second_names = set(second.names)
    if allow_intersection:
        ordered = sorted(first_names & second_names)
    else:
        if first_names != second_names:
            raise ValueError(
                "Paired feature names must match exactly. Use --allow-intersection to pair only common names."
            )
        ordered = sorted(first_names)
    if len(ordered) < 2:
        raise ValueError("Need at least two common generated images for paired metrics.")
    first_map = {name: idx for idx, name in enumerate(first.names)}
    second_map = {name: idx for idx, name in enumerate(second.names)}
    first_idx = np.asarray([first_map[name] for name in ordered], dtype=np.int64)
    second_idx = np.asarray([second_map[name] for name in ordered], dtype=np.int64)
    provenance = {
        "num_first": int(len(first.names)),
        "num_second": int(len(second.names)),
        "num_common": int(len(ordered)),
        "excluded_from_first": sorted(first_names - second_names),
        "excluded_from_second": sorted(second_names - first_names),
    }
    return first.features[first_idx], second.features[second_idx], ordered, provenance


def paired_delta_fid_ci(
    ref_feats: np.ndarray,
    quant_feats: np.ndarray,
    qdrift_feats: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
    device: str,
) -> Dict[str, float]:
    values: List[float] = []
    ref = torch.as_tensor(ref_feats, dtype=torch.float64, device=device)
    quant = torch.as_tensor(quant_feats, dtype=torch.float64, device=device)
    qdrift = torch.as_tensor(qdrift_feats, dtype=torch.float64, device=device)
    for _ in tqdm(range(n_bootstrap), desc="paired Delta FID", leave=False):
        ref_idx = torch.as_tensor(rng.integers(0, len(ref), size=len(ref)), device=device)
        gen_idx = torch.as_tensor(rng.integers(0, len(quant), size=len(quant)), device=device)
        mean_ref, cov_ref = feature_moments(ref[ref_idx])
        sqrt_ref = covariance_sqrt(cov_ref)
        fid_quant = fid_from_moments(mean_ref, cov_ref, sqrt_ref, *feature_moments(quant[gen_idx]))
        fid_qdrift = fid_from_moments(mean_ref, cov_ref, sqrt_ref, *feature_moments(qdrift[gen_idx]))
        values.append(float((fid_qdrift - fid_quant).item()))
    arr = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(arr, [2.5, 97.5])
    return {
        "delta_fid_boot_mean": float(arr.mean()),
        "delta_fid_ci_low": float(low),
        "delta_fid_ci_high": float(high),
        "fid_bootstrap_replicates": int(n_bootstrap),
    }


def group_rows(rows: Sequence[Mapping[str, object]]) -> Dict[Tuple[str, str], Dict[str, Mapping[str, object]]]:
    grouped: Dict[Tuple[str, str], Dict[str, Mapping[str, object]]] = {}
    for row in rows:
        key = (str(row["label"]), str(row["setting"]))
        grouped.setdefault(key, {})[str(row["method"])] = row
    return grouped


def command_extract(args: argparse.Namespace) -> int:
    try:
        from cleanfid import fid as cleanfid
    except Exception as exc:
        raise SystemExit(f"clean-fid is required for feature extraction: {exc}") from exc

    manifest = load_manifest(args.manifest)
    output_root = args.output_root.resolve()
    image_root = args.image_root.resolve()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    feat_model = cleanfid.build_feature_extractor("clean", device)

    entries = [manifest["ref"], *list(manifest["rows"])]
    extraction_log: List[Dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="qdrift_features_") as tmp:
        tmp_root = Path(tmp)
        for idx, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ValueError(f"Invalid manifest entry: {entry}")
            image_dir = resolve_image_path(image_root, entry)
            if image_dir is None:
                continue
            feature_path = resolve_feature_path(output_root, entry)
            valid_dir = tmp_root / f"{idx:04d}_{feature_path.stem}"
            names, invalid = materialize_valid_symlinks(image_dir, valid_dir)
            if invalid and not args.allow_invalid:
                raise ValueError(f"Invalid images in {image_dir}: {invalid}. Use --allow-invalid to exclude explicitly.")
            if "expected_count" in entry and len(names) != int(entry["expected_count"]):
                raise ValueError(f"Image count mismatch for {image_dir}: {len(names)} != {entry['expected_count']}")
            features = cleanfid.get_folder_features(
                str(valid_dir),
                feat_model,
                num_workers=args.num_workers,
                num=None,
                batch_size=args.batch_size,
                device=device,
                verbose=True,
                mode="clean",
            ).astype(np.float64, copy=False)
            if features.shape[0] != len(names):
                raise RuntimeError(f"Feature count mismatch for {image_dir}: {features.shape[0]} vs {len(names)}")
            write_features(feature_path, FeatureSet(features=features, names=names))
            extraction_log.append(
                {
                    "features": str(entry["features"]),
                    "num_images": int(len(names)),
                    "num_invalid": int(len(invalid)),
                    "invalid_images": invalid,
                }
            )
            print(f"[saved] {feature_path} ({features.shape[0]} images)")

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(json.dumps({"format": "qdrift_feature_extraction_v1", "entries": extraction_log}, indent=2) + "\n")
    return 0


def command_compute(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    feature_root = args.feature_root.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    ref_entry = manifest["ref"]
    if not isinstance(ref_entry, Mapping):
        raise ValueError("'ref' must be a manifest object.")
    ref = load_features(resolve_feature_path(feature_root, ref_entry))

    cache: Dict[str, FeatureSet] = {}

    def get_features(row: Mapping[str, object]) -> FeatureSet:
        key = str(row["features"])
        if key not in cache:
            cache[key] = load_features(resolve_feature_path(feature_root, row))
        return cache[key]

    rows: List[Dict[str, object]] = []
    manifest_rows = manifest["rows"]
    for row in manifest_rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"Invalid row: {row}")
        gen = get_features(row)
        out: Dict[str, object] = {
            "label": str(row["label"]),
            "setting": str(row["setting"]),
            "method": str(row["method"]),
            "features": str(row["features"]),
            "num_ref": int(ref.features.shape[0]),
            "num_gen": int(gen.features.shape[0]),
        }
        if "fid" in row:
            out["fid"] = float(row["fid"])
        if "reported_fid" in row:
            out["fid"] = float(row["reported_fid"])
        if args.compute_fid:
            out["computed_fid"] = frechet_distance_torch(ref.features, gen.features, device)
        row_rng = np.random.default_rng(args.kid_seed)
        out.update(kid_with_subsets(ref.features, gen.features, row_rng, args.kid_subsets, args.kid_subset_size))
        rows.append(out)

    paired_rows: List[Dict[str, object]] = []
    grouped = group_rows(manifest_rows)
    for (label, setting), methods in grouped.items():
        if "Quantized" not in methods or "Q-Drift" not in methods:
            continue
        quant = get_features(methods["Quantized"])
        qdrift = get_features(methods["Q-Drift"])
        quant_feats, qdrift_feats, names, provenance = align_by_names(
            quant,
            qdrift,
            allow_intersection=args.allow_intersection,
        )
        point_quant = frechet_distance_torch(ref.features, quant_feats, device)
        point_qdrift = frechet_distance_torch(ref.features, qdrift_feats, device)
        paired: Dict[str, object] = {
            "label": label,
            "setting": setting,
            "num_ref": int(ref.features.shape[0]),
            "num_common_gen": int(len(names)),
            "paired_point_quant_fid": float(point_quant),
            "paired_point_qdrift_fid": float(point_qdrift),
            "paired_point_delta_fid": float(point_qdrift - point_quant),
            "pairing": provenance,
        }
        if args.fid_bootstrap > 0:
            group_rng = np.random.default_rng(args.fid_seed)
            paired.update(
                paired_delta_fid_ci(
                    ref.features,
                    quant_feats,
                    qdrift_feats,
                    group_rng,
                    args.fid_bootstrap,
                    device,
                )
            )
        paired_rows.append(paired)

    result = {
        "format": "qdrift_distribution_metrics_v2",
        "provenance": {
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "package_versions": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "clean-fid")},
        },
        "algorithm": {
            "fid": "clean-fid Inception features; Frechet distance from feature means/covariances.",
            "paired_delta_fid_ci": "Percentile bootstrap of Delta FID with shared reference resample and shared aligned generated-image indices.",
            "kid": "Unbiased cubic-polynomial MMD from clean-fid features, averaged over random subsets.",
        },
        "seeds": {
            "kid": int(args.kid_seed),
            "paired_delta_fid": int(args.fid_seed),
        },
        "parameters": {
            "kid_subsets": int(args.kid_subsets),
            "kid_subset_size": int(args.kid_subset_size),
            "fid_bootstrap": int(args.fid_bootstrap),
            "allow_intersection": bool(args.allow_intersection),
        },
        "rows": rows,
        "paired_rows": paired_rows,
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="extract clean-fid features from image folders")
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--image-root", type=Path, default=Path("."))
    extract.add_argument("--output-root", type=Path, required=True)
    extract.add_argument("--device", type=str, default="cuda")
    extract.add_argument("--batch-size", type=int, default=64)
    extract.add_argument("--num-workers", type=int, default=8)
    extract.add_argument("--allow-invalid", action="store_true")
    extract.add_argument("--log", type=Path, default=None)
    extract.set_defaults(func=command_extract)

    compute = subparsers.add_parser("compute", help="compute KID and paired Delta FID from feature caches")
    compute.add_argument("--manifest", type=Path, required=True)
    compute.add_argument("--feature-root", type=Path, default=Path("."))
    compute.add_argument("--output", type=Path, required=True)
    compute.add_argument("--device", type=str, default="cuda")
    compute.add_argument("--compute-fid", action="store_true")
    compute.add_argument("--fid-bootstrap", type=int, default=1000)
    compute.add_argument("--kid-subsets", type=int, default=100)
    compute.add_argument("--kid-subset-size", type=int, default=1000)
    compute.add_argument("--kid-seed", type=int, default=1234)
    compute.add_argument("--fid-seed", type=int, default=5678)
    compute.add_argument("--allow-intersection", action="store_true")
    compute.set_defaults(func=command_compute)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
