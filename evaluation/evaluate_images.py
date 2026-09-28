#!/usr/bin/env python3
"""Evaluate generated image folders with the paper metric conventions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from tqdm import tqdm

try:
    from torchmetrics.image import (
        LearnedPerceptualImagePatchSimilarity,
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )
    from torchmetrics.multimodal import CLIPScore

    TORCHMETRICS_AVAILABLE = True
except Exception as exc:  # pragma: no cover - exercised only in incomplete environments
    TORCHMETRICS_AVAILABLE = False
    TORCHMETRICS_IMPORT_ERROR = str(exc)


IMG_EXTS = (".png", ".jpg", ".jpeg")


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_cache_dirs(cache_dir: Optional[str]) -> None:
    if not cache_dir:
        return
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    torch_cache = cache / "torch"
    torch_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_cache)
    try:
        torch.hub.set_dir(str(torch_cache))
    except Exception:
        pass

    hf_cache = cache / "huggingface"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache / "hub"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache / "transformers"))


def iter_image_files(folder: Path) -> Iterable[Path]:
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            yield path


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.convert("RGB").load()
        return True
    except Exception:
        return False


def load_prompts(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    rows = obj.get("prompts") if isinstance(obj, dict) else obj
    if not isinstance(rows, list):
        raise ValueError("prompts file must be a list or an object with a 'prompts' list")

    prompts: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt") or "").strip()
        category = str(row.get("category") or "").strip()
        if not prompt or not category:
            continue
        global_idx = int(row.get("global_idx", idx))
        prompts.append({"prompt": prompt, "category": category, "global_idx": global_idx})
    if not prompts:
        raise ValueError(f"no usable prompts in {path}")
    seen: set[str] = set()
    duplicates: List[str] = []
    for row in prompts:
        name = prompt_filename(row)
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(f"duplicate prompt filename ids: {sorted(set(duplicates))}")
    return prompts


def prompt_filename(row: Dict[str, Any]) -> str:
    return f"{row['category']}_{int(row['global_idx']):05d}.png"


def find_prompt_image_path(image_dir: Path, row: Dict[str, Any]) -> Path:
    stem = f"{row['category']}_{int(row['global_idx']):05d}"
    for ext in IMG_EXTS:
        path = image_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return image_dir / f"{stem}.png"


def collect_image_names(image_dir: Path, allow_missing: bool) -> tuple[List[str], List[str]]:
    names: List[str] = []
    invalid: List[str] = []
    for path in iter_image_files(image_dir):
        if is_valid_image(path):
            names.append(path.name)
        else:
            invalid.append(path.name)
    if invalid and not allow_missing:
        raise ValueError(f"image directory contains invalid image files: invalid={len(invalid)}")
    return sorted(names), sorted(invalid)


def select_prompt_pairs(
    image_dir: Path,
    prompts: Sequence[Dict[str, Any]],
    allow_missing: bool,
) -> tuple[List[Dict[str, Any]], List[str], List[str]]:
    kept: List[Dict[str, Any]] = []
    missing: List[str] = []
    invalid: List[str] = []
    for row in prompts:
        expected_name = prompt_filename(row)
        image_path = find_prompt_image_path(image_dir, row)
        if not image_path.exists():
            missing.append(expected_name)
            continue
        if not is_valid_image(image_path):
            invalid.append(image_path.name)
            continue
        kept.append({**row, "filename": image_path.name})
    if (missing or invalid) and not allow_missing:
        problems = []
        if missing:
            problems.append(f"missing={len(missing)}")
        if invalid:
            problems.append(f"invalid={len(invalid)}")
        raise ValueError("prompt/image pairing is incomplete: " + ", ".join(problems))
    return kept, missing, invalid


def select_image_names(
    image_dir: Path,
    prompts: Optional[Sequence[Dict[str, Any]]],
    allow_missing: bool,
) -> tuple[List[str], List[Dict[str, Any]], Dict[str, List[str]]]:
    if prompts is None:
        names, invalid = collect_image_names(image_dir, allow_missing=allow_missing)
        return names, [], {"missing": [], "invalid": invalid}

    prompt_pairs, missing, invalid = select_prompt_pairs(image_dir, prompts, allow_missing=allow_missing)
    names = [row["filename"] for row in prompt_pairs]
    return names, prompt_pairs, {"missing": missing, "invalid": invalid}


def pil_to_tensor_01(image: Image.Image) -> torch.Tensor:
    arr = np.array(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected RGB image array, got shape={arr.shape}")
    return torch.from_numpy(arr).permute(2, 0, 1).to(torch.float32) / 255.0


class PairImageDataset(data.Dataset):
    def __init__(self, fp_dir: Path, image_dir: Path, names: Sequence[str]):
        self.fp_dir = fp_dir
        self.image_dir = image_dir
        self.names = list(names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        with Image.open(self.fp_dir / name) as fp_image:
            fp_image = fp_image.convert("RGB")
            with Image.open(self.image_dir / name) as gen_image:
                gen_image = gen_image.convert("RGB")
                if fp_image.size != gen_image.size:
                    fp_image = fp_image.resize(gen_image.size, Image.Resampling.BICUBIC)
                return [pil_to_tensor_01(gen_image), pil_to_tensor_01(fp_image), name]


class PromptImageDataset(data.Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], image_dir: Path):
        self.rows = list(rows)
        self.image_dir = image_dir

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        with Image.open(self.image_dir / row["filename"]) as image:
            arr = np.array(image.convert("RGB"))
        return [torch.from_numpy(arr).permute(2, 0, 1), row["prompt"]]


def compute_clip_score(
    prompt_pairs: Sequence[Dict[str, Any]],
    image_dir: Path,
    batch_size: int,
    num_workers: int,
    device: str,
) -> Dict[str, Any]:
    if not TORCHMETRICS_AVAILABLE:
        raise RuntimeError(f"torchmetrics is not available: {TORCHMETRICS_IMPORT_ERROR}")
    metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
    loader = data.DataLoader(
        PromptImageDataset(prompt_pairs, image_dir),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"clip_score ({image_dir.name})"):
            metric.update(batch[0].to(device), list(batch[1]))
    return {"clip_score": float(metric.compute().mean().item()), "clip_score_num_pairs": len(prompt_pairs)}


def compute_similarity(
    fp_dir: Path,
    image_dir: Path,
    names: Sequence[str],
    batch_size: int,
    num_workers: int,
    device: str,
) -> Dict[str, Any]:
    if not TORCHMETRICS_AVAILABLE:
        raise RuntimeError(f"torchmetrics is not available: {TORCHMETRICS_IMPORT_ERROR}")
    metric_psnr = PeakSignalNoiseRatio(data_range=1.0, reduction="elementwise_mean", dim=(1, 2, 3)).to(device)
    metric_lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    metric_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    loader = data.DataLoader(
        PairImageDataset(fp_dir, image_dir, names),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"similarity ({image_dir.name} vs fp)"):
            gen = batch[0].to(device)
            ref = batch[1].to(device)
            metric_psnr.update(gen, ref)
            metric_lpips.update(gen, ref)
            metric_ssim.update(gen, ref)
    return {
        "psnr_vs_fp16": float(metric_psnr.compute().item()),
        "lpips_vs_fp16": float(metric_lpips.compute().item()),
        "ssim_vs_fp16": float(metric_ssim.compute().item()),
        "similarity_num_pairs": len(names),
    }


def validate_similarity_names(fp_dir: Path, names: Sequence[str], allow_missing: bool) -> tuple[List[str], List[str]]:
    missing_or_invalid = [name for name in names if not (fp_dir / name).exists() or not is_valid_image(fp_dir / name)]
    if missing_or_invalid and not allow_missing:
        raise ValueError(f"fp pairing is incomplete: missing_or_invalid={len(missing_or_invalid)}")
    return [name for name in names if name not in set(missing_or_invalid)], missing_or_invalid


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(convert(obj), handle, indent=2, ensure_ascii=False)


def package_versions() -> Dict[str, Optional[str]]:
    packages = {
        "numpy": "numpy",
        "pillow": "Pillow",
        "torch": "torch",
        "torchmetrics": "torchmetrics",
        "torchvision": "torchvision",
    }
    versions: Dict[str, Optional[str]] = {}
    for key, dist_name in packages.items():
        try:
            versions[key] = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None
    return versions


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--fp-dir", type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", choices=("clip", "psnr", "lpips", "ssim"), default=("clip", "psnr", "lpips", "ssim"))
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--expected-pairs", type=int)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--cache-dir")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.image_dir.is_dir():
        raise SystemExit(f"image-dir is not a directory: {args.image_dir}")
    ensure_cache_dirs(args.cache_dir)

    prompts = load_prompts(args.prompts) if args.prompts else None
    names, prompt_pairs, excluded = select_image_names(args.image_dir, prompts, allow_missing=args.allow_missing)
    if args.expected_count is not None and len(names) != args.expected_count:
        raise SystemExit(f"selected image count {len(names)} does not match --expected-count {args.expected_count}")
    if not names:
        raise SystemExit("no valid generated images selected")

    requested = set(args.metrics)
    out: Dict[str, Any] = {
        "format": "qdrift_evaluate_images_v1",
        "provenance": {
            "evaluator": str(Path(__file__).name),
            "evaluator_sha256": file_sha256(Path(__file__)),
            "package_versions": package_versions(),
        },
        "protocol": {
            "clip": "torchmetrics CLIPScore openai/clip-vit-large-patch14 on uint8 RGB images",
            "similarity": "torchmetrics PSNR(data_range=1, per-image mean), LPIPS(normalize=True), SSIM(data_range=1)",
        },
        "inputs": {
            "image_dir": str(args.image_dir),
            "fp_dir": str(args.fp_dir) if args.fp_dir else None,
            "prompts": str(args.prompts) if args.prompts else None,
        },
        "metrics_requested": sorted(requested),
        "num_images_selected": len(names),
        "excluded": excluded,
        "results": {},
    }

    if "clip" in requested:
        if prompts is None:
            raise SystemExit("--prompts is required for CLIPScore")
        out["results"].update(
            compute_clip_score(prompt_pairs, args.image_dir, args.batch_size, args.num_workers, args.device)
        )

    similarity_metrics = requested.intersection({"psnr", "lpips", "ssim"})
    if similarity_metrics:
        if args.fp_dir is None:
            raise SystemExit("--fp-dir is required for PSNR/LPIPS/SSIM")
        if not args.fp_dir.is_dir():
            raise SystemExit(f"fp-dir is not a directory: {args.fp_dir}")
        sim_names, missing_fp = validate_similarity_names(args.fp_dir, names, allow_missing=args.allow_missing)
        expected_pairs = args.expected_pairs if args.expected_pairs is not None else args.expected_count
        if expected_pairs is not None and len(sim_names) != expected_pairs:
            raise SystemExit(
                f"similarity pair count {len(sim_names)} does not match expected pairs {expected_pairs}"
            )
        out["excluded"]["missing_or_invalid_fp"] = missing_fp
        sim = compute_similarity(args.fp_dir, args.image_dir, sim_names, args.batch_size, args.num_workers, args.device)
        for key, value in sim.items():
            if key.startswith("psnr") and "psnr" not in requested:
                continue
            if key.startswith("lpips") and "lpips" not in requested:
                continue
            if key.startswith("ssim") and "ssim" not in requested:
                continue
            out["results"][key] = value
        out["results"]["similarity_num_pairs"] = sim["similarity_num_pairs"]

    save_json(args.output, out)
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
