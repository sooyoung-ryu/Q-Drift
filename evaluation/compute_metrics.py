"""
Metrics for generated image folders. Computes:

  1) FID (vs MJHQ real-image subset)      -> `data/mjhq_fid_reference/`
  2) CLIPScore (image-text)              -> prompts from `evaluation_results.json`
  3) LPIPS  (vs fp16 baseline images)
  4) PSNR   (vs fp16 baseline images)
  5) SSIM   (vs fp16 baseline images)

FID uses the local reference folder built by `scripts/prepare_data.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from tqdm import tqdm

try:
    from cleanfid import fid as cleanfid

    CLEANFID_AVAILABLE = True
except Exception as e:
    CLEANFID_AVAILABLE = False
    CLEANFID_IMPORT_ERROR = str(e)

try:
    from torchmetrics.image import (
        LearnedPerceptualImagePatchSimilarity,
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )
    from torchmetrics.multimodal import CLIPScore

    TORCHMETRICS_AVAILABLE = True
except Exception as e:
    TORCHMETRICS_AVAILABLE = False
    TORCHMETRICS_IMPORT_ERROR = str(e)


_IMG_EXTS = (".png", ".jpg", ".jpeg")


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_cache_dirs(cache_dir: Optional[str]) -> None:
    """
    Ensure common model-weight caches are writable.

    Notes:
    - LPIPS (torchmetrics/torchvision) may download weights into torch hub cache.
    - CLIPScore (torchmetrics) may use Hugging Face under the hood.
    """
    if not cache_dir:
        return
    cache = Path(cache_dir).expanduser().resolve()
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


def _iter_image_files(folder: Path) -> Iterable[Path]:
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in _IMG_EXTS:
            yield p


def _list_image_filenames(folder: Path) -> List[str]:
    return sorted([p.name for p in _iter_image_files(folder)])


def _is_valid_image_file(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.convert("RGB").load()
        return True
    except Exception:
        return False


def _list_valid_image_filenames(folder: Path) -> List[str]:
    return sorted([p.name for p in _iter_image_files(folder) if _is_valid_image_file(p)])


def _materialize_valid_image_symlinks(src_dir: Path, dst_dir: Path) -> Tuple[int, int]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    valid = 0
    for p in _iter_image_files(src_dir):
        total += 1
        if not _is_valid_image_file(p):
            continue
        link = dst_dir / p.name
        if link.exists():
            continue
        try:
            os.symlink(str(p), str(link))
        except FileExistsError:
            pass
        valid += 1
    return total, valid


def _fid_reference_dir() -> Path:
    ref_dir = Path(__file__).resolve().parents[1] / "data" / "mjhq_fid_reference"
    if not ref_dir.is_dir():
        raise FileNotFoundError(f"FID reference images not found: {ref_dir}. Run scripts/prepare_data.py.")
    return ref_dir


def _load_prompts_from_evaluation_results(results_path: Path) -> List[Dict]:
    """
    Expected format: dict with "prompts": list[dict(prompt, category, global_idx, ...)].
    """
    with results_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    prompts = None
    if isinstance(obj, dict) and isinstance(obj.get("prompts"), list):
        prompts = obj["prompts"]
    elif isinstance(obj, list):
        prompts = obj
    else:
        raise ValueError(
            f"Unsupported prompts file format: {results_path} (expected dict['prompts'] or a list[dict])"
        )

    normalized: List[Dict] = []
    for idx, row in enumerate(prompts):
        if not isinstance(row, dict):
            continue
        prompt = (row.get("prompt") or "").strip()
        category = (row.get("category") or "").strip()
        global_idx = row.get("global_idx", idx)
        if not prompt or not category:
            continue
        try:
            global_idx = int(global_idx)
        except Exception as e:
            raise ValueError(f"Invalid global_idx at prompts[{idx}]: {global_idx!r}") from e
        normalized.append(
            {
                "prompt": prompt,
                "id": row.get("id", ""),
                "category": category,
                "global_idx": global_idx,
            }
        )

    if len(normalized) == 0:
        raise ValueError(f"No valid prompts found in: {results_path}")
    return normalized


def _prompt_filename(row: Dict) -> str:
    return f"{row['category']}_{int(row['global_idx']):05d}.png"


def _find_prompt_image_path(gen_dirpath: Path, row: Dict) -> Path:
    stem = f"{row['category']}_{int(row['global_idx']):05d}"
    for ext in _IMG_EXTS:
        p = gen_dirpath / f"{stem}{ext}"
        if p.exists():
            return p
    return gen_dirpath / f"{stem}.png"


def _filter_prompts_to_existing(prompts: List[Dict], image_dir: Path) -> List[Dict]:
    if not image_dir.is_dir():
        return []
    kept: List[Dict] = []
    for p in prompts:
        img_path = _find_prompt_image_path(image_dir, p)
        if img_path.exists() and _is_valid_image_file(img_path):
            kept.append(p)
    return kept


def _pil_to_tensor_01(img: Image.Image) -> torch.Tensor:
    arr = np.array(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image array, got shape={arr.shape}")
    return torch.from_numpy(arr).permute(2, 0, 1).to(torch.float32) / 255.0


class PairImageByNameDataset(data.Dataset):
    def __init__(self, ref_dir: Path, gen_dir: Path, names: Sequence[str]):
        super().__init__()
        self.ref_dir = ref_dir
        self.gen_dir = gen_dir
        self.names = list(names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        ref_path = self.ref_dir / name
        gen_path = self.gen_dir / name
        with Image.open(ref_path) as ref_img:
            ref_img = ref_img.convert("RGB")
            with Image.open(gen_path) as gen_img:
                gen_img = gen_img.convert("RGB")
                if ref_img.size != gen_img.size:
                    ref_img = ref_img.resize(gen_img.size, Image.Resampling.BICUBIC)
                ref_tensor = _pil_to_tensor_01(ref_img)
                gen_tensor = _pil_to_tensor_01(gen_img)
        return [gen_tensor, ref_tensor]


def compute_image_similarity_metrics_vs_fp16(
    fp16_dir: Path,
    model_dir: Path,
    batch_size: int,
    num_workers: int,
    device: str,
) -> Dict[str, Optional[float]]:
    if not TORCHMETRICS_AVAILABLE:
        raise RuntimeError(f"torchmetrics is not available: {TORCHMETRICS_IMPORT_ERROR}")

    fp16_names = set(_list_valid_image_filenames(fp16_dir))
    model_names = set(_list_valid_image_filenames(model_dir))
    common = sorted(fp16_names.intersection(model_names))
    if len(common) == 0:
        raise RuntimeError("No overlapping filenames between fp16 and model.")

    metric_psnr = PeakSignalNoiseRatio(data_range=1.0, reduction="elementwise_mean", dim=(1, 2, 3)).to(device)
    metric_lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    metric_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    dataset = PairImageByNameDataset(fp16_dir, model_dir, common)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"similarity ({model_dir.name} vs fp16)"):
            gen, ref = batch[0].to(device), batch[1].to(device)
            metric_psnr.update(gen, ref)
            metric_lpips.update(gen, ref)
            metric_ssim.update(gen, ref)

    return {
        "psnr_vs_fp16": float(metric_psnr.compute().item()),
        "lpips_vs_fp16": float(metric_lpips.compute().item()),
        "ssim_vs_fp16": float(metric_ssim.compute().item()),
        "num_pairs": int(len(common)),
    }


class PromptImageDataset(data.Dataset):
    def __init__(self, prompts: List[Dict], gen_dir: Path):
        super().__init__()
        self.prompts = prompts
        self.gen_dir = gen_dir

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int):
        row = self.prompts[idx]
        path = _find_prompt_image_path(self.gen_dir, row)
        with Image.open(path) as img:
            img = img.convert("RGB")
            arr = np.array(img)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return [tensor, row["prompt"]]


def compute_clip_score(
    prompts: List[Dict],
    gen_dir: Path,
    batch_size: int,
    num_workers: int,
    device: str,
) -> Dict[str, Optional[float]]:
    if not TORCHMETRICS_AVAILABLE:
        raise RuntimeError(f"torchmetrics is not available: {TORCHMETRICS_IMPORT_ERROR}")

    metric_score = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
    dataset = PromptImageDataset(prompts, gen_dir)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"clip_score ({gen_dir.name})"):
            images = batch[0].to(device)
            metric_score.update(images, list(batch[1]))

    return {"clip_score": float(metric_score.compute().mean().item()), "num_pairs": int(len(prompts))}


def compute_fid_vs_reference(
    ref_dir: Path,
    gen_dir: Path,
    device: str,
    num_workers: int,
    batch_size: int,
) -> float:
    if not CLEANFID_AVAILABLE:
        raise RuntimeError(f"clean-fid is not available: {CLEANFID_IMPORT_ERROR}")

    feat_model = cleanfid.build_feature_extractor("clean", device)
    ref_feats = cleanfid.get_folder_features(
        str(ref_dir),
        feat_model,
        num_workers=num_workers,
        num=None,
        batch_size=batch_size,
        device=device,
        verbose=True,
        mode="clean",
    )
    gen_feats = cleanfid.get_folder_features(
        str(gen_dir),
        feat_model,
        num_workers=num_workers,
        num=None,
        batch_size=batch_size,
        device=device,
        verbose=True,
        mode="clean",
    )
    mu1, sigma1 = np.mean(ref_feats, axis=0), np.cov(ref_feats, rowvar=False)
    mu2, sigma2 = np.mean(gen_feats, axis=0), np.cov(gen_feats, rowvar=False)
    return float(cleanfid.frechet_distance(mu1, sigma1, mu2, sigma2))


def _save_json(path: Path, obj: Dict) -> None:
    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        if isinstance(o, list):
            return [convert(v) for v in o]
        return o

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(convert(obj), f, indent=2, ensure_ascii=False)


def _write_summary(path: Path, results: Dict) -> None:
    lines: List[str] = []
    lines.append("Metrics summary (FID, CLIPScore, LPIPS, PSNR, SSIM)")
    lines.append("")
    lines.append(f"FID reference: {results.get('fid_ref_dir')}")
    lines.append(f"Images root:   {results.get('images_root')}")
    lines.append("")

    models: Dict[str, Dict] = results.get("models", {}) or {}
    for model_name in sorted(models.keys()):
        m = models[model_name] or {}
        lines.append(f"[{model_name}]")
        if m.get("clip_score") is not None:
            lines.append(f"  CLIPScore:   {m['clip_score']:.4f}")
        if m.get("psnr_vs_fp16") is not None:
            lines.append(f"  PSNR:        {m['psnr_vs_fp16']:.4f}")
        if m.get("lpips_vs_fp16") is not None:
            lines.append(f"  LPIPS:       {m['lpips_vs_fp16']:.4f}")
        if m.get("ssim_vs_fp16") is not None:
            lines.append(f"  SSIM:        {m['ssim_vs_fp16']:.4f}")
        if m.get("fid") is not None:
            lines.append(f"  FID:         {m['fid']:.4f}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _discover_model_sources(evaluate_dir: Path) -> Tuple[Dict[str, List[Path]], str]:
    """
    Returns:
      model_sources: model_name -> list[source_dirpaths]
      images_root_label: a printable description for logs/json
    """
    images_root = evaluate_dir / "images"
    model_sources: Dict[str, List[Path]] = defaultdict(list)

    if images_root.is_dir():
        # Treat as a valid images root only if it contains at least one model subdirectory with image files.
        model_dirs = [p for p in images_root.iterdir() if p.is_dir()]
        has_any_images = False
        for md in model_dirs:
            if any(True for _ in _iter_image_files(md)):
                has_any_images = True
                break
        if has_any_images:
            for md in model_dirs:
                if any(True for _ in _iter_image_files(md)):
                    model_sources[md.name].append(md)
            return dict(model_sources), str(images_root)

    # Otherwise, look for rank*/images/<model> structure.
    rank_image_roots = []
    for rank_dir in sorted(evaluate_dir.glob("rank*")):
        ir = rank_dir / "images"
        if ir.is_dir():
            rank_image_roots.append(ir)

    for ir in rank_image_roots:
        for md in sorted([p for p in ir.iterdir() if p.is_dir()]):
            if any(True for _ in _iter_image_files(md)):
                model_sources[md.name].append(md)

    if len(model_sources) == 0:
        raise FileNotFoundError(
            f"Could not find any generated images under {evaluate_dir}.\n"
            f"Expected either:\n"
            f"  - {evaluate_dir/'images'}/<model>/*.png, or\n"
            f"  - {evaluate_dir}/rank*/images/<model>/*.png"
        )

    return dict(model_sources), f"{evaluate_dir}/rank*/images"


def _materialize_models_to_tmp(model_sources: Dict[str, List[Path]], tmp_images_root: Path) -> Dict[str, Path]:
    """
    Create a per-model directory of symlinks so downstream metrics can treat each model as a single folder.
    """
    out: Dict[str, Path] = {}
    tmp_images_root.mkdir(parents=True, exist_ok=True)

    for model_name, src_dirs in sorted(model_sources.items()):
        dst = tmp_images_root / model_name
        dst.mkdir(parents=True, exist_ok=True)
        out[model_name] = dst

        for src in src_dirs:
            for p in _iter_image_files(src):
                link = dst / p.name
                if link.exists():
                    # If duplicates exist, assume they're identical and keep the first one.
                    continue
                try:
                    os.symlink(str(p), str(link))
                except FileExistsError:
                    pass
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate_dir", type=str, required=True, help="Evaluation directory (contains images/ or rank*/)")
    ap.add_argument("--seed", type=int, default=42, help="Prompt sampling seed (used only for record-keeping).")
    ap.add_argument("--num_samples", type=int, default=None, help="Number of prompts/images (optional).")
    ap.add_argument(
        "--prompts_path",
        type=str,
        default=None,
        help="Path to evaluation_results.json. Default: <evaluate_dir>/evaluation_results.json if present.",
    )
    ap.add_argument("--models", type=str, nargs="*", default=None, help="Models to evaluate (default: all found).")
    ap.add_argument("--device", type=str, default=_default_device())
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size for similarity/CLIPScore.")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--fid_num_workers", type=int, default=8)
    ap.add_argument("--fid_batch_size", type=int, default=64)
    ap.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Writable cache root for model weights (TORCH_HOME/HF_*).",
    )
    ap.add_argument(
        "--results_path",
        type=str,
        default=None,
        help="Output JSON path (default: <evaluate_dir>/evaluation_metrics_only.json)",
    )
    ap.add_argument(
        "--summary_path",
        type=str,
        default=None,
        help="Output summary text path (default: <evaluate_dir>/evaluation_metrics_only_summary.txt)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    evaluate_dir = Path(args.evaluate_dir).resolve()
    if not evaluate_dir.is_dir():
        raise SystemExit(f"evaluate_dir is not a directory: {evaluate_dir}")

    _ensure_cache_dirs(args.cache_dir)

    fid_ref_dir = _fid_reference_dir()

    prompts_path = Path(args.prompts_path) if args.prompts_path else (evaluate_dir / "evaluation_results.json")
    if not prompts_path.exists():
        raise SystemExit(
            f"prompts_path not found: {prompts_path}\n"
            f"Expected evaluation_results.json in {evaluate_dir} (no dataset download is performed)."
        )
    prompts = _load_prompts_from_evaluation_results(prompts_path)
    num_samples = args.num_samples if args.num_samples is not None else len(prompts)

    model_sources, images_root_label = _discover_model_sources(evaluate_dir)
    discovered_models = sorted(model_sources.keys())
    models = args.models if args.models else discovered_models
    missing_models = [m for m in models if m not in model_sources]
    if missing_models:
        raise SystemExit(f"Requested models not found: {missing_models}. Available: {discovered_models}")

    device = args.device
    print(f"Device:       {device}")
    print(f"Evaluate dir: {evaluate_dir}")
    print(f"Prompts:      {len(prompts)} ({prompts_path})")
    print(f"Num samples:  {num_samples}")
    print(f"FID ref:      {fid_ref_dir}  (images={len(_list_image_filenames(fid_ref_dir))})")
    print(f"Images root:  {images_root_label}")
    print(f"Models:       {models}")

    results_path = Path(args.results_path) if args.results_path else (evaluate_dir / "evaluation_metrics_only.json")
    summary_path = Path(args.summary_path) if args.summary_path else (evaluate_dir / "evaluation_metrics_only_summary.txt")

    out: Dict = {
        "format": "evaluation_metrics_only_v2_unified",
        "evaluate_dir": str(evaluate_dir),
        "images_root": str(images_root_label),
        "prompts_path": str(prompts_path),
        "seed": int(args.seed),
        "num_samples": int(num_samples),
        "fid_ref_dir": str(fid_ref_dir),
        "models": {},
    }

    with tempfile.TemporaryDirectory(prefix="metrics_tmp_", dir=str(evaluate_dir)) as tmp:
        tmp_root = Path(tmp)
        tmp_images_root = tmp_root / "images"
        model_dirs = _materialize_models_to_tmp({m: model_sources[m] for m in models}, tmp_images_root)
        fid_ref_valid_dir = tmp_root / "fid_ref_valid"
        fid_ref_total, fid_ref_valid = _materialize_valid_image_symlinks(fid_ref_dir, fid_ref_valid_dir)
        if fid_ref_valid < fid_ref_total:
            print(
                f"FID reference: skipped {fid_ref_total - fid_ref_valid} invalid image(s) "
                f"(using {fid_ref_valid}/{fid_ref_total})."
            )

        fp16_dir = model_dirs.get("fp16")
        for model_name in models:
            model_dir = model_dirs[model_name]
            model_metrics: Dict[str, Optional[float]] = {}

            # CLIPScore (image-text) for the model's existing images.
            try:
                clip_prompts = _filter_prompts_to_existing(prompts, model_dir)
                if len(clip_prompts) == 0:
                    raise RuntimeError("No prompt/image pairs found for CLIPScore.")
                clip = compute_clip_score(
                    prompts=clip_prompts,
                    gen_dir=model_dir,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=device,
                )
                model_metrics["clip_score"] = clip["clip_score"]
                model_metrics["clip_score_num_pairs"] = clip["num_pairs"]
            except Exception as e:
                print(f"✗ CLIPScore failed for {model_name}: {e}")
                model_metrics["clip_score"] = None

            # Similarity metrics vs fp16 baseline (if available).
            if fp16_dir is not None and fp16_dir.is_dir() and model_name != "fp16":
                try:
                    sim = compute_image_similarity_metrics_vs_fp16(
                        fp16_dir=fp16_dir,
                        model_dir=model_dir,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        device=device,
                    )
                    model_metrics["psnr_vs_fp16"] = sim["psnr_vs_fp16"]
                    model_metrics["lpips_vs_fp16"] = sim["lpips_vs_fp16"]
                    model_metrics["ssim_vs_fp16"] = sim["ssim_vs_fp16"]
                    model_metrics["similarity_num_pairs"] = sim["num_pairs"]
                except Exception as e:
                    print(f"✗ Similarity metrics failed for {model_name}: {e}")
                    model_metrics["psnr_vs_fp16"] = None
                    model_metrics["lpips_vs_fp16"] = None
                    model_metrics["ssim_vs_fp16"] = None
            else:
                model_metrics["psnr_vs_fp16"] = None
                model_metrics["lpips_vs_fp16"] = None
                model_metrics["ssim_vs_fp16"] = None

            # FID vs local MJHQ real-image subset (seed42).
            try:
                fid_model_valid_dir = tmp_root / "fid_models_valid" / model_name
                gen_total, gen_valid = _materialize_valid_image_symlinks(model_dir, fid_model_valid_dir)
                model_metrics["fid_num_images_used"] = gen_valid
                model_metrics["fid_num_images_total"] = gen_total

                if fid_ref_valid == 0:
                    raise RuntimeError("No valid reference images found in FID directory.")
                if gen_valid == 0:
                    raise RuntimeError("No valid generated images found.")
                if gen_valid < gen_total:
                    print(
                        f"FID input ({model_name}): skipped {gen_total - gen_valid} invalid image(s) "
                        f"(using {gen_valid}/{gen_total})."
                    )
                model_metrics["fid"] = compute_fid_vs_reference(
                    ref_dir=fid_ref_valid_dir,
                    gen_dir=fid_model_valid_dir,
                    device=device,
                    num_workers=args.fid_num_workers,
                    batch_size=args.fid_batch_size,
                )
            except Exception as e:
                print(f"✗ FID failed for {model_name}: {e}")
                model_metrics["fid"] = None

            out["models"][model_name] = model_metrics

    _save_json(results_path, out)
    _write_summary(summary_path, out)
    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
