"""
Build a deterministic MJHQ real-image folder that matches the MJHQ prompt sampling
used by the evaluate scripts (`load_prompts` in experiments/svdquant/*/scripts/evaluate.py).

It samples MJHQ-30K in a stratified way (equal samples per category) with a given
seed, then writes the corresponding *real* MJHQ images into an output folder.

Images are read from `mjhq30k_imgs.zip` in the HF dataset repo (playgroundai/MJHQ-30K).
No `datasets` dependency is required (avoids PIL/datasets compatibility issues).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

from PIL import Image
from huggingface_hub import hf_hub_download

MJHQ_REPO_ID = "playgroundai/MJHQ-30K"
MJHQ_META_FILENAME = "meta_data.json"
MJHQ_ZIP_FILENAME = "mjhq30k_imgs.zip"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class SampleItem:
    image_id: str
    category: str
    prompt: str
    global_idx: int


def _download_meta() -> str:
    return hf_hub_download(repo_id=MJHQ_REPO_ID, filename=MJHQ_META_FILENAME, repo_type="dataset")


def _download_zip() -> str:
    return hf_hub_download(repo_id=MJHQ_REPO_ID, filename=MJHQ_ZIP_FILENAME, repo_type="dataset")


def _load_meta(meta_path: str) -> dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_mjhq_items(*, meta_path: str | None, num_samples: int, seed: int) -> list[SampleItem]:
    """
    Matches `load_prompts_from_mjhq()` behavior in the evaluate scripts,
    but keeps (id, category, prompt) so we can fetch the corresponding real images.
    """
    if not meta_path or not os.path.exists(meta_path):
        print("Downloading MJHQ-30K metadata from HF Hub...")
        meta_path = _download_meta()

    metadata = _load_meta(meta_path)

    by_category: dict[str, list[tuple[str, str]]] = defaultdict(list)  # category -> [(id, prompt)]
    for image_id, info in metadata.items():
        if not isinstance(info, dict):
            continue
        prompt = (info.get("prompt") or "").strip()
        category = (info.get("category") or "").strip()
        if not prompt or not category:
            continue
        by_category[category].append((image_id, prompt))

    if num_samples % 10 != 0:
        raise ValueError(f"num_samples must be divisible by 10 for stratified sampling, got {num_samples}")

    random.seed(seed)
    samples_per_category = num_samples // 10

    picked: list[tuple[str, str, str]] = []  # (category, id, prompt)
    for category in sorted(by_category.keys()):
        pool = by_category[category]
        if len(pool) < samples_per_category:
            raise ValueError(
                f"Not enough items in category '{category}': need {samples_per_category}, have {len(pool)}"
            )
        sampled = random.sample(pool, samples_per_category)
        picked.extend([(category, image_id, prompt) for image_id, prompt in sampled])

    # Same final shuffle step as the original prompt loader.
    random.shuffle(picked)

    out: list[SampleItem] = []
    for idx, (category, image_id, prompt) in enumerate(picked):
        out.append(SampleItem(image_id=image_id, category=category, prompt=prompt, global_idx=idx))
    return out


def _iter_zip_images(zip_path: str) -> Iterable[tuple[str, str]]:
    """
    Yield (image_id, member_name) pairs for image files inside the MJHQ zip.
    The image_id is derived from the basename stem (without extension).
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            p = Path(name)
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            yield (p.stem, name)


def build_zip_index(zip_path: str) -> dict[str, str]:
    """
    Map image_id -> zip member name.
    """
    index: dict[str, str] = {}
    for image_id, member in _iter_zip_images(zip_path):
        # Keep the first occurrence if duplicates exist.
        index.setdefault(image_id, member)
    return index


def write_images(
    *,
    zip_path: str,
    zip_index: dict[str, str],
    items: list[SampleItem],
    output_dir: str,
    manifest_path: str | None,
    overwrite: bool,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    failed: list[str] = []
    written = 0
    skipped = 0

    manifest_f = None
    if manifest_path is not None:
        manifest_f = open(manifest_path, "w", encoding="utf-8")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for it in items:
                member = zip_index.get(it.image_id)
                if member is None:
                    missing.append(it.image_id)
                    continue

                # Match the evaluate scripts' image naming exactly.
                out_name = f"{it.category}_{it.global_idx:05d}.png"
                out_path = out_dir / out_name

                if out_path.exists() and not overwrite:
                    skipped += 1
                else:
                    try:
                        with zf.open(member, "r") as src:
                            raw = src.read()
                        img = Image.open(BytesIO(raw))
                        img = img.convert("RGB")
                        img.save(out_path, format="PNG")
                        written += 1
                    except Exception:
                        failed.append(it.image_id)
                        continue

                if manifest_f is not None:
                    manifest_f.write(
                        json.dumps(
                            {
                                "global_idx": it.global_idx,
                                "category": it.category,
                                "image_id": it.image_id,
                                "prompt": it.prompt,
                                "zip_member": member,
                                "output_file": str(out_path),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    finally:
        if manifest_f is not None:
            manifest_f.close()

    print("\n" + "=" * 70)
    print("MJHQ seed folder build complete")
    print("=" * 70)
    print(f"Output dir: {out_dir.resolve()}")
    print(f"Total requested: {len(items)}")
    print(f"Written: {written}, skipped: {skipped}")
    if manifest_path is not None:
        print(f"Manifest: {Path(manifest_path).resolve()}")
    if missing:
        print(f"⚠️  Missing {len(missing)} ids in zip index (showing up to 20):")
        for mid in missing[:20]:
            print(f"  - {mid}")
    if failed:
        print(f"⚠️  Failed to decode/convert {len(failed)} image(s) (showing up to 20):")
        for mid in failed[:20]:
            print(f"  - {mid}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build MJHQ real-image folder matching the evaluation MJHQ prompt sampling."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to write sampled real MJHQ images (flat folder).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of samples to select (default: 5000). Must be divisible by 10.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="MJHQ prompt sampling seed (default: 42).",
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default=None,
        help="Optional path to MJHQ meta_data.json; if not provided, downloads from HF Hub.",
    )
    parser.add_argument(
        "--zip_path",
        type=str,
        default=None,
        help="Optional path to mjhq30k_imgs.zip; if not provided, downloads from HF Hub.",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Optional path to write a JSONL manifest (recommended).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in output_dir.",
    )
    args = parser.parse_args()

    items = sample_mjhq_items(meta_path=args.meta_path, num_samples=int(args.num_samples), seed=int(args.seed))
    print(f"Sampled {len(items)} items (seed={args.seed}, num_samples={args.num_samples})")

    zip_path = args.zip_path
    if not zip_path or not os.path.exists(zip_path):
        print("Downloading MJHQ-30K image zip from HF Hub...")
        zip_path = _download_zip()
    print(f"Zip: {zip_path}")

    print("Indexing zip members (image_id -> member path)...")
    zip_index = build_zip_index(zip_path)
    print(f"Indexed {len(zip_index)} image files")

    manifest_path = args.manifest_path
    if manifest_path is None:
        manifest_path = str(Path(args.output_dir) / "manifest.jsonl")

    write_images(
        zip_path=zip_path,
        zip_index=zip_index,
        items=items,
        output_dir=args.output_dir,
        manifest_path=manifest_path,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
