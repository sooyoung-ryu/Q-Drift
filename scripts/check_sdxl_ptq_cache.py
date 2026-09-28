#!/usr/bin/env python3
"""Validate the exact SDXL DeepCompressor PTQ cache inventory."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "manifests" / "sdxl_ptq_selection.json"


@dataclass(frozen=True)
class CacheCheck:
    expected: int
    present: int
    missing: int
    empty: int
    extra: int

    @property
    def valid(self) -> bool:
        return self.missing == 0 and self.empty == 0 and self.extra == 0


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    stems = manifest.get("sample_stems")
    steps = manifest.get("steps")
    guidance_branches = manifest.get("guidance_branches")
    if not isinstance(stems, list) or not all(isinstance(stem, str) for stem in stems):
        raise ValueError("manifest sample_stems must be a list of strings")
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("manifest steps must be a positive integer")
    if not isinstance(guidance_branches, int) or guidance_branches <= 0:
        raise ValueError("manifest guidance_branches must be a positive integer")
    return manifest


def expected_cache_names(manifest: dict) -> set[str]:
    return {
        f"{stem}-{step:05d}-{guidance}.pt"
        for stem in manifest["sample_stems"]
        for step in range(manifest["steps"])
        for guidance in range(manifest["guidance_branches"])
    }


def check_cache(cache_dir: Path, manifest_path: Path = DEFAULT_MANIFEST) -> CacheCheck:
    manifest = load_manifest(manifest_path)
    expected = expected_cache_names(manifest)

    actual_paths = {path.name: path for path in cache_dir.glob("*.pt")} if cache_dir.is_dir() else {}
    missing_names = expected - set(actual_paths)
    extra_names = set(actual_paths) - expected
    empty_names = {name for name in expected & set(actual_paths) if actual_paths[name].stat().st_size == 0}
    present = len(expected) - len(missing_names)

    return CacheCheck(
        expected=len(expected),
        present=present,
        missing=len(missing_names),
        empty=len(empty_names),
        extra=len(extra_names),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate SDXL PTQ cache files against manifests/sdxl_ptq_selection.json."
    )
    parser.add_argument("cache_dir", type=Path, help="Directory containing DeepCompressor .pt cache files.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Selection manifest path. Defaults to {DEFAULT_MANIFEST.relative_to(REPO_ROOT)}.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = check_cache(args.cache_dir, args.manifest)
    status = "valid" if result.valid else "invalid"
    print(
        f"{status}: expected={result.expected} present={result.present} "
        f"missing={result.missing} empty={result.empty} extra={result.extra}"
    )
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
