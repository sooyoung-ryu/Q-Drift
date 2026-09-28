#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_BASE_MODEL = "black-forest-labs/FLUX.1-schnell"


def _repo_paths() -> tuple[Path, Path, Path, Path]:
    """
    Returns:
        exp_root: .../experiments/svdquant/flux.1-schnell_W3A4
        repo_root: repository root
        deepcompressor_root: .../third_party/deepcompressor
        deepcompressor_examples_dir: .../third_party/deepcompressor/examples/diffusion
    """
    exp_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[4]
    deepcompressor_root = repo_root / "third_party" / "deepcompressor"
    deepcompressor_examples_dir = deepcompressor_root / "examples" / "diffusion"
    return exp_root, repo_root, deepcompressor_root, deepcompressor_examples_dir


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    if proc.returncode != 0:
        raise SystemExit(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _resolve_base_model(base_model: str, *, local_files_only: bool) -> str:
    p = Path(base_model)
    if p.exists():
        return str(p.resolve())
    if not local_files_only:
        return base_model
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=base_model, local_files_only=True)
    except Exception as e:
        raise SystemExit(
            "\n".join(
                [
                    "Cannot find the FLUX.1-schnell model locally (offline mode).",
                    f"- base_model: {base_model}",
                    "- Fix options:",
                    "  1) Pass `--base_model /path/to/local/FLUX.1-schnell` (a diffusers pipeline folder).",
                    "  2) Or re-run with `--allow_downloads` in an environment with network access.",
                    f"- Original error: {e}",
                ]
            )
        )


def _write_deepcompressor_model_cfg(*, src_cfg: Path, dst_cfg: Path, resolved_base_model: str) -> None:
    data = yaml.safe_load(src_cfg.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("pipeline"), dict):
        raise SystemExit(f"Unexpected YAML structure (expected mapping with `pipeline`): {src_cfg}")

    data["pipeline"]["path"] = resolved_base_model
    dumped = yaml.safe_dump(data, sort_keys=False)
    dst_cfg.write_text(dumped + ("" if dumped.endswith("\n") else "\n"), encoding="utf-8")


def main() -> None:
    exp_root, repo_root, deepcompressor_root, examples_dir = _repo_paths()
    if not examples_dir.is_dir():
        raise SystemExit(f"DeepCompressor examples dir not found: {examples_dir}")

    parser = argparse.ArgumentParser(
        description=(
            "Quantize FLUX.1-schnell Transformer to W3A4 using DeepCompressor and save checkpoint under "
            "`flux.1-schnell_w3a4/model/transformer_w3a4_g64/`.\n\n"
            "This is the STAGE=quantize step of run_paper.sh."
        )
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="HF repo id or local directory containing FLUX.1-schnell (diffusers pipeline).",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        default=False,
        help="If set, do not try to download from the internet (recommended on restricted networks).",
    )
    parser.add_argument(
        "--allow_downloads",
        action="store_true",
        default=False,
        help="If set, allow downloading model files from the internet (requires network access).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="deepcompressor",
        choices=["deepcompressor"],
        help="Quantization backend (currently only deepcompressor is supported here).",
    )
    parser.add_argument(
        "--quant_config",
        type=str,
        default="flux_w3a4.yaml",
        help=(
            "DeepCompressor quant config path. Accepts:\n"
            "  - a filename under `examples/diffusion/configs/svdquant/` (e.g. flux_w3a4.yaml, flux_w3a4_full.yaml)\n"
            "  - or a path relative to the DeepCompressor examples dir\n"
            "  - or an absolute path"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="model",
        help="Root directory (relative to this experiment folder unless absolute).",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="",
        help='Subdirectory name under output_root (default: "transformer_w3a4_g{group_size}").',
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="If set, remove existing output directory before writing.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=64,
        help="Group size along input channels/features for weight quantization (must match quant config).",
    )
    parser.add_argument(
        "--skip_collect",
        action="store_true",
        default=False,
        help="If set, skip DeepCompressor calibration cache collection even if caches are missing.",
    )
    args = parser.parse_args()

    if args.allow_downloads and args.local_files_only:
        raise SystemExit("Only one of --allow_downloads or --local_files_only can be set.")

    output_name = args.output_name.strip() or f"transformer_w3a4_g{int(args.group_size)}"
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (exp_root / output_root).resolve()
    output_dir = (output_root / output_name).resolve()

    model_pt = output_dir / "model.pt"
    transformer_dir = output_dir / "transformer"
    if (model_pt.exists() or transformer_dir.is_dir()) and not args.overwrite:
        print(f"⏭️  SKIPPING: quantization (found {model_pt} or {transformer_dir})")
        return

    if args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_base_model = _resolve_base_model(args.base_model, local_files_only=bool(args.local_files_only))

    model_cfg_src = (examples_dir / "configs" / "model" / "flux.1-schnell.yaml").resolve()
    if not model_cfg_src.is_file():
        raise SystemExit(f"DeepCompressor model config not found: {model_cfg_src}")

    def _slug(s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
        return s[:120] or "unnamed"

    generated_cfg_dir = (examples_dir / ".cache" / "model_configs").resolve()
    generated_cfg_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = (generated_cfg_dir / f"flux.1-schnell__{_slug(output_name)}.yaml").resolve()
    _write_deepcompressor_model_cfg(src_cfg=model_cfg_src, dst_cfg=model_cfg, resolved_base_model=resolved_base_model)

    collect_cfg = (examples_dir / "configs" / "collect" / "qdiff.yaml").resolve()

    def _resolve_quant_cfg(arg: str) -> Path:
        arg = (arg or "").strip()
        if not arg:
            raise SystemExit("--quant_config cannot be empty")
        p = Path(arg)
        if p.is_absolute():
            return p
        # relative to examples dir
        cand = (examples_dir / p).resolve()
        if cand.is_file():
            return cand
        # filename under configs/svdquant
        cand = (examples_dir / "configs" / "svdquant" / arg).resolve()
        if cand.is_file():
            return cand
        # relative to current cwd
        cand = p.resolve()
        if cand.is_file():
            return cand
        raise SystemExit(f"Quant config not found: {arg}")

    quant_cfg = _resolve_quant_cfg(args.quant_config)

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{deepcompressor_root}:{env.get('PYTHONPATH','')}"
    env["PATH"] = f"{Path(sys.executable).resolve().parent}:{env.get('PATH','')}"

    # DeepCompressor calibration cache directory is deterministic from configs:
    #   datasets/{dtype}/{pipeline.name}/{eval.protocol}/{dataset_name}/s128/caches/*.pt
    calib_dir = examples_dir / "datasets" / "torch.bfloat16" / "flux.1-schnell" / "fmeuler4-g0" / "qdiff" / "s128"
    caches_dir = calib_dir / "caches"

    if args.skip_collect:
        print("⏭️  Skipping calibration cache collection (--skip_collect)")
    else:
        has_cache = caches_dir.is_dir() and any(caches_dir.glob("*.pt"))
        if has_cache:
            print(f"⏭️  SKIPPING: calibration cache collection (found {caches_dir}/*.pt)")
        else:
            print("Collecting DeepCompressor calibration caches (qdiff, 128 prompts)...")
            model_cfg_rel = os.path.relpath(model_cfg, examples_dir)
            collect_cfg_rel = os.path.relpath(collect_cfg, examples_dir)
            collect_env = dict(env)
            collect_env.pop("DEEPCOMPRESSOR_DIFFUSION_MODEL_ONLY_ON_GPU", None)
            _run(
                [
                    sys.executable,
                    "-m",
                    "deepcompressor.app.diffusion.dataset.collect.calib",
                    model_cfg_rel,
                    collect_cfg_rel,
                ],
                cwd=examples_dir,
                env=collect_env,
            )

    print(f"Running DeepCompressor PTQ (SVDQuant) and saving checkpoint to {output_dir}...")
    model_cfg_rel = os.path.relpath(model_cfg, examples_dir)
    quant_cfg_rel = os.path.relpath(quant_cfg, examples_dir)
    ptq_env = dict(env)
    ptq_env["DEEPCOMPRESSOR_DIFFUSION_MODEL_ONLY_ON_GPU"] = "1"
    _run(
        [
            sys.executable,
            "-m",
            "deepcompressor.app.diffusion.ptq",
            model_cfg_rel,
            quant_cfg_rel,
            "--skip-gen",
            "true",
            "--skip-eval",
            "true",
            "--copy-on-save",
            "true",
            "--save-model",
            str(output_dir),
        ],
        cwd=examples_dir,
        env=ptq_env,
    )

    config_json = output_dir / "deepcompressor_config.json"
    config_json.write_text(
        json.dumps(
            {
                "backend": "deepcompressor",
                "examples_dir": str(examples_dir),
                "model_config": str(model_cfg),
                "quant_config": str(quant_cfg),
                "collect_config": str(collect_cfg),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("✓ Quantization completed")
    print(f"  - checkpoint: {output_dir}")
    print(f"  - config:     {config_json}")


if __name__ == "__main__":
    main()
