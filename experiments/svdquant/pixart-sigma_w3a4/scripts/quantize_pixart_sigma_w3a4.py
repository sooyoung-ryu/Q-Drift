#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path


DEFAULT_BASE_MODEL = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"


def _repo_paths() -> tuple[Path, Path, Path, Path]:
    """
    Returns:
        exp_root: .../experiments/svdquant/pixart-sigma_w3a4
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
                    "Cannot find the PixArt-Sigma model locally (offline mode).",
                    f"- base_model: {base_model}",
                    "- Fix options:",
                    "  1) Pass `--base_model /path/to/local/PixArt-Sigma-XL-2-1024-MS` (a diffusers pipeline folder).",
                    "  2) Or re-run with `--allow_downloads` in an environment with network access.",
                    f"- Original error: {e}",
                ]
            )
        )


def _write_deepcompressor_model_cfg(*, src_cfg: Path, dst_cfg: Path, resolved_base_model: str) -> None:
    """
    Ensure `pipeline.path` exists and points to `resolved_base_model`.
    """
    text = src_cfg.read_text(encoding="utf-8")

    # Case 1: pipeline.path exists -> replace it.
    replaced = re.sub(r'(?m)^  path: .*$', f'  path: "{resolved_base_model}"', text, count=1)
    if replaced != text:
        dst_cfg.write_text(replaced + ("\n" if not replaced.endswith("\n") else ""), encoding="utf-8")
        return

    # Case 2: pipeline.path missing -> inject after `pipeline:` block header.
    m = re.search(r"(?m)^pipeline:\s*$", text)
    if not m:
        raise SystemExit(f"Failed to locate `pipeline:` section in model config: {src_cfg}")

    insert_at = m.end()
    injected = text[:insert_at] + f'\n  path: "{resolved_base_model}"' + text[insert_at:]
    dst_cfg.write_text(injected + ("\n" if not injected.endswith("\n") else ""), encoding="utf-8")


def _sha256_text(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    exp_root, repo_root, deepcompressor_root, examples_dir = _repo_paths()
    if not examples_dir.is_dir():
        raise SystemExit(f"DeepCompressor examples dir not found: {examples_dir}")

    parser = argparse.ArgumentParser(
        description=(
            "Quantize PixArt-Sigma DiT (Transformer2D) to W3A4 using DeepCompressor (SVDQuant PTQ) and save a reusable "
            "checkpoint for Q-Drift under `pixart-sigma_w3a4/model/transformer_w3a4/`.\n\n"
            "This is the STAGE=quantize step of run_paper.sh."
        )
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="HF repo id or local directory containing PixArt-Sigma (diffusers pipeline).",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        default=False,
        help="If set, do not try to download from the internet.",
    )
    parser.add_argument(
        "--allow_downloads",
        action="store_true",
        default=False,
        help="If set, allow downloading model files from the internet.",
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
        help='Subdirectory name under output_root (default: "transformer_w3a4").',
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="If set, remove existing output directory before writing.",
    )
    args = parser.parse_args()

    if args.allow_downloads and args.local_files_only:
        raise SystemExit("Only one of --allow_downloads or --local_files_only can be set.")

    output_name = args.output_name.strip() or "transformer_w3a4"
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (exp_root / output_root).resolve()
    output_dir = (output_root / output_name).resolve()

    model_pt = output_dir / "model.pt"
    if model_pt.exists() and not args.overwrite:
        print(f"⏭️  SKIPPING: quantization (found {model_pt})")
        return

    print("PixArt-Sigma DeepCompressor PTQ (W3A4)")
    print(f"- base_model:  {args.base_model}")
    print(f"- output_dir:  {output_dir}")

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_base_model = _resolve_base_model(args.base_model, local_files_only=bool(args.local_files_only))

    model_cfg_src = (examples_dir / "configs" / "model" / "pixart-sigma.yaml").resolve()
    if not model_cfg_src.is_file():
        raise SystemExit(f"DeepCompressor model config not found: {model_cfg_src}")

    def _slug(s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
        return s[:120] or "unnamed"

    generated_cfg_dir = (exp_root / ".cache" / "model_configs").resolve()
    generated_cfg_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = (generated_cfg_dir / f"pixart-sigma__{_slug(output_name)}.yaml").resolve()
    _write_deepcompressor_model_cfg(src_cfg=model_cfg_src, dst_cfg=model_cfg, resolved_base_model=resolved_base_model)

    cfg_default = (examples_dir / "configs" / "__default__.yaml").resolve()
    quant_cfg = (examples_dir / "configs" / "svdquant" / "pixart_sigma_w3a4.yaml").resolve()
    prompt_qdiff = (examples_dir / "prompts" / "qdiff.yaml").resolve()

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{deepcompressor_root}:{env.get('PYTHONPATH','')}"
    env.setdefault("DEEPCOMPRESSOR_EXT_VERBOSE", "1")

    # Make the first-time extension build deterministic & easier to inspect.
    torch_ext_dir = env.get("TORCH_EXTENSIONS_DIR", "").strip()
    if not torch_ext_dir:
        torch_ext_dir = str((exp_root / ".torch_extensions").resolve())
        env["TORCH_EXTENSIONS_DIR"] = torch_ext_dir
    Path(torch_ext_dir).mkdir(parents=True, exist_ok=True)

    if not env.get("TORCH_CUDA_ARCH_LIST", "").strip():
        try:
            import torch

            caps = set()
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    major, minor = torch.cuda.get_device_capability(i)
                    caps.add(f"{major}.{minor}")
            if caps:
                env["TORCH_CUDA_ARCH_LIST"] = ";".join(sorted(caps))
        except Exception:
            pass

    print(f"- TORCH_EXTENSIONS_DIR: {env.get('TORCH_EXTENSIONS_DIR')}")
    if env.get("TORCH_CUDA_ARCH_LIST"):
        print(f"- TORCH_CUDA_ARCH_LIST: {env.get('TORCH_CUDA_ARCH_LIST')}")
    print("- NOTE: First run may compile a CUDA extension (can take 5–60+ min).", flush=True)

    # DeepCompressor reads calibration caches from `quant.calib.path` in
    # `deepcompressor/examples/diffusion/configs/__default__.yaml`, which is a relative path:
    #   datasets/{dtype}/{model}/{protocol}/{data}/s128
    #
    # Since we run DeepCompressor with cwd=repo_root, caches must live under:
    #   {repo_root}/datasets/...
    datasets_root = (repo_root / "datasets").resolve()

    # Cache path (W3A4 uses a distinct dataset name to avoid clobbering W4A4 caches):
    #   {datasets_root}/torch.float16/pixart-sigma/dpm20-g4.5/qdiff_w3a4/s128/caches/*.pt
    calib_dir = (
        datasets_root / "torch.float16" / "pixart-sigma" / "dpm20-g4.5" / "qdiff_w3a4" / "s128"
    ).resolve()
    caches_dir = calib_dir / "caches"
    meta_path = calib_dir / "cache_meta.json"

    has_cache = caches_dir.is_dir() and any(caches_dir.glob("*.pt"))
    if has_cache:
        expected_meta = {
            "default_cfg_sha256": _sha256_text(cfg_default),
            "model_cfg_sha256": _sha256_text(model_cfg),
            "quant_cfg_sha256": _sha256_text(quant_cfg),
            "prompt_path": str(prompt_qdiff),
            "collect_num_samples": 128,
        }
        have_meta = {}
        try:
            have_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        except Exception:
            have_meta = {}

        if have_meta == expected_meta:
            print(f"⏭️  SKIPPING: calibration cache collection (found {caches_dir}/*.pt)")
        else:
            print("⚠️  Found existing calibration caches, but metadata does not match current configs.")
            print(f"- cache_dir: {caches_dir}")
            print(f"- meta:      {meta_path} ({'present' if meta_path.exists() else 'missing'})")
            print("Deleting and re-collecting caches for the current W3A4 config...")
            if calib_dir.exists():
                shutil.rmtree(calib_dir)
            has_cache = False

    if not has_cache:
        print("Collecting DeepCompressor calibration caches (qdiff_w3a4, 128 prompts)...")
        _run(
            [
                sys.executable,
                "-m",
                "deepcompressor.app.diffusion.dataset.collect.calib",
                str(cfg_default),
                str(model_cfg),
                str(quant_cfg),
                "--collect-root",
                str(datasets_root),
                "--collect-dataset-name",
                "qdiff_w3a4",
                "--collect-data-path",
                str(prompt_qdiff),
                "--collect-num-samples",
                "128",
                "--output-root",
                str(exp_root / "tmp"),
                "--eval-benchmarks",
                "MJHQ",
                "--eval-num-samples",
                "1",
            ],
            # omniconfig asserts that config paths are under os.getcwd().
            # Use a common ancestor so both DeepCompressor configs and our generated model config pass.
            cwd=repo_root,
            env=env,
        )
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(
                {
                    "default_cfg_sha256": _sha256_text(cfg_default),
                    "model_cfg_sha256": _sha256_text(model_cfg),
                    "quant_cfg_sha256": _sha256_text(quant_cfg),
                    "prompt_path": str(prompt_qdiff),
                    "collect_num_samples": 128,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print(f"Running DeepCompressor PTQ (W3A4) and saving checkpoint to {output_dir}...")
    _run(
        [
            sys.executable,
            "-m",
            "deepcompressor.app.diffusion.ptq",
            str(cfg_default),
            str(model_cfg),
            str(quant_cfg),
            "--skip-gen",
            "true",
            "--skip-eval",
            "true",
            "--copy-on-save",
            "true",
            "--save-model",
            str(output_dir),
            "--output-root",
            str(exp_root / "runs"),
            "--cache-root",
            str(exp_root / "runs"),
        ],
        # omniconfig asserts that config paths are under os.getcwd().
        cwd=repo_root,
        env=env,
    )

    expected_files = [
        output_dir / "model.pt",
        output_dir / "wgts.pt",
        output_dir / "branch.pt",
        output_dir / "smooth.pt",
    ]
    missing = [str(p) for p in expected_files if not p.exists()]
    if missing:
        raise SystemExit(
            "\n".join(
                [
                    "DeepCompressor PTQ finished, but expected checkpoint files are missing:",
                    *[f"- {p}" for p in missing],
                    "",
                    "This usually means the quant config did not enable smooth/low-rank as intended, or the run failed silently.",
                    f"- quant_cfg: {quant_cfg}",
                    f"- output_dir: {output_dir}",
                ]
            )
        )

    config_json = output_dir / "deepcompressor_config.json"
    config_json.write_text(
        json.dumps(
            {
                "backend": "deepcompressor",
                "examples_dir": str(examples_dir),
                "default_config": str(cfg_default),
                "model_config": str(model_cfg),
                "quant_config": str(quant_cfg),
                "collect_prompt": str(prompt_qdiff),
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
