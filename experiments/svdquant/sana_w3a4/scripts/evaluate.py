"""
Sampling-only evaluation script for Q-Drift correction on MJHQ-30K.

This script generates and saves images for:
  - BF16 Sana baseline
  - Quantized baseline (no correction)
  - Quantized + Q-Drift (multiple bias/qdrift scales)

It intentionally does NOT compute metrics. Metrics are computed separately by:
  - evaluation/compute_metrics.py

Reproducibility design:
  - Prompt sampling is controlled by: --mjhq_prompt_sample_seed
  - Initial noise is controlled independently by: --noise_seed_start
    (per-prompt deterministic seed = noise_seed_start + global_idx)

Optional logging:
  - Saves full per-step x_t latents as sharded `.pth` files under `--xt_output_dir/xt_latents/`.
    You can enable this separately for:
      * Q-Drift variants: --save_xt (default: True)
      * BF16 baseline: --save_xt_fp16
      * Quant baseline: --save_xt_quant_baseline
    Rank 0 can also merge shards into a single large `.pth` per variant.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline, SanaPipeline
from huggingface_hub import hf_hub_download

# Some environments ship a `SanaPipeline.__call__` that expects `self._execution_device`,
# but `DiffusionPipeline._execution_device` may be missing or may raise (e.g., missing
# offload hook helpers). Patch in a safe fallback that always returns `self.device`.
_orig_exec_prop = getattr(DiffusionPipeline, "_execution_device", None)
if isinstance(_orig_exec_prop, property):
    _orig_exec_fget = _orig_exec_prop.fget

    def _safe_execution_device(self):
        try:
            if _orig_exec_fget is not None:
                return _orig_exec_fget(self)
        except Exception:
            pass
        return getattr(self, "device", torch.device("cpu"))

    DiffusionPipeline._execution_device = property(_safe_execution_device)  # type: ignore[assignment]
else:
    DiffusionPipeline._execution_device = property(lambda self: getattr(self, "device", torch.device("cpu")))  # type: ignore[attr-defined,assignment]

# Additionally, some Sana builds look up `_execution_device` directly on the pipeline instance/class.
# Ensure SanaPipeline has a robust execution-device property that we can override per-rank via an
# instance attribute.
def _sana_safe_execution_device(self):
    forced = getattr(self, "_forced_execution_device", None)
    if forced is not None:
        return forced
    return getattr(self, "device", torch.device("cpu"))

SanaPipeline._execution_device = property(_sana_safe_execution_device)  # type: ignore[assignment]

# Some older/forked diffusers builds don't define `DiffusionPipeline.device`, but Sana's
# pipeline implementation may still try to access `self.device`. Ensure it exists.
if not hasattr(SanaPipeline, "device"):
    SanaPipeline.device = property(lambda self: getattr(self, "_forced_execution_device", torch.device("cpu")))  # type: ignore[attr-defined,assignment]

# Allow importing the project-local Q-Drift schedulers at:
# `schedulers/*` at the repo root.
_PROJECT_ROOT = None
for _p in Path(__file__).resolve().parents:
    if (_p / "schedulers").is_dir():
        _PROJECT_ROOT = _p
        break
if _PROJECT_ROOT is not None and str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from schedulers.dpm_solver_qdrift import DPMSolverQDriftScheduler
from tqdm import tqdm

from deepcompressor_sana_loader import load_quant_transformer

# MJHQ-30K constants
MJHQ_REPO_ID = "playgroundai/MJHQ-30K"
MJHQ_META_FILENAME = "meta_data.json"

DEFAULT_BASE_MODEL = "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
DEFAULT_QUANT_MODEL_DIR = str((Path(__file__).resolve().parents[1] / "model" / "transformer_w3a4").resolve())


def download_mjhq_metadata() -> str:
    return hf_hub_download(repo_id=MJHQ_REPO_ID, filename=MJHQ_META_FILENAME, repo_type="dataset")


def load_mjhq_metadata(meta_path: str) -> Dict[str, Dict[str, Any]]:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompts(meta_path: Optional[str], num_samples: int, mjhq_prompt_sample_seed: int) -> List[Dict[str, Any]]:
    if not meta_path or not os.path.exists(meta_path):
        meta_path = download_mjhq_metadata()

    metadata = load_mjhq_metadata(meta_path)

    prompts_by_category: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for image_id, info in metadata.items():
        if isinstance(info, dict) and (info.get("prompt") or "").strip():
            prompts_by_category[str(info["category"])].append(
                {"prompt": info["prompt"].strip(), "id": image_id, "category": str(info["category"])}
            )

    # Stratified sampling: 10 categories, equal allocation.
    random.seed(mjhq_prompt_sample_seed)
    samples_per_category = num_samples // 10

    prompts: List[Dict[str, Any]] = []
    for category in sorted(prompts_by_category.keys()):
        sampled = random.sample(prompts_by_category[category], samples_per_category)
        prompts.extend(sampled)

    random.shuffle(prompts)

    # Add a stable global index so:
    #  - file naming is consistent across ranks
    #  - noise seeding can be tied to global_idx
    for i, row in enumerate(prompts):
        row["global_idx"] = i

    return prompts


def _parse_global_indices(spec: Optional[str]) -> Optional[List[int]]:
    if spec is None or not spec.strip():
        return None
    indices = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid --global_indices range: {part}")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    return sorted(indices)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _all_images_exist(output_dir: str, prompts: List[Dict[str, Any]]) -> bool:
    for row in prompts:
        category = row["category"]
        global_idx = int(row["global_idx"])
        filename = f"{category}_{global_idx:05d}.png"
        if not os.path.exists(os.path.join(output_dir, filename)):
            return False
    return True


def _existing_image_dirs(output_dir: str, model_key: str) -> List[str]:
    dirs = [os.path.join(output_dir, "images", model_key)]
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            if name.startswith("rank"):
                dirs.append(os.path.join(output_dir, name, "images", model_key))
    return dirs


def _all_images_exist_any(dirs: List[str], prompts: List[Dict[str, Any]]) -> bool:
    for row in prompts:
        category = row["category"]
        global_idx = int(row["global_idx"])
        filename = f"{category}_{global_idx:05d}.png"
        if not any(d and os.path.exists(os.path.join(d, filename)) for d in dirs):
            return False
    return True


def _parse_xt_dtype(name: str) -> torch.dtype:
    name = (name or "").lower().strip()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported --xt_dtype: {name} (expected bfloat16|float16|float32)")


class XTLatentsLogger:
    """
    Save full per-step x_t (latents) during sampling, sharded to disk.

    Each shard stores:
      - xt[t]: [N, C, H, W] (on CPU) for each timestep t
      - global_indices: which prompts (global_idx) are included
    """

    def __init__(
        self,
        *,
        model_key: str,
        xt_output_dir: str,
        rank: int,
        world_size: int,
        prompt_seed: int,
        noise_seed_start: int,
        num_inference_steps: int,
        shard_size: int,
        xt_dtype: torch.dtype,
    ):
        self.model_key = model_key
        self.xt_output_dir = xt_output_dir
        self.rank = rank
        self.world_size = world_size
        self.prompt_seed = prompt_seed
        self.noise_seed_start = noise_seed_start
        self.num_inference_steps = num_inference_steps
        self.shard_size = int(shard_size)
        if self.shard_size <= 0:
            raise ValueError("--xt_shard_size must be > 0")
        self.xt_dtype = xt_dtype

        self._shard_id = 0
        self._global_indices_in_shard: List[int] = []
        self._xt_by_t: DefaultDict[float, List[torch.Tensor]] = defaultdict(list)

    def begin_sample(self, global_idx: int) -> None:
        self._global_indices_in_shard.append(int(global_idx))

    def record_step(self, timestep: float, latents: torch.Tensor) -> None:
        # latents: [B, C, H, W]; pipeline generation uses B==1 here.
        latents_cpu = latents.detach().to("cpu")
        if latents_cpu.dtype != self.xt_dtype:
            latents_cpu = latents_cpu.to(dtype=self.xt_dtype)
        self._xt_by_t[float(timestep)].append(latents_cpu[0].contiguous())

    def _flush_shard(self) -> Optional[str]:
        if len(self._global_indices_in_shard) == 0:
            return None

        _ensure_dir(self.xt_output_dir)
        shard_path = os.path.join(
            self.xt_output_dir, f"xt_latents_{self.model_key}_rank{self.rank}_shard{self._shard_id:05d}.pth"
        )

        timesteps = sorted(self._xt_by_t.keys())
        xt_dict: Dict[float, torch.Tensor] = {}
        for t in timesteps:
            xt_dict[t] = torch.stack(self._xt_by_t[t], dim=0)

        payload = {
            "model_key": self.model_key,
            "rank": int(self.rank),
            "world_size": int(self.world_size),
            "prompt_seed": int(self.prompt_seed),
            "noise_seed_start": int(self.noise_seed_start),
            "num_inference_steps": int(self.num_inference_steps),
            "xt_dtype": str(self.xt_dtype).replace("torch.", ""),
            "global_indices": list(self._global_indices_in_shard),
            "timesteps": timesteps,
            "xt": xt_dict,  # {timestep: [N, C, H, W]}
            "num_samples": int(len(self._global_indices_in_shard)),
        }
        torch.save(payload, shard_path)

        self._shard_id += 1
        self._global_indices_in_shard = []
        self._xt_by_t = defaultdict(list)
        return shard_path

    def end_sample(self) -> Optional[str]:
        if len(self._global_indices_in_shard) >= self.shard_size:
            return self._flush_shard()
        return None

    def finalize(self) -> List[str]:
        paths: List[str] = []
        last = self._flush_shard()
        if last is not None:
            paths.append(last)
        return paths


def _write_xt_index(xt_output_dir: str, model_key: str, world_size: int) -> Optional[str]:
    """
    Write a small `.pth` index describing where the sharded x_t latent files are.
    Avoids merging huge tensors into a single file.
    """
    if not os.path.isdir(xt_output_dir):
        return None
    shards = []
    for name in os.listdir(xt_output_dir):
        if name.startswith(f"xt_latents_{model_key}_rank") and name.endswith(".pth"):
            shards.append(os.path.join(xt_output_dir, name))
    shards = sorted(shards)
    if len(shards) == 0:
        return None

    index_path = os.path.join(xt_output_dir, f"xt_latents_{model_key}_index.pth")
    payload = {
        "format": "xt_latents_sharded_v1",
        "model_key": model_key,
        "world_size": int(world_size),
        "shards": shards,
    }
    torch.save(payload, index_path)
    return index_path


def _merge_xt_latents_single(xt_output_dir: str, model_key: str, world_size: int) -> Optional[str]:
    """
    Merge sharded x_t latents into a single large `.pth` file per variant.

    Warning: this is large (GBs) and requires substantial CPU RAM. Implemented to match the
    collect_statistics-style "single merged .pth" workflow.
    """
    if not os.path.isdir(xt_output_dir):
        return None

    shard_paths = []
    for name in os.listdir(xt_output_dir):
        if name.startswith(f"xt_latents_{model_key}_rank") and name.endswith(".pth"):
            shard_paths.append(os.path.join(xt_output_dir, name))
    shard_paths = sorted(shard_paths)
    if len(shard_paths) == 0:
        return None

    # Load and concatenate (order mirrors the shard_paths order).
    xt_chunks_by_t: DefaultDict[float, List[torch.Tensor]] = defaultdict(list)
    global_indices: List[int] = []

    prompt_seed = None
    noise_seed_start = None
    num_inference_steps = None
    xt_dtype = None

    for p in shard_paths:
        try:
            data = torch.load(p, map_location="cpu", weights_only=True)
        except TypeError:
            data = torch.load(p, map_location="cpu")
        prompt_seed = data.get("prompt_seed", prompt_seed)
        noise_seed_start = data.get("noise_seed_start", noise_seed_start)
        num_inference_steps = data.get("num_inference_steps", num_inference_steps)
        xt_dtype = data.get("xt_dtype", xt_dtype)
        global_indices.extend([int(x) for x in data.get("global_indices", [])])

        xt_by_t = data.get("xt", {})
        for t in data.get("timesteps", []):
            t_key = float(t)
            if t in xt_by_t:
                xt_chunks_by_t[t_key].append(xt_by_t[t])
            else:
                xt_chunks_by_t[t_key].append(xt_by_t[t_key])

    timesteps = sorted(xt_chunks_by_t.keys())
    xt_merged: Dict[float, torch.Tensor] = {}
    for t in timesteps:
        xt_merged[t] = torch.cat(xt_chunks_by_t[t], dim=0)

    # Reorder by global_idx for a stable alignment with prompts/images.
    if len(global_indices) > 0:
        gi = torch.tensor(global_indices, dtype=torch.int64)
        order = torch.argsort(gi)
        global_indices_sorted = gi[order].tolist()
        for t in timesteps:
            xt_merged[t] = xt_merged[t].index_select(0, order)
        global_indices = global_indices_sorted

    merged_path = os.path.join(xt_output_dir, f"xt_latents_{model_key}.pth")
    payload = {
        "format": "xt_latents_merged_v1",
        "model_key": model_key,
        "world_size": int(world_size),
        "prompt_seed": int(prompt_seed) if prompt_seed is not None else None,
        "noise_seed_start": int(noise_seed_start) if noise_seed_start is not None else None,
        "num_inference_steps": int(num_inference_steps) if num_inference_steps is not None else None,
        "xt_dtype": xt_dtype,
        "global_indices": global_indices,
        "timesteps": timesteps,
        "xt": xt_merged,
        "num_samples": int(len(global_indices)),
        "source_shards": shard_paths,
    }
    torch.save(payload, merged_path)
    return merged_path


def _generate_images(
    *,
    pipeline: SanaPipeline,
    prompts: List[Dict[str, Any]],
    output_dir: str,
    existing_dirs: Optional[List[str]] = None,
    prefix: str,
    noise_seed_start: int,
    num_inference_steps: int,
    guidance_scale: float,
    negative_prompt: str,
    height: int,
    width: int,
    device: torch.device,
    batch_size: int = 1,
    xt_logger: Optional[XTLatentsLogger] = None,
) -> None:
    _ensure_dir(output_dir)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("--batch_size must be >= 1")
    if batch_size != 1 and xt_logger is not None:
        raise ValueError("x_t logging supports only --batch_size 1. Use --no-save_xt for batched generation.")

    # Make sure SanaPipeline uses the correct per-rank CUDA device even if diffusers'
    # internal `_execution_device` helper is missing/broken in this environment.
    pipeline._forced_execution_device = device
    # Some Sana builds access `self.device` directly (and may not define it). Provide a
    # rank-local instance attribute as a fallback.
    try:
        _ = pipeline.device  # type: ignore[attr-defined]
    except Exception:
        try:
            setattr(pipeline, "device", device)
        except Exception:
            try:
                pipeline.__dict__["device"] = device
            except Exception:
                pass
    # Some Sana builds also access `self._execution_device` directly.
    try:
        setattr(pipeline, "_execution_device", device)
    except Exception:
        pass

    # Diffusers callback-based x_t logging (requires a recent diffusers).
    can_callback = True
    try:
        import inspect

        sig = inspect.signature(pipeline.__call__)
        can_callback = "callback_on_step_end" in sig.parameters
    except Exception:
        can_callback = False

    def _already_done(row: Dict[str, Any]) -> bool:
        category = row["category"]
        global_idx = int(row["global_idx"])
        filename = f"{category}_{global_idx:05d}.png"
        if existing_dirs is not None:
            for d in existing_dirs:
                if d and os.path.exists(os.path.join(d, filename)):
                    return True
        return os.path.exists(os.path.join(output_dir, filename))

    pending_prompts = [row for row in prompts if not _already_done(row)]
    if batch_size != 1:
        for start in tqdm(range(0, len(pending_prompts), batch_size), desc=f"Generating {prefix}"):
            batch = pending_prompts[start : start + batch_size]
            if not batch:
                continue
            prompt_batch = [row["prompt"] for row in batch]
            generators = [
                torch.Generator(device=device).manual_seed(int(noise_seed_start) + int(row["global_idx"]))
                for row in batch
            ]
            images = pipeline(
                prompt=prompt_batch,
                negative_prompt=[negative_prompt] * len(batch),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generators,
            ).images
            for row, image in zip(batch, images):
                category = row["category"]
                global_idx = int(row["global_idx"])
                filename = f"{category}_{global_idx:05d}.png"
                image.save(os.path.join(output_dir, filename))
        return

    for row in tqdm(pending_prompts, desc=f"Generating {prefix}"):
        prompt = row["prompt"]
        category = row["category"]
        global_idx = int(row["global_idx"])
        filename = f"{category}_{global_idx:05d}.png"
        out_path = os.path.join(output_dir, filename)

        generator = torch.Generator(device=device).manual_seed(int(noise_seed_start) + global_idx)

        callback = None
        callback_inputs = None
        if xt_logger is not None:
            if not can_callback:
                raise RuntimeError(
                    "x_t logging requested, but diffusers pipeline does not support `callback_on_step_end`.\n"
                    "Please upgrade diffusers, or disable with `--no-save_xt`."
                )

            xt_logger.begin_sample(global_idx)

            def _on_step_end(_pipe, _step_index: int, timestep, callback_kwargs):
                latents = callback_kwargs.get("latents", None)
                if latents is not None:
                    t_val = float(timestep.item()) if isinstance(timestep, torch.Tensor) else float(timestep)
                    xt_logger.record_step(t_val, latents)
                return callback_kwargs

            callback = _on_step_end
            callback_inputs = ["latents"]

        extra = {}
        if callback is not None:
            extra["callback_on_step_end"] = callback
            extra["callback_on_step_end_tensor_inputs"] = callback_inputs

        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            **extra,
        ).images[0]
        image.save(out_path)
        if xt_logger is not None:
            flushed = xt_logger.end_sample()
            if flushed is not None:
                print(f"✓ Flushed x_t shard: {flushed}")


def _apply_generation_args(args):
    if args.xt_output_dir is None:
        args.xt_output_dir = args.output_dir
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.qdrift_only:
        args.skip_fp16 = True
        args.skip_quant_baseline = True
    if args.batch_size != 1 and (args.save_xt or args.save_xt_fp16 or args.save_xt_quant_baseline):
        raise ValueError("Batched generation does not support x_t logging. Use --no-save_xt for Q-Drift batches.")
    return args

def main():
    parser = argparse.ArgumentParser(description="Sampling-only evaluation for Sana Q-Drift on MJHQ-30K")
    parser.add_argument("--mu_dict", type=str, required=True, help="Path to mu_dict.npy")
    parser.add_argument("--cov_dict", type=str, required=True, help="Path to cov_dict.npy")
    parser.add_argument("--output_dir", type=str, default="evaluation", help="Output directory for results")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of MJHQ prompts to sample")
    parser.add_argument(
        "--global_indices",
        type=str,
        default=None,
        help="Optional comma-separated global indices or inclusive ranges to generate from the full --num_samples draw.",
    )
    parser.add_argument(
        "--mjhq_prompt_sample_seed",
        type=int,
        default=42,
        help="Random seed for MJHQ prompt sampling (must match metrics-only stage)",
    )
    parser.add_argument(
        "--noise_seed_start",
        type=int,
        default=42,
        help="Base seed for initial noise (per-prompt seed = noise_seed_start + global_idx)",
    )
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--meta_path", type=str, default=None, help="Optional path to MJHQ meta_data.json")
    parser.add_argument("--qdrift_scales", type=float, nargs="+", default=[1.0])
    parser.add_argument(
        "--qdrift_scalar",
        action="store_true",
        help="Ablation: one scalar V per step (channel-averaged) instead of per-channel V.",
    )
    parser.add_argument("--bias_scales", type=float, nargs="+", default=[0.0])
    parser.add_argument("--batch_size", type=int, default=1, help="Prompt batch size for image generation")
    parser.add_argument(
        "--qdrift_only",
        action="store_true",
        help="Generate only Q-Drift variants; skip BF16/FP16 and quantized-baseline image generation.",
    )
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--quant_model_dir", type=str, default=DEFAULT_QUANT_MODEL_DIR)
    parser.add_argument(
        "--skip_fp16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip BF16/FP16 baseline image generation (useful when images already exist under --output_dir/images/fp16/)",
    )
    parser.add_argument(
        "--skip_quant_baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip quantized-baseline (no correction) image generation (useful when images already exist under --output_dir/images/quant_baseline/)",
    )
    parser.add_argument(
        "--save_xt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save full per-step x_t latents for Q-Drift variants to --xt_output_dir/xt_latents/",
    )
    parser.add_argument(
        "--save_xt_fp16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save per-step x_t latents for the BF16 baseline to --xt_output_dir/xt_latents/ (model_key='fp16')",
    )
    parser.add_argument(
        "--save_xt_quant_baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save per-step x_t latents for the quant baseline to --xt_output_dir/xt_latents/ (model_key='quant_baseline')",
    )
    parser.add_argument(
        "--xt_output_dir",
        type=str,
        default=None,
        help="Directory to save x_t latents (.pth). Default: same as --output_dir",
    )
    parser.add_argument("--xt_shard_size", type=int, default=32, help="Samples per x_t shard file")
    parser.add_argument("--xt_dtype", type=str, default="bfloat16", help="x_t dtype: bfloat16|float16|float32")
    parser.add_argument(
        "--xt_merge_single",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On rank 0, merge x_t shards into a single large `.pth` per variant",
    )
    parser.add_argument(
        "--xt_delete_shards_after_merge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete per-rank shard files after successfully writing the merged `.pth`",
    )
    parser.add_argument(
        "--dist_timeout_minutes",
        type=int,
        default=240,
        help="Process-group collective timeout (minutes). Increase for long CPU-side postprocessing/merges.",
    )
    args = parser.parse_args()
    args = _apply_generation_args(args)

    # torchrun sets env vars automatically
    torch.distributed.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=int(args.dist_timeout_minutes)),
    )
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if rank == 0:
        print(f"[Multi-GPU] world_size={world_size} output_dir={args.output_dir}")

    # Pre-download shared assets on rank 0 to reduce cache races.
    mjhq_meta_path = args.meta_path
    if rank == 0:
        try:
            if mjhq_meta_path is None or not os.path.exists(mjhq_meta_path):
                mjhq_meta_path = download_mjhq_metadata()
            _ = SanaPipeline.from_pretrained(args.base_model, variant="bf16", torch_dtype=torch.bfloat16)
            _ = load_quant_transformer(
                base_model_id=args.base_model,
                quant_model_dir=args.quant_model_dir,
                device=device,
                torch_dtype=torch.bfloat16,
            )
        except Exception as e:
            print(f"⚠️  Rank 0 prefetch failed: {e}")

    torch.distributed.barrier()
    if mjhq_meta_path is None or not os.path.exists(mjhq_meta_path):
        mjhq_meta_path = download_mjhq_metadata()

    # All ranks load identical prompt list, then slice by rank.
    all_prompts = load_prompts(mjhq_meta_path, args.num_samples, args.mjhq_prompt_sample_seed)
    selected_indices = _parse_global_indices(args.global_indices)
    if selected_indices is None:
        generation_prompts = all_prompts
    else:
        selected_set = set(selected_indices)
        generation_prompts = [row for row in all_prompts if int(row["global_idx"]) in selected_set]
        missing_selected = sorted(selected_set.difference(int(row["global_idx"]) for row in generation_prompts))
        if missing_selected:
            raise ValueError(f"--global_indices contains indices outside the prompt set: {missing_selected[:20]}")
        if rank == 0:
            print(f"Generating selected global indices: {len(generation_prompts)} / {len(all_prompts)}")

    samples_per_gpu = len(generation_prompts) // world_size
    start_idx = rank * samples_per_gpu
    end_idx = len(generation_prompts) if rank == world_size - 1 else start_idx + samples_per_gpu
    prompts = generation_prompts[start_idx:end_idx]
    if prompts:
        min_global = min(int(row["global_idx"]) for row in prompts)
        max_global = max(int(row["global_idx"]) for row in prompts)
        print(
            f"[Rank {rank}/{world_size}] prompts: {len(prompts)} "
            f"(selected idx {start_idx}:{end_idx}, global {min_global}:{max_global})"
        )
    else:
        print(f"[Rank {rank}/{world_size}] prompts: 0 (selected idx {start_idx}:{end_idx})")

    # Rank-local output dirs
    rank_output_dir = os.path.join(args.output_dir, f"rank{rank}")
    images_root = os.path.join(rank_output_dir, "images")
    _ensure_dir(images_root)
    merged_images_root = os.path.join(args.output_dir, "images")

    # Save prompt list once (rank 0).
    if rank == 0:
        _ensure_dir(args.output_dir)
        eval_results_path = os.path.join(args.output_dir, "evaluation_results.json")
        payload = {
            "benchmark": "MJHQ-30K",
            "num_samples": int(args.num_samples),
            "mjhq_prompt_sample_seed": int(args.mjhq_prompt_sample_seed),
            "noise_seed_start": int(args.noise_seed_start),
            "num_inference_steps": int(args.num_inference_steps),
            "guidance_scale": float(args.guidance_scale),
            "height": int(args.height),
            "width": int(args.width),
            "negative_prompt": str(args.negative_prompt),
            "base_model": str(args.base_model),
            "quant_model_dir": str(args.quant_model_dir),
            "qdrift_scales": list(args.qdrift_scales),
            "qdrift_scalar": bool(args.qdrift_scalar),
            "qdrift_only": bool(args.qdrift_only),
            "batch_size": int(args.batch_size),
            "bias_scales": list(args.bias_scales),
            "selected_global_indices": selected_indices,
            "prompts": all_prompts,
        }
        with open(eval_results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved prompts + config: {eval_results_path}")

    torch.distributed.barrier()

    # 1) FP16 images (optional x_t logging)
    fp16_dir = os.path.join(images_root, "fp16")
    merged_fp16_dir = os.path.join(merged_images_root, "fp16")
    existing_fp16_dirs = _existing_image_dirs(args.output_dir, "fp16")
    if (not args.skip_fp16) and (not _all_images_exist_any(existing_fp16_dirs, prompts)):
        base_scheduler = DPMSolverMultistepScheduler.from_pretrained(args.base_model, subfolder="scheduler")
        fp16_pipeline = SanaPipeline.from_pretrained(
            args.base_model, scheduler=base_scheduler, variant="bf16", torch_dtype=torch.bfloat16
        ).to(device)
        fp16_pipeline.vae.to(torch.bfloat16)
        if fp16_pipeline.text_encoder is not None:
            fp16_pipeline.text_encoder.to(torch.bfloat16)
        fp16_xt_logger = None
        if args.save_xt_fp16:
            xt_latents_dir = os.path.join(args.xt_output_dir, "xt_latents")
            fp16_xt_logger = XTLatentsLogger(
                model_key="fp16",
                xt_output_dir=xt_latents_dir,
                rank=rank,
                world_size=world_size,
                prompt_seed=args.mjhq_prompt_sample_seed,
                noise_seed_start=args.noise_seed_start,
                num_inference_steps=args.num_inference_steps,
                shard_size=args.xt_shard_size,
                xt_dtype=_parse_xt_dtype(args.xt_dtype),
            )
        _generate_images(
            pipeline=fp16_pipeline,
            prompts=prompts,
            output_dir=fp16_dir,
            existing_dirs=existing_fp16_dirs,
            prefix="fp16",
            noise_seed_start=args.noise_seed_start,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            device=device,
            batch_size=args.batch_size,
            xt_logger=fp16_xt_logger,
        )
        if fp16_xt_logger is not None:
            shard_paths = fp16_xt_logger.finalize()
            for p in shard_paths:
                print(f"[Rank {rank}] ✓ Saved x_t shard: {p}")
        del fp16_pipeline
        torch.cuda.empty_cache()

    # 2) Quant baseline images (optional x_t logging)
    baseline_dir = os.path.join(images_root, "quant_baseline")
    qdrift_variants = [(bias, drift) for bias in args.bias_scales for drift in args.qdrift_scales]
    merged_baseline_dir = os.path.join(merged_images_root, "quant_baseline")
    existing_baseline_dirs = _existing_image_dirs(args.output_dir, "quant_baseline")
    def _qdrift_model_key(bias_scale: float, qdrift_scale: float) -> str:
        return f"quant_bias{bias_scale}_qdrift_scale{qdrift_scale}" + ("_scalar" if args.qdrift_scalar else "")

    need_quant_pipeline = any(
        not _all_images_exist_any(_existing_image_dirs(args.output_dir, _qdrift_model_key(b, d)), prompts)
        for b, d in qdrift_variants
    )
    if not args.skip_quant_baseline:
        need_quant_pipeline = need_quant_pipeline or (not _all_images_exist_any(existing_baseline_dirs, prompts))

    quant_pipeline = None
    if need_quant_pipeline:
        base_scheduler = DPMSolverMultistepScheduler.from_pretrained(args.base_model, subfolder="scheduler")
        quant_transformer = load_quant_transformer(
            base_model_id=args.base_model,
            quant_model_dir=args.quant_model_dir,
            device=device,
            torch_dtype=torch.bfloat16,
        )
        quant_pipeline = SanaPipeline.from_pretrained(
            args.base_model,
            transformer=quant_transformer,
            scheduler=base_scheduler,
            variant="bf16",
            torch_dtype=torch.bfloat16,
        ).to(device)
        quant_pipeline.vae.to(torch.bfloat16)
        if quant_pipeline.text_encoder is not None:
            quant_pipeline.text_encoder.to(torch.bfloat16)

    if (not args.skip_quant_baseline) and (quant_pipeline is not None) and (not _all_images_exist_any(existing_baseline_dirs, prompts)):
        baseline_xt_logger = None
        if args.save_xt_quant_baseline:
            xt_latents_dir = os.path.join(args.xt_output_dir, "xt_latents")
            baseline_xt_logger = XTLatentsLogger(
                model_key="quant_baseline",
                xt_output_dir=xt_latents_dir,
                rank=rank,
                world_size=world_size,
                prompt_seed=args.mjhq_prompt_sample_seed,
                noise_seed_start=args.noise_seed_start,
                num_inference_steps=args.num_inference_steps,
                shard_size=args.xt_shard_size,
                xt_dtype=_parse_xt_dtype(args.xt_dtype),
            )
        _generate_images(
            pipeline=quant_pipeline,
            prompts=prompts,
            output_dir=baseline_dir,
            existing_dirs=existing_baseline_dirs,
            prefix="quant_baseline",
            noise_seed_start=args.noise_seed_start,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            device=device,
            batch_size=args.batch_size,
            xt_logger=baseline_xt_logger,
        )
        if baseline_xt_logger is not None:
            shard_paths = baseline_xt_logger.finalize()
            for p in shard_paths:
                print(f"[Rank {rank}] ✓ Saved x_t shard: {p}")

    # 3) Q-Drift variants (optionally log x_t)
    for bias_scale, qdrift_scale in qdrift_variants:
        model_key = _qdrift_model_key(bias_scale, qdrift_scale)
        qdrift_dir = os.path.join(images_root, model_key)
        merged_qdrift_dir = os.path.join(merged_images_root, model_key)
        existing_qdrift_dirs = _existing_image_dirs(args.output_dir, model_key)
        if _all_images_exist_any(existing_qdrift_dirs, prompts):
            continue
        if quant_pipeline is None:
            raise RuntimeError("Internal error: expected quant_pipeline to be initialized for Q-Drift variants")
        variant_log_dir = os.path.join(rank_output_dir, f"logs_bias{bias_scale}_scale{qdrift_scale}")
        scheduler = DPMSolverQDriftScheduler.from_pretrained(
            args.base_model,
            subfolder="scheduler",
            mu_dict_path=args.mu_dict,
            cov_dict_path=args.cov_dict,
            log_dir=variant_log_dir,
            bias_scale=bias_scale,
            qdrift_scale=qdrift_scale,
            qdrift_scalar=args.qdrift_scalar,
        )
        quant_pipeline.scheduler = scheduler

        xt_logger = None
        if args.save_xt:
            xt_latents_dir = os.path.join(args.xt_output_dir, "xt_latents")
            xt_logger = XTLatentsLogger(
                model_key=model_key,
                xt_output_dir=xt_latents_dir,
                rank=rank,
                world_size=world_size,
                prompt_seed=args.mjhq_prompt_sample_seed,
                noise_seed_start=args.noise_seed_start,
                num_inference_steps=args.num_inference_steps,
                shard_size=args.xt_shard_size,
                xt_dtype=_parse_xt_dtype(args.xt_dtype),
            )

        _generate_images(
            pipeline=quant_pipeline,
            prompts=prompts,
            output_dir=qdrift_dir,
            existing_dirs=existing_qdrift_dirs,
            prefix=model_key,
            noise_seed_start=args.noise_seed_start,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            device=device,
            batch_size=args.batch_size,
            xt_logger=xt_logger,
        )

        if xt_logger is not None:
            shard_paths = xt_logger.finalize()
            for p in shard_paths:
                print(f"[Rank {rank}] ✓ Saved x_t shard: {p}")

        del scheduler

    if quant_pipeline is not None:
        del quant_pipeline
        torch.cuda.empty_cache()

    # Synchronize before merge on rank 0.
    torch.distributed.barrier()

    # We are done with distributed GPU work. Tear down the NCCL process group before the
    # (potentially long) CPU-side merge, to avoid watchdog timeouts on other ranks.
    torch.distributed.destroy_process_group()

    # ========== Merge results (rank 0 only) ==========
    if rank == 0:
        merged_images_root = os.path.join(args.output_dir, "images")
        _ensure_dir(merged_images_root)

        # Copy images from all ranks into <output_dir>/images/<model>/
        for r in range(world_size):
            rank_dir = os.path.join(args.output_dir, f"rank{r}", "images")
            if not os.path.exists(rank_dir):
                continue
            for model_name in os.listdir(rank_dir):
                src_model_dir = os.path.join(rank_dir, model_name)
                if not os.path.isdir(src_model_dir):
                    continue
                dst_model_dir = os.path.join(merged_images_root, model_name)
                _ensure_dir(dst_model_dir)
                for img_file in os.listdir(src_model_dir):
                    src = os.path.join(src_model_dir, img_file)
                    dst = os.path.join(dst_model_dir, img_file)
                    if os.path.isfile(src) and not os.path.exists(dst):
                        shutil.copy2(src, dst)

        # Merge qdrift logs for each variant.
        for bias_scale, qdrift_scale in qdrift_variants:
            merged_log_dir = os.path.join(args.output_dir, f"logs_bias{bias_scale}_scale{qdrift_scale}")
            _ensure_dir(merged_log_dir)
            merged_log_file = os.path.join(merged_log_dir, "qdrift_log.jsonl")

            with open(merged_log_file, "w", encoding="utf-8") as outf:
                for r in range(world_size):
                    rank_log_file = os.path.join(
                        args.output_dir,
                        f"rank{r}",
                        f"logs_bias{bias_scale}_scale{qdrift_scale}",
                        "qdrift_log.jsonl",
                    )
                    if os.path.exists(rank_log_file):
                        with open(rank_log_file, "r", encoding="utf-8") as inf:
                            outf.write(inf.read())

        # Write x_t indices (one per variant).
        if args.save_xt or args.save_xt_fp16 or args.save_xt_quant_baseline:
            xt_latents_dir = os.path.join(args.xt_output_dir, "xt_latents")
            _ensure_dir(xt_latents_dir)
            extra_keys = []
            if args.save_xt_fp16:
                extra_keys.append("fp16")
            if args.save_xt_quant_baseline:
                extra_keys.append("quant_baseline")
            for model_key in extra_keys:
                xt_index = _write_xt_index(xt_latents_dir, model_key, world_size)
                if xt_index is not None:
                    print(f"✓ Wrote x_t index: {xt_index}")
                if args.xt_merge_single:
                    merged_xt = _merge_xt_latents_single(xt_latents_dir, model_key, world_size)
                    if merged_xt is not None:
                        print(f"✓ Merged x_t latents: {merged_xt}")
            for bias_scale, qdrift_scale in qdrift_variants:
                model_key = _qdrift_model_key(bias_scale, qdrift_scale)
                xt_index = _write_xt_index(xt_latents_dir, model_key, world_size)
                if xt_index is not None:
                    print(f"✓ Wrote x_t index: {xt_index}")
                if args.xt_merge_single:
                    merged_xt = _merge_xt_latents_single(xt_latents_dir, model_key, world_size)
                    if merged_xt is not None:
                        print(f"✓ Merged x_t latents: {merged_xt}")
                        if args.xt_delete_shards_after_merge:
                            # Best-effort cleanup.
                            for name in list(os.listdir(xt_latents_dir)):
                                if name.startswith(f"xt_latents_{model_key}_rank") and name.endswith(".pth"):
                                    try:
                                        os.remove(os.path.join(xt_latents_dir, name))
                                    except Exception:
                                        pass

        # Clean up rank dirs to save space (best effort).
        for r in range(world_size):
            rank_path = os.path.join(args.output_dir, f"rank{r}")
            try:
                if os.path.exists(rank_path):
                    shutil.rmtree(rank_path)
            except Exception:
                pass

        print(f"✓ Sampling complete. Images saved under: {os.path.join(args.output_dir, 'images')}")
    return


if __name__ == "__main__":
    main()
