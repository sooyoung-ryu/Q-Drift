"""
Sampling-only evaluation script for Q-Drift correction on MJHQ-30K for FLUX.1-schnell.

This script generates and saves images for:
  - FP16 FLUX.1-schnell baseline
  - Quantized baseline (no correction)
  - Quantized + Q-Drift (bias/qdrift scale sweeps)

It intentionally does NOT compute metrics. Metrics are computed separately by:
  - evaluation/compute_metrics.py

Reproducibility design:
  - Prompt sampling is controlled by: --mjhq_prompt_sample_seed
  - Initial noise is controlled independently by: --noise_seed_start
    (per-prompt deterministic seed = noise_seed_start + global_idx)

Model specifics:
  - FLUX.1-schnell typically uses guidance_scale=0.0 and 4 inference steps.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional

import torch
from diffusers import FluxPipeline
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# Allow importing `schedulers/` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if _REPO_ROOT.is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from schedulers.flow_match_euler_qdrift import FlowMatchEulerQDriftScheduler  # noqa: E402

# Allow importing `experiments/svdquant/deepcompressor_loader.py`.
_QDRIFT_DIR = Path(__file__).resolve().parents[2]
if _QDRIFT_DIR.is_dir() and str(_QDRIFT_DIR) not in sys.path:
    sys.path.insert(0, str(_QDRIFT_DIR))
from deepcompressor_loader import load_quant_flux_transformer  # noqa: E402

# MJHQ-30K constants
MJHQ_REPO_ID = "playgroundai/MJHQ-30K"
MJHQ_META_FILENAME = "meta_data.json"

DEFAULT_BASE_MODEL = "black-forest-labs/FLUX.1-schnell"
DEFAULT_QUANT_MODEL_DIR = str((Path(__file__).resolve().parents[1] / "model" / "transformer_w3a4_g64").resolve())


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

    random.seed(mjhq_prompt_sample_seed)
    samples_per_category = num_samples // 10

    prompts: List[Dict[str, Any]] = []
    for category in sorted(prompts_by_category.keys()):
        sampled = random.sample(prompts_by_category[category], samples_per_category)
        prompts.extend(sampled)

    random.shuffle(prompts)
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


def _model_image_dirs(output_dir: str, model_key: str) -> List[str]:
    dirs = [os.path.join(output_dir, "images", model_key)]
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            if name.startswith("rank"):
                dirs.append(os.path.join(output_dir, name, "images", model_key))
    return dirs


def _all_images_exist_in_any_dir(dirs: List[str], prompts: List[Dict[str, Any]]) -> bool:
    for row in prompts:
        category = row["category"]
        global_idx = int(row["global_idx"])
        filename = f"{category}_{global_idx:05d}.png"
        if not any(os.path.exists(os.path.join(image_dir, filename)) for image_dir in dirs):
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
        self._xt_by_t: DefaultDict[int, List[torch.Tensor]] = defaultdict(list)

    def begin_sample(self, global_idx: int) -> None:
        self._global_indices_in_shard.append(int(global_idx))

    def record_step(self, timestep: int, latents: torch.Tensor) -> None:
        latents_cpu = latents.detach().to("cpu")
        if latents_cpu.dtype != self.xt_dtype:
            latents_cpu = latents_cpu.to(dtype=self.xt_dtype)
        self._xt_by_t[int(timestep)].append(latents_cpu[0].contiguous())

    def _flush_shard(self) -> Optional[str]:
        if len(self._global_indices_in_shard) == 0:
            return None

        _ensure_dir(self.xt_output_dir)
        shard_path = os.path.join(
            self.xt_output_dir, f"xt_latents_{self.model_key}_rank{self.rank}_shard{self._shard_id:05d}.pth"
        )

        timesteps = sorted(self._xt_by_t.keys())
        xt_dict: Dict[int, torch.Tensor] = {}
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
            "xt": xt_dict,
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


@torch.inference_mode()
def _generate_images(
    *,
    pipeline: FluxPipeline,
    prompts: List[Dict[str, Any]],
    output_dir: str,
    existing_dirs: Optional[List[str]],
    prefix: str,
    noise_seed_start: int,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    device: torch.device,
    xt_logger: Optional[XTLatentsLogger],
) -> None:
    _ensure_dir(output_dir)
    pipeline.set_progress_bar_config(disable=True)

    for row in tqdm(prompts, desc=f"Generating {prefix}"):
        prompt = row["prompt"]
        category = row["category"]
        global_idx = int(row["global_idx"])
        filename = f"{category}_{global_idx:05d}.png"
        out_path = os.path.join(output_dir, filename)
        if os.path.exists(out_path):
            continue
        if existing_dirs is not None and any(
            os.path.exists(os.path.join(existing_dir, filename)) for existing_dir in existing_dirs
        ):
            continue

        generator = torch.Generator(device=device).manual_seed(int(noise_seed_start) + global_idx)

        callback = None
        callback_inputs: Optional[List[str]] = None
        if xt_logger is not None:
            xt_logger.begin_sample(global_idx)

            def _on_step_end(_pipe, _step_index: int, timestep, callback_kwargs):
                latents = callback_kwargs.get("latents", None)
                if latents is not None:
                    t_val = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Sampling-only evaluation for FLUX.1-schnell Q-Drift on MJHQ-30K")
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
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--meta_path", type=str, default=None, help="Optional path to MJHQ meta_data.json")
    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="HF repo id or local directory for the FLUX.1-schnell diffusers pipeline.",
    )
    parser.add_argument(
        "--quant_model_dir",
        type=str,
        required=True,
        help=(
            "DeepCompressor quantized Transformer checkpoint directory "
            "(expects `model.pt` or `transformer/`)."
        ),
    )
    parser.add_argument("--qdrift_scales", type=float, nargs="+", default=[1.0])
    parser.add_argument(
        "--skip_fp16",
        action="store_true",
        help="Skip FP16 generation.",
    )
    parser.add_argument(
        "--skip_quant_baseline",
        action="store_true",
        help="Skip quantized baseline generation.",
    )
    parser.add_argument(
        "--qdrift_only",
        action="store_true",
        help="Generate only Q-Drift variants; equivalent to --skip_fp16 --skip_quant_baseline.",
    )
    parser.add_argument(
        "--qdrift_scalar",
        action="store_true",
        help="Ablation: one scalar V per step (channel-averaged) instead of per-channel V.",
    )
    parser.add_argument("--bias_scales", type=float, nargs="+", default=None, help="Scale(s) applied to E[error|output]")
    parser.add_argument(
        "--save_xt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save full per-step x_t latents for Q-Drift variants to --xt_output_dir/xt_latents/",
    )
    parser.add_argument(
        "--xt_output_dir",
        type=str,
        default=None,
        help="Directory to save x_t latents (.pth). Default: --output_dir",
    )
    parser.add_argument("--xt_shard_size", type=int, default=16, help="Samples per x_t shard file")
    parser.add_argument("--xt_dtype", type=str, default="bfloat16", help="x_t dtype: bfloat16|float16|float32")
    parser.add_argument(
        "--xt_merge_single",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="On rank 0, merge x_t shards into a single large `.pth` per variant (can require large CPU RAM)",
    )
    parser.add_argument(
        "--xt_delete_shards_after_merge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete per-rank shard files after successfully writing the merged `.pth`",
    )
    args = parser.parse_args()
    if args.qdrift_only:
        args.skip_fp16 = True
        args.skip_quant_baseline = True

    if args.xt_output_dir is None:
        args.xt_output_dir = args.output_dir
    if args.bias_scales is None:
        args.bias_scales = [1.0]

    torch.distributed.init_process_group(backend="nccl", timeout=timedelta(hours=12))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if rank == 0:
        print(f"[Multi-GPU] world_size={world_size} output_dir={args.output_dir}")

    mjhq_meta_path = args.meta_path
    if rank == 0:
        try:
            if mjhq_meta_path is None or not os.path.exists(mjhq_meta_path):
                mjhq_meta_path = download_mjhq_metadata()
            _ = FluxPipeline.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
        except Exception as e:
            print(f"⚠️  Rank 0 prefetch failed: {e}")

    torch.distributed.barrier()
    if mjhq_meta_path is None or not os.path.exists(mjhq_meta_path):
        mjhq_meta_path = download_mjhq_metadata()

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

    rank_output_dir = os.path.join(args.output_dir, f"rank{rank}")
    images_root = os.path.join(rank_output_dir, "images")
    _ensure_dir(images_root)

    if rank == 0:
        _ensure_dir(args.output_dir)
        eval_results_path = os.path.join(args.output_dir, "evaluation_results.json")
        payload = {
            "benchmark": "MJHQ-30K",
            "model": "FLUX.1-schnell",
            "num_samples": int(args.num_samples),
            "mjhq_prompt_sample_seed": int(args.mjhq_prompt_sample_seed),
            "noise_seed_start": int(args.noise_seed_start),
            "num_inference_steps": int(args.num_inference_steps),
            "guidance_scale": float(args.guidance_scale),
            "height": int(args.height),
            "width": int(args.width),
            "qdrift_scales": list(args.qdrift_scales),
            "qdrift_scalar": bool(args.qdrift_scalar),
            "bias_scales": list(args.bias_scales),
            "selected_global_indices": selected_indices,
            "prompts": all_prompts,
        }
        with open(eval_results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved prompts + config: {eval_results_path}")

    torch.distributed.barrier()

    fp16_dir = os.path.join(images_root, "fp16")
    fp16_existing_dirs = _model_image_dirs(args.output_dir, "fp16")
    if args.skip_fp16 or _all_images_exist_in_any_dir(fp16_existing_dirs, prompts):
        if rank == 0:
            print("✓ Skipping fp16 generation (already exists)")
    else:
        fp16_pipeline = FluxPipeline.from_pretrained(args.base_model, torch_dtype=torch.bfloat16).to(device)
        _generate_images(
            pipeline=fp16_pipeline,
            prompts=prompts,
            output_dir=fp16_dir,
            existing_dirs=fp16_existing_dirs,
            prefix="fp16",
            noise_seed_start=args.noise_seed_start,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            device=device,
            xt_logger=None,
        )
        del fp16_pipeline
        torch.cuda.empty_cache()

    baseline_dir = os.path.join(images_root, "quant_baseline")
    baseline_existing_dirs = _model_image_dirs(args.output_dir, "quant_baseline")
    if args.skip_quant_baseline or _all_images_exist_in_any_dir(baseline_existing_dirs, prompts):
        if rank == 0:
            print("✓ Skipping quant_baseline generation (already exists or disabled)")
    else:
        quant_transformer = load_quant_flux_transformer(
            base_model_id=args.base_model,
            quant_model_dir=args.quant_model_dir,
            device=device,
            torch_dtype=torch.bfloat16,
        )
        quant_baseline_pipeline = FluxPipeline.from_pretrained(
            args.base_model, transformer=quant_transformer, torch_dtype=torch.bfloat16
        ).to(device)
        _generate_images(
            pipeline=quant_baseline_pipeline,
            prompts=prompts,
            output_dir=baseline_dir,
            existing_dirs=baseline_existing_dirs,
            prefix="quant_baseline",
            noise_seed_start=args.noise_seed_start,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            device=device,
            xt_logger=None,
        )
        del quant_baseline_pipeline, quant_transformer
        torch.cuda.empty_cache()

    qdrift_variants = [(bias, drift) for bias in args.bias_scales for drift in args.qdrift_scales]
    for bias_scale, qdrift_scale in qdrift_variants:
        model_key = f"quant_bias{bias_scale}_qdrift_scale{qdrift_scale}" + ("_scalar" if args.qdrift_scalar else "")
        qdrift_dir = os.path.join(images_root, model_key)
        qdrift_existing_dirs = _model_image_dirs(args.output_dir, model_key)
        if _all_images_exist_in_any_dir(qdrift_existing_dirs, prompts):
            continue

        quant_transformer = load_quant_flux_transformer(
            base_model_id=args.base_model,
            quant_model_dir=args.quant_model_dir,
            device=device,
            torch_dtype=torch.bfloat16,
        )

        variant_log_dir = os.path.join(rank_output_dir, f"logs_bias{bias_scale}_scale{qdrift_scale}")
        scheduler = FlowMatchEulerQDriftScheduler.from_pretrained(
            args.base_model,
            subfolder="scheduler",
            mu_dict_path=args.mu_dict,
            cov_dict_path=args.cov_dict,
            log_dir=variant_log_dir,
            bias_scale=bias_scale,
            qdrift_scale=qdrift_scale,
            qdrift_scalar=args.qdrift_scalar,
        )
        qdrift_pipeline = FluxPipeline.from_pretrained(
            args.base_model,
            transformer=quant_transformer,
            scheduler=scheduler,
            torch_dtype=torch.bfloat16,
        ).to(device)

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
            pipeline=qdrift_pipeline,
            prompts=prompts,
            output_dir=qdrift_dir,
            existing_dirs=qdrift_existing_dirs,
            prefix=model_key,
            noise_seed_start=args.noise_seed_start,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            device=device,
            xt_logger=xt_logger,
        )

        if xt_logger is not None:
            shard_paths = xt_logger.finalize()
            for p in shard_paths:
                print(f"[Rank {rank}] ✓ Saved x_t shard: {p}")

        del qdrift_pipeline, scheduler, quant_transformer
        torch.cuda.empty_cache()

    torch.distributed.barrier()

    if rank == 0:
        merged_images_root = os.path.join(args.output_dir, "images")
        _ensure_dir(merged_images_root)

        rank_names = sorted(
            name
            for name in os.listdir(args.output_dir)
            if name.startswith("rank") and os.path.isdir(os.path.join(args.output_dir, name, "images"))
        )
        for rank_name in rank_names:
            rank_dir = os.path.join(args.output_dir, rank_name, "images")
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

        for bias_scale, qdrift_scale in qdrift_variants:
            merged_log_dir = os.path.join(args.output_dir, f"logs_bias{bias_scale}_scale{qdrift_scale}")
            _ensure_dir(merged_log_dir)
            merged_log_file = os.path.join(merged_log_dir, "qdrift_log.jsonl")

            with open(merged_log_file, "w", encoding="utf-8") as outf:
                for rank_name in rank_names:
                    rank_log_file = os.path.join(
                        args.output_dir,
                        rank_name,
                        f"logs_bias{bias_scale}_scale{qdrift_scale}",
                        "qdrift_log.jsonl",
                    )
                    if os.path.exists(rank_log_file):
                        with open(rank_log_file, "r", encoding="utf-8") as inf:
                            shutil.copyfileobj(inf, outf)

        if args.save_xt:
            xt_latents_dir = os.path.join(args.xt_output_dir, "xt_latents")
            for bias_scale, qdrift_scale in qdrift_variants:
                model_key = f"quant_bias{bias_scale}_qdrift_scale{qdrift_scale}" + ("_scalar" if args.qdrift_scalar else "")
                index = _write_xt_index(xt_latents_dir, model_key, world_size)
                if index is not None:
                    print(f"✓ Wrote x_t index: {index}")

        print(f"✓ Merged images under: {merged_images_root}")


if __name__ == "__main__":
    main()
