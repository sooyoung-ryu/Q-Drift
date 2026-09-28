"""
Generate MJHQ-30K evaluation images for SDXL-Turbo SVDQuant W3A4: FP16, quantized baseline, and Q-Drift.

Prompts are a stratified sample of MJHQ-30K (`--mjhq_prompt_sample_seed`). SDXL-Turbo uses
guidance_scale=0 and 4 inference steps. Metrics are computed separately by
`evaluation/compute_metrics.py` (see run_paper.sh).
"""

import argparse
import os
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from diffusers import StableDiffusionXLPipeline
import random



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


def _rank_local_seed_offset(global_idx: int, num_prompts: int, seed_world_size: int) -> int:
    if seed_world_size <= 0:
        raise ValueError("seed_world_size must be positive")
    samples_per_rank = num_prompts // seed_world_size
    if samples_per_rank <= 0:
        raise ValueError("seed_world_size exceeds the number of prompts")
    rank = min(int(global_idx) // samples_per_rank, seed_world_size - 1)
    local_idx = int(global_idx) - rank * samples_per_rank
    return rank * 10000 + local_idx


def _select_prompts_by_global_indices(all_prompts: List[Dict], selected_indices: List[int]) -> List[Dict]:
    selected_set = set(selected_indices)
    selected_prompts = [dict(row) for row in all_prompts if int(row["global_idx"]) in selected_set]
    missing_selected = sorted(selected_set.difference(int(row["global_idx"]) for row in selected_prompts))
    if missing_selected:
        raise ValueError(f"--global_indices contains indices outside the prompt set: {missing_selected[:20]}")
    return selected_prompts


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_xt_dtype(name: str) -> torch.dtype:
    name = str(name).strip().lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("fp32", "float32"):
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
        self.model_key = str(model_key)
        self.xt_output_dir = str(xt_output_dir)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.prompt_seed = int(prompt_seed)
        self.noise_seed_start = int(noise_seed_start)
        self.num_inference_steps = int(num_inference_steps)
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
        xt_dict: Dict[int, torch.Tensor] = {t: torch.stack(self._xt_by_t[t], dim=0) for t in timesteps}
        payload = {
            "format": "xt_latents_sharded_v1",
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
    payload = {"format": "xt_latents_index_v1", "model_key": model_key, "world_size": int(world_size), "shards": shards}
    torch.save(payload, index_path)
    return index_path


def _merge_xt_latents_single(xt_output_dir: str, model_key: str, world_size: int) -> Optional[str]:
    """
    Merge sharded x_t latents into a single `.pth` file per model_key.

    Warning: this can be very large for big runs. Use `--xt_merge_single` intentionally.
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

    xt_chunks_by_t: DefaultDict[int, List[torch.Tensor]] = defaultdict(list)
    global_indices: List[int] = []

    prompt_seed = None
    noise_seed_start = None
    num_inference_steps = None
    xt_dtype = None

    for p in shard_paths:
        data = torch.load(p, map_location="cpu")
        prompt_seed = data.get("prompt_seed", prompt_seed)
        noise_seed_start = data.get("noise_seed_start", noise_seed_start)
        num_inference_steps = data.get("num_inference_steps", num_inference_steps)
        xt_dtype = data.get("xt_dtype", xt_dtype)
        global_indices.extend([int(x) for x in data.get("global_indices", [])])

        xt_by_t = data.get("xt", {})
        for t in data.get("timesteps", []):
            xt_chunks_by_t[int(t)].append(xt_by_t[int(t)])

    timesteps = sorted(xt_chunks_by_t.keys())
    xt_merged: Dict[int, torch.Tensor] = {}
    for t in timesteps:
        xt_merged[t] = torch.cat(xt_chunks_by_t[t], dim=0)

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
from collections import defaultdict
from huggingface_hub import hf_hub_download

DEFAULT_BASE_MODEL = "stabilityai/sdxl-turbo"
DEFAULT_QUANT_MODEL_DIR = str((Path(__file__).resolve().parents[1] / "model" / "unet_w3a4_g64").resolve())

# Allow importing the project-local Q-Drift schedulers at:
# `schedulers/*` at the repo root.
_PROJECT_ROOT = None
for _p in Path(__file__).resolve().parents:
    if (_p / "schedulers").is_dir():
        _PROJECT_ROOT = _p
        break
if _PROJECT_ROOT is not None and str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from schedulers.euler_ancestral_qdrift import EulerAncestralQDriftScheduler

# Allow importing `experiments/svdquant/deepcompressor_loader.py`.
_QDRIFT_DIR = Path(__file__).resolve().parents[2]
if str(_QDRIFT_DIR) not in sys.path:
    sys.path.insert(0, str(_QDRIFT_DIR))
from deepcompressor_loader import load_quant_unet



# MJHQ-30K constants
MJHQ_REPO_ID = "playgroundai/MJHQ-30K"
MJHQ_META_FILENAME = "meta_data.json"


def download_mjhq_metadata() -> str:
    """Download MJHQ-30K meta_data.json from HF Hub."""
    return hf_hub_download(
        repo_id=MJHQ_REPO_ID,
        filename=MJHQ_META_FILENAME,
        repo_type="dataset",
    )


def load_mjhq_metadata(meta_path: str) -> Dict[str, Dict]:
    """Load MJHQ-30K meta_data.json."""
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def configure_torch_cache(torch_cache_dir: Optional[str] = None):
    """Configure torch hub cache directory."""
    if torch_cache_dir:
        cache_dir = Path(torch_cache_dir).expanduser()
    else:
        cache_dir = Path.home() / ".cache" / "torch"
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(cache_dir)
    torch.hub.set_dir(str(cache_dir))


def load_prompts(meta_path: str = None, num_samples: int = 5000, mjhq_prompt_sample_seed: int = 42) -> List[Dict]:
    """
    Load prompts from MJHQ-30K using stratified sampling.
    MJHQ-30K: 10 categories, samples 500 per category for 5000 total.
    """
    # Download or load meta_data.json
    if not meta_path or not os.path.exists(meta_path):
        meta_path = download_mjhq_metadata()
    
    metadata = load_mjhq_metadata(meta_path)
    
    # Group prompts by category
    prompts_by_category = defaultdict(list)
    for image_id, info in metadata.items():
        if isinstance(info, dict) and info.get('prompt', '').strip():
            prompts_by_category[info['category']].append({
                'prompt': info['prompt'].strip(),
                'id': image_id,
                'category': info['category']
            })
    
    # Stratified sampling: 500 per category (10 categories)
    random.seed(mjhq_prompt_sample_seed)
    samples_per_category = num_samples // 10
    
    formatted_prompts = []
    for category in sorted(prompts_by_category.keys()):
        sampled = random.sample(prompts_by_category[category], samples_per_category)
        formatted_prompts.extend(sampled)
    
    random.shuffle(formatted_prompts)
    
    # Add global index to each prompt for consistent file naming across ranks
    for idx, prompt_dict in enumerate(formatted_prompts):
        prompt_dict['global_idx'] = idx
    
    print(f"Loaded {len(formatted_prompts)} prompts from MJHQ-30K ({samples_per_category} per category)")
    
    return formatted_prompts


def load_existing_images(
    output_dir: str,
    prompts: List[Dict],
) -> Optional[List[Image.Image]]:
    """
    Load existing images from directory if they all exist.
    
    Returns:
        List of images if all exist, None otherwise
    """
    if not os.path.exists(output_dir):
        return None
    
    images = []
    for prompt_dict in prompts:
        # Use global_idx for consistent naming across ranks
        global_idx = prompt_dict.get('global_idx', 0)
        filename = f"{prompt_dict['category']}_{global_idx:05d}.png"
        filepath = os.path.join(output_dir, filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            # Load image and immediately close file handle to avoid "too many open files"
            img = Image.open(filepath)
            img.load()  # Force load into memory and close file handle
            images.append(img)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    return images


def generate_images(
    pipeline,
    prompts: List[Dict],
    output_dir: str,
    prefix: str,
    mjhq_prompt_sample_seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    device: str,
    xt_logger: Optional[XTLatentsLogger] = None,
    seed_by_global_idx: bool = False,
):
    """
    Generate images for all prompts using the given pipeline, or load from disk if they exist.
    
    Returns:
        List of generated PIL Images
    """
    # Try to load existing images first
    existing_images = load_existing_images(output_dir, prompts)
    if existing_images is not None:
        print(f"✓ Found existing images, loading from {output_dir}")
        return existing_images
    
    # Generate new images
    os.makedirs(output_dir, exist_ok=True)
    images = []
    
    for i, prompt_dict in enumerate(tqdm(prompts, desc=f"Generating {prefix}")):
        prompt = prompt_dict['prompt']
        category = prompt_dict['category']
        global_idx = prompt_dict['global_idx']
        
        if "_seed_offset" in prompt_dict:
            seed_offset = int(prompt_dict["_seed_offset"])
        else:
            seed_offset = int(global_idx) if seed_by_global_idx else i
        generator = torch.Generator(device=device).manual_seed(mjhq_prompt_sample_seed + seed_offset)
        if xt_logger is not None:
            xt_logger.begin_sample(global_idx)

        callback_kwargs = {}
        if xt_logger is not None:
            callback_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

            def _cb(_pipe, _step_index, timestep, kwargs):
                latents = kwargs.get("latents")
                if latents is not None:
                    t_val = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
                    xt_logger.record_step(t_val, latents)
                return kwargs

            callback_kwargs["callback_on_step_end"] = _cb

        image = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **callback_kwargs,
        ).images[0]
        
        images.append(image)
        
        # Save image with category_globalidx format
        filename = f"{category}_{global_idx:05d}.png"
        image.save(os.path.join(output_dir, filename))
        if xt_logger is not None:
            xt_logger.end_sample()
    
    if xt_logger is not None:
        shard_paths = xt_logger.finalize()
        for p in shard_paths:
            print(f"✓ Saved x_t shard: {p}")

    return images


def evaluate_benchmark(
    benchmark_name: str,
    prompts: List[Dict],
    mu_dict_path: str,
    cov_dict_path: str,
    output_dir: str,
    mjhq_prompt_sample_seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    device: str,
    base_model_id: str = DEFAULT_BASE_MODEL,
    quant_model_dir: str = DEFAULT_QUANT_MODEL_DIR,
    evaluate_fp16: bool = True,
    evaluate_baseline: bool = True,
    bias_scales: List[float] = [1.0],
    qdrift_scales: List[float] = [1.0],
    qdrift_scalar: bool = False,
    local_rank: int = 0,
    world_size: int = 1,
    merged_output_dir: str = None,  # For checking already-merged images
    save_xt: bool = False,
    save_xt_fp16: bool = False,
    save_xt_quant_baseline: bool = False,
    xt_output_dir: Optional[str] = None,
    xt_shard_size: int = 32,
    xt_dtype: str = "bfloat16",
    seed_by_global_idx: bool = False,
):
    """
    Evaluate models on a benchmark dataset.
    
    Args:
        benchmark_name: Name of the benchmark (e.g., "MJHQ-30K")
        prompts: List of prompt dictionaries
        mu_dict_path: Path to mu_dict.npy
        cov_dict_path: Path to cov_dict.npy
        output_dir: Directory to save results
        mjhq_prompt_sample_seed: Random seed for prompt sampling and image generation
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale
        device: Device to use
        base_model_id: HF repo id or local directory for SDXL-Turbo diffusers pipeline
        quant_model_dir: Path to W3A4 quantized UNet checkpoint directory (DeepCompressor)
        evaluate_fp16: Whether to evaluate FP16 baseline
        evaluate_baseline: Whether to evaluate quantized baseline
    """
    print("\n" + "="*70)
    print(f"Evaluating on {benchmark_name}")
    print("="*70)
    print(f"Number of prompts: {len(prompts)}")
    print(f"Output directory: {output_dir}")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    if xt_output_dir is None:
        xt_output_dir = output_dir
    
    results = {
        'benchmark': benchmark_name,
        'num_samples': len(prompts),
        'prompts': prompts,
        'metrics': {}
    }
    
    # Dictionary to store all generated images
    all_images = {}
    
    # ========== Step 1: Generate/Load All Images ==========
    print("\n" + "="*70)
    print("STEP 1: Image Generation/Loading")
    print("="*70)
    
    # ========== FP16 Reference ==========
    fp16_images = None
    if evaluate_fp16:
        print("\n[FP16] Loading/Generating FP16 reference images...")
        
        fp16_dir = os.path.join(output_dir, "images", "fp16")
        
        # Only rank 0 checks if images exist (to avoid "too many open files" on all ranks)
        images_exist = False
        if local_rank == 0 and merged_output_dir:
            merged_fp16_dir = os.path.join(merged_output_dir, "images", "fp16")
            test_images = load_existing_images(merged_fp16_dir, prompts)
            if test_images is not None:
                images_exist = True
                print(f"✓ Found existing FP16 images in merged directory: {merged_fp16_dir}")
        
        # Broadcast result to all ranks (if distributed)
        if world_size > 1:
            import torch.distributed as dist
            images_exist_tensor = torch.tensor([1 if images_exist else 0], dtype=torch.int, device=device)
            dist.broadcast(images_exist_tensor, src=0)
            images_exist = bool(images_exist_tensor.item())
        
        # Skip generation if images exist
        if images_exist:
            print(f"✓ Skipping FP16 generation (images already exist)")
            fp16_images = None  # Will be loaded later by rank 0 for metrics
        elif fp16_images is None:
            # Generate new images
            try:
                fp16_pipeline = StableDiffusionXLPipeline.from_pretrained(
                    base_model_id,
                    torch_dtype=torch.bfloat16,
                    variant="fp16",
                ).to(device)
            except Exception:
                fp16_pipeline = StableDiffusionXLPipeline.from_pretrained(
                    base_model_id,
                    torch_dtype=torch.bfloat16,
                ).to(device)
            
            os.makedirs(fp16_dir, exist_ok=True)
            fp16_xt_logger = None
            if save_xt_fp16:
                xt_latents_dir = os.path.join(str(xt_output_dir), "xt_latents")
                fp16_xt_logger = XTLatentsLogger(
                    model_key="fp16",
                    xt_output_dir=xt_latents_dir,
                    rank=int(local_rank),
                    world_size=int(world_size),
                    prompt_seed=int(mjhq_prompt_sample_seed),
                    noise_seed_start=int(mjhq_prompt_sample_seed),
                    num_inference_steps=int(num_inference_steps),
                    shard_size=int(xt_shard_size),
                    xt_dtype=_parse_xt_dtype(xt_dtype),
                )
            fp16_images = generate_images(
                fp16_pipeline,
                prompts,
                fp16_dir,
                "fp16",
                mjhq_prompt_sample_seed,
                num_inference_steps,
                guidance_scale,
                device,
                xt_logger=fp16_xt_logger,
                seed_by_global_idx=seed_by_global_idx,
            )
            
            print(f"✓ Generated {len(fp16_images)} FP16 reference images")
            
            del fp16_pipeline
            torch.cuda.empty_cache()
        
        all_images['fp16'] = fp16_images
    
    # ========== Quantized Baseline ==========
    quant_baseline_images = None
    if evaluate_baseline:
        print("\n[Baseline] Loading/Generating quantized baseline images...")
        
        quant_baseline_dir = os.path.join(output_dir, "images", "quant_baseline")
        
        # Only rank 0 checks if images exist
        images_exist = False
        if local_rank == 0 and merged_output_dir:
            merged_baseline_dir = os.path.join(merged_output_dir, "images", "quant_baseline")
            test_images = load_existing_images(merged_baseline_dir, prompts)
            if test_images is not None:
                images_exist = True
                print(f"✓ Found existing baseline images in merged directory: {merged_baseline_dir}")
        
        # Broadcast result to all ranks
        if world_size > 1:
            import torch.distributed as dist
            images_exist_tensor = torch.tensor([1 if images_exist else 0], dtype=torch.int, device=device)
            dist.broadcast(images_exist_tensor, src=0)
            images_exist = bool(images_exist_tensor.item())
        
        # Skip generation if images exist
        if images_exist:
            print(f"✓ Skipping baseline generation (images already exist)")
            quant_baseline_images = None
        elif quant_baseline_images is None:
            # Generate new images
            quant_unet = load_quant_unet(
                base_model_id=base_model_id,
                quant_model_dir=quant_model_dir,
                device=device,
                torch_dtype=torch.bfloat16,
            )
            
            try:
                quant_baseline_pipeline = StableDiffusionXLPipeline.from_pretrained(
                    base_model_id,
                    unet=quant_unet,
                    torch_dtype=torch.bfloat16,
                    variant="fp16",
                ).to(device)
            except Exception:
                quant_baseline_pipeline = StableDiffusionXLPipeline.from_pretrained(
                    base_model_id,
                    unet=quant_unet,
                    torch_dtype=torch.bfloat16,
                ).to(device)
            
            os.makedirs(quant_baseline_dir, exist_ok=True)
            baseline_xt_logger = None
            if save_xt_quant_baseline:
                xt_latents_dir = os.path.join(str(xt_output_dir), "xt_latents")
                baseline_xt_logger = XTLatentsLogger(
                    model_key="quant_baseline",
                    xt_output_dir=xt_latents_dir,
                    rank=int(local_rank),
                    world_size=int(world_size),
                    prompt_seed=int(mjhq_prompt_sample_seed),
                    noise_seed_start=int(mjhq_prompt_sample_seed),
                    num_inference_steps=int(num_inference_steps),
                    shard_size=int(xt_shard_size),
                    xt_dtype=_parse_xt_dtype(xt_dtype),
                )
            quant_baseline_images = generate_images(
                quant_baseline_pipeline,
                prompts,
                quant_baseline_dir,
                "baseline",
                mjhq_prompt_sample_seed,
                num_inference_steps,
                guidance_scale,
                device,
                xt_logger=baseline_xt_logger,
                seed_by_global_idx=seed_by_global_idx,
            )
            
            print(f"✓ Generated {len(quant_baseline_images)} quantized baseline images")
            
            del quant_baseline_pipeline, quant_unet
            torch.cuda.empty_cache()
        
        all_images['quant_baseline'] = quant_baseline_images
    
    # ========== Quantized with Bias/Q-Drift (multiple scales) ==========
    for bias_idx, bias_scale in enumerate(bias_scales):
        for drift_idx, qdrift_scale in enumerate(qdrift_scales):
            combo_idx = bias_idx * len(qdrift_scales) + drift_idx + 1
            combo_total = len(bias_scales) * len(qdrift_scales)
            print(
                f"\n[Bias={bias_scale}, Drift={qdrift_scale}] "
                f"({combo_idx}/{combo_total}) Loading/Generating corrected images..."
            )
            
            model_key = f"quant_bias_{bias_scale}_drift{qdrift_scale}" + ("_scalar" if qdrift_scalar else "")
            quant_qdrift_dir = os.path.join(output_dir, "images", model_key)
            
            # Only rank 0 checks if images exist
            images_exist = False
            if local_rank == 0 and merged_output_dir:
                merged_qdrift_dir = os.path.join(merged_output_dir, "images", model_key)
                test_images = load_existing_images(merged_qdrift_dir, prompts)
                if test_images is not None:
                    images_exist = True
                    print(
                        f"✓ Found existing images (bias={bias_scale}, drift={qdrift_scale}) "
                        f"in merged directory: {merged_qdrift_dir}"
                    )
            
            # Broadcast result to all ranks
            if world_size > 1:
                import torch.distributed as dist
                images_exist_tensor = torch.tensor([1 if images_exist else 0], dtype=torch.int, device=device)
                dist.broadcast(images_exist_tensor, src=0)
                images_exist = bool(images_exist_tensor.item())
            
            # Skip generation if images exist
            quant_qdrift_images = None
            if images_exist:
                print("✓ Skipping generation (images already exist)")
            else:
                # Generate new images
                quant_unet = load_quant_unet(
                    base_model_id=base_model_id,
                    quant_model_dir=quant_model_dir,
                    device=device,
                    torch_dtype=torch.bfloat16,
                )
                
                # Create scale-specific log directory
                scale_log_dir = os.path.join(output_dir, f"logs_bias_{bias_scale}_drift{qdrift_scale}")
                
                scheduler = EulerAncestralQDriftScheduler.from_pretrained(
                    base_model_id,
                    subfolder="scheduler",
                    mu_dict_path=mu_dict_path,
                    cov_dict_path=cov_dict_path,
                    log_dir=scale_log_dir,
                    bias_scale=bias_scale,
                    qdrift_scale=qdrift_scale,
                    qdrift_scalar=qdrift_scalar,
                )
                
                try:
                    quant_qdrift_pipeline = StableDiffusionXLPipeline.from_pretrained(
                        base_model_id,
                        unet=quant_unet,
                        scheduler=scheduler,
                        torch_dtype=torch.bfloat16,
                        variant="fp16",
                    ).to(device)
                except Exception:
                    quant_qdrift_pipeline = StableDiffusionXLPipeline.from_pretrained(
                        base_model_id,
                        unet=quant_unet,
                        scheduler=scheduler,
                        torch_dtype=torch.bfloat16,
                    ).to(device)
                
                os.makedirs(quant_qdrift_dir, exist_ok=True)
                qdrift_xt_logger = None
                if save_xt:
                    xt_latents_dir = os.path.join(str(xt_output_dir), "xt_latents")
                    qdrift_xt_logger = XTLatentsLogger(
                        model_key=model_key,
                        xt_output_dir=xt_latents_dir,
                        rank=int(local_rank),
                        world_size=int(world_size),
                        prompt_seed=int(mjhq_prompt_sample_seed),
                        noise_seed_start=int(mjhq_prompt_sample_seed),
                        num_inference_steps=int(num_inference_steps),
                        shard_size=int(xt_shard_size),
                        xt_dtype=_parse_xt_dtype(xt_dtype),
                    )
                quant_qdrift_images = generate_images(
                    quant_qdrift_pipeline,
                    prompts,
                    quant_qdrift_dir,
                    f"bias_{bias_scale}_drift{qdrift_scale}",
                    mjhq_prompt_sample_seed,
                    num_inference_steps,
                    guidance_scale,
                    device,
                    xt_logger=qdrift_xt_logger,
                    seed_by_global_idx=seed_by_global_idx,
                )
                
                print(f"✓ Generated {len(quant_qdrift_images)} images (bias={bias_scale}, drift={qdrift_scale})")
                
                del quant_qdrift_pipeline, quant_unet, scheduler
                torch.cuda.empty_cache()
            
            all_images[model_key] = quant_qdrift_images
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark evaluation for SDXL-Turbo Q-Drift correction on MJHQ-30K")
    parser.add_argument(
        "--mu_dict",
        type=str,
        required=True,
        help="Path to mu_dict.npy"
    )
    parser.add_argument(
        "--cov_dict",
        type=str,
        required=True,
        help="Path to cov_dict.npy"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of samples to evaluate from MJHQ-30K"
    )
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
        help="Random seed for MJHQ prompt sampling and image generation"
    )
    parser.add_argument(
        "--global_index_seed_mode",
        choices=["global", "rank_local"],
        default="global",
        help=(
            "Seed mode for deterministic initial noise. 'global' uses seed + global_idx for "
            "--global_indices smoke subsets; 'rank_local' reproduces the full-run "
            "SDXL-Turbo seeds from seed + rank*10000 + local_idx and can be used for full runs."
        ),
    )
    parser.add_argument(
        "--seed_world_size",
        type=int,
        default=None,
        help="World size of the full run whose seed layout is reproduced by --global_index_seed_mode rank_local.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=4,
        help="Number of denoising steps (default: 4 for SDXL-Turbo)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale (default: 0.0 for SDXL-Turbo)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="HF repo id or local directory for SDXL-Turbo diffusers pipeline.",
    )
    parser.add_argument(
        "--quant_model_dir",
        type=str,
        default=DEFAULT_QUANT_MODEL_DIR,
        help="Path to the W3A4 quantized UNet checkpoint directory (DeepCompressor).",
    )
    parser.add_argument(
        "--torch_cache_dir",
        type=str,
        default=None,
        help="Directory to store torch hub checkpoints (default: ~/.cache/torch)"
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default=None,
        help="Path to MJHQ-30K meta_data.json. If not specified, downloads from HF Hub"
    )
    parser.add_argument(
        "--bias_scales",
        type=float,
        nargs="+",
        default=[1.0],
        help="Bias correction scale values to test (default: [1.0]). Example: --bias_scales 0.0 0.5 1.0 1.5"
    )
    parser.add_argument(
        "--qdrift_scales",
        type=float,
        nargs="+",
        default=[1.0],
        help="Q-Drift scale values to test (default: [1.0]). Example: --qdrift_scales -1.0 0.0 0.5 1.0 1.5 2.0"
    )
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
    parser.add_argument(
        "--save_xt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save per-step x_t latents for Q-Drift variants to --xt_output_dir/xt_latents/",
    )
    parser.add_argument(
        "--save_xt_fp16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save per-step x_t latents for FP16 to --xt_output_dir/xt_latents/ (model_key='fp16')",
    )
    parser.add_argument(
        "--save_xt_quant_baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save per-step x_t latents for quant baseline to --xt_output_dir/xt_latents/ (model_key='quant_baseline')",
    )
    parser.add_argument("--xt_output_dir", type=str, default=None, help="Directory to save x_t latents (default: --output_dir)")
    parser.add_argument("--xt_shard_size", type=int, default=32, help="Samples per x_t shard file")
    parser.add_argument("--xt_dtype", type=str, default="bfloat16", help="x_t dtype: bfloat16|float16|float32")
    parser.add_argument(
        "--xt_merge_single",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="On rank 0, merge x_t shards into a single large `.pth` per model key (can be very large).",
    )
    parser.add_argument(
        "--xt_delete_shards_after_merge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete per-rank shard files after successfully writing the merged `.pth` (best effort).",
    )
    args = parser.parse_args()
    if args.qdrift_only:
        args.skip_fp16 = True
        args.skip_quant_baseline = True
    if args.xt_output_dir is None:
        args.xt_output_dir = args.output_dir
    
    # Setup device (multi-GPU only - requires torchrun)
    # torchrun sets environment variables automatically
    torch.distributed.init_process_group(backend="nccl")
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    print(f"[Multi-GPU Mode] Rank {rank}/{world_size}, Device: {device}")
    
    # Configure torch cache directory to avoid permission errors
    configure_torch_cache(args.torch_cache_dir)
    
    # Only rank 0 creates output directory
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    
    torch.distributed.barrier()
    
    # ========== Pre-download models (Rank 0 only) to avoid race conditions ==========
    if rank == 0:
        print("\n" + "="*70)
        print("Pre-downloading models and metadata (Rank 0 only)")
        print("="*70)
        
        try:
            # Download MJHQ metadata
            print("Downloading MJHQ-30K metadata...")
            mjhq_meta_path = download_mjhq_metadata()
            print(f"✓ MJHQ metadata downloaded to: {mjhq_meta_path}")
            
            # Download FP16 model
            print("Downloading SDXL-Turbo FP16 model...")
            try:
                _ = StableDiffusionXLPipeline.from_pretrained(
                    args.base_model,
                    torch_dtype=torch.bfloat16,
                    variant="fp16",
                )
            except Exception:
                _ = StableDiffusionXLPipeline.from_pretrained(
                    args.base_model,
                    torch_dtype=torch.bfloat16,
                )
            print("✓ SDXL-Turbo FP16 model downloaded")
            
            # Warm up the quantized UNet load path (loads base UNet + applies PTQ checkpoint).
            print("Warming up quantized UNet load (DeepCompressor checkpoint)...")
            _ = load_quant_unet(
                base_model_id=args.base_model,
                quant_model_dir=args.quant_model_dir,
                device=torch.device("cpu"),
                torch_dtype=torch.bfloat16,
            )
            print("✓ Quantized UNet load warmed up")
            
            print("="*70)
            print("✓ All models pre-downloaded successfully")
            print("="*70)
        except Exception as e:
            print(f"✗ Pre-download failed: {e}")
            print("Warning: Other ranks may encounter download conflicts")
    
    # Synchronize to ensure all models are downloaded before others proceed
    torch.distributed.barrier()
    
    # All non-rank-0 processes get the metadata path from HF cache
    if rank != 0:
        # Download from cache (fast, already downloaded by rank 0)
        if args.meta_path is None or not os.path.exists(args.meta_path):
            mjhq_meta_path = download_mjhq_metadata()  # This will be fast (from cache)
        else:
            mjhq_meta_path = args.meta_path
    
    # ========== Load Prompts ==========
    if rank == 0:
        print("\n" + "="*70)
        print("Loading MJHQ-30K Prompts")
        print("="*70)
    
    # All ranks load the full prompt list. Optionally restrict generation to selected
    # global indices while preserving the original full --num_samples draw.
    all_prompts = load_prompts(mjhq_meta_path, args.num_samples, args.mjhq_prompt_sample_seed)
    selected_indices = _parse_global_indices(args.global_indices)
    use_rank_local_seeds = args.global_index_seed_mode == "rank_local"
    seed_world_size = int(args.seed_world_size or world_size)
    selected_seed_by_global_idx = selected_indices is not None and not use_rank_local_seeds
    if selected_indices is None:
        generation_prompts = all_prompts
    else:
        generation_prompts = _select_prompts_by_global_indices(all_prompts, selected_indices)
        if rank == 0:
            print(f"Generating selected global indices: {len(generation_prompts)} / {len(all_prompts)}")
            print(f"Selected-index seed mode: {args.global_index_seed_mode}")

    if use_rank_local_seeds:
        for row in generation_prompts:
            row["_seed_offset"] = _rank_local_seed_offset(
                int(row["global_idx"]), len(all_prompts), seed_world_size
            )
        if rank == 0:
            print(f"Using rank-local seed mapping from {seed_world_size} source ranks")

    # Split selected prompts across GPUs.
    samples_per_gpu = len(generation_prompts) // world_size
    start_idx = rank * samples_per_gpu
    if rank == world_size - 1:
        end_idx = len(generation_prompts)
    else:
        end_idx = start_idx + samples_per_gpu

    mjhq_prompts = generation_prompts[start_idx:end_idx]
    if mjhq_prompts:
        min_global = min(int(row["global_idx"]) for row in mjhq_prompts)
        max_global = max(int(row["global_idx"]) for row in mjhq_prompts)
        print(
            f"Rank {rank}: Processing {len(mjhq_prompts)} prompts "
            f"(selected idx {start_idx}-{end_idx}, global {min_global}-{max_global})"
        )
    else:
        print(f"Rank {rank}: Processing 0 prompts (selected idx {start_idx}-{end_idx})")
    
    # Use rank-specific output directory
    rank_output_dir = os.path.join(args.output_dir, f"rank{rank}")
    os.makedirs(rank_output_dir, exist_ok=True)
    seed_base = args.mjhq_prompt_sample_seed if (selected_indices is not None or use_rank_local_seeds) else args.mjhq_prompt_sample_seed + rank * 10000
    
    # Generate images only (skip metrics computation - will compute after merge)
    # Pass merged directory to check for already-completed images (for resume)
    mjhq_results = evaluate_benchmark(
        benchmark_name=f"MJHQ-30K (Rank {rank}/{world_size})",
        prompts=mjhq_prompts,
        mu_dict_path=args.mu_dict,
        cov_dict_path=args.cov_dict,
        output_dir=rank_output_dir,
        mjhq_prompt_sample_seed=seed_base,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        device=device,
        base_model_id=args.base_model,
        quant_model_dir=args.quant_model_dir,
        evaluate_fp16=not args.skip_fp16,
        evaluate_baseline=not args.skip_quant_baseline,
        bias_scales=args.bias_scales,
        qdrift_scales=args.qdrift_scales,
        qdrift_scalar=args.qdrift_scalar,
        local_rank=rank,
        world_size=world_size,
        merged_output_dir=args.output_dir,  # Check merged directory for resume
        seed_by_global_idx=selected_seed_by_global_idx,
        save_xt=bool(args.save_xt),
        save_xt_fp16=bool(args.save_xt_fp16),
        save_xt_quant_baseline=bool(args.save_xt_quant_baseline),
        xt_output_dir=str(args.xt_output_dir),
        xt_shard_size=int(args.xt_shard_size),
        xt_dtype=str(args.xt_dtype),
    )
    
    # Synchronize all ranks before merging
    torch.distributed.barrier()
    
    # ========== Merge results on rank 0 ==========
    if rank == 0:
        print("\n" + "="*70)
        print("Merging results from all ranks...")
        print("="*70)
        
        # Copy all rank images to main output directory
        for r in range(world_size):
            rank_dir = os.path.join(args.output_dir, f"rank{r}", "images")
            if os.path.exists(rank_dir):
                dest_dir = os.path.join(args.output_dir, "images")
                
                # Copy each model's images
                for model_name in os.listdir(rank_dir):
                    src_model_dir = os.path.join(rank_dir, model_name)
                    dest_model_dir = os.path.join(dest_dir, model_name)
                    
                    if os.path.isdir(src_model_dir):
                        os.makedirs(dest_model_dir, exist_ok=True)
                        for img_file in os.listdir(src_model_dir):
                            src_path = os.path.join(src_model_dir, img_file)
                            dest_path = os.path.join(dest_model_dir, img_file)
                            if os.path.isfile(src_path) and not os.path.exists(dest_path):
                                shutil.copy2(src_path, dest_path)
        
        print("✓ Images merged from all ranks")
        
        # Merge log files for Q-Drift analysis
        for bias_scale in args.bias_scales:
            for qdrift_scale in args.qdrift_scales:
                merged_log_dir = os.path.join(args.output_dir, f"logs_bias_{bias_scale}_drift{qdrift_scale}")
                os.makedirs(merged_log_dir, exist_ok=True)
                merged_log_file = os.path.join(merged_log_dir, "qdrift_log.jsonl")
                
                with open(merged_log_file, 'w') as outf:
                    for r in range(world_size):
                        rank_log_file = os.path.join(
                            args.output_dir,
                            f"rank{r}",
                            f"logs_bias_{bias_scale}_drift{qdrift_scale}",
                            "qdrift_log.jsonl",
                        )
                        if os.path.exists(rank_log_file):
                            with open(rank_log_file, 'r') as inf:
                                outf.write(inf.read())
        
        print("✓ Log files merged")

        # Write x_t indices (best-effort).
        if args.save_xt or args.save_xt_fp16 or args.save_xt_quant_baseline:
            xt_latents_dir = os.path.join(str(args.xt_output_dir), "xt_latents")
            _ensure_dir(xt_latents_dir)
            if args.save_xt_fp16:
                _ = _write_xt_index(xt_latents_dir, "fp16", world_size)
                if args.xt_merge_single:
                    merged = _merge_xt_latents_single(xt_latents_dir, "fp16", world_size)
                    if merged is not None:
                        print(f"✓ Merged x_t latents: {merged}")
                        if args.xt_delete_shards_after_merge:
                            for name in list(os.listdir(xt_latents_dir)):
                                if name.startswith("xt_latents_fp16_rank") and name.endswith(".pth"):
                                    try:
                                        os.remove(os.path.join(xt_latents_dir, name))
                                    except Exception:
                                        pass
            if args.save_xt_quant_baseline:
                _ = _write_xt_index(xt_latents_dir, "quant_baseline", world_size)
                if args.xt_merge_single:
                    merged = _merge_xt_latents_single(xt_latents_dir, "quant_baseline", world_size)
                    if merged is not None:
                        print(f"✓ Merged x_t latents: {merged}")
                        if args.xt_delete_shards_after_merge:
                            for name in list(os.listdir(xt_latents_dir)):
                                if name.startswith("xt_latents_quant_baseline_rank") and name.endswith(".pth"):
                                    try:
                                        os.remove(os.path.join(xt_latents_dir, name))
                                    except Exception:
                                        pass
            if args.save_xt:
                for bias_scale in args.bias_scales:
                    for qdrift_scale in args.qdrift_scales:
                        model_key = f"quant_bias_{bias_scale}_drift{qdrift_scale}" + ("_scalar" if args.qdrift_scalar else "")
                        _ = _write_xt_index(xt_latents_dir, model_key, world_size)
                        if args.xt_merge_single:
                            merged = _merge_xt_latents_single(xt_latents_dir, model_key, world_size)
                            if merged is not None:
                                print(f"✓ Merged x_t latents: {merged}")
                                if args.xt_delete_shards_after_merge:
                                    for name in list(os.listdir(xt_latents_dir)):
                                        if name.startswith(f"xt_latents_{model_key}_rank") and name.endswith(".pth"):
                                            try:
                                                os.remove(os.path.join(xt_latents_dir, name))
                                            except Exception:
                                                pass
        
        # Clean up rank directories after successful merge
        print("\nCleaning up rank directories...")
        for r in range(world_size):
            rank_dir_path = os.path.join(args.output_dir, f"rank{r}")
            if os.path.exists(rank_dir_path):
                shutil.rmtree(rank_dir_path)
                print(f"  ✓ Removed rank{r}/")
        print("✓ Rank directories cleaned up")
    
    # Synchronize again
    torch.distributed.barrier()

    if selected_indices is not None:
        if rank == 0:
            print("\n" + "="*70)
            print("Selected global-index smoke generation complete; skipping full metrics/resume stage.")
            print("="*70)
            print(f"Generated selected global indices: {selected_indices}")
            print(f"All smoke images saved to: {args.output_dir}")
        torch.distributed.destroy_process_group()
        return
    
    torch.distributed.destroy_process_group()
    if rank == 0:
        print("\n" + "="*70)
        print("✓ Benchmark evaluation complete!")
        print("="*70)
        print(f"All results saved to: {args.output_dir}")
        print(f"Evaluated bias scales: {args.bias_scales}")
        print(f"Evaluated drift scales: {args.qdrift_scales}")


if __name__ == "__main__":
    main()
