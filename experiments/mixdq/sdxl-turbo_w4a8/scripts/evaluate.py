"""
Generate MJHQ-30K evaluation images for SDXL-Turbo MixDQ W4A8: FP16, quantized baseline, and Q-Drift.

Prompts are a stratified sample of MJHQ-30K (`--mjhq_prompt_sample_seed`). SDXL-Turbo uses
guidance_scale=0 and 4 inference steps. Metrics are computed separately by
`evaluation/compute_metrics.py` (see run_paper.sh).
"""

import argparse
import os
import json
import shutil
import sys
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import yaml

# Compatibility: some libraries still import `cached_download` from huggingface_hub.
import huggingface_hub as _huggingface_hub
if not hasattr(_huggingface_hub, "cached_download"):
    from huggingface_hub import hf_hub_download as _hf_hub_download

    _huggingface_hub.cached_download = _hf_hub_download  # type: ignore[attr-defined]

from diffusers import StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

# Allow importing local helper modules in this experiment folder.
_LOCAL_ROOT = Path(__file__).resolve().parents[1]
if str(_LOCAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOCAL_ROOT))

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



# MJHQ-30K constants
MJHQ_REPO_ID = "playgroundai/MJHQ-30K"
MJHQ_META_FILENAME = "meta_data.json"


def _default_mp_paths():
    mixdq_root = Path(__file__).resolve().parents[4] / "third_party" / "mixdq"
    mp_root = mixdq_root / "mixed_precision_scripts" / "mixed_percision_config" / "sdxl_turbo" / "final_config"
    return (
        str(mp_root / "weight" / "weight_4.00.yaml"),
        str(mp_root / "act" / "act_8.00.yaml"),
        str(mp_root / "act" / "act_sensitivie_a8_1%.pt"),
    )


def _apply_author_mp_and_protect(unet_quant, config_weight_mp: str, config_act_mp: str, act_protect: str):
    with open(config_weight_mp, "r", encoding="utf-8") as f:
        w_bits = yaml.safe_load(f)
    with open(config_act_mp, "r", encoding="utf-8") as f:
        a_bits = yaml.safe_load(f)

    protected = torch.load(act_protect, map_location="cpu")
    unet_quant.set_layer_quant(
        model=unet_quant,
        module_name_list=protected,
        quant_level="per_layer",
        weight_quant=False,
        act_quant=False,
    )
    unet_quant.load_bitwidth_config(model=unet_quant, bit_config=w_bits, bit_type="weight")
    unet_quant.load_bitwidth_config(model=unet_quant, bit_config=a_bits, bit_type="act")


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


def _resolve_mixdq_paths(mixdq_base_path: Optional[str], mixdq_config: Optional[str], mixdq_ckpt: Optional[str]):
    if mixdq_base_path is not None:
        cfg_path = os.path.join(mixdq_base_path, "config.yaml")
        ckpt_path = os.path.join(mixdq_base_path, "ckpt.pth")
    else:
        cfg_path = mixdq_config
        ckpt_path = mixdq_ckpt

    if cfg_path is None or ckpt_path is None:
        raise ValueError("Provide --mixdq_base_path or both --mixdq_config and --mixdq_ckpt")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"MixDQ config not found: {cfg_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"MixDQ ckpt not found: {ckpt_path}")
    return cfg_path, ckpt_path


def build_mixdq_quant_pipeline(
    mixdq_cfg,
    mixdq_ckpt_path: str,
    device: torch.device,
    w_bit: int = 8,
    a_bit: int = 8,
    config_weight_mp: str | None = None,
    config_act_mp: str | None = None,
    act_protect: str | None = None,
):
    """
    Build a StableDiffusionXLPipeline whose UNet is quantized by MixDQ (qdiff) and loaded from `ckpt.pth`.

    Note: `load_quant_params` loads tensors on CPU; calling `pipe.to(device)` after loading moves them to GPU.
    """
    # qdiff backend is optional and depends on a compatible diffusers version.
    _MIXDQ_ROOT = Path(__file__).resolve().parents[4] / "third_party" / "mixdq"
    _QUANT_UTILS = _MIXDQ_ROOT / "quant_utils"
    if _QUANT_UTILS.is_dir() and str(_QUANT_UTILS) not in sys.path:
        sys.path.insert(0, str(_QUANT_UTILS))

    from qdiff.models.quant_model import QuantModel
    from qdiff.utils import get_model, load_quant_params

    base_unet, pipe = get_model(
        mixdq_cfg.model,
        fp16=True,
        return_pipe=True,
        convert_model_for_quant=True,
    )

    wq_params = mixdq_cfg.quant.weight.quantizer
    aq_params = mixdq_cfg.quant.activation.quantizer
    if mixdq_cfg.get("mixed_precision", False):
        wq_params["mixed_precision"] = mixdq_cfg.mixed_precision
        aq_params["mixed_precision"] = mixdq_cfg.mixed_precision

    qnn = QuantModel(
        model=base_unet,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
    )
    qnn.set_quant_state(True, True)
    load_quant_params(qnn, mixdq_ckpt_path, dtype=torch.float16)

    if config_weight_mp and config_act_mp and act_protect:
        _apply_author_mp_and_protect(qnn, config_weight_mp, config_act_mp, act_protect)
    else:
        # Fallback: uniform refactor bitwidth (power-of-two only).
        if w_bit is not None:
            if w_bit not in (2, 4, 8, 16):
                raise ValueError(f"quant_backend=qdiff only supports w_bit in (2,4,8,16). Got w_bit={w_bit}.")
            qnn.set_layer_bit(model=qnn, n_bit=w_bit, quant_level="reset", bit_type="weight")
        if a_bit is not None:
            if a_bit not in (2, 4, 8, 16):
                raise ValueError(f"quant_backend=qdiff only supports a_bit in (2,4,8,16). Got a_bit={a_bit}.")
            qnn.set_layer_bit(model=qnn, n_bit=a_bit, quant_level="reset", bit_type="act")

    pipe.unet = qnn.half().eval()
    pipe = pipe.to(device)
    return pipe


def build_mixdq_hf_quant_pipeline(device: torch.device, w_bit: int = 8, a_bit: int = 8):
    """
    Build the authors' released HuggingFace MixDQ pipeline and quantize the UNet.

    This downloads the prepackaged quantization parameters and (optional) BOS tensors from HF.
    """
    try:
        import mixdq_extension._C  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "mixdq-extension is required for quant_backend=hf. "
            "Install: `pip install -i https://pypi.org/simple/ mixdq-extension`"
        ) from e

    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        custom_pipeline="nics-efc/MixDQ",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    # The authors' custom pipeline uses a module-level global counter `_NUM` to
    # index `_SPLIT`. If `quantize_unet()` is called more than once in the same
    # Python process, `_NUM` can overflow and crash. Reset it defensively.
    try:
        import importlib

        _pipe_mod = importlib.import_module(pipe.__class__.__module__)
        if hasattr(_pipe_mod, "_NUM"):
            setattr(_pipe_mod, "_NUM", 0)
    except Exception:
        pass

    if w_bit != 8 or a_bit != 8:
        raise ValueError(
            f"quant_backend=hf only supports int8 quantization in the authors' pipeline (got w_bit={w_bit}, a_bit={a_bit}). "
            "Use w_bit=8,a_bit=8 or switch to quant_backend=qdiff with a compatible ckpt/config."
        )
    pipe.quantize_unet(w_bit=w_bit, a_bit=a_bit, bos=True)
    pipe = pipe.to(device)
    return pipe

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


def parse_global_indices(global_indices: Optional[str]) -> Optional[List[int]]:
    if global_indices is None or str(global_indices).strip() == "":
        return None

    parsed: List[int] = []
    seen = set()
    for token in str(global_indices).split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if idx < 0:
            raise ValueError(f"--global_indices contains a negative index: {idx}")
        if idx in seen:
            raise ValueError(f"--global_indices contains a duplicate index: {idx}")
        seen.add(idx)
        parsed.append(idx)

    if not parsed:
        raise ValueError("--global_indices did not contain any valid indices")
    return parsed


def select_prompts_by_global_indices(prompts: List[Dict], global_indices: Optional[List[int]]) -> List[Dict]:
    if global_indices is None:
        return prompts

    by_idx = {int(prompt["global_idx"]): prompt for prompt in prompts}
    missing = [idx for idx in global_indices if idx not in by_idx]
    if missing:
        available_max = max(by_idx.keys()) if by_idx else -1
        raise ValueError(
            f"--global_indices contains indices outside the sampled prompt list: {missing}. "
            f"Available global_idx range is 0..{available_max} for --num_samples={len(prompts)}."
        )
    return [by_idx[idx] for idx in global_indices]


def attach_original_shard_seed_offsets(prompts: List[Dict], num_samples: int, source_world_size: int) -> None:
    if source_world_size <= 0:
        raise ValueError("--global_indices_seed_world_size must be a positive integer")

    samples_per_gpu = num_samples // source_world_size
    if samples_per_gpu <= 0:
        raise ValueError(
            f"--global_indices_seed_world_size={source_world_size} is too large for --num_samples={num_samples}"
        )

    for prompt in prompts:
        global_idx = int(prompt["global_idx"])
        if global_idx >= samples_per_gpu * (source_world_size - 1):
            original_rank = source_world_size - 1
            shard_start = samples_per_gpu * (source_world_size - 1)
        else:
            original_rank = global_idx // samples_per_gpu
            shard_start = original_rank * samples_per_gpu
        local_i = global_idx - shard_start
        prompt["_qdrift_seed_offset"] = int(original_rank * 10000 + local_i)


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
    seed_by_global_index: bool = False,
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
        
        if "_qdrift_seed_offset" in prompt_dict:
            seed_offset = int(prompt_dict["_qdrift_seed_offset"])
        else:
            seed_offset = int(global_idx) if seed_by_global_index else i
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
        filepath = os.path.join(output_dir, filename)
        # Be extra defensive in distributed runs / different CWDs.
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        image.save(filepath)
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
    quant_backend: str,
    mixdq_config_path: Optional[str],
    mixdq_ckpt_path: Optional[str],
    fp16_reference_dir: Optional[str],
    output_dir: str,
    mjhq_prompt_sample_seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    device: str,
    evaluate_fp16: bool = True,
    evaluate_baseline: bool = True,
    bias_scales: List[float] = [1.0],
    qdrift_scales: List[float] = [1.0],
    local_rank: int = 0,
    world_size: int = 1,
    merged_output_dir: str = None,  # For checking already-merged images
    w_bit: int = 8,  # quant_backend=hf only
    a_bit: int = 8,  # quant_backend=hf only
    config_weight_mp: str | None = None,  # quant_backend=qdiff only (author inference style)
    config_act_mp: str | None = None,  # quant_backend=qdiff only (author inference style)
    act_protect: str | None = None,  # quant_backend=qdiff only (author inference style)
    save_xt: bool = False,
    save_xt_fp16: bool = False,
    save_xt_quant_baseline: bool = False,
    xt_output_dir: Optional[str] = None,
    xt_shard_size: int = 100,
    xt_dtype: str = "bfloat16",
    generate_fp16: bool = False,
    qdrift_scalar: bool = False,
    seed_by_global_index: bool = False,
):
    """
    Evaluate models on a benchmark dataset.
    
    Args:
        benchmark_name: Name of the benchmark (e.g., "MJHQ-30K")
        prompts: List of prompt dictionaries
        mu_dict_path: Path to mu_dict.npy
        cov_dict_path: Path to cov_dict.npy
        mixdq_config_path: Path to MixDQ config.yaml
        mixdq_ckpt_path: Path to MixDQ ckpt.pth
        fp16_reference_dir: Directory containing pre-generated FP16 images (optional; used when generate_fp16=False)
        output_dir: Directory to save results
        mjhq_prompt_sample_seed: Random seed for prompt sampling and image generation
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale
        device: Device to use
        evaluate_fp16: Whether to evaluate FP16 baseline
        evaluate_baseline: Whether to evaluate quantized baseline
    """
    print("\n" + "="*70)
    print(f"Evaluating on {benchmark_name}")
    print("="*70)
    print(f"Number of prompts: {len(prompts)}")
    print(f"Output directory: {output_dir}")
    print("="*70)
    
    mixdq_cfg = None
    if quant_backend == "qdiff":
        if mixdq_config_path is None or mixdq_ckpt_path is None:
            raise ValueError("For quant_backend=qdiff, provide --mixdq_base_path or both --mixdq_config and --mixdq_ckpt")
        mixdq_cfg = OmegaConf.load(mixdq_config_path)

    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'benchmark': benchmark_name,
        'num_samples': len(prompts),
        'prompts': prompts,
        'metrics': {}
    }
    
    # Dictionary to store all generated images
    all_images = {}

    # Lazily constructed quantized pipeline (built only if we need to generate any quant images).
    quant_pipeline = None
    quant_pipeline_default_scheduler = None
    
    # ========== Step 1: Generate/Load All Images ==========
    print("\n" + "="*70)
    print("STEP 1: Image Generation/Loading")
    print("="*70)
    
    # ========== FP16 Reference ==========
    fp16_images = None
    if evaluate_fp16:
        if generate_fp16:
            print("\n[FP16] Loading/Generating FP16 images...")

            fp16_dir = os.path.join(output_dir, "images", "fp16")
            os.makedirs(fp16_dir, exist_ok=True)

            fp16_pipeline = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=torch.float16,
                variant="fp16",
            ).to(device)
            fp16_pipeline.scheduler = EulerAncestralQDriftScheduler.from_pretrained(
                "stabilityai/sdxl-turbo", subfolder="scheduler"
            )
            fp16_pipeline.set_progress_bar_config(disable=True)

            fp16_xt_logger = None
            if save_xt_fp16 and xt_output_dir is not None:
                xt_latents_dir = os.path.join(str(xt_output_dir), "xt_latents")
                _ensure_dir(xt_latents_dir)
                fp16_xt_logger = XTLatentsLogger(
                    model_key="fp16",
                    xt_output_dir=xt_latents_dir,
                    rank=local_rank,
                    world_size=world_size,
                    prompt_seed=mjhq_prompt_sample_seed,
                    noise_seed_start=mjhq_prompt_sample_seed,
                    num_inference_steps=num_inference_steps,
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
                seed_by_global_index=seed_by_global_index,
            )

            print(f"✓ Generated {len(fp16_images)} FP16 images")
            torch.cuda.empty_cache()

            fp16_images = None
            print("✓ Skipping FP16 image loading during generation-only stage")

            all_images["fp16"] = fp16_images
        else:
            print("\n[FP16] Loading/Generating FP16 reference images...")

            # Load from an existing directory (prompt seed must match).
            if not fp16_reference_dir:
                raise ValueError(
                    "FP16 generation is disabled. Provide --fp16_reference_dir pointing to pre-generated fp16 images, "
                    "or enable --generate_fp16."
                )

            # Only rank 0 checks to avoid too many open files.
            images_exist = False
            if local_rank == 0:
                test_images = load_existing_images(fp16_reference_dir, prompts)
                if test_images is not None:
                    images_exist = True
                    print(f"✓ Found FP16 images in: {fp16_reference_dir}")

            # Broadcast result to all ranks (if distributed)
            if world_size > 1:
                import torch.distributed as dist

                images_exist_tensor = torch.tensor([1 if images_exist else 0], dtype=torch.int, device=device)
                dist.broadcast(images_exist_tensor, src=0)
                images_exist = bool(images_exist_tensor.item())

            if not images_exist:
                raise FileNotFoundError(
                    "FP16 reference images not found/mismatched for the current prompt set. "
                    f"Check --fp16_reference_dir='{fp16_reference_dir}', --num_samples, and --mjhq_prompt_sample_seed."
                )

            # Images are loaded later by the metrics stage.
            print("✓ Skipping FP16 image loading during generation-only stage")
            fp16_images = None

            all_images["fp16"] = fp16_images
    
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
            if quant_pipeline is None:
                if quant_backend == "hf":
                    quant_pipeline = build_mixdq_hf_quant_pipeline(device=device, w_bit=w_bit, a_bit=a_bit)
                elif quant_backend == "qdiff":
                    quant_pipeline = build_mixdq_quant_pipeline(
                        mixdq_cfg,
                        mixdq_ckpt_path,
                        device=device,
                        w_bit=w_bit,
                        a_bit=a_bit,
                        config_weight_mp=config_weight_mp,
                        config_act_mp=config_act_mp,
                        act_protect=act_protect,
                    )
                else:
                    raise ValueError(f"Unknown quant_backend: {quant_backend} (expected: hf|qdiff)")
                quant_pipeline_default_scheduler = quant_pipeline.scheduler
            else:
                quant_pipeline.scheduler = quant_pipeline_default_scheduler
            
            os.makedirs(quant_baseline_dir, exist_ok=True)
            baseline_xt_logger = None
            if save_xt_quant_baseline and xt_output_dir is not None:
                xt_latents_dir = os.path.join(str(xt_output_dir), "xt_latents")
                _ensure_dir(xt_latents_dir)
                baseline_xt_logger = XTLatentsLogger(
                    model_key="quant_baseline",
                    xt_output_dir=xt_latents_dir,
                    rank=local_rank,
                    world_size=world_size,
                    prompt_seed=mjhq_prompt_sample_seed,
                    noise_seed_start=mjhq_prompt_sample_seed,
                    num_inference_steps=num_inference_steps,
                    shard_size=int(xt_shard_size),
                    xt_dtype=_parse_xt_dtype(xt_dtype),
                )
            quant_baseline_images = generate_images(
                quant_pipeline,
                prompts,
                quant_baseline_dir,
                "baseline",
                mjhq_prompt_sample_seed,
                num_inference_steps,
                guidance_scale,
                device,
                xt_logger=baseline_xt_logger,
                seed_by_global_index=seed_by_global_index,
            )
            
            print(f"✓ Generated {len(quant_baseline_images)} quantized baseline images")
            
            # qdiff activation quantizers update their parameters on use, so each
            # image set starts from a freshly loaded quantized pipeline.
            quant_pipeline = None
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
                if quant_pipeline is None:
                    if quant_backend == "hf":
                        quant_pipeline = build_mixdq_hf_quant_pipeline(device=device, w_bit=w_bit, a_bit=a_bit)
                    elif quant_backend == "qdiff":
                        quant_pipeline = build_mixdq_quant_pipeline(
                            mixdq_cfg,
                            mixdq_ckpt_path,
                            device=device,
                            w_bit=w_bit,
                            a_bit=a_bit,
                            config_weight_mp=config_weight_mp,
                            config_act_mp=config_act_mp,
                            act_protect=act_protect,
                        )
                    else:
                        raise ValueError(f"Unknown quant_backend: {quant_backend} (expected: hf|qdiff)")
                    quant_pipeline_default_scheduler = quant_pipeline.scheduler

                # Create scale-specific log directory
                scale_log_dir = os.path.join(output_dir, f"logs_bias_{bias_scale}_drift{qdrift_scale}")
                
                scheduler = EulerAncestralQDriftScheduler.from_pretrained(
                    "stabilityai/sdxl-turbo",
                    subfolder="scheduler",
                    mu_dict_path=mu_dict_path,
                    cov_dict_path=cov_dict_path,
                    log_dir=scale_log_dir,
                    bias_scale=bias_scale,
                    qdrift_scale=qdrift_scale,
                    qdrift_scalar=qdrift_scalar,
                )
                quant_pipeline.scheduler = scheduler
                
                os.makedirs(quant_qdrift_dir, exist_ok=True)
                qdrift_xt_logger = None
                if save_xt and xt_output_dir is not None:
                    xt_latents_dir = os.path.join(str(xt_output_dir), "xt_latents")
                    _ensure_dir(xt_latents_dir)
                    qdrift_xt_logger = XTLatentsLogger(
                        model_key=model_key,
                        xt_output_dir=xt_latents_dir,
                        rank=local_rank,
                        world_size=world_size,
                        prompt_seed=mjhq_prompt_sample_seed,
                        noise_seed_start=mjhq_prompt_sample_seed,
                        num_inference_steps=num_inference_steps,
                        shard_size=int(xt_shard_size),
                        xt_dtype=_parse_xt_dtype(xt_dtype),
                    )
                quant_qdrift_images = generate_images(
                    quant_pipeline,
                    prompts,
                    quant_qdrift_dir,
                    f"bias_{bias_scale}_drift{qdrift_scale}",
                    mjhq_prompt_sample_seed,
                    num_inference_steps,
                    guidance_scale,
                    device,
                    xt_logger=qdrift_xt_logger,
                    seed_by_global_index=seed_by_global_index,
                )
                
                print(f"✓ Generated {len(quant_qdrift_images)} images (bias={bias_scale}, drift={qdrift_scale})")
                
                del scheduler
                quant_pipeline = None
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
        "--mjhq_prompt_sample_seed",
        type=int,
        default=42,
        help="Random seed for MJHQ prompt sampling and image generation"
    )
    parser.add_argument(
        "--global_indices",
        type=str,
        default=None,
        help="Optional comma-separated MJHQ global_idx values to generate after drawing the full --num_samples prompt set. "
             "Use this for small pixel-wise smoke tests without changing paper defaults.",
    )
    parser.add_argument(
        "--global_indices_seed_world_size",
        type=int,
        default=None,
        help="When --global_indices is set, reproduce the full-run seed layout of this many ranks. "
             "The MixDQ paper row was generated with 4 ranks; full runs also use this option when set.",
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
        "--fp16_reference_dir",
        type=str,
        default=None,
        help="Directory containing pre-generated FP16 images to use as reference (generation is disabled)",
    )
    parser.add_argument(
        "--bias_scales",
        type=float,
        nargs="+",
        default=[1.0],
        help="Bias correction scale values to test (default: [1.0]). Example: --bias_scales 0.0 0.5 1.0 1.5"
    )
    parser.add_argument(
        "--qdrift_scalar",
        action="store_true",
        help="Collapse V_sigma to one scalar per step.",
    )
    parser.add_argument(
        "--qdrift_scales",
        type=float,
        nargs="+",
        default=[1.0],
        help="Q-Drift scale values to test (default: [1.0]). Example: --qdrift_scales -1.0 0.0 0.5 1.0 1.5 2.0"
    )
    parser.add_argument(
        "--save_xt",
        action="store_true",
        help="Save per-step x_t latents for Q-Drift variants to --xt_output_dir/xt_latents/",
    )
    parser.add_argument(
        "--save_xt_fp16",
        action="store_true",
        help="Also save per-step x_t latents for the FP16 variant to --xt_output_dir/xt_latents/ (model_key='fp16')",
    )
    parser.add_argument(
        "--save_xt_quant_baseline",
        action="store_true",
        help="Also save per-step x_t latents for the quant baseline to --xt_output_dir/xt_latents/ (model_key='quant_baseline')",
    )
    parser.add_argument(
        "--xt_output_dir",
        type=str,
        default=None,
        help="Base directory for x_t logging (defaults to --output_dir). Shards are written under xt_latents/.",
    )
    parser.add_argument(
        "--xt_shard_size",
        type=int,
        default=100,
        help="Number of samples per x_t shard file (default: 100)",
    )
    parser.add_argument(
        "--xt_dtype",
        type=str,
        default="bfloat16",
        help="x_t dtype: bfloat16|float16|float32 (default: bfloat16)",
    )
    parser.add_argument(
        "--skip_fp16",
        action="store_true",
        help="Skip FP16 generation/loading.",
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
        "--generate_fp16",
        action="store_true",
        help="Generate FP16 images (and optionally x_t) instead of requiring --fp16_reference_dir.",
    )
    parser.add_argument(
        "--quant_backend",
        type=str,
        choices=["hf", "qdiff"],
        default="hf",
        help="Quantized UNet backend: hf (authors' MixDQ pipeline) or qdiff (local PTQ ckpt.pth)",
    )
    parser.add_argument(
        "--mixdq_base_path",
        type=str,
        default=None,
        help="MixDQ PTQ log dir containing config.yaml and ckpt.pth (recommended)",
    )
    parser.add_argument(
        "--mixdq_config",
        type=str,
        default=None,
        help="Path to MixDQ config.yaml (used if --mixdq_base_path is not set)",
    )
    parser.add_argument(
        "--mixdq_ckpt",
        type=str,
        default=None,
        help="Path to MixDQ ckpt.pth (used if --mixdq_base_path is not set)",
    )
    parser.add_argument(
        "--w_bit",
        type=int,
        default=8,
        help="Weight bitwidth for quant_backend=hf (default: 8)",
    )
    parser.add_argument(
        "--a_bit",
        type=int,
        default=8,
        help="Activation bitwidth for quant_backend=hf (default: 8)",
    )
    default_w_mp, default_a_mp, default_protect = _default_mp_paths()
    parser.add_argument(
        "--config_weight_mp",
        type=str,
        default=default_w_mp,
        help="Weight mixed-precision bit config (author inference style; used for quant_backend=qdiff)",
    )
    parser.add_argument(
        "--config_act_mp",
        type=str,
        default=default_a_mp,
        help="Activation mixed-precision bit config (author inference style; used for quant_backend=qdiff)",
    )
    parser.add_argument(
        "--act_protect",
        type=str,
        default=default_protect,
        help="Path to act_protect list (.pt) to keep extremely sensitive layers as FP16 (quant_backend=qdiff)",
    )
    parser.add_argument(
        "--disable_mp",
        action="store_true",
        help="Disable author-style mixed precision and use uniform w_bit/a_bit refactor instead (quant_backend=qdiff).",
    )
    args = parser.parse_args()
    if args.qdrift_only:
        args.skip_fp16 = True
        args.skip_quant_baseline = True

    xt_output_dir = args.xt_output_dir if args.xt_output_dir is not None else args.output_dir

    mixdq_config_path = None
    mixdq_ckpt_path = None
    mixdq_cfg = None
    if args.quant_backend == "qdiff":
        mixdq_config_path, mixdq_ckpt_path = _resolve_mixdq_paths(
            args.mixdq_base_path, args.mixdq_config, args.mixdq_ckpt
        )
        mixdq_cfg = OmegaConf.load(mixdq_config_path)

    config_weight_mp = None if args.disable_mp else args.config_weight_mp
    config_act_mp = None if args.disable_mp else args.config_act_mp
    act_protect = None if args.disable_mp else args.act_protect
    
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
            # Resolve MJHQ metadata. Respect --meta_path when provided so
            # smoke tests can run from local artifacts without a metadata fetch.
            if args.meta_path is not None and os.path.exists(args.meta_path):
                mjhq_meta_path = args.meta_path
                print(f"✓ Using MJHQ metadata from: {mjhq_meta_path}")
            else:
                print("Downloading MJHQ-30K metadata...")
                mjhq_meta_path = download_mjhq_metadata()
                print(f"✓ MJHQ metadata downloaded to: {mjhq_meta_path}")
            
            # Warm up / pre-download weights to avoid rank races.
            if args.quant_backend == "hf":
                # Download only (do NOT call quantize_unet here) to avoid module-level
                # global state issues in the custom pipeline.
                print("Warming up MixDQ HF pipeline (download only)...")
                from diffusers import DiffusionPipeline

                _ = DiffusionPipeline.from_pretrained(
                    "stabilityai/sdxl-turbo",
                    custom_pipeline="nics-efc/MixDQ",
                    torch_dtype=torch.float16,
                    variant="fp16",
                )
                print("✓ MixDQ HF pipeline downloaded")
            else:
                print("Warming up SDXL-Turbo weights (MixDQ qdiff cache)...")
                _ = get_model(mixdq_cfg.model, fp16=True, return_pipe=False, convert_model_for_quant=True)
                print("✓ SDXL-Turbo weights ready")
            
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
    
    # All ranks load the full prompt list first so optional smoke-test subsets
    # preserve paper global_idx values and their corresponding initial-noise seeds.
    full_mjhq_prompts = load_prompts(mjhq_meta_path, args.num_samples, args.mjhq_prompt_sample_seed)
    selected_global_indices = parse_global_indices(args.global_indices)
    mjhq_prompts = select_prompts_by_global_indices(full_mjhq_prompts, selected_global_indices)
    use_rank_local_seeds = args.global_indices_seed_world_size is not None
    seed_by_global_index = selected_global_indices is not None and not use_rank_local_seeds
    if use_rank_local_seeds:
        attach_original_shard_seed_offsets(mjhq_prompts, args.num_samples, args.global_indices_seed_world_size)
    if rank == 0:
        if selected_global_indices is not None:
            print(f"Using global-index smoke subset: {selected_global_indices}")
        if use_rank_local_seeds:
            print(f"Using rank-local seed mapping from {args.global_indices_seed_world_size} source ranks")
    
    # Split prompts across GPUs
    samples_per_gpu = len(mjhq_prompts) // world_size
    start_idx = rank * samples_per_gpu
    
    # Last GPU handles remainder
    if rank == world_size - 1:
        end_idx = len(mjhq_prompts)
    else:
        end_idx = start_idx + samples_per_gpu
    
    mjhq_prompts = mjhq_prompts[start_idx:end_idx]
    print(f"Rank {rank}: Processing {len(mjhq_prompts)} prompts (indices {start_idx}-{end_idx})")
    
    # Use rank-specific output directory
    rank_output_dir = os.path.join(args.output_dir, f"rank{rank}")
    os.makedirs(rank_output_dir, exist_ok=True)
    
    # Generate images only (skip metrics computation - will compute after merge)
    # Pass merged directory to check for already-completed images (for resume)
    mjhq_results = evaluate_benchmark(
        benchmark_name=f"MJHQ-30K (Rank {rank}/{world_size})",
        prompts=mjhq_prompts,
        mu_dict_path=args.mu_dict,
        cov_dict_path=args.cov_dict,
        quant_backend=args.quant_backend,
        mixdq_config_path=mixdq_config_path,
        mixdq_ckpt_path=mixdq_ckpt_path,
        fp16_reference_dir=args.fp16_reference_dir,
        output_dir=rank_output_dir,
        mjhq_prompt_sample_seed=(
            args.mjhq_prompt_sample_seed
            if (seed_by_global_index or use_rank_local_seeds)
            else args.mjhq_prompt_sample_seed + rank * 10000
        ),
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        device=device,
        evaluate_fp16=not args.skip_fp16,
        evaluate_baseline=not args.skip_quant_baseline,
        bias_scales=args.bias_scales,
        qdrift_scales=args.qdrift_scales,
        local_rank=rank,
        world_size=world_size,
        merged_output_dir=args.output_dir,  # Check merged directory for resume
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        config_weight_mp=config_weight_mp,
        config_act_mp=config_act_mp,
        act_protect=act_protect,
        save_xt=args.save_xt,
        save_xt_fp16=args.save_xt_fp16,
        save_xt_quant_baseline=args.save_xt_quant_baseline,
        xt_output_dir=xt_output_dir,
        xt_shard_size=args.xt_shard_size,
        xt_dtype=args.xt_dtype,
        generate_fp16=args.generate_fp16,
        qdrift_scalar=args.qdrift_scalar,
        seed_by_global_index=seed_by_global_index,
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

        # Write x_t indices (shared xt_output_dir across ranks).
        xt_latents_dir = os.path.join(str(xt_output_dir), "xt_latents")
        if args.save_xt_fp16:
            _ = _write_xt_index(xt_latents_dir, "fp16", world_size)
        if args.save_xt_quant_baseline:
            _ = _write_xt_index(xt_latents_dir, "quant_baseline", world_size)
        if args.save_xt:
            for bias_scale in args.bias_scales:
                for qdrift_scale in args.qdrift_scales:
                    model_key = f"quant_bias_{bias_scale}_drift{qdrift_scale}" + ("_scalar" if args.qdrift_scalar else "")
                    _ = _write_xt_index(xt_latents_dir, model_key, world_size)
        
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
