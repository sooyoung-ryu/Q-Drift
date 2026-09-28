"""
Collect UNet outputs for Q-Drift gaussian modeling.

This script implements manual denoising loops to collect UNet output pairs from FP16 and quantized models.

Based on D2-DPM's approach adapted for SDXL-Turbo + Euler Ancestral sampling.
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import yaml
from tqdm import tqdm

# Compatibility: some libraries still import `cached_download` from huggingface_hub.
import huggingface_hub as _huggingface_hub
if not hasattr(_huggingface_hub, "cached_download"):
    from huggingface_hub import hf_hub_download as _hf_hub_download

    _huggingface_hub.cached_download = _hf_hub_download  # type: ignore[attr-defined]

from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

# Allow importing MixDQ quantization code (qdiff) without requiring installation (optional backend).
_MIXDQ_ROOT = Path(__file__).resolve().parents[4] / "third_party" / "mixdq"
_QUANT_UTILS = _MIXDQ_ROOT / "quant_utils"
if _QUANT_UTILS.is_dir() and str(_QUANT_UTILS) not in sys.path:
    sys.path.insert(0, str(_QUANT_UTILS))

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
    # Preserve extremely sensitive layers as FP16 (disable quant).
    unet_quant.set_layer_quant(
        model=unet_quant,
        module_name_list=protected,
        quant_level="per_layer",
        weight_quant=False,
        act_quant=False,
    )
    # Apply per-layer bitwidth configs.
    unet_quant.load_bitwidth_config(model=unet_quant, bit_config=w_bits, bit_type="weight")
    unet_quant.load_bitwidth_config(model=unet_quant, bit_config=a_bits, bit_type="act")


class OutputCollector:
    """Collect UNet outputs from fp16 and quantized models."""
    
    def __init__(self):
        self.fp16_output = {}  # {timestep: List[tensor]}
        self.quant_output = {}  # {timestep: List[tensor]}
        
    def add_pair(self, timestep: int, fp16_output: torch.Tensor, quant_output: torch.Tensor):
        """
        Add a pair of UNet outputs from the same x_t.
        
        Args:
            timestep: Current timestep
            fp16_output: FP16 UNet output, shape [B, C, H, W]
            quant_output: Quantized UNet output, shape [B, C, H, W]
        """
        if timestep not in self.fp16_output:
            self.fp16_output[timestep] = []
            self.quant_output[timestep] = []
        
        # Store as [C, H, W] (remove batch dim)
        fp16_cpu = fp16_output.detach().cpu()
        quant_cpu = quant_output.detach().cpu()
        
        if fp16_cpu.dim() == 4 and fp16_cpu.size(0) == 1:
            fp16_cpu = fp16_cpu.squeeze(0)
        if quant_cpu.dim() == 4 and quant_cpu.size(0) == 1:
            quant_cpu = quant_cpu.squeeze(0)
        
        self.fp16_output[timestep].append(fp16_cpu)
        self.quant_output[timestep].append(quant_cpu)
    
    def get_stacked_data(self) -> Dict:
        """
        Stack collected UNet outputs.
        
        Returns:
            Dict with:
                - fp16_output: {timestep: tensor [N, C, H, W]}
                - quant_output: {timestep: tensor [N, C, H, W]}
                - timesteps: sorted list of timesteps
                - num_samples: total number of samples
        """
        fp16_dict = {}
        quant_dict = {}
        
        timesteps = sorted(self.fp16_output.keys())
        
        for t in timesteps:
            fp16_dict[t] = torch.stack(self.fp16_output[t])
            quant_dict[t] = torch.stack(self.quant_output[t])
        
        num_samples = len(self.fp16_output[timesteps[0]]) if timesteps else 0
        
        return {
            'fp16_output': fp16_dict,
            'quant_output': quant_dict,
            'timesteps': timesteps,
            'num_samples': num_samples,
        }


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


def load_prompts_from_mjhq(meta_path: str = None, num_samples: int = 5000, seed: int = 41) -> List[str]:
    """
    Load prompts from MJHQ-30K using stratified sampling.
    MJHQ-30K: 10 categories, samples 500 per category for 5000 total.
    
    Note: Uses different seed than evaluation (seed 41 vs evaluation's seed 42)
    to avoid train/test contamination.
    """
    # Download or load meta_data.json
    if not meta_path or not os.path.exists(meta_path):
        print("Downloading MJHQ-30K metadata from HF Hub...")
        meta_path = download_mjhq_metadata()
    
    metadata = load_mjhq_metadata(meta_path)
    
    # Group prompts by category
    prompts_by_category = defaultdict(list)
    for image_id, info in metadata.items():
        if isinstance(info, dict) and info.get('prompt', '').strip():
            prompts_by_category[info['category']].append(info['prompt'].strip())
    
    # Stratified sampling: equal samples per category
    random.seed(seed)
    samples_per_category = num_samples // 10
    
    formatted_prompts = []
    for category in sorted(prompts_by_category.keys()):
        sampled = random.sample(prompts_by_category[category], samples_per_category)
        formatted_prompts.extend(sampled)
    
    random.shuffle(formatted_prompts)
    print(f"Loaded {len(formatted_prompts)} prompts from MJHQ-30K ({samples_per_category} per category)")
    
    return formatted_prompts


def load_prompts_from_yaml(yaml_path: str, max_samples: int = -1, shuffle: bool = True) -> List[str]:
    """
    Load prompts from YAML file (Deepcompressor format).
    
    YAML format:
        '0000': 'prompt text 1'
        '0001': 'prompt text 2'
        ...
    
    Args:
        yaml_path: Path to YAML file with {filename: prompt} mapping
        max_samples: Maximum number of prompts to load (-1 for all)
        shuffle: Whether to shuffle prompts before selecting
        
    Returns:
        List of prompt strings
    """
    with open(yaml_path, 'r') as f:
        meta = yaml.safe_load(f)
    
    # Get all prompts
    names = list(meta.keys())
    
    # Optionally limit and shuffle
    if max_samples > 0 and len(names) > max_samples:
        if shuffle:
            random.Random(0).shuffle(names)  # Use fixed seed for reproducibility
        names = names[:max_samples]
        names = sorted(names)  # Sort for consistency
    
    # Extract prompts
    prompts = [meta[name] for name in names]
    return prompts


def load_prompts(prompt_file: str = None, max_samples: int = -1, use_mjhq: bool = False, seed: int = None, mjhq_seed: int = None) -> List[str]:
    """
    Load prompts for image generation.
    
    Supports four formats:
    1. MJHQ-30K dataset (use_mjhq=True)
    2. YAML file (.yaml or .yml) - Deepcompressor format {filename: prompt}
    3. Text file (.txt) - One prompt per line
    4. None - Use default prompts
    
    Args:
        prompt_file: Optional path to prompt file or MJHQ meta_data.json
        max_samples: Maximum number of prompts to load from YAML (-1 for all)
        use_mjhq: Whether to load from MJHQ-30K dataset
        seed: Deprecated, use mjhq_seed instead
        mjhq_seed: Random seed specifically for MJHQ sampling (should be same across all GPUs)
        
    Returns:
        List of prompt strings
    """
    if use_mjhq:
        # Use mjhq_seed if provided, otherwise fallback to seed (for backward compatibility)
        actual_seed = mjhq_seed if mjhq_seed is not None else (seed if seed is not None else 41)
        print(f"Loading prompts from MJHQ-30K dataset (seed={actual_seed})...")
        prompts = load_prompts_from_mjhq(meta_path=prompt_file, num_samples=max_samples, seed=actual_seed)
    elif prompt_file and os.path.exists(prompt_file):
        ext = os.path.splitext(prompt_file)[1].lower()
        
        if ext in ['.yaml', '.yml']:
            print(f"Loading prompts from YAML: {prompt_file}")
            prompts = load_prompts_from_yaml(prompt_file, max_samples=max_samples)
            print(f"Loaded {len(prompts)} prompts from YAML")
        else:
            print(f"Loading prompts from text file: {prompt_file}")
            with open(prompt_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(prompts)} prompts from text file")
    else:
        print("Using default prompts")
        prompts = [
            "A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
            "A photo of a cat sitting on a wooden table.",
            "An oil painting of a sunset over mountains.",
            "A portrait of a young woman with red hair.",
            "A futuristic city with flying cars.",
            "A close-up photo of a flower with morning dew.",
            "A dog playing in a park.",
            "An underwater scene with colorful fish.",
            "A steaming cup of coffee on a rustic table.",
            "A vintage car on an empty desert road.",
        ]
    
    return prompts


def manual_denoising_loop(
    unet_fp16,
    unet_quant,
    vae,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    scheduler,
    prompt: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    device: str = "cuda",
) -> Dict[int, tuple]:
    """
    Run manual denoising loop and collect UNet outputs from both UNets.
    
    Args:
        unet_fp16: FP16 UNet model
        unet_quant: Quantized UNet model
        vae, text_encoder, etc: Pipeline components
        prompt: Text prompt
        seed: Random seed
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale
        device: Device to run on
        
    Returns:
        Dict mapping timestep to (fp16_output, quant_output) tuples
    """
    # Encode prompt
    # For SDXL, we need both text encoders
    with torch.no_grad():
        # Tokenize with both tokenizers
        text_inputs_1 = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs_2 = tokenizer_2(
            prompt,
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        # Encode with both text encoders
        prompt_embeds_1 = text_encoder(
            text_inputs_1.input_ids.to(device),
            output_hidden_states=True,
        )
        prompt_embeds_2 = text_encoder_2(
            text_inputs_2.input_ids.to(device),
            output_hidden_states=True,
        )
        
        # Pool and concatenate (SDXL-specific)
        pooled_prompt_embeds = prompt_embeds_2[0]
        prompt_embeds_1 = prompt_embeds_1.hidden_states[-2]
        prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
        
        # For SDXL-Turbo, we also need add_text_embeds and add_time_ids
        # Prepare added conditioning (simplified for SDXL-Turbo)
        add_text_embeds = pooled_prompt_embeds
        
        # Original size, crops coords, target size for SDXL
        original_size = (1024, 1024)
        crops_coords_top_left = (0, 0)
        target_size = (1024, 1024)
        
        add_time_ids = torch.tensor([
            list(original_size) + list(crops_coords_top_left) + list(target_size)
        ], dtype=torch.float32).to(device)
    
    # Keep dtype aligned with FP16 UNet / MixDQ quantized UNet.
    prompt_embeds_fp16 = prompt_embeds.to(torch.float16)
    add_text_embeds_fp16 = add_text_embeds.to(torch.float16)
    add_time_ids_fp16 = add_time_ids.to(torch.float16)
    
    # Set timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    
    # Initialize latent
    height = width = 64  # SDXL latent size for 1024x1024
    num_channels = 4
    shape = (1, num_channels, height, width)
    
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    latents = latents * scheduler.init_noise_sigma
    
    # Collect UNet outputs
    output_pairs = {}
    
    # Denoising loop
    for i, t in enumerate(timesteps):
        # Scale latent
        latent_model_input = scheduler.scale_model_input(latents, t)
        
        # Convert latent to FP16 for UNet inference
        latent_model_input_fp16 = latent_model_input.to(torch.float16)
        
        # Call both UNets on the SAME x_t
        with torch.no_grad():
            # FP16 UNet
            output_fp16 = unet_fp16(
                latent_model_input_fp16,
                t,
                encoder_hidden_states=prompt_embeds_fp16,
                added_cond_kwargs={"text_embeds": add_text_embeds_fp16, "time_ids": add_time_ids_fp16},
                return_dict=False,
            )[0]
            
            # Quantized UNet
            output_quant = unet_quant(
                latent_model_input_fp16,
                t,
                encoder_hidden_states=prompt_embeds_fp16,
                added_cond_kwargs={"text_embeds": add_text_embeds_fp16, "time_ids": add_time_ids_fp16},
                return_dict=False,
            )[0]
        
        # Convert UNet outputs back to Float32 for storage and scheduler
        output_fp16_f32 = output_fp16.float()
        output_quant_f32 = output_quant.float()
        
        # Store output pair
        timestep_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
        output_pairs[timestep_int] = (output_fp16_f32.clone(), output_quant_f32.clone())
        
        # Advance latent using FP16 output (to maintain clean trajectory)
        latents = scheduler.step(output_fp16_f32, t, latents, return_dict=False)[0]
    
    return output_pairs


def collect_statistics(
    num_samples: int,
    output_dir: str,
    prompt_file: str = None,
    noise_seed_start: int = 0,
    num_inference_steps: int = 4,
    guidance_scale: float = 0.0,
    use_mjhq: bool = True,
    rank: int = 0,
    world_size: int = 1,
    device: torch.device = None,
    total_samples: int = None,  # Add total samples for correct prompt loading
    mjhq_prompt_sample_seed: int = None,  # Separate seed for MJHQ sampling (same across all GPUs)
    quant_backend: str = "hf",  # ["hf", "qdiff"]
    mixdq_config_path: str = None,  # qdiff backend only
    mixdq_ckpt_path: str = None,  # qdiff backend only
    w_bit: int = 8,  # hf backend only
    a_bit: int = 8,  # hf backend only
    config_weight_mp_path: str = None,  # qdiff backend only (author inference style)
    config_act_mp_path: str = None,  # qdiff backend only (author inference style)
    act_protect_path: str = None,  # qdiff backend only (author inference style)
):
    """
    Collect output statistics from FP16 and quantized UNets on identical x_t.
    
    Args:
        num_samples: Number of samples to collect (per GPU)
        output_dir: Output directory for statistics
        prompt_file: Path to prompt file (YAML or TXT) or MJHQ meta_data.json
        noise_seed_start: Starting seed for initial noise generation (different per GPU)
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale (0.0 for SDXL-Turbo)
        use_mjhq: Whether to use MJHQ-30K dataset
        rank: GPU rank for distributed processing
        world_size: Total number of GPUs
        device: Device to use
        total_samples: Total samples across all GPUs (for correct prompt loading)
        mjhq_prompt_sample_seed: Random seed for MJHQ prompt sampling (same across all GPUs to ensure consistent stratified sampling)
    """
    print("="*60)
    print("Q-Drift Statistics Collection")
    print("="*60)
    print(f"Target samples: {num_samples}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Noise seed start: {noise_seed_start}")
    if use_mjhq:
        print(f"Dataset: MJHQ-30K (prompt sample seed {mjhq_prompt_sample_seed}, different from evaluation seed 42)")
    print("="*60)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load prompts
    # Use total_samples if provided (multi-GPU), otherwise use num_samples
    max_samples_to_load = total_samples if total_samples is not None else num_samples
    prompts = load_prompts(prompt_file, max_samples=max_samples_to_load, use_mjhq=use_mjhq, mjhq_seed=mjhq_prompt_sample_seed)
    
    # Split prompts across GPUs
    if world_size > 1:
        samples_per_gpu = num_samples
        start_idx = rank * samples_per_gpu
        end_idx = start_idx + samples_per_gpu
        prompts = prompts[start_idx:end_idx]
        print(f"Rank {rank}: Processing {len(prompts)} prompts (indices {start_idx}-{end_idx})")
    print(f"\nLoaded {len(prompts)} prompts")
    
    # Load models
    print("\n" + "="*60)
    print("Loading models...")
    print("="*60)
    
    model_id = "stabilityai/sdxl-turbo"

    print("\nLoading FP16 pipeline...")
    fp16_pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)
    print("✓ FP16 pipeline loaded")

    if quant_backend == "hf":
        # Use the authors' released HuggingFace custom pipeline.
        if w_bit != 8 or a_bit != 8:
            raise ValueError(
                f"quant_backend=hf only supports int8 quantization in the authors' pipeline (got w_bit={w_bit}, a_bit={a_bit}). "
                "Use w_bit=8,a_bit=8 or switch to quant_backend=qdiff with a compatible ckpt/config."
            )
        print(f"\nLoading MixDQ HuggingFace pipeline and quantizing UNet (W{w_bit}A{a_bit})...")
        try:
            import mixdq_extension._C  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "mixdq-extension is required for quant_backend=hf. "
                "Install: `pip install -i https://pypi.org/simple/ mixdq-extension`"
            ) from e

        quant_pipe = DiffusionPipeline.from_pretrained(
            model_id,
            custom_pipeline="nics-efc/MixDQ",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        quant_pipe.quantize_unet(w_bit=w_bit, a_bit=a_bit, bos=True)
        quant_pipe = quant_pipe.to(device)
        unet_quant = quant_pipe.unet
        print("✓ Quantized UNet ready (HF MixDQ)")
    elif quant_backend == "qdiff":
        # Optional fallback: algorithm-level MixDQ (qdiff) + local PTQ ckpt.pth.
        if mixdq_config_path is None or mixdq_ckpt_path is None:
            raise ValueError("For quant_backend=qdiff, pass --mixdq_base_path or both --mixdq_config/--mixdq_ckpt")

        from qdiff.models.quant_model import QuantModel
        from qdiff.utils import get_model, load_quant_params

        mixdq_cfg = OmegaConf.load(mixdq_config_path)
        model_id = mixdq_cfg.model.model_id

        print("\nLoading MixDQ quantized UNet (qdiff + ckpt.pth)...")
        quant_base_unet = get_model(
            mixdq_cfg.model,
            fp16=True,
            return_pipe=False,
            convert_model_for_quant=True,
        )

        wq_params = mixdq_cfg.quant.weight.quantizer
        aq_params = mixdq_cfg.quant.activation.quantizer
        if mixdq_cfg.get("mixed_precision", False):
            wq_params["mixed_precision"] = mixdq_cfg.mixed_precision
            aq_params["mixed_precision"] = mixdq_cfg.mixed_precision

        unet_quant = QuantModel(
            model=quant_base_unet,
            weight_quant_params=wq_params,
            act_quant_params=aq_params,
        ).cuda().eval().half()
        unet_quant.set_quant_state(True, True)
        load_quant_params(unet_quant, mixdq_ckpt_path, dtype=torch.float16)
        if config_weight_mp_path and config_act_mp_path and act_protect_path:
            print("\nApplying MixDQ mixed-precision configs + act_protect (author inference style)...")
            _apply_author_mp_and_protect(unet_quant, config_weight_mp_path, config_act_mp_path, act_protect_path)
        else:
            # Fallback: uniform refactor bitwidth (power-of-two only).
            if w_bit is not None:
                if w_bit not in (2, 4, 8, 16):
                    raise ValueError(
                        f"quant_backend=qdiff only supports power-of-two bitwidth refactor with this codebase (2/4/8/16). Got w_bit={w_bit}."
                    )
                unet_quant.set_layer_bit(model=unet_quant, n_bit=w_bit, quant_level="reset", bit_type="weight")
            if a_bit is not None:
                if a_bit not in (2, 4, 8, 16):
                    raise ValueError(
                        f"quant_backend=qdiff only supports power-of-two bitwidth refactor with this codebase (2/4/8/16). Got a_bit={a_bit}."
                    )
                unet_quant.set_layer_bit(model=unet_quant, n_bit=a_bit, quant_level="reset", bit_type="act")
        unet_quant.cuda()
        print("✓ Quantized UNet ready (qdiff)")
    else:
        raise ValueError(f"Unknown --quant_backend: {quant_backend} (expected: hf|qdiff)")

    # Extract components
    unet_fp16 = fp16_pipeline.unet
    vae = fp16_pipeline.vae
    text_encoder = fp16_pipeline.text_encoder
    text_encoder_2 = fp16_pipeline.text_encoder_2
    tokenizer = fp16_pipeline.tokenizer
    tokenizer_2 = fp16_pipeline.tokenizer_2
    
    # Create scheduler (Q-Drift scheduler; correction disabled during statistics collection).
    scheduler = EulerAncestralQDriftScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        mu_dict_path=None,
        cov_dict_path=None,
        log_dir=None,
        bias_scale=0.0,
        qdrift_scale=0.0,
    )
    
    # Collector
    collector = OutputCollector()
    
    # Collect statistics
    print("\n" + "="*60)
    print("Collecting output pairs...")
    print("="*60)
    
    samples_collected = 0
    prompt_idx = 0
    
    pbar = tqdm(total=num_samples, desc="Samples")
    
    while samples_collected < num_samples:
        # Get prompt (cycle through prompts)
        prompt = prompts[prompt_idx % len(prompts)]
        current_seed = noise_seed_start + samples_collected
        
        # Run manual denoising loop
        output_pairs = manual_denoising_loop(
            unet_fp16=unet_fp16,
            unet_quant=unet_quant,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
            prompt=prompt,
            seed=current_seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            device=device,
        )
        
        # Add to collector
        for timestep, (output_fp16, output_quant) in output_pairs.items():
            collector.add_pair(timestep, output_fp16, output_quant)
        
        samples_collected += 1
        prompt_idx += 1
        pbar.update(1)
    
    pbar.close()
    
    # Stack and save
    print("\n" + "="*60)
    print("Saving statistics...")
    print("="*60)
    
    data = collector.get_stacked_data()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data
    if world_size > 1:
        output_path = os.path.join(output_dir, f"data_output_pairs_rank{rank}.pth")
    else:
        output_path = os.path.join(output_dir, "data_output_pairs.pth")
    torch.save(data, output_path)
    
    print(f"\n✓ Statistics saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Samples collected: {data['num_samples']}")
    print(f"  Timesteps: {data['timesteps']}")
    
    for t in data['timesteps']:
        fp16_shape = data['fp16_output'][t].shape
        quant_shape = data['quant_output'][t].shape
        print(f"  Timestep {t}: FP16 {fp16_shape}, Quant {quant_shape}")
    
    print("\n" + "="*60)
    print("Collection complete!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Collect output statistics for Q-Drift Gaussian modeling"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of samples to collect (default: 5000 for MJHQ)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for statistics"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Path to prompt file (YAML or TXT) or MJHQ meta_data.json. If not provided, uses default prompts."
    )
    parser.add_argument(
        "--noise_seed_start",
        type=int,
        default=0,
        help="Starting seed for initial noise generation (default: 0, will use seeds from 0 to num_samples-1)"
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
        help="Guidance scale (default: 0.0 for SDXL-Turbo)"
    )
    parser.add_argument(
        "--use_mjhq",
        action="store_true",
        help="Use MJHQ-30K dataset for prompts"
    )
    parser.add_argument(
        "--mjhq_prompt_sample_seed",
        type=int,
        default=41,
        help="Random seed for MJHQ prompt sampling (default: 41, different from evaluation seed 42)"
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
        help="Weight mixed-precision bit config (author inference style; requires qdiff ckpt)",
    )
    parser.add_argument(
        "--config_act_mp",
        type=str,
        default=default_a_mp,
        help="Activation mixed-precision bit config (author inference style; requires qdiff ckpt)",
    )
    parser.add_argument(
        "--act_protect",
        type=str,
        default=default_protect,
        help="Path to act_protect list (.pt) to keep extremely sensitive layers as FP16",
    )
    parser.add_argument(
        "--disable_mp",
        action="store_true",
        help="Disable author-style mixed precision and use uniform w_bit/a_bit refactor instead.",
    )
    args = parser.parse_args()

    mixdq_config_path = None
    mixdq_ckpt_path = None
    if args.quant_backend == "qdiff":
        if args.mixdq_base_path is not None:
            mixdq_config_path = os.path.join(args.mixdq_base_path, "config.yaml")
            mixdq_ckpt_path = os.path.join(args.mixdq_base_path, "ckpt.pth")
        else:
            mixdq_config_path = args.mixdq_config
            mixdq_ckpt_path = args.mixdq_ckpt

        if mixdq_config_path is None or mixdq_ckpt_path is None:
            raise ValueError("For quant_backend=qdiff, provide --mixdq_base_path or both --mixdq_config and --mixdq_ckpt")
        if not os.path.exists(mixdq_config_path):
            raise FileNotFoundError(f"MixDQ config not found: {mixdq_config_path}")
        if not os.path.exists(mixdq_ckpt_path):
            raise FileNotFoundError(f"MixDQ ckpt not found: {mixdq_ckpt_path}")
    
    # Setup device (multi-GPU support via torchrun)
    # torchrun automatically sets RANK, LOCAL_RANK, and WORLD_SIZE environment variables
    torch.distributed.init_process_group(backend="nccl")
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    # Check if output file already exists (only rank 0 checks, then broadcast)
    final_output_path = os.path.join(args.output_dir, "data_output_pairs.pth")
    should_skip = False
    
    if rank == 0:
        if os.path.exists(final_output_path):
            should_skip = True
            print("="*60)
            print("⏭️  SKIPPING: Statistics collection")
            print("="*60)
            print(f"Output file already exists: {final_output_path}")
            print("To re-run statistics collection, delete this file first.")
            print("="*60)
    
    # Broadcast skip decision to all ranks
    if world_size > 1:
        should_skip_tensor = torch.tensor([1 if should_skip else 0], dtype=torch.int, device=device)
        torch.distributed.broadcast(should_skip_tensor, src=0)
        should_skip = bool(should_skip_tensor.item())
    
    # All ranks skip if decided
    if should_skip:
        torch.distributed.barrier()
        return
    
    # Split samples across GPUs
    num_samples_per_gpu = args.num_samples // world_size
    start_idx = rank * num_samples_per_gpu
    
    # Adjust for last GPU to handle remainder
    if rank == world_size - 1:
        num_samples_per_gpu = args.num_samples - start_idx
    
    print(f"Rank {rank}/{world_size}: Processing {num_samples_per_gpu} samples (starting from {start_idx})")
    
    collect_statistics(
        num_samples=num_samples_per_gpu,
        output_dir=args.output_dir,
        prompt_file=args.prompt_file,
        noise_seed_start=args.noise_seed_start + start_idx,  # Different starting seed per GPU for noise generation
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        use_mjhq=args.use_mjhq,
        rank=rank,
        world_size=world_size,
        device=device if world_size > 1 else None,
        total_samples=args.num_samples,  # Pass total samples for correct prompt loading
        mjhq_prompt_sample_seed=args.mjhq_prompt_sample_seed,  # Same seed for all GPUs for MJHQ prompt sampling
        quant_backend=args.quant_backend,
        mixdq_config_path=mixdq_config_path,
        mixdq_ckpt_path=mixdq_ckpt_path,
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        # author-style MP/protect (qdiff only)
        config_weight_mp_path=None if args.disable_mp else args.config_weight_mp,
        config_act_mp_path=None if args.disable_mp else args.config_act_mp,
        act_protect_path=None if args.disable_mp else args.act_protect,
    )
    
    # Merge rank-specific outputs (multi-GPU only)
    if world_size > 1:
        # Synchronize all ranks
        torch.distributed.barrier()
        
        # Merge on rank 0
        if rank == 0:
            print("\nMerging statistics from all ranks...")
            merged_data = {"fp16_output": {}, "quant_output": {}, "timesteps": [], "num_samples": 0}
            output_file = os.path.join(args.output_dir, "data_output_pairs.pth")
            
            for r in range(world_size):
                rank_path = os.path.join(args.output_dir, f"data_output_pairs_rank{r}.pth")
                if os.path.exists(rank_path):
                    data = torch.load(rank_path)
                    
                    # Merge timestep data
                    for t in data['timesteps']:
                        if t not in merged_data['fp16_output']:
                            merged_data['fp16_output'][t] = []
                            merged_data['quant_output'][t] = []
                        merged_data['fp16_output'][t].append(data['fp16_output'][t])
                        merged_data['quant_output'][t].append(data['quant_output'][t])
                    
                    merged_data['num_samples'] += data['num_samples']
                else:
                    print(f"⚠️  Warning: Rank {r} file not found: {rank_path}")
            
            # Stack concatenated tensors
            merged_data['timesteps'] = sorted(merged_data['fp16_output'].keys())
            for t in merged_data['timesteps']:
                merged_data['fp16_output'][t] = torch.cat(merged_data['fp16_output'][t], dim=0)
                merged_data['quant_output'][t] = torch.cat(merged_data['quant_output'][t], dim=0)
            
            # Save merged file
            torch.save(merged_data, output_file)
            print(f"✓ Merged statistics saved to: {output_file}")
            print(f"  Total samples: {merged_data['num_samples']}")
            print(f"  Timesteps: {merged_data['timesteps']}")
            
            # Clean up rank files after successful merge
            print("\nCleaning up rank files...")
            for r in range(world_size):
                rank_path = os.path.join(args.output_dir, f"data_output_pairs_rank{r}.pth")
                if os.path.exists(rank_path):
                    os.remove(rank_path)
                    print(f"  ✓ Removed data_output_pairs_rank{r}.pth")
            print("✓ Rank files cleaned up")


if __name__ == "__main__":
    main()
