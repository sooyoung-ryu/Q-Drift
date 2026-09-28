"""
Collect UNet outputs for Q-Drift gaussian modeling.

This script implements manual denoising loops to collect UNet output pairs from FP16 and quantized models.

SDXL-Turbo + Euler Ancestral sampling.
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
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import hf_hub_download

DEFAULT_BASE_MODEL = "stabilityai/sdxl-turbo"
DEFAULT_QUANT_MODEL_DIR = str((Path(__file__).resolve().parents[1] / "model" / "unet_w3a4_g64").resolve())

# Allow importing `experiments/svdquant/deepcompressor_loader.py`.
_QDRIFT_DIR = Path(__file__).resolve().parents[2]
if str(_QDRIFT_DIR) not in sys.path:
    sys.path.insert(0, str(_QDRIFT_DIR))
from deepcompressor_loader import load_quant_unet

# MJHQ-30K constants
MJHQ_REPO_ID = "playgroundai/MJHQ-30K"
MJHQ_META_FILENAME = "meta_data.json"


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


def _shard_output_path(output_dir: str, rank: int, shard_id: int) -> Path:
    return Path(output_dir) / f"data_output_pairs_rank{rank}_shard{shard_id:05d}.pth"


def _save_output_shard(collector: OutputCollector, output_dir: str, rank: int, shard_id: int, start_index: int, end_index: int) -> Path | None:
    data = collector.get_stacked_data()
    if data["num_samples"] == 0:
        return None

    os.makedirs(output_dir, exist_ok=True)
    output_path = _shard_output_path(output_dir, rank, shard_id)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(data, tmp_path)
    os.replace(tmp_path, output_path)

    metadata = {
        "rank": rank,
        "shard_id": shard_id,
        "num_samples": data["num_samples"],
        "start_index": start_index,
        "end_index": end_index,
        "timesteps": data["timesteps"],
    }
    meta_path = output_path.with_suffix(".json")
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    os.replace(meta_tmp, meta_path)
    return output_path


def _existing_shard_sample_count(
    output_dir: str,
    rank: int,
    shard_id: int,
    expected_start_index: int | None = None,
    expected_end_index: int | None = None,
) -> int | None:
    output_path = _shard_output_path(output_dir, rank, shard_id)
    if not output_path.exists():
        return None
    meta_path = output_path.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if expected_start_index is not None and metadata.get("start_index") != expected_start_index:
            raise RuntimeError(
                f"Existing shard {output_path} starts at {metadata.get('start_index')}; "
                f"expected {expected_start_index}. Delete it before resuming."
            )
        if expected_end_index is not None and metadata.get("end_index") != expected_end_index:
            raise RuntimeError(
                f"Existing shard {output_path} ends at {metadata.get('end_index')}; "
                f"expected {expected_end_index}. Delete it before resuming."
            )
        return int(metadata.get("num_samples", 0))
    data = torch.load(output_path, map_location="cpu")
    return int(data.get("num_samples", 0))

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
    
    # Convert embeddings to BFloat16 once (UNets expect BFloat16)
    prompt_embeds_bf16 = prompt_embeds.to(torch.bfloat16)
    add_text_embeds_bf16 = add_text_embeds.to(torch.bfloat16)
    add_time_ids_bf16 = add_time_ids.to(torch.bfloat16)
    
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
        
        # Convert latent to BFloat16 for UNet inference
        latent_model_input_bf16 = latent_model_input.to(torch.bfloat16)
        
        # Call both UNets on the SAME x_t
        with torch.no_grad():
            # FP16 UNet
            output_fp16 = unet_fp16(
                latent_model_input_bf16,
                t,
                encoder_hidden_states=prompt_embeds_bf16,
                added_cond_kwargs={"text_embeds": add_text_embeds_bf16, "time_ids": add_time_ids_bf16},
                return_dict=False,
            )[0]
            
            # Quantized UNet
            output_quant = unet_quant(
                latent_model_input_bf16,
                t,
                encoder_hidden_states=prompt_embeds_bf16,
                added_cond_kwargs={"text_embeds": add_text_embeds_bf16, "time_ids": add_time_ids_bf16},
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
    base_model_id: str = DEFAULT_BASE_MODEL,
    quant_model_dir: str = DEFAULT_QUANT_MODEL_DIR,
    use_mjhq: bool = True,
    rank: int = 0,
    world_size: int = 1,
    device: torch.device = None,
    total_samples: int = None,  # Add total samples for correct prompt loading
    mjhq_prompt_sample_seed: int = None,  # Separate seed for MJHQ sampling (same across all GPUs)
    shard_size: int = 0,
    global_start_idx: int = 0,
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
        base_model_id: HF repo id or local directory for SDXL-Turbo diffusers pipeline
        quant_model_dir: Path to W3A4 quantized UNet checkpoint directory (DeepCompressor)
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
        start_idx = global_start_idx
        end_idx = start_idx + num_samples
        prompts = prompts[start_idx:end_idx]
        print(f"Rank {rank}: Processing {len(prompts)} prompts (indices {start_idx}-{end_idx})")
    print(f"\nLoaded {len(prompts)} prompts")
    
    # Load models
    print("\n" + "="*60)
    print("Loading models...")
    print("="*60)
    
    # Load FP16 pipeline to get components
    print("\nLoading FP16 pipeline...")
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
    print("✓ FP16 pipeline loaded")
    
    # Load quantized UNet
    print("\nLoading quantized UNet...")
    unet_quant = load_quant_unet(
        base_model_id=base_model_id,
        quant_model_dir=quant_model_dir,
        device=device,
        torch_dtype=torch.bfloat16,
    )
    print("✓ Quantized UNet loaded")
    
    # Extract components
    unet_fp16 = fp16_pipeline.unet
    vae = fp16_pipeline.vae
    text_encoder = fp16_pipeline.text_encoder
    text_encoder_2 = fp16_pipeline.text_encoder_2
    tokenizer = fp16_pipeline.tokenizer
    tokenizer_2 = fp16_pipeline.tokenizer_2
    
    # Create scheduler
    scheduler = EulerAncestralDiscreteScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        timestep_spacing="trailing",
        prediction_type="epsilon",
    )
    
    # Collector
    collector = OutputCollector()
    
    # Collect statistics
    print("\n" + "="*60)
    print("Collecting output pairs...")
    print("="*60)
    
    samples_collected = 0
    prompt_idx = 0

    shard_size = max(int(shard_size or 0), 0)
    use_shards = shard_size > 0
    shard_id = 0
    shard_collected = 0
    shard_start_index = global_start_idx
    
    pbar = tqdm(total=num_samples, desc="Samples")
    
    while samples_collected < num_samples:

        if use_shards and shard_collected == 0:
            expected_shard_len = min(shard_size, num_samples - samples_collected)
            existing_count = _existing_shard_sample_count(
                output_dir,
                rank,
                shard_id,
                expected_start_index=shard_start_index,
                expected_end_index=shard_start_index + expected_shard_len,
            )
            if existing_count == expected_shard_len:
                print(f"Rank {rank}: Skipping existing shard {shard_id:05d} ({existing_count} samples)")
                samples_collected += expected_shard_len
                prompt_idx += expected_shard_len
                shard_start_index += expected_shard_len
                shard_id += 1
                pbar.update(expected_shard_len)
                continue
            if existing_count is not None:
                raise RuntimeError(
                    f"Existing shard {_shard_output_path(output_dir, rank, shard_id)} has {existing_count} samples; "
                    f"expected {expected_shard_len}. Delete it before resuming."
                )
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

        if use_shards:
            shard_collected += 1
            if shard_collected >= shard_size or samples_collected >= num_samples:
                saved_path = _save_output_shard(
                    collector=collector,
                    output_dir=output_dir,
                    rank=rank,
                    shard_id=shard_id,
                    start_index=shard_start_index,
                    end_index=shard_start_index + shard_collected,
                )
                if saved_path is not None:
                    print(f"Rank {rank}: Saved shard {shard_id:05d} to {saved_path}")
                collector = OutputCollector()
                shard_start_index += shard_collected
                shard_collected = 0
                shard_id += 1
    
    pbar.close()

    if use_shards:
        print(f"\nSharded collection complete for rank {rank}: {shard_id} shard(s) in {output_dir}")
        return
    
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
        "--shard_size",
        type=int,
        default=0,
        help="If positive, save calibration pairs as rank-local shards of this many samples and skip merged output.",
    )
    args = parser.parse_args()
    
    # Setup device (multi-GPU support via torchrun)
    # torchrun automatically sets RANK, LOCAL_RANK, and WORLD_SIZE environment variables
    torch.distributed.init_process_group(backend="nccl")
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    use_shards = args.shard_size is not None and args.shard_size > 0

    # Check if output file already exists (only rank 0 checks, then broadcast)
    final_output_path = os.path.join(args.output_dir, "data_output_pairs.pth")
    should_skip = False
    
    if rank == 0:
        if (not use_shards) and os.path.exists(final_output_path):
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
        base_model_id=args.base_model,
        quant_model_dir=args.quant_model_dir,
        use_mjhq=args.use_mjhq,
        rank=rank,
        world_size=world_size,
        device=device if world_size > 1 else None,
        total_samples=args.num_samples,  # Pass total samples for correct prompt loading
        mjhq_prompt_sample_seed=args.mjhq_prompt_sample_seed,  # Same seed for all GPUs for MJHQ prompt sampling
        shard_size=args.shard_size,
        global_start_idx=start_idx,
    )
    
    # Merge rank-specific outputs (multi-GPU only)
    if world_size > 1:
        # Synchronize all ranks
        torch.distributed.barrier()

        if use_shards:
            if rank == 0:
                print("Sharded collection complete; keeping rank-local shard files for pooled fitting.")
            return
        
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
