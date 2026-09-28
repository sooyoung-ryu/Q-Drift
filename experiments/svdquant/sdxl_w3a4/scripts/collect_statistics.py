"""
Collect UNet outputs for Q-Drift correction on SDXL (base model).

This script mirrors the SDXL-Turbo pipeline but targets the full SDXL model,
including classifier-free guidance. It runs a manual denoising loop that calls
both the FP16 and quantized UNets on identical latents (x_t) so that we can
measure the local quantization error (quant_output - fp16_output) at every
timestep without trajectory divergence.

Key properties:
- Collects UNet epsilon outputs, not latents
- Uses the same x_t for FP16 and quantized UNets
- Implements classifier-free guidance (CFG) when guidance_scale > 1
- Defines error as: error = quant_output - fp16_output
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler, UNet2DConditionModel
from huggingface_hub import hf_hub_download

from tqdm import tqdm

DEFAULT_BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_QUANT_MODEL_DIR = str((Path(__file__).resolve().parents[1] / "model" / "unet_w3a4_g64").resolve())
DEFAULT_NEGATIVE_PROMPT = ""

# Allow importing `experiments/svdquant/deepcompressor_loader.py`.
_QDRIFT_DIR = Path(__file__).resolve().parents[2]
if str(_QDRIFT_DIR) not in sys.path:
    sys.path.insert(0, str(_QDRIFT_DIR))
from deepcompressor_loader import load_quant_unet

# MJHQ-30K constants
MJHQ_REPO_ID = "playgroundai/MJHQ-30K"
MJHQ_META_FILENAME = "meta_data.json"


def _maybe_add_deepcompressor_to_syspath() -> None:
    here = Path(__file__).resolve()
    deepcompressor_root = here.parents[4] / "third_party" / "deepcompressor"
    if deepcompressor_root.is_dir() and str(deepcompressor_root) not in sys.path:
        sys.path.insert(0, str(deepcompressor_root))


def _load_quant_metadata(quant_model_dir: str) -> Dict:
    meta_path = Path(quant_model_dir) / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _apply_activation_quant_if_available(unet: UNet2DConditionModel, *, quant_model_dir: str) -> bool:
    """
    Apply activation quantization hooks if `act_quantizer_state.pt` exists in the quantized model directory.

    Note: activation quantization hooks are NOT serialized in diffusers `save_pretrained`, so this must be done
    at runtime after loading the UNet (and ideally after moving it to the target device).
    """
    act_state_path = Path(quant_model_dir) / "act_quantizer_state.pt"
    if not act_state_path.exists():
        return False

    try:
        _maybe_add_deepcompressor_to_syspath()
        import torch

        from deepcompressor.app.diffusion.dataset.calib import DiffusionCalibCacheLoaderConfig
        from deepcompressor.app.diffusion.quant import DiffusionQuantConfig, quantize_diffusion_activations
        from deepcompressor.app.diffusion.quant.quantizer.config import (
            DiffusionActivationQuantizerConfig,
            DiffusionWeightQuantizerConfig,
        )
        from deepcompressor.data.dtype import QuantDataType

        meta = _load_quant_metadata(quant_model_dir)
        act_dtype = str(meta.get("act_quant_dtype") or "sint4")
        w_dtype = str(meta.get("weight_quant_dtype") or "sint3")
        group_shape = meta.get("group_shape") or [1, 64, 1, 1, 1]
        group_shape = tuple(int(x) for x in group_shape)

        quant_config = DiffusionQuantConfig(
            wgts=DiffusionWeightQuantizerConfig(
                dtype=QuantDataType.from_str(w_dtype),
                group_shapes=(group_shape,),
                scale_dtypes=(None,),
                skips=[],
            ),
            ipts=DiffusionActivationQuantizerConfig(
                dtype=QuantDataType.from_str(act_dtype),
                group_shapes=((-1, -1, -1),),
                scale_dtypes=(None,),
                skips=[],
            ),
            opts=DiffusionActivationQuantizerConfig(
                dtype=QuantDataType.from_str(act_dtype),
                group_shapes=((-1, -1, -1),),
                scale_dtypes=(None,),
                skips=[],
            ),
            calib=DiffusionCalibCacheLoaderConfig(
                data="unused",
                num_samples=0,
                batch_size=1,
                path="",
                num_workers=0,
            ),
        )

        act_quantizer_state = torch.load(act_state_path, map_location="cpu")
        quantize_diffusion_activations(
            unet,
            quant_config,
            quantizer_state_dict=act_quantizer_state,
            orig_state_dict=None,
        )
        print(f"✓ Applied activation quantization hooks from: {act_state_path}")
        return True
    except Exception as e:
        print(f"⚠️  Failed to apply activation quantization hooks ({act_state_path}): {e}")
        return False


class OutputCollector:
    """Collect UNet outputs from fp16 and quantized models."""

    def __init__(self):
        self.fp16_output: Dict[int, List[torch.Tensor]] = {}
        self.quant_output: Dict[int, List[torch.Tensor]] = {}

    def add_pair(self, timestep: int, fp16_output: torch.Tensor, quant_output: torch.Tensor) -> None:
        if timestep not in self.fp16_output:
            self.fp16_output[timestep] = []
            self.quant_output[timestep] = []

        self.fp16_output[timestep].append(fp16_output.detach().cpu())
        self.quant_output[timestep].append(quant_output.detach().cpu())

    def get_stacked_data(self) -> Dict:
        timesteps = sorted(self.fp16_output.keys())
        fp16_dict = {t: torch.stack(self.fp16_output[t]) for t in timesteps}
        quant_dict = {t: torch.stack(self.quant_output[t]) for t in timesteps}
        num_samples = len(self.fp16_output[timesteps[0]]) if timesteps else 0

        return {
            "fp16_output": fp16_dict,
            "quant_output": quant_dict,
            "timesteps": timesteps,
            "num_samples": num_samples,
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


def load_prompts_from_mjhq(meta_path: str = None, num_samples: int = 5000, seed: int = 42) -> List[str]:
    """
    Load prompts from MJHQ-30K using stratified sampling.
    MJHQ-30K: 10 categories, samples 500 per category for 5000 total.
    
    Note: Uses seed 42 for reproducibility (same as evaluation).
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
    with open(yaml_path, "r") as f:
        meta = yaml.safe_load(f)

    names = list(meta.keys())
    if max_samples > 0 and len(names) > max_samples:
        if shuffle:
            random.Random(0).shuffle(names)
        names = sorted(names[:max_samples])

    return [meta[name] for name in names]


def load_prompts(prompt_file: str = None, max_samples: int = -1, use_mjhq: bool = False, seed: int = None, mjhq_seed: int = None) -> List[str]:
    if use_mjhq:
        # Use mjhq_seed if provided, otherwise fallback to seed (for backward compatibility)
        actual_seed = mjhq_seed if mjhq_seed is not None else (seed if seed is not None else 42)
        print(f"Loading prompts from MJHQ-30K dataset (seed={actual_seed})...")
        prompts = load_prompts_from_mjhq(meta_path=prompt_file, num_samples=max_samples, seed=actual_seed)
    elif prompt_file and os.path.exists(prompt_file):
        ext = os.path.splitext(prompt_file)[1].lower()
        if ext in {".yaml", ".yml"}:
            print(f"Loading prompts from YAML: {prompt_file}")
            prompts = load_prompts_from_yaml(prompt_file, max_samples=max_samples)
        else:
            print(f"Loading prompts from text file: {prompt_file}")
            with open(prompt_file, "r") as f:
                prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts")
    else:
        print("Using default prompts")
        prompts = [
            "A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
            "A photo of a cat sitting on a wooden table, photorealistic.",
            "An oil painting of a sunset over mountains, vibrant colors.",
            "A portrait of a young woman with red hair, studio lighting.",
            "A futuristic city with flying cars at night, neon lights.",
            "A close-up photo of a flower with morning dew, macro photography.",
            "A dog playing in a park, shallow depth of field.",
            "An underwater scene with colorful fish and coral reef.",
            "A steaming cup of coffee on a rustic table, warm lighting.",
            "A vintage car on an empty desert road, golden hour.",
        ]
    return prompts


def _encode_prompt(
    prompt: str,
    tokenizer,
    tokenizer_2,
    text_encoder,
    text_encoder_2,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    text_embeds_1 = text_encoder(text_inputs_1.input_ids.to(device), output_hidden_states=True)
    text_embeds_2 = text_encoder_2(text_inputs_2.input_ids.to(device), output_hidden_states=True)

    pooled = text_embeds_2[0]
    hidden_1 = text_embeds_1.hidden_states[-2]
    hidden_2 = text_embeds_2.hidden_states[-2]
    prompt_embeds = torch.cat([hidden_1, hidden_2], dim=-1)
    return prompt_embeds, pooled


def prepare_conditioning(
    prompt: str,
    negative_prompt: str,
    tokenizer,
    tokenizer_2,
    text_encoder,
    text_encoder_2,
    device: torch.device,
    guidance_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    do_cfg = guidance_scale > 1.0

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = _encode_prompt(
            prompt, tokenizer, tokenizer_2, text_encoder, text_encoder_2, device
        )
        negative_embeds, negative_pooled_embeds = _encode_prompt(
            negative_prompt, tokenizer, tokenizer_2, text_encoder, text_encoder_2, device
        )

        original_size = (1024, 1024)
        crops_coords_top_left = (0, 0)
        target_size = (1024, 1024)
        add_time_ids = torch.tensor(
            [list(original_size) + list(crops_coords_top_left) + list(target_size)],
            dtype=torch.float32,
            device=device,
        )

    prompt_embeds_bf16 = prompt_embeds.to(torch.bfloat16)
    negative_prompt_embeds_bf16 = negative_embeds.to(torch.bfloat16)

    add_text_embeds = pooled_prompt_embeds.to(torch.bfloat16)
    negative_add_text_embeds = negative_pooled_embeds.to(torch.bfloat16)

    add_time_ids_bf16 = add_time_ids.to(torch.bfloat16)
    negative_add_time_ids = add_time_ids_bf16.clone()

    if do_cfg:
        prompt_embeds_bf16 = torch.cat([negative_prompt_embeds_bf16, prompt_embeds_bf16], dim=0)
        add_text_embeds = torch.cat([negative_add_text_embeds, add_text_embeds], dim=0)
        add_time_ids_bf16 = torch.cat([negative_add_time_ids, add_time_ids_bf16], dim=0)

    return prompt_embeds_bf16, add_text_embeds, add_time_ids_bf16, do_cfg


def manual_denoising_loop(
    unet_fp16,
    unet_quant,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    scheduler,
    prompt: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    device: torch.device,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    prompt_embeds_bf16, add_text_embeds_bf16, add_time_ids_bf16, do_cfg = prepare_conditioning(
        prompt,
        negative_prompt,
        tokenizer,
        tokenizer_2,
        text_encoder,
        text_encoder_2,
        device,
        guidance_scale,
    )

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    latent_shape = (1, unet_fp16.in_channels, unet_fp16.sample_size, unet_fp16.sample_size)
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=torch.float32)
    latents = latents * scheduler.init_noise_sigma

    output_pairs: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    for t in timesteps:
        latent_model_input = latents
        if do_cfg:
            latent_model_input = torch.cat([latent_model_input, latent_model_input], dim=0)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        latent_model_input_bf16 = latent_model_input.to(torch.bfloat16)

        with torch.no_grad():
            fp16_noise = unet_fp16(
                latent_model_input_bf16,
                t,
                encoder_hidden_states=prompt_embeds_bf16,
                added_cond_kwargs={"text_embeds": add_text_embeds_bf16, "time_ids": add_time_ids_bf16},
                return_dict=False,
            )[0]
            quant_noise = unet_quant(
                latent_model_input_bf16,
                t,
                encoder_hidden_states=prompt_embeds_bf16,
                added_cond_kwargs={"text_embeds": add_text_embeds_bf16, "time_ids": add_time_ids_bf16},
                return_dict=False,
            )[0]

        def _apply_cfg(noise: torch.Tensor) -> torch.Tensor:
            noise = noise.to(torch.float32)
            if do_cfg:
                noise_uncond, noise_text = noise.chunk(2)
                noise = noise_uncond + guidance_scale * (noise_text - noise_uncond)
            return noise

        fp16_eps = _apply_cfg(fp16_noise)
        quant_eps = _apply_cfg(quant_noise)

        timestep_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
        output_pairs[timestep_int] = (fp16_eps.clone().detach(), quant_eps.clone().detach())

        latents = scheduler.step(fp16_eps, t, latents, return_dict=False)[0]

    return output_pairs


def collect_statistics(
    num_samples: int,
    output_dir: str,
    base_model: str,
    quant_model_dir: str,
    prompt_file: str = None,
    noise_seed_start: int = 0,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    use_mjhq: bool = False,
    rank: int = 0,
    world_size: int = 1,
    device: torch.device = None,
    total_samples: int = None,
    mjhq_prompt_sample_seed: int = None,
    shard_size: int = 0,
    global_start_idx: int = 0,
):
    print("=" * 60)
    print("SDXL Q-Drift Statistics Collection")
    print("=" * 60)
    print(f"Target samples: {num_samples}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Noise seed start: {noise_seed_start}")
    if use_mjhq:
        print(f"Dataset: MJHQ-30K (prompt sample seed {mjhq_prompt_sample_seed})")
    print("=" * 60)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    print("\n" + "=" * 60)
    print("Loading models...")
    print("=" * 60)

    print("\nLoading FP16 reference pipeline...")
    fp16_pipeline = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        variant="fp16",
    ).to(device)
    print("✓ FP16 pipeline ready")

    print("\nLoading quantized UNet...")
    unet_quant = load_quant_unet(
        base_model_id=base_model,
        quant_model_dir=quant_model_dir,
        device=device,
        torch_dtype=torch.bfloat16,
    )
    _apply_activation_quant_if_available(unet_quant, quant_model_dir=quant_model_dir)
    print("✓ Quantized UNet ready")

    unet_fp16 = fp16_pipeline.unet
    text_encoder = fp16_pipeline.text_encoder
    text_encoder_2 = fp16_pipeline.text_encoder_2
    tokenizer = fp16_pipeline.tokenizer
    tokenizer_2 = fp16_pipeline.tokenizer_2

    scheduler = EulerDiscreteScheduler.from_pretrained(
        base_model,
        subfolder="scheduler",
    )

    collector = OutputCollector()
    pbar = tqdm(total=num_samples, desc="Samples")

    samples_collected = 0
    prompt_idx = 0

    shard_size = max(int(shard_size or 0), 0)
    use_shards = shard_size > 0
    shard_id = 0
    shard_collected = 0
    shard_start_index = global_start_idx

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
        prompt = prompts[prompt_idx % len(prompts)]
        # IMPORTANT: do NOT reuse the same noise for every prompt/sample.
        # Reusing a fixed seed collapses latent diversity and can badly bias the
        # estimated error statistics (and downstream Q-Drift correction).
        #
        # Keep determinism while ensuring different noise per sample, and ensure
        # different ranks don't overlap in multi-GPU runs.
        rank_offset = global_start_idx if world_size > 1 else 0
        current_seed = noise_seed_start + rank_offset + samples_collected

        output_pairs = manual_denoising_loop(
            unet_fp16=unet_fp16,
            unet_quant=unet_quant,
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

        for timestep, (fp16_eps, quant_eps) in output_pairs.items():
            collector.add_pair(timestep, fp16_eps, quant_eps)

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

    data = collector.get_stacked_data()
    os.makedirs(output_dir, exist_ok=True)
    if world_size > 1:
        output_path = os.path.join(output_dir, f"data_output_pairs_rank{rank}.pth")
    else:
        output_path = os.path.join(output_dir, "data_output_pairs.pth")
    torch.save(data, output_path)

    print("\n" + "=" * 60)
    print("Collection complete!")
    print(f"Saved statistics to: {output_path}")
    print(f"Samples: {data['num_samples']}")
    print(f"Timesteps: {data['timesteps']}")
    print("=" * 60)


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
        help="Output directory for calibration outputs (data_output_pairs.pth)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="HF repo id or local directory containing the SDXL base model.",
    )
    parser.add_argument(
        "--quant_model_dir",
        type=str,
        default=DEFAULT_QUANT_MODEL_DIR,
        help="Directory containing the quantized UNet artifacts (expects subfolder `unet/`).",
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
        default=42,
        help="Seed for initial noise generation (default: 42, all samples use the same seed)"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of denoising steps (default: 30 for SDXL)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale (default: 7.5 for SDXL)"
    )
    parser.add_argument(
        "--use_mjhq",
        action="store_true",
        help="Use MJHQ-30K dataset for prompts"
    )
    parser.add_argument(
        "--mjhq_prompt_sample_seed",
        type=int,
        default=42,
        help="Random seed for MJHQ prompt sampling (default: 42)"
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=0,
        help="If positive, save calibration pairs as rank-local shards of this many samples and skip merged output.",
    )
    # Allow torch.distributed.launch style --local_rank without affecting logic.
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank (set by torch.distributed.launch; ignored because we use env vars)",
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
        base_model=args.base_model,
        quant_model_dir=args.quant_model_dir,
        prompt_file=args.prompt_file,
        noise_seed_start=args.noise_seed_start,  # Same seed for all GPUs and all samples
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
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
