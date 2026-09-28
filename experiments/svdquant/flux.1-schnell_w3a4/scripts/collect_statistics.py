"""
Collect Transformer outputs for Q-Drift correction on Flux.1-schnell.

This script runs a manual denoising loop that calls both the FP16 and quantized
Transformers on *identical* latents (x_t) so that we can measure the local 
quantization error (quant_output - fp16_output) at every timestep.

IMPORTANT (baseline trajectory):
This variant advances the latent trajectory using the quantized baseline model
(i.e., it follows the baseline trajectory), and evaluates fp16/quant outputs on
those baseline latents.

Key properties:
- Collects Transformer velocity outputs, not latents
- Uses the same x_t for FP16 and quantized Transformers
- Uses flow matching scheduler (not Euler discrete)
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
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift
from huggingface_hub import hf_hub_download

def _patch_flux_time_text_embed_forward(transformer: torch.nn.Module) -> None:
    """
    Make FLUX compatible across diffusers versions where `time_text_embed.forward`
    may or may not accept a `guidance` argument.

    Some envs end up with a `FluxTransformer2DModel.forward()` that calls
    `time_text_embed(timestep, guidance, pooled_projections)`, while the embedding
    module only supports `(timestep, pooled_projections)`.
    """
    time_text_embed = getattr(transformer, "time_text_embed", None)
    if time_text_embed is None or not hasattr(time_text_embed, "forward"):
        return

    try:
        import inspect
        import types

        orig_forward = time_text_embed.forward
        sig = inspect.signature(orig_forward)
    except Exception:
        return

    # `orig_forward` is a bound method, so `self` is not present in the signature.
    # Old signature: (timestep, pooled_projections)  -> 2 params
    # New signature: (timestep, guidance, pooled_projections) -> 3 params
    if len(sig.parameters) != 2:
        return

    def _forward_compat(self, timestep, *args, **kwargs):
        pooled = None

        if len(args) == 1:
            pooled = args[0]
        elif len(args) >= 2:
            # Called as (timestep, guidance, pooled_projections, ...)
            pooled = args[1]
        else:
            pooled = kwargs.get("pooled_projections", kwargs.get("pooled_prompt_embeds", None))

        if pooled is None:
            return orig_forward(timestep, *args, **kwargs)
        return orig_forward(timestep, pooled)

    time_text_embed.forward = types.MethodType(_forward_compat, time_text_embed)


from tqdm import tqdm

DEFAULT_BASE_MODEL = "black-forest-labs/FLUX.1-schnell"
DEFAULT_NEGATIVE_PROMPT = ""
DEFAULT_QUANT_MODEL_DIR = str((Path(__file__).resolve().parents[1] / "model" / "transformer_w3a4_g64").resolve())

# Allow importing `experiments/svdquant/deepcompressor_loader.py`.
_QDRIFT_DIR = Path(__file__).resolve().parents[2]
if str(_QDRIFT_DIR) not in sys.path:
    sys.path.insert(0, str(_QDRIFT_DIR))
from deepcompressor_loader import load_quant_flux_transformer  # noqa: E402

# MJHQ-30K constants
MJHQ_REPO_ID = "playgroundai/MJHQ-30K"
MJHQ_META_FILENAME = "meta_data.json"


class OutputCollector:
    """Collect Transformer outputs from fp16 and quantized models."""

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
        actual_seed = mjhq_seed if mjhq_seed is not None else (seed if seed is not None else 41)
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
            "A cat holding a sign that says hello world",
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


def manual_denoising_loop(
    transformer_fp16,
    transformer_quant,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    scheduler,
    pipeline,
    prompt: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    device: torch.device,
    height: int = 1024,
    width: int = 1024,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Manual denoising loop for Flux.1-schnell to collect outputs at each timestep.
    """
    # Use pipeline's encode_prompt to get correct text_ids
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            max_sequence_length=512,
        )
    
    prompt_embeds = prompt_embeds.to(dtype=torch.bfloat16)
    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=torch.bfloat16)
    
    # Prepare latents and latent image IDs using pipeline method
    latent_channels = transformer_fp16.config.in_channels // 4
    generator = torch.Generator(device=device).manual_seed(seed)
    
    latents, latent_image_ids = pipeline.prepare_latents(
        batch_size=1,
        num_channels_latents=latent_channels,
        height=height,
        width=width,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
        latents=None,
    )
    
    # Calculate mu for FLUX's flow matching scheduler based on packed latents
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    
    # Use the standard scheduler (not Q-Drift) for statistics collection.
    #
    # IMPORTANT: Match FluxPipeline's default per-step timestep grid used during sampling.
    # For example, when num_inference_steps=4, FluxPipeline uses:
    #   [1000, 750, 500, 250]
    # whereas the scheduler's default grid can be:
    #   [1000, 667, 334, 1]
    # If the grids differ, Q-Drift will warn at inference time and fall back to nearest
    # available statistics (e.g. 750->667, 500->334).
    num_train_timesteps = int(getattr(scheduler.config, "num_train_timesteps", 1000))
    pipeline_timesteps = np.linspace(num_train_timesteps, 0, num_inference_steps + 1, dtype=np.float32)[:-1]
    try:
        scheduler.set_timesteps(
            num_inference_steps, device=device, mu=mu, timesteps=pipeline_timesteps.tolist()
        )
    except TypeError:
        # Backward compatibility with scheduler implementations that don't accept explicit timesteps.
        scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
    timesteps = scheduler.timesteps
    
    output_pairs: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    
    for i, t in enumerate(timesteps):
        # FLUX uses internal guidance, no need to concatenate latents
        latent_model_input = latents
        
        # Prepare timestep (following FluxPipeline implementation)
        timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)
        
        # Prepare guidance (FLUX uses internal guidance embedding).
        # Some diffusers builds only support the no-guidance path; also FLUX.1-schnell
        # typically runs with `guidance_scale=0.0`.
        guidance = None
        if guidance_scale is not None and float(guidance_scale) != 0.0:
            # FluxPipeline uses float32 for guidance
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latent_model_input.shape[0])
        
        with torch.no_grad():
            common_kwargs = dict(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )
            if guidance is not None:
                common_kwargs["guidance"] = guidance

            # FP16 transformer
            fp16_noise = transformer_fp16(**common_kwargs)[0]
            
            # Quantized transformer
            quant_noise = transformer_quant(**common_kwargs)[0]
        
        # FLUX handles guidance internally, no manual CFG needed
        
        timestep_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
        output_pairs[timestep_int] = (fp16_noise.clone().detach(), quant_noise.clone().detach())
        
        # Advance the latent trajectory using the quantized baseline output (baseline trajectory),
        # while still collecting fp16/quant outputs on the same x_t.
        latents = scheduler.step(fp16_noise, t, latents, return_dict=False)[0]
    
    return output_pairs


def collect_statistics(
    num_samples: int,
    output_dir: str,
    base_model: str,
    quant_model_dir: str | None,
    prompt_file: str = None,
    noise_seed_start: int = 0,
    num_inference_steps: int = 20,
    guidance_scale: float = 3.5,
    height: int = 1024,
    width: int = 1024,
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
    print("Flux.1-schnell Q-Drift Statistics Collection")
    print("=" * 60)
    print(f"Target samples: {num_samples}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Noise seed start: {noise_seed_start}")
    print(f"Resolution: {height}x{width}")
    if use_mjhq:
        print(f"Dataset: MJHQ-30K (prompt sample seed {mjhq_prompt_sample_seed}, different from evaluation seed 42)")
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
    fp16_pipeline = FluxPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
    ).to(device)
    print("✓ FP16 pipeline ready")

    print("\nLoading quantized Transformer...")
    transformer_quant = load_quant_flux_transformer(
        base_model_id=base_model,
        quant_model_dir=quant_model_dir,
        device=device,
        torch_dtype=torch.bfloat16,
    )
    print(f"✓ Quantized Transformer ready: {quant_model_dir}")

    transformer_fp16 = fp16_pipeline.transformer
    _patch_flux_time_text_embed_forward(transformer_fp16)
    _patch_flux_time_text_embed_forward(transformer_quant)
    text_encoder = fp16_pipeline.text_encoder
    text_encoder_2 = fp16_pipeline.text_encoder_2
    tokenizer = fp16_pipeline.tokenizer
    tokenizer_2 = fp16_pipeline.tokenizer_2
    scheduler = fp16_pipeline.scheduler

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
        current_seed = noise_seed_start + samples_collected

        output_pairs = manual_denoising_loop(
            transformer_fp16=transformer_fp16,
            transformer_quant=transformer_quant,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
            pipeline=fp16_pipeline,
            prompt=prompt,
            seed=current_seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            device=device,
            height=height,
            width=width,
        )

        for timestep, (fp16_out, quant_out) in output_pairs.items():
            collector.add_pair(timestep, fp16_out, quant_out)

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
    parser = argparse.ArgumentParser(description="Collect output statistics for Flux.1-schnell Q-Drift")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of samples to collect (default: 5000 for MJHQ)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store statistics")
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
    parser.add_argument("--prompt_file", type=str, default=None, help="Optional prompt file (YAML or TXT) or path to MJHQ meta_data.json")
    parser.add_argument("--noise_seed_start", type=int, default=0, help="Starting seed for initial noise generation (default: 0, will use seeds 0 to num_samples-1)")
    parser.add_argument("--mjhq_prompt_sample_seed", type=int, default=41, help="Random seed for MJHQ prompt sampling (default: 41, different from evaluation seed 42)")
    parser.add_argument("--num_inference_steps", type=int, default=4, help="Denoising steps for Flux.1-schnell (default: 4)")
    parser.add_argument("--guidance_scale", type=float, default=0.0, help="Guidance scale (default: 0.0 for Flux.1-schnell, no CFG)")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--use_mjhq", action="store_true", help="Use MJHQ-30K dataset for prompts")
    parser.add_argument(
        "--shard_size",
        type=int,
        default=0,
        help="If positive, save calibration pairs as rank-local shards of this many samples and skip merged output.",
    )
    # Accept torch.distributed.launch style --local_rank without impacting env-based setup.
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank (ignored; we read LOCAL_RANK from the environment)",
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
        noise_seed_start=args.noise_seed_start + start_idx,  # Different starting seed per GPU for noise generation
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
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
