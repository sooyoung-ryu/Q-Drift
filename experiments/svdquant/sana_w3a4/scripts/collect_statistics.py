"""
Collect Sana transformer outputs for Q-Drift correction.

We collect paired transformer noise predictions (BF16 reference vs quantized)
on the *same* latents x_t at each timestep, to estimate per-timestep channel-wise
Gaussian models used by Q-Drift schedulers.

Design notes:
  - Collects transformer outputs (epsilon / noise prediction), not latents.
  - Avoids trajectory divergence by updating latents using the BF16 output only.
  - Supports classifier-free guidance (CFG) when guidance_scale > 1.
  - Stores output dictionaries keyed by the scheduler timestep value (float).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import torch
import yaml
from diffusers import DPMSolverMultistepScheduler, SanaPipeline
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from deepcompressor_sana_loader import apply_deepcompressor_patches, load_quant_transformer

DEFAULT_BASE_MODEL = "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
DEFAULT_QUANT_MODEL_DIR = str((Path(__file__).resolve().parents[1] / "model" / "transformer_w3a4").resolve())
DEFAULT_NEGATIVE_PROMPT = ""

# MJHQ-30K constants
MJHQ_REPO_ID = "playgroundai/MJHQ-30K"
MJHQ_META_FILENAME = "meta_data.json"


class OutputCollector:
    """Collect transformer outputs from bf16 and quantized models."""

    def __init__(self):
        self.fp16_output: Dict[float, List[torch.Tensor]] = {}
        self.quant_output: Dict[float, List[torch.Tensor]] = {}

    def add_pair(self, timestep: float, fp16_output: torch.Tensor, quant_output: torch.Tensor) -> None:
        t = float(timestep)
        if t not in self.fp16_output:
            self.fp16_output[t] = []
            self.quant_output[t] = []

        self.fp16_output[t].append(fp16_output.detach().cpu())
        self.quant_output[t].append(quant_output.detach().cpu())

    def get_stacked_data(self) -> Dict[str, Any]:
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


def download_mjhq_metadata() -> str:
    return hf_hub_download(repo_id=MJHQ_REPO_ID, filename=MJHQ_META_FILENAME, repo_type="dataset")


def load_mjhq_metadata(meta_path: str) -> Dict[str, Dict[str, Any]]:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompts_from_mjhq(meta_path: Optional[str], num_samples: int, seed: int) -> List[str]:
    if not meta_path or not os.path.exists(meta_path):
        print("Downloading MJHQ-30K metadata from HF Hub...")
        meta_path = download_mjhq_metadata()

    metadata = load_mjhq_metadata(meta_path)

    prompts_by_category: DefaultDict[str, List[str]] = defaultdict(list)
    for _image_id, info in metadata.items():
        if isinstance(info, dict) and (info.get("prompt") or "").strip():
            prompts_by_category[str(info["category"])].append(info["prompt"].strip())

    random.seed(seed)
    samples_per_category = num_samples // 10
    prompts: List[str] = []
    for category in sorted(prompts_by_category.keys()):
        prompts.extend(random.sample(prompts_by_category[category], samples_per_category))

    random.shuffle(prompts)
    print(f"Loaded {len(prompts)} prompts from MJHQ-30K ({samples_per_category} per category)")
    return prompts


def load_prompts_from_yaml(yaml_path: str, max_samples: int = -1, shuffle: bool = True) -> List[str]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    names = list(meta.keys())
    if max_samples > 0 and len(names) > max_samples:
        if shuffle:
            random.Random(0).shuffle(names)
        names = sorted(names[:max_samples])

    return [meta[name] for name in names]


def load_prompts(
    prompt_file: Optional[str],
    *,
    max_samples: int,
    use_mjhq: bool,
    mjhq_seed: int,
) -> List[str]:
    if use_mjhq:
        print(f"Loading prompts from MJHQ-30K dataset (seed={mjhq_seed})...")
        return load_prompts_from_mjhq(meta_path=prompt_file, num_samples=max_samples, seed=mjhq_seed)

    if prompt_file and os.path.exists(prompt_file):
        ext = os.path.splitext(prompt_file)[1].lower()
        if ext in {".yaml", ".yml"}:
            print(f"Loading prompts from YAML: {prompt_file}")
            prompts = load_prompts_from_yaml(prompt_file, max_samples=max_samples)
        else:
            print(f"Loading prompts from text file: {prompt_file}")
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts")
        return prompts

    print("Using default prompts")
    return [
        "A cute 🐼 eating 🎋, ink drawing style",
        "A cinematic photo of a cat sitting on a wooden table, photorealistic.",
        "An oil painting of a sunset over mountains, vibrant colors.",
        "A futuristic city with flying cars at night, neon lights.",
        "An underwater scene with colorful fish and coral reef.",
    ]


def prepare_conditioning(
    *,
    pipe: SanaPipeline,
    prompt: str | List[str],
    negative_prompt: str | List[str],
    device: torch.device,
    guidance_scale: float,
    max_sequence_length: int = 300,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    do_cfg = guidance_scale > 1.0
    (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = pipe.encode_prompt(
        prompt=prompt,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        device=device,
        clean_caption=False,
        max_sequence_length=max_sequence_length,
        complex_human_instruction=None,
    )

    if do_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

    return prompt_embeds, prompt_attention_mask, do_cfg


def manual_denoising_loop(
    *,
    pipe: SanaPipeline,
    transformer_fp16,
    transformer_quant,
    scheduler,
    prompt: str | List[str],
    seed: int | List[int],
    num_inference_steps: int,
    guidance_scale: float,
    device: torch.device,
    height: int,
    width: int,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
) -> Dict[float, Tuple[torch.Tensor, torch.Tensor]]:
    prompts = prompt if isinstance(prompt, list) else [prompt]
    seeds = seed if isinstance(seed, list) else [seed]
    if len(prompts) != len(seeds):
        raise ValueError(f"prompt/seed batch size mismatch: {len(prompts)} vs {len(seeds)}")
    batch_size = len(prompts)
    negative_prompts = [negative_prompt] * batch_size
    prompt_embeds, prompt_attention_mask, do_cfg = prepare_conditioning(
        pipe=pipe,
        prompt=prompts,
        negative_prompt=negative_prompts,
        device=device,
        guidance_scale=guidance_scale,
    )

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    generators = [torch.Generator(device=device).manual_seed(int(s)) for s in seeds]
    latent_channels = int(transformer_fp16.config.in_channels)
    latents = pipe.prepare_latents(
        batch_size,
        latent_channels,
        height,
        width,
        torch.float32,
        device,
        generators if batch_size > 1 else generators[0],
        latents=None,
    )
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generators[0], eta=0.0)
    transformer_dtype = transformer_fp16.dtype

    output_pairs: Dict[float, Tuple[torch.Tensor, torch.Tensor]] = {}

    for t in timesteps:
        latent_model_input = torch.cat([latents] * 2, dim=0) if do_cfg else latents
        # Match diffusers pipelines: some schedulers require scaling the model input.
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        timestep = t.expand(latent_model_input.shape[0])
        timestep = timestep * float(transformer_fp16.config.timestep_scale)

        with torch.no_grad():
            fp16_noise = transformer_fp16(
                latent_model_input.to(dtype=transformer_dtype),
                encoder_hidden_states=prompt_embeds.to(dtype=transformer_dtype),
                encoder_attention_mask=prompt_attention_mask,
                timestep=timestep,
                return_dict=False,
                attention_kwargs=None,
            )[0].float()
            quant_noise = transformer_quant(
                latent_model_input.to(dtype=transformer_dtype),
                encoder_hidden_states=prompt_embeds.to(dtype=transformer_dtype),
                encoder_attention_mask=prompt_attention_mask,
                timestep=timestep,
                return_dict=False,
                attention_kwargs=None,
            )[0].float()

        if do_cfg:
            fp_uncond, fp_text = fp16_noise.chunk(2)
            fp16_noise = fp_uncond + guidance_scale * (fp_text - fp_uncond)

            q_uncond, q_text = quant_noise.chunk(2)
            quant_noise = q_uncond + guidance_scale * (q_text - q_uncond)

        # Learned sigma (keep epsilon only).
        if int(transformer_fp16.config.out_channels) // 2 == latent_channels:
            fp16_noise = fp16_noise.chunk(2, dim=1)[0]
            quant_noise = quant_noise.chunk(2, dim=1)[0]

        timestep_key = float(t.item()) if isinstance(t, torch.Tensor) else float(t)
        output_pairs[timestep_key] = (fp16_noise.clone().detach(), quant_noise.clone().detach())

        # Update latents using *reference* output to keep x_t identical across models.
        latents = scheduler.step(fp16_noise, t, latents, **extra_step_kwargs, return_dict=False)[0]

    return output_pairs


def rank_index_range(total_samples: int, rank: int, world_size: int) -> Tuple[int, int]:
    """Return the half-open global-index range assigned to one rank."""
    if total_samples < 0:
        raise ValueError("total_samples must be non-negative")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} is outside world_size {world_size}")
    base = total_samples // world_size
    remainder = total_samples % world_size
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return start, end


def _shard_stem(rank: int, shard_id: int) -> str:
    return f"data_output_pairs_rank{rank}_shard{shard_id:05d}"


def _shard_paths(output_dir: str, rank: int, shard_id: int) -> Tuple[Path, Path]:
    out = Path(output_dir)
    stem = _shard_stem(rank, shard_id)
    return out / f"{stem}.pth", out / f"{stem}.json"


def _cleanup_incomplete_rank_shards(output_dir: str, rank: int) -> None:
    """Remove own-rank shard files that are unsafe for resume/fit after a crash."""
    out = Path(output_dir)
    if not out.exists():
        return

    for tmp_path in sorted(out.glob(f"data_output_pairs_rank{rank}_shard*.tmp*")):
        try:
            tmp_path.unlink()
            print(f"Removed temporary shard file: {tmp_path}")
        except FileNotFoundError:
            pass

    for pth_path in sorted(out.glob(f"data_output_pairs_rank{rank}_shard*.pth")):
        meta_path = pth_path.with_suffix(".json")
        remove_pair = False
        reason = ""
        if not meta_path.exists():
            remove_pair = True
            reason = "missing sidecar"
        else:
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)
                if not metadata.get("complete", False):
                    remove_pair = True
                    reason = "incomplete sidecar"
            except (OSError, json.JSONDecodeError):
                remove_pair = True
                reason = "unreadable sidecar"
        if not remove_pair:
            continue
        for candidate in (pth_path, meta_path):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
        print(f"Removed incomplete shard ({reason}): {pth_path}")


def _load_completed_sharded_indices(
    output_dir: str,
    rank: int,
    *,
    prompts: List[str] | None = None,
    noise_seed_start: int | None = None,
    total_samples: int | None = None,
    expected_settings: Dict[str, object] | None = None,
) -> set[int]:
    _cleanup_incomplete_rank_shards(output_dir, rank)
    completed: set[int] = set()
    out = Path(output_dir)
    if not out.exists():
        return completed
    for meta_path in sorted(out.glob("data_output_pairs_rank*_shard*.json")):
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not metadata.get("complete", False):
            continue
        pth_path = meta_path.with_suffix(".pth")
        if not pth_path.exists():
            continue

        if total_samples is not None and int(metadata.get("num_samples_total", -1)) != int(total_samples):
            raise ValueError(f"Shard total-sample metadata does not match current run: {meta_path}")
        if expected_settings is not None and metadata.get("settings") is not None:
            settings = metadata["settings"]
            for key, expected_value in expected_settings.items():
                if settings.get(key) != expected_value:
                    raise ValueError(f"Shard settings metadata does not match current run for {key}: {meta_path}")

        global_indices = [int(idx) for idx in metadata.get("global_indices", [])]
        seeds = [int(seed) for seed in metadata.get("seeds", [])]
        shard_prompts = list(metadata.get("prompts", []))
        if not (len(global_indices) == len(seeds) == len(shard_prompts) == int(metadata.get("num_samples", -1))):
            raise ValueError(f"Incomplete shard sidecar metadata: {meta_path}")
        if noise_seed_start is not None:
            expected_seeds = [int(noise_seed_start) + idx for idx in global_indices]
            if seeds != expected_seeds:
                raise ValueError(f"Shard seed metadata does not match current run: {meta_path}")
        if prompts is not None:
            expected_prompts = [prompts[idx] for idx in global_indices]
            if shard_prompts != expected_prompts:
                raise ValueError(f"Shard prompt metadata does not match current run: {meta_path}")
        completed.update(global_indices)
    return completed


def _save_sharded_pairs(
    *,
    output_dir: str,
    rank: int,
    shard_id: int,
    collector: OutputCollector,
    global_indices: List[int],
    seeds: List[int],
    prompts: List[str],
    total_samples: int,
    world_size: int,
    settings: Dict[str, object] | None = None,
) -> None:
    if not global_indices:
        return
    data = collector.get_stacked_data()
    data["global_indices"] = [int(i) for i in global_indices]
    data["seeds"] = [int(s) for s in seeds]
    data["num_samples_total"] = int(total_samples)
    pth_path, meta_path = _shard_paths(output_dir, rank, shard_id)
    pth_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_pth = pth_path.with_name(f"{pth_path.name}.tmp.{os.getpid()}")
    tmp_meta = meta_path.with_name(f"{meta_path.name}.tmp.{os.getpid()}")
    for tmp in (tmp_pth, tmp_meta):
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass

    torch.save(data, tmp_pth)
    os.replace(tmp_pth, pth_path)
    metadata = {
        "complete": True,
        "rank": int(rank),
        "world_size": int(world_size),
        "shard_id": int(shard_id),
        "num_samples": int(data["num_samples"]),
        "num_samples_total": int(total_samples),
        "global_indices": [int(i) for i in global_indices],
        "seeds": [int(s) for s in seeds],
        "prompts": prompts,
        "data_file": pth_path.name,
        "settings": settings or {},
    }
    try:
        with tmp_meta.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, sort_keys=True)
            f.write("\n")
        os.replace(tmp_meta, meta_path)
    except Exception:
        try:
            pth_path.unlink()
        except FileNotFoundError:
            pass
        try:
            tmp_meta.unlink()
        except FileNotFoundError:
            pass
        raise
    print(f"✓ Saved shard: {pth_path} ({data['num_samples']} samples)")

def collect_statistics(
    *,
    num_samples: int,
    output_dir: str,
    prompt_file: Optional[str],
    noise_seed_start: int,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    use_mjhq: bool,
    mjhq_prompt_sample_seed: int,
    rank: int,
    world_size: int,
    device: torch.device,
    total_samples: Optional[int],
    base_model: str,
    quant_model_dir: str,
) -> None:
    print("=" * 60)
    print("Sana Q-Drift Statistics Collection")
    print("=" * 60)
    print(f"Base model:        {base_model}")
    print(f"Quant model dir:   {quant_model_dir}")
    print(f"Target samples:    {num_samples}")
    print(f"Inference steps:   {num_inference_steps}")
    print(f"Guidance scale:    {guidance_scale}")
    print(f"Resolution:        {height}x{width}")
    print(f"Noise seed start:  {noise_seed_start}")
    if use_mjhq:
        print(f"Dataset:           MJHQ-30K (prompt seed {mjhq_prompt_sample_seed})")
    print("=" * 60)

    max_samples_to_load = total_samples if total_samples is not None else num_samples
    prompts = load_prompts(
        prompt_file,
        max_samples=max_samples_to_load,
        use_mjhq=use_mjhq,
        mjhq_seed=mjhq_prompt_sample_seed,
    )

    if world_size > 1:
        start_idx = rank * num_samples
        end_idx = start_idx + num_samples
        prompts = prompts[start_idx:end_idx]
        print(f"Rank {rank}: Processing {len(prompts)} prompts (indices {start_idx}-{end_idx})")

    print("\n" + "=" * 60)
    print("Loading models...")
    print("=" * 60)

    print("\nLoading BF16 reference pipeline...")
    pipe = SanaPipeline.from_pretrained(
        base_model,
        variant="bf16",
        torch_dtype=torch.bfloat16,
    ).to(device)
    pipe.vae.to(torch.bfloat16)
    if pipe.text_encoder is not None:
        pipe.text_encoder.to(torch.bfloat16)
    print("✓ BF16 pipeline ready")

    print("\nLoading quantized transformer...")
    transformer_quant = load_quant_transformer(
        base_model_id=base_model,
        quant_model_dir=quant_model_dir,
        device=device,
        torch_dtype=torch.bfloat16,
    )
    print("✓ Quantized transformer ready")

    transformer_fp16 = pipe.transformer
    apply_deepcompressor_patches(model=transformer_fp16, shift_activations=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(base_model, subfolder="scheduler")
    scheduler = pipe.scheduler

    collector = OutputCollector()
    pbar = tqdm(total=num_samples, desc="Samples")

    samples_collected = 0
    prompt_idx = 0

    while samples_collected < num_samples:
        prompt = prompts[prompt_idx % len(prompts)]
        rank_offset = rank * num_samples if world_size > 1 else 0
        current_seed = int(noise_seed_start) + int(rank_offset) + int(samples_collected)

        output_pairs = manual_denoising_loop(
            pipe=pipe,
            transformer_fp16=transformer_fp16,
            transformer_quant=transformer_quant,
            scheduler=scheduler,
            prompt=prompt,
            seed=current_seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            device=device,
            height=height,
            width=width,
        )

        for timestep, (fp16_eps, quant_eps) in output_pairs.items():
            collector.add_pair(float(timestep), fp16_eps, quant_eps)

        samples_collected += 1
        prompt_idx += 1
        pbar.update(1)

    pbar.close()

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



def collect_statistics_sharded(
    *,
    global_start: int,
    global_end: int,
    output_dir: str,
    prompt_file: Optional[str],
    noise_seed_start: int,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    use_mjhq: bool,
    mjhq_prompt_sample_seed: int,
    rank: int,
    world_size: int,
    device: torch.device,
    total_samples: int,
    base_model: str,
    quant_model_dir: str,
    shard_size: int,
    batch_size: int,
) -> None:
    print("=" * 60)
    print("Sana Q-Drift Sharded Statistics Collection")
    print("=" * 60)
    print(f"Global range: [{global_start}, {global_end})")
    print(f"Shard size: {shard_size}")
    print(f"Batch size: {batch_size}")
    print(f"Noise seed start: {noise_seed_start}")
    print("=" * 60)

    prompts = load_prompts(
        prompt_file,
        max_samples=total_samples,
        use_mjhq=use_mjhq,
        mjhq_seed=mjhq_prompt_sample_seed,
    )
    if len(prompts) < total_samples:
        raise ValueError(f"Loaded {len(prompts)} prompts but need {total_samples}")

    expected_settings = {
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "height": int(height),
        "width": int(width),
        "use_mjhq": bool(use_mjhq),
        "mjhq_prompt_sample_seed": int(mjhq_prompt_sample_seed),
    }
    completed = _load_completed_sharded_indices(
        output_dir,
        rank,
        prompts=prompts,
        noise_seed_start=noise_seed_start,
        total_samples=total_samples,
        expected_settings=expected_settings,
    )
    assigned = list(range(global_start, global_end))
    pending = [idx for idx in assigned if idx not in completed]
    print(f"Rank {rank}/{world_size}: {len(completed)} completed, {len(pending)} pending")
    if not pending:
        return

    print("\nLoading BF16 reference pipeline...")
    pipe = SanaPipeline.from_pretrained(
        base_model,
        variant="bf16",
        torch_dtype=torch.bfloat16,
    ).to(device)
    pipe.vae.to(torch.bfloat16)
    if pipe.text_encoder is not None:
        pipe.text_encoder.to(torch.bfloat16)

    print("\nLoading quantized transformer...")
    transformer_quant = load_quant_transformer(
        base_model_id=base_model,
        quant_model_dir=quant_model_dir,
        device=device,
        torch_dtype=torch.bfloat16,
    )

    transformer_fp16 = pipe.transformer
    apply_deepcompressor_patches(model=transformer_fp16, shift_activations=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(base_model, subfolder="scheduler")
    scheduler = pipe.scheduler

    shard_id = 0
    while _shard_paths(output_dir, rank, shard_id)[0].exists() or _shard_paths(output_dir, rank, shard_id)[1].exists():
        shard_id += 1

    pbar = tqdm(total=len(pending), desc=f"Rank {rank} samples")
    shard_collector = OutputCollector()
    shard_indices: List[int] = []
    shard_seeds: List[int] = []
    shard_prompts: List[str] = []

    for batch_start in range(0, len(pending), batch_size):
        batch_indices = pending[batch_start:batch_start + batch_size]
        batch_prompts = [prompts[idx] for idx in batch_indices]
        batch_seeds = [int(noise_seed_start) + int(idx) for idx in batch_indices]
        output_pairs = manual_denoising_loop(
            pipe=pipe,
            transformer_fp16=transformer_fp16,
            transformer_quant=transformer_quant,
            scheduler=scheduler,
            prompt=batch_prompts,
            seed=batch_seeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            device=device,
            height=height,
            width=width,
        )
        for local_idx, global_idx in enumerate(batch_indices):
            for timestep, (fp16_eps, quant_eps) in output_pairs.items():
                shard_collector.add_pair(
                    float(timestep),
                    fp16_eps[local_idx:local_idx + 1],
                    quant_eps[local_idx:local_idx + 1],
                )
            shard_indices.append(int(global_idx))
            shard_seeds.append(batch_seeds[local_idx])
            shard_prompts.append(batch_prompts[local_idx])
            pbar.update(1)

            if len(shard_indices) >= shard_size:
                _save_sharded_pairs(
                    output_dir=output_dir,
                    rank=rank,
                    shard_id=shard_id,
                    collector=shard_collector,
                    global_indices=shard_indices,
                    seeds=shard_seeds,
                    prompts=shard_prompts,
                    total_samples=total_samples,
                    world_size=world_size,
                    settings=expected_settings,
                )
                shard_id += 1
                shard_collector = OutputCollector()
                shard_indices = []
                shard_seeds = []
                shard_prompts = []

    if shard_indices:
        _save_sharded_pairs(
            output_dir=output_dir,
            rank=rank,
            shard_id=shard_id,
            collector=shard_collector,
            global_indices=shard_indices,
            seeds=shard_seeds,
            prompts=shard_prompts,
            total_samples=total_samples,
            world_size=world_size,
            settings=expected_settings,
        )
    pbar.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect output statistics for Sana Q-Drift Gaussian modeling")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for calibration outputs")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of samples to collect")
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Prompt file (YAML/TXT) or MJHQ meta_data.json; if omitted uses default prompts",
    )
    parser.add_argument(
        "--noise_seed_start",
        type=int,
        default=42,
        help="Base seed for initial noise (per-sample seed = noise_seed_start + global_sample_idx)",
    )
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=4.5, help="CFG guidance scale")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--use_mjhq", action="store_true", help="Use MJHQ-30K dataset for prompts")
    parser.add_argument("--mjhq_prompt_sample_seed", type=int, default=42, help="Random seed for MJHQ prompt sampling")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL, help="HF repo id / path for Sana model")
    parser.add_argument(
        "--quant_model_dir",
        type=str,
        default=DEFAULT_QUANT_MODEL_DIR,
        help="DeepCompressor PTQ checkpoint directory (contains model.pt, wgts.pt, ...).",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank (set by torch.distributed.launch/torchrun; ignored when not using DDP)",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Actual prompt batch size for sharded collection")
    parser.add_argument("--shard_size", type=int, default=0, help="If >0, save resumable raw-pair shards with this many samples per shard")
    args = parser.parse_args()

    # Setup device (multi-GPU support via torchrun)
    # torchrun sets RANK, LOCAL_RANK, and WORLD_SIZE environment variables.
    torch.distributed.init_process_group(backend="nccl")
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.shard_size < 0:
        raise ValueError("--shard_size must be non-negative")

    if args.shard_size > 0:
        start_idx, end_idx = rank_index_range(args.num_samples, rank, world_size)
        print(f"Rank {rank}/{world_size}: sharded collection for global indices [{start_idx}, {end_idx})")
        collect_statistics_sharded(
            global_start=start_idx,
            global_end=end_idx,
            output_dir=args.output_dir,
            prompt_file=args.prompt_file,
            noise_seed_start=args.noise_seed_start,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            use_mjhq=args.use_mjhq,
            mjhq_prompt_sample_seed=args.mjhq_prompt_sample_seed,
            rank=rank,
            world_size=world_size,
            device=device,
            total_samples=args.num_samples,
            base_model=args.base_model,
            quant_model_dir=args.quant_model_dir,
            shard_size=args.shard_size,
            batch_size=args.batch_size,
        )
        torch.distributed.barrier()
        return

    final_output_path = os.path.join(args.output_dir, "data_output_pairs.pth")
    should_skip = False
    if rank == 0 and os.path.exists(final_output_path):
        should_skip = True
        print("=" * 60)
        print("SKIPPING: Statistics collection")
        print("=" * 60)
        print(f"Output file already exists: {final_output_path}")
        print("To re-run, delete this file first.")
        print("=" * 60)

    if world_size > 1:
        should_skip_tensor = torch.tensor([1 if should_skip else 0], dtype=torch.int, device=device)
        torch.distributed.broadcast(should_skip_tensor, src=0)
        should_skip = bool(should_skip_tensor.item())

    if should_skip:
        torch.distributed.barrier()
        return

    num_samples_per_gpu = args.num_samples // world_size
    start_idx = rank * num_samples_per_gpu
    if rank == world_size - 1:
        num_samples_per_gpu = args.num_samples - start_idx

    print(f"Rank {rank}/{world_size}: Processing {num_samples_per_gpu} samples (starting from {start_idx})")

    collect_statistics(
        num_samples=num_samples_per_gpu,
        output_dir=args.output_dir,
        prompt_file=args.prompt_file,
        noise_seed_start=args.noise_seed_start,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        use_mjhq=args.use_mjhq,
        mjhq_prompt_sample_seed=args.mjhq_prompt_sample_seed,
        rank=rank,
        world_size=world_size,
        device=device,
        total_samples=args.num_samples,
        base_model=args.base_model,
        quant_model_dir=args.quant_model_dir,
    )

    if world_size > 1:
        torch.distributed.barrier()
        if rank == 0:
            print("\nMerging statistics from all ranks...")
            merged_data: Dict[str, Any] = {"fp16_output": {}, "quant_output": {}, "timesteps": [], "num_samples": 0}
            output_file = os.path.join(args.output_dir, "data_output_pairs.pth")

            for r in range(world_size):
                rank_path = os.path.join(args.output_dir, f"data_output_pairs_rank{r}.pth")
                if not os.path.exists(rank_path):
                    print(f"⚠️  Warning: Rank {r} file not found: {rank_path}")
                    continue
                data = torch.load(rank_path)

                for t in data["timesteps"]:
                    if t not in merged_data["fp16_output"]:
                        merged_data["fp16_output"][t] = []
                        merged_data["quant_output"][t] = []
                    merged_data["fp16_output"][t].append(data["fp16_output"][t])
                    merged_data["quant_output"][t].append(data["quant_output"][t])

                merged_data["num_samples"] += int(data["num_samples"])

            merged_data["timesteps"] = sorted(merged_data["fp16_output"].keys())
            for t in merged_data["timesteps"]:
                merged_data["fp16_output"][t] = torch.cat(merged_data["fp16_output"][t], dim=0)
                merged_data["quant_output"][t] = torch.cat(merged_data["quant_output"][t], dim=0)

            torch.save(merged_data, output_file)
            print(f"✓ Merged statistics saved to: {output_file}")
            print(f"  Total samples: {merged_data['num_samples']}")
            print(f"  Timesteps: {merged_data['timesteps']}")

            print("\nCleaning up rank files...")
            for r in range(world_size):
                rank_path = os.path.join(args.output_dir, f"data_output_pairs_rank{r}.pth")
                if os.path.exists(rank_path):
                    os.remove(rank_path)
                    print(f"  ✓ Removed data_output_pairs_rank{r}.pth")
            print("✓ Rank files cleaned up")


if __name__ == "__main__":
    main()
