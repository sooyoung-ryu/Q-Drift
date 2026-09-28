"""
Collect PixArt-Sigma Transformer outputs for Q-Drift correction (W3A4 checkpoint).

This script runs a manual denoising loop that calls both the FP16 and quantized
transformers on identical latents (x_t) to measure local quantization error:

    error = quant_output - fp16_output

Collected outputs are saved as:
  - fp16_output[t] : (N, C, H, W)
  - quant_output[t]: (N, C, H, W)
for each timestep t.
"""

from __future__ import annotations

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
from diffusers import DPMSolverMultistepScheduler, PixArtSigmaPipeline
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from deepcompressor_pixart_loader import apply_deepcompressor_patches, load_quant_transformer


DEFAULT_BASE_MODEL = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
DEFAULT_QUANT_MODEL_DIR = str((Path(__file__).resolve().parents[1] / "model" / "transformer_w3a4").resolve())
DEFAULT_NEGATIVE_PROMPT = ""

# MJHQ-30K constants
MJHQ_REPO_ID = "playgroundai/MJHQ-30K"
MJHQ_META_FILENAME = "meta_data.json"


class OutputCollector:
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


def download_mjhq_metadata() -> str:
    return hf_hub_download(repo_id=MJHQ_REPO_ID, filename=MJHQ_META_FILENAME, repo_type="dataset")


def load_mjhq_metadata(meta_path: str) -> Dict[str, Dict]:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompts_from_mjhq(meta_path: str = None, num_samples: int = 5000, seed: int = 41) -> List[str]:
    if not meta_path or not os.path.exists(meta_path):
        print("Downloading MJHQ-30K metadata from HF Hub...")
        meta_path = download_mjhq_metadata()

    metadata = load_mjhq_metadata(meta_path)
    prompts_by_category = defaultdict(list)
    for _, info in metadata.items():
        if isinstance(info, dict) and info.get("prompt", "").strip():
            prompts_by_category[info["category"]].append(info["prompt"].strip())

    random.seed(seed)
    samples_per_category = num_samples // 10
    formatted_prompts = []
    for category in sorted(prompts_by_category.keys()):
        formatted_prompts.extend(random.sample(prompts_by_category[category], samples_per_category))
    random.shuffle(formatted_prompts)
    print(f"Loaded {len(formatted_prompts)} prompts from MJHQ-30K ({samples_per_category} per category)")
    return formatted_prompts


def load_prompts(prompt_file: str = None, max_samples: int = -1, use_mjhq: bool = False, mjhq_seed: int = None) -> List[str]:
    if use_mjhq:
        return load_prompts_from_mjhq(meta_path=prompt_file, num_samples=max_samples, seed=mjhq_seed or 41)
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        return prompts[:max_samples] if max_samples > 0 else prompts
    return [
        "A cinematic shot of a baby raccoon wearing an intricate italian priest robe.",
        "A photo of a cat sitting on a wooden table, photorealistic.",
    ]


def _build_base_scheduler(
    *, base_config: dict, dpm_algorithm_type: str, dpm_solver_order: int, dpm_solver_type: str
) -> DPMSolverMultistepScheduler:
    return DPMSolverMultistepScheduler.from_config(
        base_config,
        algorithm_type=dpm_algorithm_type,
        solver_order=int(dpm_solver_order),
        solver_type=dpm_solver_type,
    )


def manual_denoising_loop(
    *,
    transformer_fp16,
    transformer_quant,
    pipeline: PixArtSigmaPipeline,
    scheduler: DPMSolverMultistepScheduler,
    prompt: str | List[str],
    negative_prompt: str,
    seed: int | List[int],
    num_inference_steps: int,
    guidance_scale: float,
    device: torch.device,
    height: int,
    width: int,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    prompts = prompt if isinstance(prompt, list) else [prompt]
    seeds = seed if isinstance(seed, list) else [seed]
    if len(prompts) != len(seeds):
        raise ValueError(f"prompt/seed batch size mismatch: {len(prompts)} vs {len(seeds)}")
    batch_size = len(prompts)
    do_cfg = guidance_scale > 1.0
    model_dtype = next(transformer_fp16.parameters()).dtype
    negative_prompts = [negative_prompt] * batch_size

    with torch.no_grad():
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = pipeline.encode_prompt(
            prompt=prompts,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=negative_prompts,
            num_images_per_prompt=1,
            device=device,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            prompt_attention_mask=None,
            negative_prompt_attention_mask=None,
            clean_caption=True,
            max_sequence_length=300,
        )
        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
        # Some diffusers schedulers (e.g., DPM-Solver) may operate in fp32 internally and
        # can upcast latents; keep transformer inputs in the transformer's dtype.
        prompt_embeds = prompt_embeds.to(dtype=model_dtype)

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    latent_channels = transformer_fp16.config.in_channels
    generators = [torch.Generator(device=device).manual_seed(int(s)) for s in seeds]
    latents = pipeline.prepare_latents(
        batch_size=batch_size,
        num_channels_latents=latent_channels,
        height=height,
        width=width,
        dtype=model_dtype,
        device=device,
        generator=generators if batch_size > 1 else generators[0],
        latents=None,
    )
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generators[0], eta=0.0)
    added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

    output_pairs: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for t in timesteps:
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        latent_model_input = latent_model_input.to(dtype=model_dtype)
        current_timestep = t
        if not torch.is_tensor(current_timestep):
            current_timestep = torch.tensor([current_timestep], device=latent_model_input.device)
        elif len(current_timestep.shape) == 0:
            current_timestep = current_timestep[None].to(latent_model_input.device)
        current_timestep = current_timestep.expand(latent_model_input.shape[0])

        with torch.no_grad():
            out_fp16 = transformer_fp16(
                latent_model_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=current_timestep,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            out_quant = transformer_quant(
                latent_model_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=current_timestep,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

        if do_cfg:
            fp16_uncond, fp16_text = out_fp16.chunk(2)
            out_fp16 = fp16_uncond + guidance_scale * (fp16_text - fp16_uncond)
            quant_uncond, quant_text = out_quant.chunk(2)
            out_quant = quant_uncond + guidance_scale * (quant_text - quant_uncond)

        if transformer_fp16.config.out_channels // 2 == latent_channels:
            out_fp16 = out_fp16.chunk(2, dim=1)[0]
            out_quant = out_quant.chunk(2, dim=1)[0]

        out_fp16_f32 = out_fp16.float()
        out_quant_f32 = out_quant.float()
        timestep_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
        output_pairs[timestep_int] = (out_fp16_f32.clone(), out_quant_f32.clone())

        latents = scheduler.step(out_fp16_f32, t, latents, **extra_step_kwargs, return_dict=False)[0]
        latents = latents.to(dtype=model_dtype)

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
    prompt_file: str,
    noise_seed_start: int,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    base_model_id: str,
    quant_model_dir: str,
    use_mjhq: bool,
    mjhq_prompt_sample_seed: int,
    dpm_algorithm_type: str,
    dpm_solver_order: int,
    dpm_solver_type: str,
    rank: int,
    world_size: int,
    device: torch.device,
    total_samples: int,
    start_idx: int,
) -> None:
    print("=" * 60)
    print("PixArt-Sigma Q-Drift Statistics Collection")
    print("=" * 60)
    print(f"Target samples: {num_samples}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Size: {height}x{width}")
    print(f"DPM: algorithm_type={dpm_algorithm_type} solver_order={dpm_solver_order} solver_type={dpm_solver_type}")
    if use_mjhq:
        print(f"Dataset: MJHQ-30K (prompt sample seed {mjhq_prompt_sample_seed})")
    print("=" * 60)

    prompts = load_prompts(
        prompt_file,
        max_samples=total_samples,
        use_mjhq=use_mjhq,
        mjhq_seed=mjhq_prompt_sample_seed,
    )

    end_idx = start_idx + num_samples
    if world_size > 1:
        prompts = prompts[start_idx:end_idx]
        print(f"Rank {rank}: Processing {len(prompts)} prompts (indices {start_idx}-{end_idx})")

    fp16_pipeline = PixArtSigmaPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16).to(device)
    fp16_pipeline.transformer.eval()
    fp16_pipeline.text_encoder.eval()
    fp16_pipeline.vae.eval()

    # Apply the same structural patch to FP16 transformer (should be math-preserving).
    apply_deepcompressor_patches(model=fp16_pipeline.transformer, shift_activations=False)

    quant_transformer = load_quant_transformer(
        base_model_id=base_model_id,
        quant_model_dir=quant_model_dir,
        device=device,
        torch_dtype=torch.float16,
    )

    scheduler = _build_base_scheduler(
        base_config=fp16_pipeline.scheduler.config,
        dpm_algorithm_type=dpm_algorithm_type,
        dpm_solver_order=dpm_solver_order,
        dpm_solver_type=dpm_solver_type,
    )

    collector = OutputCollector()
    pbar = tqdm(total=num_samples, desc="Samples")

    samples_collected = 0
    while samples_collected < num_samples:
        prompt = prompts[samples_collected % len(prompts)]
        rank_offset = rank * num_samples if world_size > 1 else 0
        seed = noise_seed_start + rank_offset + samples_collected

        pairs = manual_denoising_loop(
            transformer_fp16=fp16_pipeline.transformer,
            transformer_quant=quant_transformer,
            pipeline=fp16_pipeline,
            scheduler=scheduler,
            prompt=prompt,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            device=device,
            height=height,
            width=width,
        )
        for timestep, (fp16_out, q_out) in pairs.items():
            collector.add_pair(timestep, fp16_out, q_out)

        samples_collected += 1
        pbar.update(1)

    pbar.close()
    data = collector.get_stacked_data()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"data_output_pairs_rank{rank}.pth" if world_size > 1 else "data_output_pairs.pth")
    torch.save(data, out_path)
    print(f"✓ Saved: {out_path}")



def collect_statistics_sharded(
    *,
    global_start: int,
    global_end: int,
    output_dir: str,
    prompt_file: str,
    noise_seed_start: int,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    base_model_id: str,
    quant_model_dir: str,
    use_mjhq: bool,
    mjhq_prompt_sample_seed: int,
    dpm_algorithm_type: str,
    dpm_solver_order: int,
    dpm_solver_type: str,
    rank: int,
    world_size: int,
    device: torch.device,
    total_samples: int,
    shard_size: int,
    batch_size: int,
) -> None:
    print("=" * 60)
    print("PixArt-Sigma Q-Drift Sharded Statistics Collection")
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
        "dpm_algorithm_type": str(dpm_algorithm_type),
        "dpm_solver_order": int(dpm_solver_order),
        "dpm_solver_type": str(dpm_solver_type),
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

    fp16_pipeline = PixArtSigmaPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16).to(device)
    fp16_pipeline.transformer.eval()
    fp16_pipeline.text_encoder.eval()
    fp16_pipeline.vae.eval()
    apply_deepcompressor_patches(model=fp16_pipeline.transformer, shift_activations=False)

    quant_transformer = load_quant_transformer(
        base_model_id=base_model_id,
        quant_model_dir=quant_model_dir,
        device=device,
        torch_dtype=torch.float16,
    )

    scheduler = _build_base_scheduler(
        base_config=fp16_pipeline.scheduler.config,
        dpm_algorithm_type=dpm_algorithm_type,
        dpm_solver_order=dpm_solver_order,
        dpm_solver_type=dpm_solver_type,
    )

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
        pairs = manual_denoising_loop(
            transformer_fp16=fp16_pipeline.transformer,
            transformer_quant=quant_transformer,
            pipeline=fp16_pipeline,
            scheduler=scheduler,
            prompt=batch_prompts,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            seed=batch_seeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            device=device,
            height=height,
            width=width,
        )
        for local_idx, global_idx in enumerate(batch_indices):
            for timestep, (fp16_out, q_out) in pairs.items():
                shard_collector.add_pair(timestep, fp16_out[local_idx:local_idx + 1], q_out[local_idx:local_idx + 1])
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
    p = argparse.ArgumentParser(description="Collect PixArt-Sigma output statistics for Q-Drift Gaussian modeling")
    p.add_argument("--num_samples", type=int, default=5000)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--prompt_file", type=str, default=None)
    p.add_argument("--noise_seed_start", type=int, default=0)
    p.add_argument("--num_inference_steps", type=int, default=20)
    p.add_argument("--guidance_scale", type=float, default=4.5)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    p.add_argument("--quant_model_dir", type=str, default=DEFAULT_QUANT_MODEL_DIR)
    p.add_argument("--use_mjhq", action="store_true")
    p.add_argument("--mjhq_prompt_sample_seed", type=int, default=41)
    p.add_argument("--dpm_algorithm_type", type=str, default="dpmsolver++")
    p.add_argument("--dpm_solver_order", type=int, default=2)
    p.add_argument("--dpm_solver_type", type=str, default="midpoint")
    p.add_argument("--batch_size", type=int, default=1, help="Actual prompt batch size for sharded collection")
    p.add_argument("--shard_size", type=int, default=0, help="If >0, save resumable raw-pair shards with this many samples per shard")
    args = p.parse_args()

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
            base_model_id=args.base_model,
            quant_model_dir=args.quant_model_dir,
            use_mjhq=args.use_mjhq,
            mjhq_prompt_sample_seed=args.mjhq_prompt_sample_seed,
            dpm_algorithm_type=args.dpm_algorithm_type,
            dpm_solver_order=args.dpm_solver_order,
            dpm_solver_type=args.dpm_solver_type,
            rank=rank,
            world_size=world_size,
            device=device,
            total_samples=args.num_samples,
            shard_size=args.shard_size,
            batch_size=args.batch_size,
        )
        torch.distributed.barrier()
        return

    final_output_path = os.path.join(args.output_dir, "data_output_pairs.pth")
    should_skip = False
    if rank == 0 and os.path.exists(final_output_path):
        should_skip = True
        print(f"⏭️  SKIPPING: statistics collection (found {final_output_path})")

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

    collect_statistics(
        num_samples=num_samples_per_gpu,
        output_dir=args.output_dir,
        prompt_file=args.prompt_file,
        noise_seed_start=args.noise_seed_start + start_idx,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        base_model_id=args.base_model,
        quant_model_dir=args.quant_model_dir,
        use_mjhq=args.use_mjhq,
        mjhq_prompt_sample_seed=args.mjhq_prompt_sample_seed,
        dpm_algorithm_type=args.dpm_algorithm_type,
        dpm_solver_order=args.dpm_solver_order,
        dpm_solver_type=args.dpm_solver_type,
        rank=rank,
        world_size=world_size,
        device=device,
        total_samples=args.num_samples,
        start_idx=start_idx,
    )

    torch.distributed.barrier()
    if world_size > 1 and rank == 0:
        print("Merging rank outputs...")
        merged = {"fp16_output": {}, "quant_output": {}, "timesteps": [], "num_samples": 0}
        for r in range(world_size):
            rp = os.path.join(args.output_dir, f"data_output_pairs_rank{r}.pth")
            d = torch.load(rp)
            for t in d["timesteps"]:
                merged.setdefault("fp16_output", {}).setdefault(t, [])
                merged.setdefault("quant_output", {}).setdefault(t, [])
                merged["fp16_output"][t].append(d["fp16_output"][t])
                merged["quant_output"][t].append(d["quant_output"][t])
            merged["num_samples"] += d["num_samples"]
        merged["timesteps"] = sorted(merged["fp16_output"].keys())
        for t in merged["timesteps"]:
            merged["fp16_output"][t] = torch.cat(merged["fp16_output"][t], dim=0)
            merged["quant_output"][t] = torch.cat(merged["quant_output"][t], dim=0)
        torch.save(merged, final_output_path)
        for r in range(world_size):
            os.remove(os.path.join(args.output_dir, f"data_output_pairs_rank{r}.pth"))
        print(f"✓ Merged: {final_output_path}")


if __name__ == "__main__":
    main()
