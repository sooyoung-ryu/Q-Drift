"""
Sampling-only evaluation script for the QNCD-style SDXL W3A4 adaptation on MJHQ-30K.

This script generates only quantized + QNCD-corrected images. It intentionally
does not generate FP16 or quantized baseline images and does not compute metrics.

Reproducibility design:
  - Prompt sampling is controlled by: --mjhq_prompt_sample_seed
  - Initial noise is controlled independently by: --noise_seed_start
    (per-prompt deterministic seed = noise_seed_start + global_idx)

Optional logging:
  - Saves full per-step x_t latents during corrected sampling as sharded `.pth` files under
    `--xt_output_dir/xt_latents/`. Rank 0 can also merge shards into a single large `.pth`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from huggingface_hub import hf_hub_download

from qncd_runtime import QNCD_IMPLEMENTATION_VERSION, prepare_conditioning, renoise_euler_ve
from tqdm import tqdm

# Allow importing `experiments/svdquant/deepcompressor_loader.py`.
_QDRIFT_DIR = Path(__file__).resolve().parents[2]
if str(_QDRIFT_DIR) not in sys.path:
    sys.path.insert(0, str(_QDRIFT_DIR))
from deepcompressor_loader import load_quant_unet
# MJHQ-30K constants
MJHQ_REPO_ID = "playgroundai/MJHQ-30K"
MJHQ_META_FILENAME = "meta_data.json"

DEFAULT_BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_QUANT_MODEL_DIR = str((Path(__file__).resolve().parents[2] / "sdxl_w3a4" / "model" / "unet_w3a4_g64").resolve())
DEFAULT_NEGATIVE_PROMPT = None


def _maybe_add_deepcompressor_to_syspath() -> None:
    here = Path(__file__).resolve()
    deepcompressor_root = here.parents[4] / "third_party" / "deepcompressor"
    if deepcompressor_root.is_dir() and str(deepcompressor_root) not in sys.path:
        sys.path.insert(0, str(deepcompressor_root))


def _load_quant_metadata(quant_model_dir: str) -> Dict[str, Any]:
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
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"✓ Applied activation quantization hooks from: {act_state_path}")
        return True
    except Exception as e:
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"⚠️  Failed to apply activation quantization hooks ({act_state_path}): {e}")
        return False


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


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _image_filename(row: Dict[str, Any]) -> str:
    category = row["category"]
    global_idx = int(row["global_idx"])
    return f"{category}_{global_idx:05d}.png"


def _all_images_exist(output_dir: str, prompts: List[Dict[str, Any]]) -> bool:
    for row in prompts:
        filename = _image_filename(row)
        if not os.path.exists(os.path.join(output_dir, filename)):
            return False
    return True


def _model_image_dirs(output_dir: str, model_key: str) -> List[str]:
    dirs: List[str] = []
    rank_prefix = "rank"
    if os.path.isdir(output_dir):
        for name in sorted(os.listdir(output_dir)):
            if not name.startswith(rank_prefix):
                continue
            candidate = os.path.join(output_dir, name, "images", model_key)
            if os.path.isdir(candidate):
                dirs.append(candidate)
    merged = os.path.join(output_dir, "images", model_key)
    if os.path.isdir(merged):
        dirs.append(merged)
    return dirs


def _all_images_exist_in_any_dir(dirs: List[str], prompts: List[Dict[str, Any]]) -> bool:
    if not dirs:
        return False
    for row in prompts:
        filename = _image_filename(row)
        if not any(os.path.exists(os.path.join(existing_dir, filename)) for existing_dir in dirs):
            return False
    return True


def _output_has_pngs(output_dir: str) -> bool:
    for root, _, files in os.walk(output_dir):
        if any(name.endswith(".png") for name in files):
            return True
    return False


def _assert_output_version_is_current(output_dir: str) -> None:
    results_path = os.path.join(output_dir, "evaluation_results.json")
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            raise RuntimeError(f"Existing evaluation metadata is unreadable: {results_path}") from exc
        if payload.get("qncd_implementation_version") != QNCD_IMPLEMENTATION_VERSION and _output_has_pngs(output_dir):
            raise RuntimeError(
                "Existing QNCD outputs were generated by an older implementation. "
                "Use a fresh --output_dir or remove the old outputs before rerunning."
            )
    elif os.path.isdir(output_dir) and _output_has_pngs(output_dir):
        raise RuntimeError(
            "Existing QNCD images have no implementation metadata. "
            "Use a fresh --output_dir or remove the old outputs before rerunning."
        )


def _parse_xt_dtype(name: str) -> torch.dtype:
    name = (name or "").lower().strip()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported --xt_dtype: {name} (expected bfloat16|float16|float32)")


def _sanitize_variant_tag(tag: str) -> str:
    tag = (tag or "").strip()
    if not tag:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", tag):
        raise ValueError(f"Invalid --variant_tag: {tag!r} (allowed: [A-Za-z0-9._-]+)")
    return tag


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
        self._xt_by_t: DefaultDict[int, List[torch.Tensor]] = defaultdict(list)

    def begin_sample(self, global_idx: int) -> None:
        self._global_indices_in_shard.append(int(global_idx))

    def record_step(self, timestep: int, latents: torch.Tensor) -> None:
        # latents: [B, C, H, W]; pipeline generation uses B==1 here.
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
    xt_chunks_by_t: DefaultDict[int, List[torch.Tensor]] = defaultdict(list)
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

        for t in data.get("timesteps", []):
            xt_chunks_by_t[int(t)].append(data["xt"][t])

    timesteps = sorted(xt_chunks_by_t.keys())
    xt_merged: Dict[int, torch.Tensor] = {}
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



def _euler_prev_sample(
    scheduler: EulerDiscreteScheduler,
    *,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    sample: torch.Tensor,
    step_index: int,
) -> torch.Tensor:
    """Stateless EulerDiscreteScheduler.step() for s_churn=0."""
    sigma = scheduler.sigmas[step_index].to(device=sample.device, dtype=torch.float32)
    sample_f = sample.to(torch.float32)
    model_output_f = model_output.to(torch.float32)

    prediction_type = scheduler.config.prediction_type
    if prediction_type in ("original_sample", "sample"):
        pred_original_sample = model_output_f
    elif prediction_type == "epsilon":
        pred_original_sample = sample_f - sigma * model_output_f
    elif prediction_type == "v_prediction":
        pred_original_sample = model_output_f * (-sigma / (sigma**2 + 1) ** 0.5) + (sample_f / (sigma**2 + 1))
    else:
        raise ValueError(f"Unsupported prediction_type for runtime QNCD: {prediction_type}")

    derivative = (sample_f - pred_original_sample) / sigma
    sigma_next = scheduler.sigmas[step_index + 1].to(device=sample.device, dtype=torch.float32)
    dt = sigma_next - sigma
    return sample_f + derivative * dt


def _scale_model_input_for_sigma(sample: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return sample / ((sigma**2 + 1) ** 0.5)


@torch.no_grad()
def _generate_images_runtime_qncd(
    *,
    pipeline: StableDiffusionXLPipeline,
    prompts: List[Dict[str, Any]],
    output_dir: str,
    prefix: str,
    noise_seed_start: int,
    num_inference_steps: int,
    guidance_scale: float,
    qncd_mean_scale: float,
    qncd_reverse_interval: int,
    device: torch.device,
    xt_logger: Optional[XTLatentsLogger] = None,
    existing_dirs: Optional[List[str]] = None,
) -> None:
    """Generate with QNCD's online reverse-noise estimator adapted to Euler."""
    _ensure_dir(output_dir)
    existing_dirs = existing_dirs or []
    do_cfg = guidance_scale > 1.0
    reverse_interval = max(int(qncd_reverse_interval), 1)

    scheduler = pipeline.scheduler
    unet = pipeline.unet

    def _apply_cfg(noise: torch.Tensor) -> torch.Tensor:
        noise = noise.to(torch.float32)
        if do_cfg:
            noise_uncond, noise_text = noise.chunk(2)
            noise = noise_uncond + guidance_scale * (noise_text - noise_uncond)
        return noise

    def _apply_qncd_correction(noise: torch.Tensor, correction: Any) -> torch.Tensor:
        if correction is None:
            return noise
        return noise - qncd_mean_scale * float(correction)

    def _decode_latents(latents: torch.Tensor):
        needs_upcasting = pipeline.vae.dtype == torch.float16 and getattr(pipeline.vae.config, "force_upcast", False)
        if needs_upcasting:
            pipeline.upcast_vae()
            latents = latents.to(next(iter(pipeline.vae.post_quant_conv.parameters())).dtype)

        latents = latents / pipeline.vae.config.scaling_factor
        image = pipeline.vae.decode(latents, return_dict=False)[0]
        image = pipeline.image_processor.postprocess(image, output_type="pil")

        if needs_upcasting:
            pipeline.vae.to(dtype=torch.float16)
        return image

    for row in tqdm(prompts, desc=f"Generating {prefix}"):
        prompt = row["prompt"]
        global_idx = int(row["global_idx"])
        filename = _image_filename(row)
        out_path = os.path.join(output_dir, filename)

        if os.path.exists(out_path):
            continue
        if any(os.path.exists(os.path.join(existing_dir, filename)) for existing_dir in existing_dirs):
            continue

        prompt_embeds, add_text_embeds, add_time_ids, _ = prepare_conditioning(
            prompt,
            DEFAULT_NEGATIVE_PROMPT,
            pipeline.tokenizer,
            pipeline.tokenizer_2,
            pipeline.text_encoder,
            pipeline.text_encoder_2,
            device,
            guidance_scale,
            force_zeros_for_empty_prompt=bool(
                getattr(pipeline.config, "force_zeros_for_empty_prompt", False)
            ),
        )

        scheduler.set_timesteps(num_inference_steps, device=device)
        generator = torch.Generator(device=device).manual_seed(int(noise_seed_start) + global_idx)
        latent_shape = (1, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size)
        latents = torch.randn(latent_shape, generator=generator, device=device, dtype=torch.float32)
        latents = latents * scheduler.init_noise_sigma

        active_correction = None
        if xt_logger is not None:
            xt_logger.begin_sample(global_idx)

        for step_index, timestep in enumerate(scheduler.timesteps):
            if scheduler.step_index is None:
                scheduler._init_step_index(timestep)

            latent_model_input = latents
            if do_cfg:
                latent_model_input = torch.cat([latent_model_input, latent_model_input], dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep).to(torch.bfloat16)

            noise_pred = unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": add_text_embeds, "time_ids": add_time_ids},
                return_dict=False,
            )[0]
            noise_pred = _apply_cfg(noise_pred)

            l_step = step_index + 1
            should_refresh = (l_step - 2) % reverse_interval == reverse_interval // 2

            # Match QNCD PLMS behavior: refresh steps first make an uncorrected
            # provisional step, estimate the new correction, then recompute this step.
            correction_for_first_step = None if should_refresh else active_correction
            corrected_noise = _apply_qncd_correction(noise_pred, correction_for_first_step)
            prev_latents = _euler_prev_sample(
                scheduler,
                model_output=corrected_noise,
                timestep=timestep,
                sample=latents,
                step_index=step_index,
            )

            if should_refresh:
                sigma = scheduler.sigmas[step_index].to(device=device, dtype=torch.float32)
                sigma_next = scheduler.sigmas[step_index + 1].to(device=device, dtype=torch.float32)
                reverse_generator = torch.Generator(device=device).manual_seed(
                    int(noise_seed_start) + global_idx + 1000003 + step_index
                )
                injected_noise = torch.randn(
                    prev_latents.shape,
                    generator=reverse_generator,
                    device=device,
                    dtype=torch.float32,
                )
                x_re = renoise_euler_ve(prev_latents, sigma, sigma_next, injected_noise)

                reverse_input = x_re
                if do_cfg:
                    reverse_input = torch.cat([reverse_input, reverse_input], dim=0)
                reverse_input = _scale_model_input_for_sigma(reverse_input, sigma).to(torch.bfloat16)

                reverse_noise = unet(
                    reverse_input,
                    timestep,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": add_text_embeds, "time_ids": add_time_ids},
                    return_dict=False,
                )[0]
                reverse_eps = _apply_cfg(reverse_noise)
                reverse_residual = reverse_eps - injected_noise
                active_correction = float(torch.mean(reverse_residual).item())

                corrected_noise = _apply_qncd_correction(noise_pred, active_correction)
                prev_latents = _euler_prev_sample(
                    scheduler,
                    model_output=corrected_noise,
                    timestep=timestep,
                    sample=latents,
                    step_index=step_index,
                )

            latents = prev_latents
            scheduler._step_index += 1

            if xt_logger is not None:
                t_val = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
                xt_logger.record_step(t_val, latents)

        image = _decode_latents(latents.to(dtype=pipeline.vae.dtype))[0]
        image.save(out_path)

        if xt_logger is not None:
            flushed = xt_logger.end_sample()
            if flushed is not None:
                print(f"✓ Flushed x_t shard: {flushed}")


def main():
    parser = argparse.ArgumentParser(description="Sampling-only evaluation for SDXL QNCD adaptation on MJHQ-30K")
    parser.add_argument("--output_dir", type=str, default="evaluation", help="Output directory for results")
    parser.add_argument(
        "--variant_tag",
        type=str,
        default="",
        help=(
            "Optional suffix appended to QNCD variant folder/log names so multiple calibrations can share "
            "the same --output_dir without collisions."
        ),
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
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of MJHQ prompts to sample")
    parser.add_argument(
        "--sample_start",
        type=int,
        default=0,
        help="Optional global prompt start index for independent evaluation workers.",
    )
    parser.add_argument(
        "--sample_end",
        type=int,
        default=None,
        help="Optional global prompt end index for independent evaluation workers.",
    )
    parser.add_argument(
        "--worker_tag",
        type=str,
        default="",
        help="Optional suffix for rank-local output dirs when running independent workers.",
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
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--meta_path", type=str, default=None, help="Optional path to MJHQ meta_data.json")
    parser.add_argument("--qncd_mean_scale", type=float, default=1.0)
    parser.add_argument("--qncd_reverse_interval", type=int, default=10)
    parser.add_argument(
        "--save_xt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save full per-step x_t latents for the QNCD variant to --xt_output_dir/xt_latents/",
    )
    parser.add_argument(
        "--save_xt_fp16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save per-step x_t latents for the FP16 variant to --xt_output_dir/xt_latents/ (model_key='fp16')",
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

    _assert_output_version_is_current(args.output_dir)
    args.variant_tag = _sanitize_variant_tag(args.variant_tag)
    args.worker_tag = _sanitize_variant_tag(args.worker_tag)
    if args.xt_output_dir is None:
        args.xt_output_dir = args.output_dir

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

    # Pre-download shared assets on rank 0 to reduce cache races in multi-GPU runs.
    # With a single rank this only duplicates the later model load.
    mjhq_meta_path = args.meta_path
    if rank == 0 and world_size > 1:
        try:
            if mjhq_meta_path is None or not os.path.exists(mjhq_meta_path):
                mjhq_meta_path = download_mjhq_metadata()
            _ = StableDiffusionXLPipeline.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, variant="fp16")
            _ = load_quant_unet(
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

    # All ranks load identical prompt list. Independent workers can take a
    # deterministic global slice before the normal torchrun rank split.
    all_prompts = load_prompts(mjhq_meta_path, args.num_samples, args.mjhq_prompt_sample_seed)
    worker_start = max(0, int(args.sample_start))
    worker_end = len(all_prompts) if args.sample_end is None else min(len(all_prompts), int(args.sample_end))
    if worker_start > worker_end:
        raise ValueError(f"Invalid sample slice: start={worker_start}, end={worker_end}")
    worker_prompts = all_prompts[worker_start:worker_end]
    samples_per_gpu = len(worker_prompts) // world_size
    start_idx = rank * samples_per_gpu
    end_idx = len(worker_prompts) if rank == world_size - 1 else start_idx + samples_per_gpu
    prompts = worker_prompts[start_idx:end_idx]
    global_start = worker_start + start_idx
    global_end = worker_start + end_idx
    print(f"[Rank {rank}/{world_size}] prompts: {len(prompts)} (idx {global_start}:{global_end})")

    # Rank-local output dirs
    rank_name = f"rank{rank}_{args.worker_tag}" if args.worker_tag else f"rank{rank}"
    rank_output_dir = os.path.join(args.output_dir, rank_name)
    images_root = os.path.join(rank_output_dir, "images")
    _ensure_dir(images_root)

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
            "runtime_qncd": True,
            "qncd_implementation_version": QNCD_IMPLEMENTATION_VERSION,
            "qncd_mean_scale": float(args.qncd_mean_scale),
            "qncd_reverse_interval": int(args.qncd_reverse_interval),
            "prompts": all_prompts,
        }
        with open(eval_results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved prompts + config: {eval_results_path}")

    torch.distributed.barrier()

    # Generate only the QNCD-corrected quantized model.
    model_key = f"quant_qncd_runtime_mean{args.qncd_mean_scale}_ri{args.qncd_reverse_interval}"
    if args.variant_tag:
        model_key = f"{model_key}_{args.variant_tag}"
    qncd_dir = os.path.join(images_root, model_key)
    qncd_existing_dirs = [
        d for d in _model_image_dirs(args.output_dir, model_key) if os.path.abspath(d) != os.path.abspath(qncd_dir)
    ]
    if not _all_images_exist_in_any_dir([qncd_dir, *qncd_existing_dirs], prompts):
        quant_unet = load_quant_unet(
            base_model_id=args.base_model,
            quant_model_dir=args.quant_model_dir,
            device=device,
            torch_dtype=torch.bfloat16,
        )
        scheduler = EulerDiscreteScheduler.from_pretrained(args.base_model, subfolder="scheduler")
        qncd_pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.base_model, unet=quant_unet, scheduler=scheduler, torch_dtype=torch.bfloat16, variant="fp16"
        ).to(device)
        _apply_activation_quant_if_available(qncd_pipeline.unet, quant_model_dir=args.quant_model_dir)

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

        _generate_images_runtime_qncd(
            pipeline=qncd_pipeline,
            prompts=prompts,
            output_dir=qncd_dir,
            prefix=model_key,
            noise_seed_start=args.noise_seed_start,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            qncd_mean_scale=args.qncd_mean_scale,
            qncd_reverse_interval=args.qncd_reverse_interval,
            device=device,
            xt_logger=xt_logger,
            existing_dirs=qncd_existing_dirs,
        )

        if xt_logger is not None:
            shard_paths = xt_logger.finalize()
            for p in shard_paths:
                print(f"[Rank {rank}] ✓ Saved x_t shard: {p}")

        del qncd_pipeline, scheduler, quant_unet
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

        # Merge QNCD logs.
        merged_log_dir = os.path.join(args.output_dir, f"logs_{model_key}")
        _ensure_dir(merged_log_dir)
        merged_log_file = os.path.join(merged_log_dir, "qncd_log.jsonl")
        with open(merged_log_file, "w", encoding="utf-8") as outf:
            for rank_name in rank_names:
                rank_log_file = os.path.join(
                    args.output_dir,
                    rank_name,
                    f"logs_{model_key}",
                    "qncd_log.jsonl",
                )
                if os.path.exists(rank_log_file):
                    with open(rank_log_file, "r", encoding="utf-8") as inf:
                        outf.write(inf.read())

        # Write x_t indices (one per variant).
        if args.save_xt:
            xt_latents_dir = os.path.join(args.xt_output_dir, "xt_latents")
            _ensure_dir(xt_latents_dir)
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

        # Clean up only this invocation's rank dirs (best effort). Independent
        # sliced workers may share the same output_dir with distinct worker_tags.
        for r in range(world_size):
            cleanup_rank_name = f"rank{r}_{args.worker_tag}" if args.worker_tag else f"rank{r}"
            rank_path = os.path.join(args.output_dir, cleanup_rank_name)
            try:
                if os.path.exists(rank_path):
                    shutil.rmtree(rank_path)
            except Exception:
                pass

        print(f"✓ Sampling complete. Images saved under: {os.path.join(args.output_dir, 'images')}")
    return


if __name__ == "__main__":
    main()
