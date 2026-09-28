"""QNCD helpers for runtime estimation on SDXL: prompt conditioning and Euler VE re-noising."""

from typing import Optional, Tuple

import torch

QNCD_IMPLEMENTATION_VERSION = "qncd_euler_ve_none_negative_v2"


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


def renoise_euler_ve(
    prev_latents: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    injected_noise: torch.Tensor,
) -> torch.Tensor:
    noise_scale = (sigma.to(torch.float32) ** 2 - sigma_next.to(torch.float32) ** 2).clamp(min=0.0).sqrt()
    return prev_latents.to(torch.float32) + noise_scale.to(device=prev_latents.device) * injected_noise.to(torch.float32)


def prepare_conditioning(
    prompt: str,
    negative_prompt: Optional[str],
    tokenizer,
    tokenizer_2,
    text_encoder,
    text_encoder_2,
    device: torch.device,
    guidance_scale: float,
    force_zeros_for_empty_prompt: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    do_cfg = guidance_scale > 1.0

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = _encode_prompt(
            prompt, tokenizer, tokenizer_2, text_encoder, text_encoder_2, device
        )
        use_zero_negative = do_cfg and force_zeros_for_empty_prompt and negative_prompt is None
        if use_zero_negative:
            negative_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_cfg:
            negative_embeds, negative_pooled_embeds = _encode_prompt(
                negative_prompt or "", tokenizer, tokenizer_2, text_encoder, text_encoder_2, device
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
    add_text_embeds = pooled_prompt_embeds.to(torch.bfloat16)
    add_time_ids_bf16 = add_time_ids.to(torch.bfloat16)

    if do_cfg:
        negative_prompt_embeds_bf16 = negative_embeds.to(torch.bfloat16)
        negative_add_text_embeds = negative_pooled_embeds.to(torch.bfloat16)
        negative_add_time_ids = add_time_ids_bf16.clone()
        prompt_embeds_bf16 = torch.cat([negative_prompt_embeds_bf16, prompt_embeds_bf16], dim=0)
        add_text_embeds = torch.cat([negative_add_text_embeds, add_text_embeds], dim=0)
        add_time_ids_bf16 = torch.cat([negative_add_time_ids, add_time_ids_bf16], dim=0)

    return prompt_embeds_bf16, add_text_embeds, add_time_ids_bf16, do_cfg
