"""Converts a DeepCompressor state dict to a Nunchaku state dict."""

import argparse
import os

import safetensors.torch
import torch
import tqdm

from .utils import convert_to_nunchaku_w4x4y16_linear_weight, convert_to_nunchaku_w4x16_linear_weight


def convert_to_nunchaku_w4x4y16_linear_state_dict(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    smooth: torch.Tensor | None = None,
    lora: tuple[torch.Tensor, torch.Tensor] | None = None,
    shift: torch.Tensor | None = None,
    smooth_fused: bool = False,
    float_point: bool = False,
    subscale: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if weight.ndim > 2:  # pointwise conv
        assert weight.numel() == weight.shape[0] * weight.shape[1]
        weight = weight.view(weight.shape[0], weight.shape[1])
    if scale.numel() > 1:
        assert scale.ndim == weight.ndim * 2
        assert scale.numel() == scale.shape[0] * scale.shape[2]
        scale = scale.view(scale.shape[0], 1, scale.shape[2], 1)
        scale_key = "wcscales" if scale.shape[2] == 1 else "wscales"
    else:
        scale_key = "wtscale"
    if subscale is None:
        subscale_key = ""
    else:
        assert subscale.ndim == weight.ndim * 2
        assert subscale.numel() == subscale.shape[0] * subscale.shape[2]
        assert subscale.numel() > 1
        subscale = subscale.view(subscale.shape[0], 1, subscale.shape[2], 1)
        subscale_key = "wcscales" if subscale.shape[2] == 1 else "wscales"
    if lora is not None and (smooth is not None or shift is not None):
        # unsmooth lora down projection
        dtype = weight.dtype
        lora_down, lora_up = lora
        lora_down = lora_down.to(dtype=torch.float64)
        if smooth is not None and not smooth_fused:
            lora_down = lora_down.div_(smooth.to(torch.float64).unsqueeze(0))
        if shift is not None:
            bias = torch.zeros([lora_up.shape[0]], dtype=torch.float64) if bias is None else bias.to(torch.float64)
            if shift.numel() == 1:
                shift = shift.view(1, 1).expand(lora_down.shape[1], 1).to(torch.float64)
            else:
                shift = shift.view(-1, 1).to(torch.float64)
            bias = bias.add_((lora_up.to(dtype=torch.float64) @ lora_down @ shift).view(-1))
            bias = bias.to(dtype=dtype)
        lora = (lora_down.to(dtype=dtype), lora_up)
    weight, scale, bias, smooth, lora, subscale = convert_to_nunchaku_w4x4y16_linear_weight(
        weight, scale=scale, bias=bias, smooth=smooth, lora=lora, float_point=float_point, subscale=subscale
    )
    state_dict: dict[str, torch.Tensor] = {}
    state_dict["qweight"] = weight
    state_dict[scale_key] = scale
    if subscale is not None:
        state_dict[subscale_key] = subscale
    state_dict["bias"] = bias
    state_dict["smooth_orig"] = smooth
    state_dict["smooth"] = torch.ones_like(smooth) if smooth_fused else smooth.clone()
    if lora is not None:
        state_dict["lora_down"] = lora[0]
        state_dict["lora_up"] = lora[1]
    return state_dict


def convert_to_nunchaku_w4x16_adanorm_single_state_dict(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> dict[str, torch.Tensor]:
    weight, scale, zero, bias = convert_to_nunchaku_w4x16_linear_weight(
        weight, scale=scale, bias=bias, adanorm_splits=3
    )
    state_dict: dict[str, torch.Tensor] = {}
    state_dict = {}
    state_dict["qweight"] = weight
    state_dict["wscales"] = scale
    state_dict["wzeros"] = zero
    state_dict["bias"] = bias
    return state_dict


def convert_to_nunchaku_w4x16_adanorm_zero_state_dict(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> dict[str, torch.Tensor]:
    weight, scale, zero, bias = convert_to_nunchaku_w4x16_linear_weight(
        weight, scale=scale, bias=bias, adanorm_splits=6
    )
    state_dict: dict[str, torch.Tensor] = {}
    state_dict = {}
    state_dict["qweight"] = weight
    state_dict["wscales"] = scale
    state_dict["wzeros"] = zero
    state_dict["bias"] = bias
    return state_dict


def update_state_dict(
    lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor], prefix: str = ""
) -> dict[str, torch.Tensor]:
    for rkey, value in rhs.items():
        lkey = f"{prefix}.{rkey}" if prefix else rkey
        assert lkey not in lhs, f"Key {lkey} already exists in the state dict."
        lhs[lkey] = value
    return lhs


def convert_to_nunchaku_transformer_block_state_dict(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    block_name: str,
    local_name_map: dict[str, str | list[str]],
    smooth_name_map: dict[str, str],
    branch_name_map: dict[str, str],
    convert_map: dict[str, str],
    float_point: bool = False,
) -> dict[str, torch.Tensor]:
    print(f"Converting block {block_name}...")
    converted: dict[str, torch.Tensor] = {}
    candidates: dict[str, torch.Tensor] = {
        param_name: param for param_name, param in state_dict.items() if param_name.startswith(block_name)
    }
    for converted_local_name, candidate_local_names in tqdm.tqdm(
        local_name_map.items(), desc=f"Converting {block_name}", dynamic_ncols=True
    ):
        if isinstance(candidate_local_names, str):
            candidate_local_names = [candidate_local_names]
        candidate_names = [f"{block_name}.{candidate_local_name}" for candidate_local_name in candidate_local_names]
        weight = [candidates[f"{candidate_name}.weight"] for candidate_name in candidate_names]
        bias = [candidates.get(f"{candidate_name}.bias", None) for candidate_name in candidate_names]
        scale = [scale_dict.get(f"{candidate_name}.weight.scale.0", None) for candidate_name in candidate_names]
        subscale = [scale_dict.get(f"{candidate_name}.weight.scale.1", None) for candidate_name in candidate_names]
        if len(weight) > 1:
            bias = None if all(b is None for b in bias) else torch.concat(bias, dim=0)
            if all(s is None for s in scale):
                scale = None
            else:
                if scale[0].numel() == 1:  # switch from per-tensor to per-channel scale
                    assert all(s.numel() == 1 for s in scale)
                    scale = torch.concat(
                        [
                            s.view(-1).expand(weight[i].shape[0]).reshape(weight[i].shape[0], 1, 1, 1)
                            for i, s in enumerate(scale)
                        ],
                        dim=0,
                    )
                else:
                    scale = torch.concat(scale, dim=0)
            subscale = None if all(s is None for s in subscale) else torch.concat(subscale, dim=0)
            weight = torch.concat(weight, dim=0)
        else:
            weight, bias, scale, subscale = weight[0], bias[0], scale[0], subscale[0]
        smooth = smooth_dict.get(f"{block_name}.{smooth_name_map.get(converted_local_name, '')}", None)
        branch = branch_dict.get(f"{block_name}.{branch_name_map.get(converted_local_name, '')}", None)
        if branch is not None:
            branch = (branch["a.weight"], branch["b.weight"])
        if scale is None:
            assert smooth is None and branch is None and subscale is None
            print(f"  - Copying {block_name} weights of {candidate_local_names} as {converted_local_name}.weight")
            converted[f"{converted_local_name}.weight"] = weight.clone().cpu()
            if bias is not None:
                print(f"  - Copying {block_name} biases of {candidate_local_names} as {converted_local_name}.bias")
                converted[f"{converted_local_name}.bias"] = bias.clone().cpu()
            continue
        if convert_map[converted_local_name] == "adanorm_single":
            print(f"  - Converting {block_name} weights of {candidate_local_names} to {converted_local_name}.")
            update_state_dict(
                converted,
                convert_to_nunchaku_w4x16_adanorm_single_state_dict(weight=weight, scale=scale, bias=bias),
                prefix=converted_local_name,
            )
        elif convert_map[converted_local_name] == "adanorm_zero":
            print(f"  - Converting {block_name} weights of {candidate_local_names} to {converted_local_name}.")
            update_state_dict(
                converted,
                convert_to_nunchaku_w4x16_adanorm_zero_state_dict(weight=weight, scale=scale, bias=bias),
                prefix=converted_local_name,
            )
        elif convert_map[converted_local_name] == "linear":
            smooth_fused = "out_proj" in converted_local_name and smooth_dict.get("proj.fuse_when_possible", True)
            shift = [candidates.get(f"{candidate_name[:-7]}.shift", None) for candidate_name in candidate_names]
            assert all(s == shift[0] for s in shift)
            shift = shift[0]
            print(
                f"  - Converting {block_name} weights of {candidate_local_names} to {converted_local_name}."
                f" (smooth_fused={smooth_fused}, shifted={shift is not None}, float_point={float_point})"
            )
            update_state_dict(
                converted,
                convert_to_nunchaku_w4x4y16_linear_state_dict(
                    weight=weight,
                    scale=scale,
                    bias=bias,
                    smooth=smooth,
                    lora=branch,
                    shift=shift,
                    smooth_fused=smooth_fused,
                    float_point=float_point,
                    subscale=subscale,
                ),
                prefix=converted_local_name,
            )
        else:
            raise NotImplementedError(f"Conversion of {convert_map[converted_local_name]} is not implemented.")
    return converted


def convert_to_nunchaku_flux_single_transformer_block_state_dict(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    block_name: str,
    float_point: bool = False,
) -> dict[str, torch.Tensor]:
    down_proj_local_name = "proj_out.linears.1.linear"
    if f"{block_name}.{down_proj_local_name}.weight" not in state_dict:
        down_proj_local_name = "proj_out.linears.1"
        assert f"{block_name}.{down_proj_local_name}.weight" in state_dict

    return convert_to_nunchaku_transformer_block_state_dict(
        state_dict=state_dict,
        scale_dict=scale_dict,
        smooth_dict=smooth_dict,
        branch_dict=branch_dict,
        block_name=block_name,
        local_name_map={
            "norm.linear": "norm.linear",
            "qkv_proj": ["attn.to_q", "attn.to_k", "attn.to_v"],
            "norm_q": "attn.norm_q",
            "norm_k": "attn.norm_k",
            "out_proj": "proj_out.linears.0",
            "mlp_fc1": "proj_mlp",
            "mlp_fc2": down_proj_local_name,
        },
        smooth_name_map={
            "qkv_proj": "attn.to_q",
            "out_proj": "proj_out.linears.0",
            "mlp_fc1": "attn.to_q",
            "mlp_fc2": down_proj_local_name,
        },
        branch_name_map={
            "qkv_proj": "attn.to_q",
            "out_proj": "proj_out.linears.0",
            "mlp_fc1": "proj_mlp",
            "mlp_fc2": down_proj_local_name,
        },
        convert_map={
            "norm.linear": "adanorm_single",
            "qkv_proj": "linear",
            "out_proj": "linear",
            "mlp_fc1": "linear",
            "mlp_fc2": "linear",
        },
        float_point=float_point,
    )


def convert_to_nunchaku_flux_transformer_block_state_dict(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    block_name: str,
    float_point: bool = False,
) -> dict[str, torch.Tensor]:
    down_proj_local_name = "ff.net.2.linear"
    if f"{block_name}.{down_proj_local_name}.weight" not in state_dict:
        down_proj_local_name = "ff.net.2"
        assert f"{block_name}.{down_proj_local_name}.weight" in state_dict
    context_down_proj_local_name = "ff_context.net.2.linear"
    if f"{block_name}.{context_down_proj_local_name}.weight" not in state_dict:
        context_down_proj_local_name = "ff_context.net.2"
        assert f"{block_name}.{context_down_proj_local_name}.weight" in state_dict

    return convert_to_nunchaku_transformer_block_state_dict(
        state_dict=state_dict,
        scale_dict=scale_dict,
        smooth_dict=smooth_dict,
        branch_dict=branch_dict,
        block_name=block_name,
        local_name_map={
            "norm1.linear": "norm1.linear",
            "norm1_context.linear": "norm1_context.linear",
            "qkv_proj": ["attn.to_q", "attn.to_k", "attn.to_v"],
            "qkv_proj_context": ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"],
            "norm_q": "attn.norm_q",
            "norm_k": "attn.norm_k",
            "norm_added_q": "attn.norm_added_q",
            "norm_added_k": "attn.norm_added_k",
            "out_proj": "attn.to_out.0",
            "out_proj_context": "attn.to_add_out",
            "mlp_fc1": "ff.net.0.proj",
            "mlp_fc2": down_proj_local_name,
            "mlp_context_fc1": "ff_context.net.0.proj",
            "mlp_context_fc2": context_down_proj_local_name,
        },
        smooth_name_map={
            "qkv_proj": "attn.to_q",
            "qkv_proj_context": "attn.add_k_proj",
            "out_proj": "attn.to_out.0",
            "out_proj_context": "attn.to_out.0",
            "mlp_fc1": "ff.net.0.proj",
            "mlp_fc2": down_proj_local_name,
            "mlp_context_fc1": "ff_context.net.0.proj",
            "mlp_context_fc2": context_down_proj_local_name,
        },
        branch_name_map={
            "qkv_proj": "attn.to_q",
            "qkv_proj_context": "attn.add_k_proj",
            "out_proj": "attn.to_out.0",
            "out_proj_context": "attn.to_add_out",
            "mlp_fc1": "ff.net.0.proj",
            "mlp_fc2": down_proj_local_name,
            "mlp_context_fc1": "ff_context.net.0.proj",
            "mlp_context_fc2": context_down_proj_local_name,
        },
        convert_map={
            "norm1.linear": "adanorm_zero",
            "norm1_context.linear": "adanorm_zero",
            "qkv_proj": "linear",
            "qkv_proj_context": "linear",
            "out_proj": "linear",
            "out_proj_context": "linear",
            "mlp_fc1": "linear",
            "mlp_fc2": "linear",
            "mlp_context_fc1": "linear",
            "mlp_context_fc2": "linear",
        },
        float_point=float_point,
    )


def convert_to_nunchaku_flux_state_dicts(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    float_point: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    block_names: set[str] = set()
    other: dict[str, torch.Tensor] = {}
    for param_name in state_dict.keys():
        if param_name.startswith(("transformer_blocks.", "single_transformer_blocks.")):
            block_names.add(".".join(param_name.split(".")[:2]))
        else:
            other[param_name] = state_dict[param_name]
    block_names = sorted(block_names, key=lambda x: (x.split(".")[0], int(x.split(".")[-1])))
    print(f"Converting {len(block_names)} transformer blocks...")
    converted: dict[str, torch.Tensor] = {}
    for block_name in block_names:
        convert_fn = convert_to_nunchaku_flux_single_transformer_block_state_dict
        if block_name.startswith("transformer_blocks"):
            convert_fn = convert_to_nunchaku_flux_transformer_block_state_dict
        update_state_dict(
            converted,
            convert_fn(
                state_dict=state_dict,
                scale_dict=scale_dict,
                smooth_dict=smooth_dict,
                branch_dict=branch_dict,
                block_name=block_name,
                float_point=float_point,
            ),
            prefix=block_name,
        )
    return converted, other


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant-path", type=str, required=True, help="path to the quantization checkpoint directory.")
    parser.add_argument("--output-root", type=str, default="", help="root to the output checkpoint directory.")
    parser.add_argument("--model-name", type=str, default=None, help="name of the model.")
    parser.add_argument("--float-point", action="store_true", help="use float-point 4-bit quantization.")
    args = parser.parse_args()
    if not args.output_root:
        args.output_root = args.quant_path
    if args.model_name is None:
        assert args.model_path is not None, "model name or path is required."
        model_name = args.model_path.rstrip(os.sep).split(os.sep)[-1]
        print(f"Model name not provided, using {model_name} as the model name.")
    else:
        model_name = args.model_name
    assert model_name, "Model name must be provided."
    assert "flux" in model_name.lower(), "Only Flux models are supported."
    state_dict_path = os.path.join(args.quant_path, "model.pt")
    scale_dict_path = os.path.join(args.quant_path, "scale.pt")
    smooth_dict_path = os.path.join(args.quant_path, "smooth.pt")
    branch_dict_path = os.path.join(args.quant_path, "branch.pt")
    map_location = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    state_dict = torch.load(state_dict_path, map_location=map_location)
    scale_dict = torch.load(scale_dict_path, map_location="cpu")
    smooth_dict = torch.load(smooth_dict_path, map_location=map_location) if os.path.exists(smooth_dict_path) else {}
    branch_dict = torch.load(branch_dict_path, map_location=map_location) if os.path.exists(branch_dict_path) else {}
    converted_state_dict, other_state_dict = convert_to_nunchaku_flux_state_dicts(
        state_dict=state_dict,
        scale_dict=scale_dict,
        smooth_dict=smooth_dict,
        branch_dict=branch_dict,
        float_point=args.float_point,
    )
    output_dirpath = os.path.join(args.output_root, model_name)
    os.makedirs(output_dirpath, exist_ok=True)
    safetensors.torch.save_file(converted_state_dict, os.path.join(output_dirpath, "transformer_blocks.safetensors"))
    safetensors.torch.save_file(other_state_dict, os.path.join(output_dirpath, "unquantized_layers.safetensors"))
    print(f"Quantized model saved to {output_dirpath}.")
