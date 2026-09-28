# -*- coding: utf-8 -*-
"""QServe state dict converter module."""

import argparse
import os

import safetensors.torch
import torch
import tqdm

from .utils import convert_to_tinychat_w4x16y16_linear_weight


def convert_to_tinychat_w4x16y16_linear_state_dict(
    param_name: str,
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    zero_pre_scaled: bool = False,
) -> dict[str, torch.Tensor]:
    """Convert a weight tensor to TinyChat W4-X16-Y16 linear state dictionary.

    Args:
        param_name (`str`):
            parameter name.
        weight (`torch.Tensor`):
            weight tensor to be converted.
        scale (`torch.Tensor`):
            scale tensor for the weight tensor.
        zero (`torch.Tensor`):
            zero point tensor for the weight tensor.
        zero_pre_scaled (`bool`, *optional*, defaults to `False`):
            whether zero point tensor is pre-scaled.

    Returns:
        `dict[str, torch.Tensor]`:
            state dictionary for the quantized weight tensor.
    """
    module_name = param_name[:-7]
    weight, scale, zero = convert_to_tinychat_w4x16y16_linear_weight(
        weight, scale=scale, zero=zero, zero_pre_scaled=zero_pre_scaled
    )
    state_dict: dict[str, torch.Tensor] = {}
    state_dict[f"{module_name}.qweight"] = weight.cpu()
    state_dict[f"{module_name}.scales"] = scale.cpu()
    state_dict[f"{module_name}.scaled_zeros"] = zero.cpu()
    return state_dict


def convert_to_tinychat_state_dict(
    state_dict: dict[str, torch.Tensor], scale_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    scales: dict[str, dict[tuple[int, ...], torch.Tensor]] = {}
    zeros: dict[str, tuple[torch.Tensor | None, bool]] = {}
    print("Loading scale tensors...")
    for name, tensor in tqdm.tqdm(scale_dict.items(), desc="Loading scale tensors", leave=False, dynamic_ncols=True):
        print(f"  - Loading tensor {name} (dtype: {tensor.dtype}, shape: {tensor.shape}, device: {tensor.device})")
        if name.endswith("zero"):
            # this is a zero point tensor
            zero = None if tensor is None or all(t.item() == 0 for t in tensor.flatten()) else tensor
            if name.endswith(".scaled_zero"):
                zeros[name[:-12]] = (zero, False)  # zero point tensor is post-scaled
            else:
                zeros[name[:-5]] = (zero, True)  # zero point tensor is pre-scaled
        else:
            assert ".weight.scale" in name
            # this is a scale tensor
            idx = name.index(".weight.scale")
            param_name = name[: idx + 7]
            scale_level = tuple(map(int, name[idx + 14 :].split(".")))
            scales.setdefault(param_name, {})[scale_level] = tensor
    for param_name in zeros.keys():
        assert param_name in state_dict, f"zero point tensor {param_name} not found in state dict."
        assert param_name in scales, f"scale tensor {param_name} not found in scale dict."
    converted: dict[str, torch.Tensor] = {}
    print("Converting state dict...")
    for param_name, param in tqdm.tqdm(state_dict.items(), desc="Converting state dict", dynamic_ncols=True):
        if param_name in scales:
            print(f"  - Converting {param_name} (dtype: {param.dtype}, shape: {param.shape}, device: {param.device})")
            weight = param.data.clone()
            if param_name in zeros:
                zero, zero_pre_scaled = zeros[param_name]
                zero = zero.clone() if zero is not None else None
            else:
                zero, zero_pre_scaled = None, False
            level_scales = sorted(scales[param_name].items(), key=lambda x: x[0])
            assert len(level_scales) == 1, "more than one scale levels are not supported."
            scale = level_scales[0][1].clone()
            converted.update(
                convert_to_tinychat_w4x16y16_linear_state_dict(
                    param_name, weight, scale=scale, zero=zero, zero_pre_scaled=zero_pre_scaled
                )
            )
        else:
            if isinstance(param, torch.Tensor):
                print(f"  - Copying {param_name} (dtype: {param.dtype}, shape: {param.shape}, device: {param.device})")
                converted[param_name] = param.clone().cpu()
            else:
                print(f"  - Copying {param_name} (type: {type(param)}, value: {param})")
                converted[param_name] = param
    return converted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant-path", type=str, required=True, help="path to the quantization checkpoint directory.")
    parser.add_argument("--output-root", type=str, default="", help="root to the output checkpoint directory.")
    parser.add_argument("--model-name", type=str, default=None, help="model name.")
    parser.add_argument("--model-path", type=str, default=None, help="path to the huggingface model directory.")
    parser.add_argument("--copy-on-save", action="store_true", help="copy files on save.")
    args = parser.parse_args()
    if not args.output_root:
        args.output_root = args.quant_path
    if args.model_name is None:
        assert args.model_path is not None, "model name or path is required."
        model_name = args.model_path.rstrip(os.sep).split(os.sep)[-1]
        print(f"Model name not provided. Using model name {model_name}.")
    else:
        model_name = args.model_name
    state_dict = torch.load(
        os.path.join(args.quant_path, "model.pt"),
        map_location="cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu",
    )
    scale_dict = torch.load(os.path.join(args.quant_path, "scale.pt"), map_location="cpu")
    converted = convert_to_tinychat_state_dict(state_dict, scale_dict)
    model_name = f"{args.model_name}-w4a16"
    output_dirpath = os.path.join(args.output_root, model_name)

    os.makedirs(output_dirpath, exist_ok=True)
    if args.model_path and os.path.exists(args.model_path):
        output_path = os.path.join(output_dirpath, "model.safetensors")
        safetensors.torch.save_file(converted, output_path)
        print(f"Quantized model checkpoint saved to {output_path}.")
        for filename in os.listdir(args.model_path):
            if filename == "tokenizer.model" or (
                filename.endswith(".json") and filename != "pytorch_model.bin.index.json"
            ):
                filepath = os.path.abspath(os.path.join(args.model_path, filename))
                if args.copy_on_save:
                    os.system(f"cp {filepath} {output_dirpath}/")
                else:
                    os.system(f"ln -s {filepath} {output_dirpath}/{filename}")
    else:
        output_path = os.path.join(output_dirpath, "tinychat-v2.pt")
        torch.save(converted, output_path)
        print(f"Quantized model checkpoint saved to {output_path}.")
    print(f"Quantized model saved to {output_dirpath}.")
