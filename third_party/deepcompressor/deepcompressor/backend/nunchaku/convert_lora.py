"""Convert LoRA weights to Nunchaku format."""

import argparse
import os

import safetensors
import safetensors.torch
import torch
import tqdm

from ..utils import load_state_dict_in_safetensors, pad
from .convert import update_state_dict
from .utils import NunchakuWeightPacker


def reorder_adanorm_lora_up(lora_up: torch.Tensor, splits: int) -> torch.Tensor:
    c, r = lora_up.shape
    assert c % splits == 0
    return lora_up.view(splits, c // splits, r).transpose(0, 1).reshape(c, r).contiguous()


def convert_to_nunchaku_transformer_block_lowrank_dict(  # noqa: C901
    orig_state_dict: dict[str, torch.Tensor],
    extra_lora_dict: dict[str, torch.Tensor],
    converted_block_name: str,
    candidate_block_name: str,
    local_name_map: dict[str, str | list[str]],
    convert_map: dict[str, str],
    default_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    print(f"Converting LoRA branch for block {candidate_block_name}...")
    converted: dict[str, torch.Tensor] = {}
    packer = NunchakuWeightPacker(bits=4)
    for converted_local_name, candidate_local_names in tqdm.tqdm(
        local_name_map.items(), desc=f"Converting {candidate_block_name}", dynamic_ncols=True
    ):
        if isinstance(candidate_local_names, str):
            candidate_local_names = [candidate_local_names]
        # region original LoRA
        orig_lora = (
            orig_state_dict.get(f"{converted_block_name}.{converted_local_name}.lora_down", None),
            orig_state_dict.get(f"{converted_block_name}.{converted_local_name}.lora_up", None),
        )
        if orig_lora[0] is None or orig_lora[1] is None:
            assert orig_lora[0] is None and orig_lora[1] is None
            orig_lora = None
        else:
            assert orig_lora[0] is not None and orig_lora[1] is not None
            orig_lora = (
                packer.unpack_lowrank_weight(orig_lora[0], down=True),
                packer.unpack_lowrank_weight(orig_lora[1], down=False),
            )
            print(f" - Found {converted_block_name} LoRA of {converted_local_name} (rank: {orig_lora[0].shape[0]})")
        # endregion
        # region extra LoRA
        extra_lora = [
            (
                extra_lora_dict.get(f"{candidate_block_name}.{candidate_local_name}.lora_A.weight", None),
                extra_lora_dict.get(f"{candidate_block_name}.{candidate_local_name}.lora_B.weight", None),
            )
            for candidate_local_name in candidate_local_names
        ]
        # if any of the extra LoRA is None, all of them should be None
        if any(lora[0] is not None or lora[1] is not None for lora in extra_lora):
            # merge extra LoRAs into one LoRA
            if len(extra_lora) > 1:
                first_lora = None
                for lora in extra_lora:
                    if lora[0] is not None:
                        assert lora[1] is not None
                        first_lora = lora
                        break
                assert first_lora is not None
                for lora_index in range(len(extra_lora)):
                    if extra_lora[lora_index][0] is None:
                        assert extra_lora[lora_index][1] is None
                        extra_lora[lora_index] = (first_lora[0].clone(), torch.zeros_like(first_lora[1]))
                if all(lora[0].equal(extra_lora[0][0]) for lora in extra_lora):
                    # if all extra LoRAs have the same lora_down, use it
                    extra_lora_down = extra_lora[0][0]
                    extra_lora_up = torch.cat([lora[1] for lora in extra_lora], dim=0)
                else:
                    extra_lora_down = torch.cat([lora[0] for lora in extra_lora], dim=0)
                    extra_lora_up_c = sum(lora[1].shape[0] for lora in extra_lora)
                    extra_lora_up_r = sum(lora[1].shape[1] for lora in extra_lora)
                    assert extra_lora_up_r == extra_lora_down.shape[0]
                    extra_lora_up = torch.zeros((extra_lora_up_c, extra_lora_up_r), dtype=extra_lora_down.dtype)
                    c, r = 0, 0
                    for lora in extra_lora:
                        c_next, r_next = c + lora[1].shape[0], r + lora[1].shape[1]
                        extra_lora_up[c:c_next, r:r_next] = lora[1]
                        c, r = c_next, r_next
            else:
                extra_lora_down, extra_lora_up = extra_lora[0]
            extra_lora: tuple[torch.Tensor, torch.Tensor] = (extra_lora_down, extra_lora_up)
            print(f" - Found {candidate_block_name} LoRA of {candidate_local_names} (rank: {extra_lora[0].shape[0]})")
        # endregion
        # region merge LoRA
        if orig_lora is None:
            if extra_lora is None:
                lora = None
            else:
                print("    - Using extra LoRA")
                lora = (extra_lora[0].to(default_dtype), extra_lora[1].to(default_dtype))
        elif extra_lora is None:
            print("    - Using original LoRA")
            lora = orig_lora
        else:
            lora = (
                torch.cat([orig_lora[0], extra_lora[0].to(orig_lora[0].dtype)], dim=0),
                torch.cat([orig_lora[1], extra_lora[1].to(orig_lora[1].dtype)], dim=1),
            )
            print(f"    - Merging original and extra LoRA (rank: {lora[0].shape[0]})")
        # endregion
        if lora is not None:
            if convert_map[converted_local_name] == "adanorm_single":
                update_state_dict(
                    converted,
                    {
                        "lora_down": pad(lora[0], divisor=16, dim=0),
                        "lora_up": pad(reorder_adanorm_lora_up(lora[1], splits=3), divisor=16, dim=1),
                    },
                    prefix=converted_local_name,
                )
            elif convert_map[converted_local_name] == "adanorm_zero":
                update_state_dict(
                    converted,
                    {
                        "lora_down": pad(lora[0], divisor=16, dim=0),
                        "lora_up": pad(reorder_adanorm_lora_up(lora[1], splits=6), divisor=16, dim=1),
                    },
                    prefix=converted_local_name,
                )
            elif convert_map[converted_local_name] == "linear":
                update_state_dict(
                    converted,
                    {
                        "lora_down": packer.pack_lowrank_weight(lora[0], down=True),
                        "lora_up": packer.pack_lowrank_weight(lora[1], down=False),
                    },
                    prefix=converted_local_name,
                )
    return converted


def convert_to_nunchaku_flux_single_transformer_block_lowrank_dict(
    orig_state_dict: dict[str, torch.Tensor],
    extra_lora_dict: dict[str, torch.Tensor],
    converted_block_name: str,
    candidate_block_name: str,
    default_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    if f"{candidate_block_name}.proj_out.lora_A.weight" in extra_lora_dict:
        assert f"{converted_block_name}.out_proj.qweight" in orig_state_dict
        assert f"{converted_block_name}.mlp_fc2.qweight" in orig_state_dict
        n1 = orig_state_dict[f"{converted_block_name}.out_proj.qweight"].shape[1] * 2
        n2 = orig_state_dict[f"{converted_block_name}.mlp_fc2.qweight"].shape[1] * 2
        lora_down = extra_lora_dict[f"{candidate_block_name}.proj_out.lora_A.weight"]
        lora_up = extra_lora_dict[f"{candidate_block_name}.proj_out.lora_B.weight"]
        assert lora_down.shape[1] == n1 + n2
        extra_lora_dict[f"{candidate_block_name}.proj_out.linears.0.lora_A.weight"] = lora_down[:, :n1].clone()
        extra_lora_dict[f"{candidate_block_name}.proj_out.linears.0.lora_B.weight"] = lora_up.clone()
        extra_lora_dict[f"{candidate_block_name}.proj_out.linears.1.lora_A.weight"] = lora_down[:, n1:].clone()
        extra_lora_dict[f"{candidate_block_name}.proj_out.linears.1.lora_B.weight"] = lora_up.clone()
        extra_lora_dict.pop(f"{candidate_block_name}.proj_out.lora_A.weight")
        extra_lora_dict.pop(f"{candidate_block_name}.proj_out.lora_B.weight")

    return convert_to_nunchaku_transformer_block_lowrank_dict(
        orig_state_dict=orig_state_dict,
        extra_lora_dict=extra_lora_dict,
        converted_block_name=converted_block_name,
        candidate_block_name=candidate_block_name,
        local_name_map={
            "norm.linear": "norm.linear",
            "qkv_proj": ["attn.to_q", "attn.to_k", "attn.to_v"],
            "norm_q": "attn.norm_q",
            "norm_k": "attn.norm_k",
            "out_proj": "proj_out.linears.0",
            "mlp_fc1": "proj_mlp",
            "mlp_fc2": "proj_out.linears.1",
        },
        convert_map={
            "norm.linear": "adanorm_single",
            "qkv_proj": "linear",
            "out_proj": "linear",
            "mlp_fc1": "linear",
            "mlp_fc2": "linear",
        },
        default_dtype=default_dtype,
    )


def convert_to_nunchaku_flux_transformer_block_lowrank_dict(
    orig_state_dict: dict[str, torch.Tensor],
    extra_lora_dict: dict[str, torch.Tensor],
    converted_block_name: str,
    candidate_block_name: str,
    default_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    return convert_to_nunchaku_transformer_block_lowrank_dict(
        orig_state_dict=orig_state_dict,
        extra_lora_dict=extra_lora_dict,
        converted_block_name=converted_block_name,
        candidate_block_name=candidate_block_name,
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
            "mlp_fc2": "ff.net.2",
            "mlp_context_fc1": "ff_context.net.0.proj",
            "mlp_context_fc2": "ff_context.net.2",
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
        default_dtype=default_dtype,
    )


def convert_to_nunchaku_flux_lowrank_dict(
    orig_state_dict: dict[str, torch.Tensor],
    extra_lora_dict: dict[str, torch.Tensor],
    default_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    block_names: set[str] = set()
    for param_name in orig_state_dict.keys():
        if param_name.startswith(("transformer_blocks.", "single_transformer_blocks.")):
            block_names.add(".".join(param_name.split(".")[:2]))
    block_names = sorted(block_names, key=lambda x: (x.split(".")[0], int(x.split(".")[-1])))
    print(f"Converting {len(block_names)} transformer blocks...")
    converted: dict[str, torch.Tensor] = {}
    for block_name in block_names:
        if block_name.startswith("transformer_blocks"):
            convert_fn = convert_to_nunchaku_flux_transformer_block_lowrank_dict
        else:
            convert_fn = convert_to_nunchaku_flux_single_transformer_block_lowrank_dict
        update_state_dict(
            converted,
            convert_fn(
                orig_state_dict=orig_state_dict,
                extra_lora_dict=extra_lora_dict,
                converted_block_name=block_name,
                candidate_block_name=block_name,
                default_dtype=default_dtype,
            ),
            prefix=block_name,
        )
    return converted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant-path", type=str, required=True, help="path to the quantized model safetensor file")
    parser.add_argument("--lora-path", type=str, required=True, help="path to LoRA weights safetensor file")
    parser.add_argument("--output-root", type=str, default="", help="root to the output safetensor file")
    parser.add_argument("--lora-name", type=str, default=None, help="name of the LoRA weights")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="default data type of the converted LoRA weights",
    )
    args = parser.parse_args()

    if not args.output_root:
        # output to the parent directory of the quantized model safetensor file
        args.output_root = os.path.dirname(args.quant_path)
    if args.lora_name is None:
        assert args.lora_path is not None, "LoRA name or path must be provided"
        lora_name = args.lora_path.rstrip(os.sep).split(os.sep)[-1].replace(".safetensors", "")
        print(f"Lora name not provided, using {lora_name} as the LoRA name")
    else:
        lora_name = args.lora_name
    assert lora_name, "LoRA name must be provided."

    assert args.quant_path.endswith(".safetensors"), "Quantized model must be a safetensor file"
    assert args.lora_path.endswith(".safetensors"), "LoRA weights must be a safetensor file"
    orig_state_dict = load_state_dict_in_safetensors(args.quant_path)
    extra_lora_dict = load_state_dict_in_safetensors(args.lora_path, filter_prefix="transformer.")
    converted = convert_to_nunchaku_flux_lowrank_dict(
        orig_state_dict=orig_state_dict,
        extra_lora_dict=extra_lora_dict,
        default_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16,
    )
    os.makedirs(args.output_root, exist_ok=True)
    safetensors.torch.save_file(converted, os.path.join(args.output_root, f"{lora_name}.safetensors"))
    print(f"Saved LoRA weights to {args.output_root}.")
