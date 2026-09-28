from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fit_scalar_gaussian import _load_pairs, _resolve_input_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize the first N paired calibration samples into one data_output_pairs.pth file.")
    parser.add_argument("--data_output_pairs_path", action="append", help="Input data_output_pairs.pth file. May be repeated.")
    parser.add_argument("--data_output_pairs_glob", action="append", help="Glob for rank-sharded data_output_pairs files.")
    parser.add_argument("--output_path", required=True, help="Output data_output_pairs.pth path.")
    parser.add_argument("--expected_num_samples", type=int, default=None)
    parser.add_argument("--max_samples", type=int, required=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    output_path = Path(args.output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    paths = _resolve_input_paths(args)
    loaded = _load_pairs(paths, expected_num_samples=args.expected_num_samples, max_samples=args.max_samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fp16_output = {t: torch.cat(loaded.fp16_by_t[t], dim=0).contiguous() for t in loaded.timesteps}
    quant_output = {t: torch.cat(loaded.quant_by_t[t], dim=0).contiguous() for t in loaded.timesteps}
    torch.save(
        {
            "fp16_output": fp16_output,
            "quant_output": quant_output,
            "timesteps": loaded.timesteps,
            "num_samples": loaded.inferred_num_samples,
        },
        output_path,
    )
    metadata = {
        "source_files": [p.name for p in paths],
        "num_samples": loaded.inferred_num_samples,
        "num_samples_available": loaded.total_num_samples_available,
        "file_sample_counts": loaded.file_sample_counts,
        "file_sample_counts_available": loaded.file_sample_counts_available,
        "timesteps": [str(t) for t in loaded.timesteps],
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
