import ast
import json
import os
from pathlib import Path
from typing import Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

COLLECTOR_PATHS = [
    REPO_ROOT / Path("experiments/svdquant/flux.1-dev_w3a4/scripts/collect_statistics.py"),
    REPO_ROOT / Path("experiments/svdquant/flux.1-schnell_w3a4/scripts/collect_statistics.py"),
    REPO_ROOT / Path("experiments/svdquant/sdxl_w3a4/scripts/collect_statistics.py"),
    REPO_ROOT / Path("experiments/svdquant/sdxl-turbo_w3a4/scripts/collect_statistics.py"),
]


def _load_shard_symbols(source_path: Path):
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    keep_names = {
        "OutputCollector",
        "_shard_output_path",
        "_save_output_shard",
        "_existing_shard_sample_count",
    }
    nodes = [node for node in tree.body if getattr(node, "name", None) in keep_names]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Dict": Dict,
        "List": List,
        "Path": Path,
        "json": json,
        "os": os,
        "torch": torch,
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace


def _collector_with_two_samples(output_collector):
    collector = output_collector()
    collector.add_pair(10, torch.ones(1, 2), torch.ones(1, 2) * 2)
    collector.add_pair(20, torch.ones(1, 2) * 3, torch.ones(1, 2) * 4)
    collector.add_pair(10, torch.ones(1, 2) * 5, torch.ones(1, 2) * 6)
    collector.add_pair(20, torch.ones(1, 2) * 7, torch.ones(1, 2) * 8)
    return collector


def test_shard_helpers_save_metadata_and_resume_count(tmp_path):
    for source_path in COLLECTOR_PATHS:
        ns = _load_shard_symbols(source_path)
        output_path = ns["_save_output_shard"](
            collector=_collector_with_two_samples(ns["OutputCollector"]),
            output_dir=str(tmp_path / source_path.parent.parent.name),
            rank=2,
            shard_id=3,
            start_index=7,
            end_index=9,
        )
        assert output_path.exists()
        assert not output_path.with_suffix(output_path.suffix + ".tmp").exists()
        metadata = json.loads(output_path.with_suffix(".json").read_text())
        assert metadata["rank"] == 2
        assert metadata["shard_id"] == 3
        assert metadata["num_samples"] == 2
        assert metadata["start_index"] == 7
        assert metadata["end_index"] == 9
        assert metadata["timesteps"] == [10, 20]
        data = torch.load(output_path, map_location="cpu")
        assert data["num_samples"] == 2
        assert data["timesteps"] == [10, 20]
        assert ns["_existing_shard_sample_count"](
            str(output_path.parent),
            rank=2,
            shard_id=3,
            expected_start_index=7,
            expected_end_index=9,
        ) == 2


def test_shard_resume_rejects_mismatched_metadata(tmp_path):
    for source_path in COLLECTOR_PATHS:
        ns = _load_shard_symbols(source_path)
        output_path = ns["_save_output_shard"](
            collector=_collector_with_two_samples(ns["OutputCollector"]),
            output_dir=str(tmp_path / source_path.parent.parent.name),
            rank=0,
            shard_id=0,
            start_index=0,
            end_index=2,
        )
        for kwargs in [
            {"expected_start_index": 1, "expected_end_index": 2},
            {"expected_start_index": 0, "expected_end_index": 3},
        ]:
            try:
                ns["_existing_shard_sample_count"](
                    str(output_path.parent),
                    rank=0,
                    shard_id=0,
                    **kwargs,
                )
            except RuntimeError as exc:
                assert "Delete it before resuming" in str(exc)
            else:
                raise AssertionError(f"{source_path} accepted mismatched shard metadata")


def test_incomplete_tmp_shard_is_not_treated_as_complete(tmp_path):
    for source_path in COLLECTOR_PATHS:
        ns = _load_shard_symbols(source_path)
        output_dir = tmp_path / source_path.parent.parent.name
        output_dir.mkdir()
        tmp_only = output_dir / "data_output_pairs_rank0_shard00000.pth.tmp"
        torch.save({"num_samples": 2}, tmp_only)
        assert ns["_existing_shard_sample_count"](
            str(output_dir),
            rank=0,
            shard_id=0,
            expected_start_index=0,
            expected_end_index=2,
        ) is None
