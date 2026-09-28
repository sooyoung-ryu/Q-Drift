from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SVD_TURBO_EVAL = ROOT / "experiments" / "svdquant" / "sdxl-turbo_w3a4" / "scripts" / "evaluate.py"
MIXDQ_TURBO_EVAL = ROOT / "experiments" / "mixdq" / "sdxl-turbo_w4a8" / "scripts" / "evaluate.py"
MIXDQ_RUN_PAPER = ROOT / "experiments" / "mixdq" / "sdxl-turbo_w4a8" / "run_paper.sh"


def _load_functions(path: Path, names: set[str]):
    source = path.read_text()
    tree = ast.parse(source)
    wanted = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            wanted.append(ast.get_source_segment(source, node))
    if len(wanted) != len(names):
        found = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        raise AssertionError(f"missing {names - found} in {path}")
    namespace = {"List": list, "Dict": dict, "Optional": object, "ValueError": ValueError}
    exec("from __future__ import annotations\n" + "\n\n".join(wanted), namespace)
    return {name: namespace[name] for name in names}


class TurboSeedMappingTest(unittest.TestCase):
    def test_svd_rank_local_mapping_is_independent_of_selected_subset(self):
        funcs = _load_functions(SVD_TURBO_EVAL, {"_rank_local_seed_offset"})
        offset = funcs["_rank_local_seed_offset"]
        selected = [0, 1, 832, 833, 4164, 4165, 4999]

        full_assigned = {idx: 42 + offset(idx, 5000, 6) for idx in range(5000)}
        selected_assigned = {idx: 42 + offset(idx, 5000, 6) for idx in selected}
        self.assertEqual(selected_assigned, {idx: full_assigned[idx] for idx in selected})
        self.assertEqual(selected_assigned[0], 42)
        self.assertEqual(selected_assigned[833], 10042)
        self.assertEqual(selected_assigned[4165], 50042)
        self.assertEqual(selected_assigned[4999], 50876)

    def test_svd_world1_rank_local_mapping_matches_global_seed_mapping(self):
        funcs = _load_functions(SVD_TURBO_EVAL, {"_rank_local_seed_offset"})
        offset = funcs["_rank_local_seed_offset"]
        for idx in [0, 7, 137, 4999]:
            self.assertEqual(offset(idx, 5000, 1), idx)
            self.assertEqual(42 + offset(idx, 5000, 1), 42 + idx)

    def test_mixdq_rank_local_mapping_is_independent_of_selected_subset(self):
        funcs = _load_functions(MIXDQ_TURBO_EVAL, {"attach_original_shard_seed_offsets"})
        attach = funcs["attach_original_shard_seed_offsets"]
        selected = [0, 1, 1249, 1250, 3749, 3750, 4999]

        full_prompts = [{"global_idx": idx} for idx in range(5000)]
        attach(full_prompts, 5000, 4)
        full_assigned = {row["global_idx"]: 42 + row["_qdrift_seed_offset"] for row in full_prompts}

        selected_prompts = [{"global_idx": idx} for idx in selected]
        attach(selected_prompts, 5000, 4)
        selected_assigned = {row["global_idx"]: 42 + row["_qdrift_seed_offset"] for row in selected_prompts}
        self.assertEqual(selected_assigned, {idx: full_assigned[idx] for idx in selected})
        self.assertEqual(selected_assigned[0], 42)
        self.assertEqual(selected_assigned[1250], 10042)
        self.assertEqual(selected_assigned[3750], 30042)
        self.assertEqual(selected_assigned[4999], 31291)

    def test_mixdq_world1_rank_local_mapping_matches_global_seed_mapping(self):
        funcs = _load_functions(MIXDQ_TURBO_EVAL, {"attach_original_shard_seed_offsets"})
        attach = funcs["attach_original_shard_seed_offsets"]
        prompts = [{"global_idx": idx} for idx in [0, 7, 137, 4999]]
        attach(prompts, 5000, 1)
        for row in prompts:
            self.assertEqual(row["_qdrift_seed_offset"], row["global_idx"])
            self.assertEqual(42 + row["_qdrift_seed_offset"], 42 + row["global_idx"])

    def test_mixdq_wrapper_builds_disjoint_calibration_split(self):
        text = MIXDQ_RUN_PAPER.read_text()
        self.assertIn("prepare_calibration_split.py", text)
        self.assertIn("--calibration_num_samples \"$CALIB_NUM_SAMPLES\"", text)
        self.assertIn("--evaluation_seed \"$EVALUATION_PROMPT_SAMPLING_SEED\"", text)
        self.assertIn("--calibration_seed \"$CALIBRATION_PROMPT_SAMPLING_SEED\"", text)
        self.assertIn("CALIBRATION_PROMPT_FILE=\"$SPLIT_DIR/eligible_calibration_metadata.json\"", text)
        self.assertIn("--prompt_file \"$CALIBRATION_PROMPT_FILE\"", text)


if __name__ == "__main__":
    unittest.main()
