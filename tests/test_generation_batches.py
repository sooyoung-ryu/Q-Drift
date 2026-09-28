from __future__ import annotations

import ast
import os
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIXART_EVAL = ROOT / "experiments" / "svdquant" / "pixart-sigma_w3a4" / "scripts" / "evaluate.py"
SANA_EVAL = ROOT / "experiments" / "svdquant" / "sana_w3a4" / "scripts" / "evaluate.py"


class _FakeTensor:
    pass


class _FakeGenerator:
    def __init__(self, device=None):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = int(seed)
        return self


class _FakeTorch:
    Tensor = _FakeTensor
    Generator = _FakeGenerator


class _FakeImage:
    def __init__(self, tag, saved):
        self.tag = tag
        self._saved = saved

    def save(self, path):
        Path(path).write_text(str(self.tag))
        self._saved.append(os.path.basename(path))


class _FakePipeline:
    def __init__(self):
        self.calls = []
        self.saved = []

    def __call__(self, **kwargs):
        prompts = kwargs["prompt"]
        generators = kwargs["generator"]
        prompt_list = prompts if isinstance(prompts, list) else [prompts]
        generator_list = generators if isinstance(generators, list) else [generators]
        seeds = [gen.seed for gen in generator_list]
        self.calls.append(
            {
                "prompts": list(prompt_list),
                "seeds": seeds,
                "negative_prompt": kwargs.get("negative_prompt"),
                "num_inference_steps": kwargs.get("num_inference_steps"),
                "guidance_scale": kwargs.get("guidance_scale"),
                "height": kwargs.get("height"),
                "width": kwargs.get("width"),
            }
        )
        return types.SimpleNamespace(images=[_FakeImage((prompt, seed), self.saved) for prompt, seed in zip(prompt_list, seeds)])


def _load_functions(path: Path):
    source = path.read_text()
    tree = ast.parse(source)
    wanted = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"_generate_images", "_apply_generation_args"}:
            segment = ast.get_source_segment(source, node)
            wanted.append(segment)
    namespace = {
        "os": os,
        "torch": _FakeTorch,
        "tqdm": lambda iterable, desc=None: iterable,
        "Dict": dict,
        "Any": object,
        "List": list,
        "Optional": object,
        "DEFAULT_NEGATIVE_PROMPT": "",
        "_ensure_dir": lambda path: os.makedirs(path, exist_ok=True),
        "ValueError": ValueError,
    }
    exec("from __future__ import annotations\n" + "\n\n".join(wanted), namespace)
    return namespace["_generate_images"], namespace["_apply_generation_args"]


def _prompts(n=6):
    return [{"prompt": f"prompt-{i}", "category": "cat", "global_idx": i} for i in range(n)]


class GenerationBatchTest(unittest.TestCase):
    def _run_batch_case(self, path: Path):
        generate_images, _ = _load_functions(path)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "out"
            existing_dir = tmp_path / "existing"
            existing_dir.mkdir()
            (existing_dir / "cat_00001.png").write_text("existing")

            pipe = _FakePipeline()
            kwargs = dict(
                pipeline=pipe,
                prompts=_prompts(),
                output_dir=str(output_dir),
                existing_dirs=[str(existing_dir)],
                prefix="qdrift",
                noise_seed_start=100,
                num_inference_steps=20,
                guidance_scale=4.5,
                height=64,
                width=64,
                device="cpu",
                batch_size=2,
                xt_logger=None,
            )
            if "sana_w3a4" in str(path):
                kwargs["negative_prompt"] = "neg"

            generate_images(**kwargs)

            self.assertEqual(
                [call["prompts"] for call in pipe.calls],
                [["prompt-0", "prompt-2"], ["prompt-3", "prompt-4"], ["prompt-5"]],
            )
            self.assertEqual([call["seeds"] for call in pipe.calls], [[100, 102], [103, 104], [105]])
            self.assertEqual(
                sorted(pipe.saved),
                ["cat_00000.png", "cat_00002.png", "cat_00003.png", "cat_00004.png", "cat_00005.png"],
            )
            self.assertFalse((output_dir / "cat_00001.png").exists())
            self.assertTrue(all(call["num_inference_steps"] == 20 for call in pipe.calls))
            self.assertTrue(all(call["guidance_scale"] == 4.5 for call in pipe.calls))
            self.assertTrue(all(call["height"] == 64 for call in pipe.calls))
            self.assertTrue(all(call["width"] == 64 for call in pipe.calls))
            if "sana_w3a4" in str(path):
                self.assertTrue(all(call["negative_prompt"] == ["neg"] * len(call["prompts"]) for call in pipe.calls))
            else:
                self.assertTrue(all(call["negative_prompt"] == [""] * len(call["prompts"]) for call in pipe.calls))

    def test_pixart_batch_preserves_global_seed_filename_and_resume_mapping(self):
        self._run_batch_case(PIXART_EVAL)

    def test_sana_batch_preserves_global_seed_filename_and_resume_mapping(self):
        self._run_batch_case(SANA_EVAL)

    def test_qdrift_only_sets_baseline_skips_for_both_scripts(self):
        for path in [PIXART_EVAL, SANA_EVAL]:
            _, apply_generation_args = _load_functions(path)
            args = types.SimpleNamespace(
                xt_output_dir=None,
                output_dir="eval",
                batch_size=1,
                qdrift_only=True,
                skip_fp16=False,
                skip_quant_baseline=False,
                save_xt=False,
                save_xt_fp16=False,
                save_xt_quant_baseline=False,
            )
            out = apply_generation_args(args)
            self.assertEqual(out.xt_output_dir, "eval")
            self.assertTrue(out.skip_fp16)
            self.assertTrue(out.skip_quant_baseline)

    def test_batched_generation_requires_no_xt_logging_for_both_scripts(self):
        for path in [PIXART_EVAL, SANA_EVAL]:
            _, apply_generation_args = _load_functions(path)
            args = types.SimpleNamespace(
                xt_output_dir="xt",
                output_dir="eval",
                batch_size=2,
                qdrift_only=False,
                skip_fp16=False,
                skip_quant_baseline=False,
                save_xt=True,
                save_xt_fp16=False,
                save_xt_quant_baseline=False,
            )
            with self.assertRaisesRegex(ValueError, "Batched generation does not support x_t logging"):
                apply_generation_args(args)


if __name__ == "__main__":
    unittest.main()
