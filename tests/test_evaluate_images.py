from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "evaluation" / "evaluate_images.py"


def load_module():
    spec = importlib.util.spec_from_file_location("evaluate_images", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_png(path: Path, color=(0, 0, 0)):
    Image.new("RGB", (4, 4), color=color).save(path)


class EvaluateImagesTest(unittest.TestCase):
    def test_prompt_selection_strict_mode_rejects_missing_and_invalid_images(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            write_png(image_dir / "cat_00000.png")
            (image_dir / "cat_00001.png").write_bytes(b"")
            prompts = [
                {"prompt": "a", "category": "cat", "global_idx": 0},
                {"prompt": "b", "category": "cat", "global_idx": 1},
                {"prompt": "c", "category": "cat", "global_idx": 2},
            ]

            with self.assertRaisesRegex(ValueError, "prompt/image pairing is incomplete"):
                module.select_prompt_pairs(image_dir, prompts, allow_missing=False)

            kept, missing, invalid = module.select_prompt_pairs(image_dir, prompts, allow_missing=True)
            self.assertEqual([row["filename"] for row in kept], ["cat_00000.png"])
            self.assertEqual(missing, ["cat_00002.png"])
            self.assertEqual(invalid, ["cat_00001.png"])


    def test_duplicate_prompt_filename_ids_are_rejected(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            prompts_path = Path(tmp) / "prompts.json"
            prompts_path.write_text(
                json.dumps(
                    {
                        "prompts": [
                            {"prompt": "a", "category": "cat", "global_idx": 0},
                            {"prompt": "b", "category": "cat", "global_idx": 0},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate prompt filename ids"):
                module.load_prompts(prompts_path)

    def test_no_prompt_selection_rejects_invalid_images_unless_allowed(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            write_png(image_dir / "ok.png")
            (image_dir / "bad.png").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "invalid image files"):
                module.select_image_names(image_dir, prompts=None, allow_missing=False)
            names, prompt_pairs, excluded = module.select_image_names(image_dir, prompts=None, allow_missing=True)
            self.assertEqual(names, ["ok.png"])
            self.assertEqual(prompt_pairs, [])
            self.assertEqual(excluded["invalid"], ["bad.png"])

    def test_main_writes_counted_clip_only_result_with_allowed_missing_images(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            write_png(image_dir / "cat_00000.png")
            (image_dir / "cat_00001.png").write_bytes(b"")
            prompts_path = root / "prompts.json"
            prompts_path.write_text(
                json.dumps(
                    {
                        "prompts": [
                            {"prompt": "a", "category": "cat", "global_idx": 0},
                            {"prompt": "b", "category": "cat", "global_idx": 1},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "metrics.json"
            calls = []

            def fake_clip(prompt_pairs, image_dir_arg, batch_size, num_workers, device):
                calls.append((prompt_pairs, image_dir_arg, batch_size, num_workers, device))
                return {"clip_score": 12.5, "clip_score_num_pairs": len(prompt_pairs)}

            module.compute_clip_score = fake_clip
            rc = module.main(
                [
                    "--image-dir",
                    str(image_dir),
                    "--prompts",
                    str(prompts_path),
                    "--output",
                    str(output),
                    "--metrics",
                    "clip",
                    "--expected-count",
                    "1",
                    "--allow-missing",
                    "--device",
                    "cpu",
                ]
            )

            self.assertEqual(rc, 0)
            self.assertEqual(len(calls), 1)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["num_images_selected"], 1)
            self.assertEqual(result["excluded"]["invalid"], ["cat_00001.png"])
            self.assertRegex(result["provenance"]["evaluator_sha256"], r"^[0-9a-f]{64}$")
            self.assertIn("torch", result["provenance"]["package_versions"])
            self.assertEqual(result["results"]["clip_score"], 12.5)
            self.assertEqual(result["results"]["clip_score_num_pairs"], 1)

    def test_expected_pairs_can_differ_from_generated_count(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, fp = root / "images", root / "fp"
            images.mkdir()
            fp.mkdir()
            for name in ("a.png", "b.png"):
                write_png(images / name)
            write_png(fp / "a.png")
            module.compute_similarity = lambda *args: {
                "psnr_vs_fp16": 20.0, "similarity_num_pairs": len(args[2])
            }
            output = root / "metrics.json"
            argv = ["--image-dir", str(images), "--fp-dir", str(fp),
                    "--output", str(output), "--metrics", "psnr",
                    "--expected-count", "2", "--allow-missing", "--device", "cpu"]
            with self.assertRaisesRegex(SystemExit, "similarity pair count"):
                module.main(argv)
            self.assertEqual(module.main(argv + ["--expected-pairs", "1"]), 0)
            result = json.loads(output.read_text())
            self.assertEqual(result["num_images_selected"], 2)
            self.assertEqual(result["results"]["similarity_num_pairs"], 1)
            self.assertEqual(result["excluded"]["missing_or_invalid_fp"], ["b.png"])

    def test_similarity_pairing_rejects_missing_fp_by_default(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fp_dir = root / "fp"
            fp_dir.mkdir()
            write_png(fp_dir / "cat_00000.png")

            with self.assertRaisesRegex(ValueError, "fp pairing is incomplete"):
                module.validate_similarity_names(
                    fp_dir,
                    ["cat_00000.png", "cat_00001.png"],
                    allow_missing=False,
                )

            names, missing = module.validate_similarity_names(
                fp_dir,
                ["cat_00000.png", "cat_00001.png"],
                allow_missing=True,
            )
            self.assertEqual(names, ["cat_00000.png"])
            self.assertEqual(missing, ["cat_00001.png"])


if __name__ == "__main__":
    unittest.main()
