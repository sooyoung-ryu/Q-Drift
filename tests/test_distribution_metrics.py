from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "evaluation" / "compute_distribution_metrics.py"
SPEC = importlib.util.spec_from_file_location("compute_distribution_metrics", MODULE_PATH)
metrics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = metrics
SPEC.loader.exec_module(metrics)


def _features(values, names):
    return metrics.FeatureSet(np.asarray(values, dtype=np.float64), list(names))


def test_align_by_names_requires_exact_match_by_default():
    first = _features([[0.0], [1.0]], ["a.png", "b.png"])
    second = _features([[0.0], [1.0]], ["a.png", "c.png"])

    try:
        metrics.align_by_names(first, second, allow_intersection=False)
    except ValueError as exc:
        assert "must match exactly" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_align_by_names_records_intersection_provenance():
    first = _features([[0.0], [1.0], [2.0]], ["a.png", "b.png", "d.png"])
    second = _features([[3.0], [2.0], [4.0]], ["c.png", "a.png", "d.png"])

    left, right, names, provenance = metrics.align_by_names(first, second, allow_intersection=True)

    assert names == ["a.png", "d.png"]
    np.testing.assert_allclose(left, [[0.0], [2.0]])
    np.testing.assert_allclose(right, [[2.0], [4.0]])
    assert provenance["excluded_from_first"] == ["b.png"]
    assert provenance["excluded_from_second"] == ["c.png"]


def test_polynomial_kid_matches_manual_unbiased_estimator():
    x = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    y = np.asarray([[1.0, 0.0], [3.0, 2.0]], dtype=np.float64)
    dim = x.shape[1]
    k_xx = (x @ x.T / dim + 1.0) ** 3
    k_yy = (y @ y.T / dim + 1.0) ** 3
    k_xy = (x @ y.T / dim + 1.0) ** 3
    expected = (
        (k_xx.sum() - np.trace(k_xx)) / 2
        + (k_yy.sum() - np.trace(k_yy)) / 2
        - 2 * k_xy.mean()
    )
    assert metrics.polynomial_mmd2_unbiased(x, y) == expected




def test_compute_command_is_independent_of_manifest_splitting(tmp_path: Path):
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    ref = _features(np.arange(15, dtype=np.float64).reshape(5, 3), [f"{i}.png" for i in range(5)])
    quant = _features(ref.features + 0.25, ref.names)
    qdrift = _features(ref.features + 0.10, ref.names)
    other = _features(ref.features + 0.50, ref.names)
    metrics.write_features(feature_root / "ref.npz", ref)
    metrics.write_features(feature_root / "quant.npz", quant)
    metrics.write_features(feature_root / "qdrift.npz", qdrift)
    metrics.write_features(feature_root / "other.npz", other)

    all_manifest = {
        "ref": {"features": "ref.npz"},
        "rows": [
            {"label": "Toy", "setting": "W1A1", "method": "Quantized", "features": "quant.npz"},
            {"label": "Toy", "setting": "W1A1", "method": "Q-Drift", "features": "qdrift.npz"},
            {"label": "Other", "setting": "W2A2", "method": "Quantized", "features": "other.npz"},
        ],
    }
    split_manifest = {"ref": all_manifest["ref"], "rows": all_manifest["rows"][:2]}
    all_path = tmp_path / "all.json"
    split_path = tmp_path / "split.json"
    all_out = tmp_path / "all_out.json"
    split_out = tmp_path / "split_out.json"
    all_path.write_text(json.dumps(all_manifest), encoding="utf-8")
    split_path.write_text(json.dumps(split_manifest), encoding="utf-8")

    for manifest_path, output_path in [(all_path, all_out), (split_path, split_out)]:
        args = metrics.build_parser().parse_args(
            [
                "compute",
                "--manifest",
                str(manifest_path),
                "--feature-root",
                str(feature_root),
                "--output",
                str(output_path),
                "--device",
                "cpu",
                "--kid-subsets",
                "2",
                "--kid-subset-size",
                "3",
                "--fid-bootstrap",
                "1",
            ]
        )
        assert metrics.command_compute(args) == 0

    all_result = json.loads(all_out.read_text(encoding="utf-8"))
    split_result = json.loads(split_out.read_text(encoding="utf-8"))
    assert all_result["rows"][:2] == split_result["rows"]
    assert all_result["paired_rows"][:1] == split_result["paired_rows"]


def test_compute_command_writes_kid_and_paired_ci(tmp_path: Path):
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    ref = _features(np.arange(12, dtype=np.float64).reshape(4, 3), ["a.png", "b.png", "c.png", "d.png"])
    quant = _features(ref.features + 0.25, ref.names)
    qdrift = _features(ref.features + 0.10, ref.names)
    metrics.write_features(feature_root / "ref.npz", ref)
    metrics.write_features(feature_root / "quant.npz", quant)
    metrics.write_features(feature_root / "qdrift.npz", qdrift)

    manifest = {
        "ref": {"features": "ref.npz"},
        "rows": [
            {"label": "Toy", "setting": "W1A1", "method": "Quantized", "features": "quant.npz"},
            {"label": "Toy", "setting": "W1A1", "method": "Q-Drift", "features": "qdrift.npz"},
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "out.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    args = metrics.build_parser().parse_args(
        [
            "compute",
            "--manifest",
            str(manifest_path),
            "--feature-root",
            str(feature_root),
            "--output",
            str(output_path),
            "--device",
            "cpu",
            "--kid-subsets",
            "2",
            "--kid-subset-size",
            "3",
            "--fid-bootstrap",
            "1",
        ]
    )
    assert metrics.command_compute(args) == 0

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["format"] == "qdrift_distribution_metrics_v2"
    assert len(result["rows"]) == 2
    assert "kid_x1000" in result["rows"][0]
    paired = result["paired_rows"][0]
    assert paired["num_common_gen"] == 4
    assert "delta_fid_ci_low" in paired
    assert "delta_fid_ci_high" in paired


def test_fid_matches_scipy_and_bootstrap_pairing():
    from scipy.linalg import sqrtm
    rng = np.random.default_rng(91)
    ref = rng.normal(size=(40, 6))
    quant = rng.normal(size=(40, 6)) * 1.2
    corrected = quant * 0.9 + 0.1
    def scipy_fid(a, b):
        ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
        return float(np.sum((a.mean(0) - b.mean(0)) ** 2) + np.trace(ca + cb - 2 * sqrtm(ca @ cb).real))
    np.testing.assert_allclose(metrics.frechet_distance_torch(ref, quant, 'cpu'), scipy_fid(ref, quant), atol=1e-10)
    actual = metrics.paired_delta_fid_ci(ref, quant, corrected, np.random.default_rng(32), 8, 'cpu')
    rng = np.random.default_rng(32)
    differences = []
    for _ in range(8):
        ri = rng.integers(0, 40, size=40)
        gi = rng.integers(0, 40, size=40)
        differences.append(scipy_fid(ref[ri], corrected[gi]) - scipy_fid(ref[ri], quant[gi]))
    np.testing.assert_allclose([actual['delta_fid_ci_low'], actual['delta_fid_ci_high']], np.percentile(differences, [2.5, 97.5]), atol=1e-10)
