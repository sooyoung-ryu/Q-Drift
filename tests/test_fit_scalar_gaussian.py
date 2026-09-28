from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def _find_module_path() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates = [parent / "fit_scalar_gaussian.py", parent / "scripts" / "fit_scalar_gaussian.py"]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    raise FileNotFoundError("fit_scalar_gaussian.py")


_MODULE_PATH = _find_module_path()
_SPEC = importlib.util.spec_from_file_location("fit_scalar_gaussian", _MODULE_PATH)
fit_scalar_gaussian = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = fit_scalar_gaussian
_SPEC.loader.exec_module(fit_scalar_gaussian)


def _write_pairs(
    path: Path,
    fp16_by_t: dict[int | float, torch.Tensor],
    quant_by_t: dict[int | float, torch.Tensor],
    *,
    timesteps: list[int | float] | None = None,
    num_samples: int | None = None,
) -> None:
    if timesteps is None:
        timesteps = sorted(fp16_by_t.keys())
    if num_samples is None:
        first = fp16_by_t[timesteps[0]]
        num_samples = int(first.shape[0])
    torch.save(
        {
            "fp16_output": fp16_by_t,
            "quant_output": quant_by_t,
            "timesteps": timesteps,
            "num_samples": num_samples,
        },
        path,
    )


def _write_single_timestep(path: Path, fp16: torch.Tensor, quant: torch.Tensor, *, timestep: int = 0, num_samples: int | None = None) -> None:
    _write_pairs(path, {timestep: fp16}, {timestep: quant}, timesteps=[timestep], num_samples=num_samples)


def _conditional_variance(cov: np.ndarray) -> float:
    return float(cov[1, 1] - cov[0, 1] * cov[0, 1] / cov[0, 0])


def test_scalar_fit_pools_all_channels_before_covariance(tmp_path: Path) -> None:
    quant_np = np.array(
        [
            [[[0.0]], [[10.0]]],
            [[[1.0]], [[11.0]]],
            [[[2.0]], [[12.0]]],
            [[[3.0]], [[13.0]]],
        ],
        dtype=np.float32,
    )
    error_np = np.array(
        [
            [[[0.0]], [[5.0]]],
            [[[0.0]], [[6.0]]],
            [[[1.0]], [[6.0]]],
            [[[1.0]], [[7.0]]],
        ],
        dtype=np.float32,
    )
    fp16 = torch.from_numpy(quant_np - error_np)
    quant = torch.from_numpy(quant_np)
    input_path = tmp_path / "data_output_pairs.pth"
    _write_single_timestep(input_path, fp16, quant)

    mu_dict, cov_dict, metadata = fit_scalar_gaussian.fit_scalar_gaussian_models(
        data_output_pairs_paths=[input_path],
        output_dir=tmp_path / "scalar",
        chunk_elements=3,
        expected_num_samples=4,
    )

    joint = np.vstack([quant_np.reshape(-1), error_np.reshape(-1)])
    expected_mu = joint.mean(axis=1).reshape(2, 1)
    expected_cov = np.cov(joint).reshape(2, 2, 1)
    np.testing.assert_allclose(mu_dict[0], expected_mu)
    np.testing.assert_allclose(cov_dict[0], expected_cov)
    assert mu_dict[0].shape == (2, 1)
    assert cov_dict[0].shape == (2, 2, 1)
    assert metadata["num_samples_inferred"] == 4

    mean_channel_v = np.mean([
        _conditional_variance(np.cov(joint[:, [0, 2, 4, 6]])),
        _conditional_variance(np.cov(joint[:, [1, 3, 5, 7]])),
    ])
    pooled_v = _conditional_variance(cov_dict[0][:, :, 0])
    assert not np.isclose(pooled_v, mean_channel_v)


def test_scalar_fit_trims_global_error_outlier(tmp_path: Path) -> None:
    quant_np = np.array([0.0, 1.0, 2.0, 3.0, 1000.0], dtype=np.float32).reshape(5, 1, 1, 1)
    error_np = np.array([0.0, 0.1, 0.2, 0.3, 100.0], dtype=np.float32).reshape(5, 1, 1, 1)
    fp16 = torch.from_numpy(quant_np - error_np)
    quant = torch.from_numpy(quant_np)
    input_path = tmp_path / "data_output_pairs.pth"
    _write_single_timestep(input_path, fp16, quant)

    _mu_dict, _cov_dict, metadata = fit_scalar_gaussian.fit_scalar_gaussian_models(
        data_output_pairs_paths=[input_path],
        output_dir=tmp_path / "scalar",
        outlier_threshold=1.0,
        chunk_elements=2,
    )

    counts = metadata["counts"]["0"]
    assert counts["total_values"] == 5
    assert counts["retained_values"] == 4
    assert counts["trimmed_values"] == 1


def test_scalar_fit_refuses_to_overwrite(tmp_path: Path) -> None:
    fp16 = torch.zeros(2, 1, 1, 1)
    quant = torch.ones(2, 1, 1, 1)
    input_path = tmp_path / "data_output_pairs.pth"
    output_dir = tmp_path / "scalar"
    _write_single_timestep(input_path, fp16, quant)

    fit_scalar_gaussian.fit_scalar_gaussian_models(data_output_pairs_paths=[input_path], output_dir=output_dir)
    try:
        fit_scalar_gaussian.fit_scalar_gaussian_models(data_output_pairs_paths=[input_path], output_dir=output_dir)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Expected FileExistsError")


def test_cli_resolution_rejects_merged_plus_rank_shards(tmp_path: Path) -> None:
    merged = tmp_path / "data_output_pairs.pth"
    shard = tmp_path / "data_output_pairs_rank0.pth"
    merged.touch()
    shard.touch()
    parser = fit_scalar_gaussian.build_arg_parser()
    args = parser.parse_args([
        "--data_output_pairs_glob",
        str(tmp_path / "data_output_pairs*.pth"),
        "--output_dir",
        str(tmp_path / "out"),
    ])
    try:
        fit_scalar_gaussian._resolve_input_paths(args)
    except ValueError as exc:
        assert "double-counting" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_duplicate_explicit_paths_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "data_output_pairs.pth"
    path.touch()
    parser = fit_scalar_gaussian.build_arg_parser()
    args = parser.parse_args([
        "--data_output_pairs_path",
        str(path),
        "--data_output_pairs_path",
        str(path),
        "--output_dir",
        str(tmp_path / "out"),
    ])
    try:
        fit_scalar_gaussian._resolve_input_paths(args)
    except ValueError as exc:
        assert "Duplicate input path" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_expected_num_samples_and_metadata_counts_are_validated_before_write(tmp_path: Path) -> None:
    fp16 = torch.zeros(2, 1, 1, 1)
    quant = torch.ones(2, 1, 1, 1)
    bad_metadata = tmp_path / "bad_metadata.pth"
    bad_expected = tmp_path / "bad_expected.pth"
    _write_single_timestep(bad_metadata, fp16, quant, num_samples=3)
    _write_single_timestep(bad_expected, fp16, quant)

    for path, kwargs in [
        (bad_metadata, {}),
        (bad_expected, {"expected_num_samples": 3}),
    ]:
        out = tmp_path / f"out_{path.stem}"
        try:
            fit_scalar_gaussian.fit_scalar_gaussian_models(data_output_pairs_paths=[path], output_dir=out, **kwargs)
        except ValueError as exc:
            assert "num_samples" in str(exc) or "Expected 3" in str(exc)
            assert not (out / "mu_dict.npy").exists()
            assert not (out / "cov_dict.npy").exists()
        else:
            raise AssertionError("Expected ValueError")


def test_inconsistent_timesteps_shapes_and_nonfinite_are_rejected(tmp_path: Path) -> None:
    fp16 = torch.zeros(2, 1, 1, 1)
    quant = torch.ones(2, 1, 1, 1)

    missing_timestep = tmp_path / "missing_timestep.pth"
    _write_pairs(missing_timestep, {0: fp16}, {0: quant}, timesteps=[0, 1])
    try:
        fit_scalar_gaussian.fit_scalar_gaussian_models(data_output_pairs_paths=[missing_timestep], output_dir=tmp_path / "out_missing")
    except ValueError as exc:
        assert "timestep sets" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

    shard0 = tmp_path / "data_output_pairs_rank0.pth"
    shard1 = tmp_path / "data_output_pairs_rank1.pth"
    _write_single_timestep(shard0, fp16, quant)
    _write_single_timestep(shard1, torch.zeros(2, 1, 2, 1), torch.ones(2, 1, 2, 1))
    try:
        fit_scalar_gaussian.fit_scalar_gaussian_models(data_output_pairs_paths=[shard0, shard1], output_dir=tmp_path / "out_shape")
    except ValueError as exc:
        assert "trailing shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

    nonfinite = tmp_path / "nonfinite.pth"
    q_bad = quant.clone()
    q_bad[0, 0, 0, 0] = float("nan")
    _write_single_timestep(nonfinite, fp16, q_bad)
    try:
        fit_scalar_gaussian.fit_scalar_gaussian_models(data_output_pairs_paths=[nonfinite], output_dir=tmp_path / "out_nonfinite")
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_duplicate_input_basenames_get_stable_metadata_keys(tmp_path: Path) -> None:
    fp16 = torch.zeros(2, 1, 1, 1)
    quant = torch.ones(2, 1, 1, 1)
    dir0 = tmp_path / "rank0"
    dir1 = tmp_path / "rank1"
    dir0.mkdir()
    dir1.mkdir()
    path0 = dir0 / "data_output_pairs.pth"
    path1 = dir1 / "data_output_pairs.pth"
    _write_single_timestep(path0, fp16, quant)
    _write_single_timestep(path1, fp16, quant)

    _mu_dict, _cov_dict, metadata = fit_scalar_gaussian.fit_scalar_gaussian_models(
        data_output_pairs_paths=[path0, path1],
        output_dir=tmp_path / "out",
        expected_num_samples=4,
    )

    assert metadata["file_sample_counts"] == {
        "00000_data_output_pairs.pth": 2,
        "00001_data_output_pairs.pth": 2,
    }
    assert metadata["file_sample_counts_available"] == metadata["file_sample_counts"]


def test_max_samples_uses_deterministic_prefix_across_inputs(tmp_path: Path) -> None:
    quant = torch.arange(10, dtype=torch.float32).reshape(10, 1, 1, 1)
    error = (torch.arange(10, dtype=torch.float32).reshape(10, 1, 1, 1) % 3) / 10.0
    fp16 = quant - error
    shard0 = tmp_path / "data_output_pairs_rank0_shard00000.pth"
    shard1 = tmp_path / "data_output_pairs_rank1_shard00000.pth"
    _write_single_timestep(shard0, fp16[:4], quant[:4])
    _write_single_timestep(shard1, fp16[4:], quant[4:])

    mu_dict, cov_dict, metadata = fit_scalar_gaussian.fit_scalar_gaussian_models(
        data_output_pairs_paths=[shard0, shard1],
        output_dir=tmp_path / "subset_out",
        expected_num_samples=6,
        max_samples=6,
    )

    actual_error = quant[:6].double() - fp16[:6].double()
    joint = np.vstack([quant[:6].double().numpy().reshape(-1), actual_error.numpy().reshape(-1)])
    np.testing.assert_allclose(mu_dict[0], joint.mean(axis=1).reshape(2, 1))
    np.testing.assert_allclose(cov_dict[0], np.cov(joint).reshape(2, 2, 1))
    assert metadata["num_samples_inferred"] == 6
    assert metadata["num_samples_available"] == 10
    assert metadata["file_sample_counts"] == {
        "data_output_pairs_rank0_shard00000.pth": 4,
        "data_output_pairs_rank1_shard00000.pth": 2,
    }
    assert metadata["file_sample_counts_available"] == {
        "data_output_pairs_rank0_shard00000.pth": 4,
        "data_output_pairs_rank1_shard00000.pth": 6,
    }
    assert metadata["max_samples"] == 6


def test_shard_equivalence_and_chunk_equivalence(tmp_path: Path) -> None:
    quant = torch.arange(24, dtype=torch.float32).reshape(6, 1, 2, 2)
    error = (torch.arange(24, dtype=torch.float32).reshape(6, 1, 2, 2) % 5) / 10.0
    fp16 = quant - error
    merged = tmp_path / "merged.pth"
    shard0 = tmp_path / "shard0.pth"
    shard1 = tmp_path / "shard1.pth"
    _write_single_timestep(merged, fp16, quant)
    _write_single_timestep(shard0, fp16[:2], quant[:2])
    _write_single_timestep(shard1, fp16[2:], quant[2:])

    mu_merged, cov_merged, _ = fit_scalar_gaussian.fit_scalar_gaussian_models(
        data_output_pairs_paths=[merged],
        output_dir=tmp_path / "merged_out",
        chunk_elements=7,
        expected_num_samples=6,
    )
    mu_shards, cov_shards, metadata = fit_scalar_gaussian.fit_scalar_gaussian_models(
        data_output_pairs_paths=[shard0, shard1],
        output_dir=tmp_path / "shard_out",
        chunk_elements=5,
        expected_num_samples=6,
    )
    mu_chunk, cov_chunk, _ = fit_scalar_gaussian.fit_scalar_gaussian_models(
        data_output_pairs_paths=[merged],
        output_dir=tmp_path / "chunk_out",
        chunk_elements=3,
        expected_num_samples=6,
    )

    np.testing.assert_allclose(mu_merged[0], mu_shards[0])
    np.testing.assert_allclose(cov_merged[0], cov_shards[0])
    np.testing.assert_allclose(mu_merged[0], mu_chunk[0])
    np.testing.assert_allclose(cov_merged[0], cov_chunk[0])
    assert metadata["num_samples_inferred"] == 6
    assert metadata["file_sample_counts"] == {"shard0.pth": 2, "shard1.pth": 4}
