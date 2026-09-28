import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER_PATH = ROOT / "schedulers" / "dpm_solver_qdrift.py"


spec = importlib.util.spec_from_file_location("dpm_solver_qdrift", SCHEDULER_PATH)
dpm_solver_qdrift = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dpm_solver_qdrift)
DPMSolverQDriftScheduler = dpm_solver_qdrift.DPMSolverQDriftScheduler


def _stats_files(num_channels=3):
    mu = np.zeros((2, num_channels), dtype=np.float32)
    cov = np.zeros((2, 2, num_channels), dtype=np.float32)
    cov[0, 0] = np.array([2.0, 3.0, 5.0], dtype=np.float32)[:num_channels]
    cov[1, 1] = np.array([1.0, 4.0, 9.0], dtype=np.float32)[:num_channels]

    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)
    mu_path = temp_path / "mu.npy"
    cov_path = temp_path / "cov.npy"
    np.save(mu_path, {0.0: mu, 999.0: mu})
    np.save(cov_path, {0.0: cov, 999.0: cov})
    return temp_dir, str(mu_path), str(cov_path)


def _scalar_stats_files():
    mu = np.zeros((2, 1), dtype=np.float32)
    cov = np.zeros((2, 2, 1), dtype=np.float32)
    cov[0, 0, 0] = 2.0
    cov[1, 1, 0] = 7.0

    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)
    mu_path = temp_path / "mu.npy"
    cov_path = temp_path / "cov.npy"
    np.save(mu_path, {0.0: mu, 999.0: mu})
    np.save(cov_path, {0.0: cov, 999.0: cov})
    return temp_dir, str(mu_path), str(cov_path)


def _scheduler(mu_path, cov_path, scalar, prediction_type="epsilon", use_flow_sigmas=False):
    base = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        algorithm_type="dpmsolver++",
        solver_order=2,
        solver_type="midpoint",
        prediction_type=prediction_type,
        use_flow_sigmas=use_flow_sigmas,
        lower_order_final=True,
    )
    wrapper = DPMSolverQDriftScheduler(
        base_scheduler=base,
        mu_dict_path=mu_path,
        cov_dict_path=cov_path,
        bias_scale=0.0,
        qdrift_scale=1.0,
        qdrift_scalar=scalar,
    )
    wrapper.set_timesteps(5)
    return wrapper


def _recorded_step_sequence(scheduler, channels=3):
    calls = []
    original = scheduler._qdrift_scale_from_coeffs

    def record_scale(*args, **kwargs):
        scale = original(*args, **kwargs)
        calls.append((len(kwargs["coeffs_x"]), scale.detach().clone()))
        return scale

    scheduler._qdrift_scale_from_coeffs = record_scale
    sample = torch.linspace(-0.4, 0.5, channels * 2 * 2, dtype=torch.float32).reshape(1, channels, 2, 2)

    for i, timestep in enumerate(scheduler.timesteps):
        model_output = torch.full_like(sample, 0.01 * (i + 1))
        sample = scheduler.step(model_output, timestep, sample).prev_sample

    return calls, sample


class DpmQDriftScalarTest(unittest.TestCase):
    def test_qdrift_scalar_collapses_actual_warmup_midpoint_and_final_steps(self):
        stats_dir, mu_path, cov_path = _stats_files()
        try:
            scalar_scheduler = _scheduler(mu_path, cov_path, scalar=True)
            channel_scheduler = _scheduler(mu_path, cov_path, scalar=False)

            scalar_calls, scalar_sample = _recorded_step_sequence(scalar_scheduler)
            channel_calls, channel_sample = _recorded_step_sequence(channel_scheduler)

            self.assertEqual([kind for kind, _ in scalar_calls], [1, 2, 2, 2, 1])
            self.assertEqual([kind for kind, _ in channel_calls], [1, 2, 2, 2, 1])

            for _, scale in scalar_calls:
                self.assertEqual(scale.shape[1], 1)

            for _, scale in channel_calls:
                self.assertEqual(scale.shape[1], 3)
                self.assertFalse(torch.allclose(scale[:, 0], scale[:, 1]))

            self.assertFalse(torch.allclose(scalar_sample, channel_sample))
        finally:
            stats_dir.cleanup()

    def test_scalar_shape_statistics_broadcast_through_pixart_epsilon_path(self):
        stats_dir, mu_path, cov_path = _scalar_stats_files()
        try:
            scheduler = _scheduler(mu_path, cov_path, scalar=True, prediction_type="epsilon")
            calls, _ = _recorded_step_sequence(scheduler)
            self.assertEqual([kind for kind, _ in calls], [1, 2, 2, 2, 1])
            self.assertTrue(all(scale.shape[1] == 1 for _, scale in calls))
        finally:
            stats_dir.cleanup()

    def test_scalar_shape_statistics_broadcast_through_sana_flow_prediction_path(self):
        stats_dir, mu_path, cov_path = _scalar_stats_files()
        try:
            scheduler = _scheduler(mu_path, cov_path, scalar=True, prediction_type="flow_prediction", use_flow_sigmas=True)
            calls, _ = _recorded_step_sequence(scheduler)
            self.assertEqual([kind for kind, _ in calls], [1, 2, 2, 2, 1])
            self.assertTrue(all(scale.shape[1] == 1 for _, scale in calls))
        finally:
            stats_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
