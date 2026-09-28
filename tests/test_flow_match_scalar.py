import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER_PATH = ROOT / "schedulers" / "flow_match_euler_qdrift.py"

spec = importlib.util.spec_from_file_location("flow_match_euler_qdrift", SCHEDULER_PATH)
flow_match_euler_qdrift = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flow_match_euler_qdrift)
FlowMatchEulerQDriftScheduler = flow_match_euler_qdrift.FlowMatchEulerQDriftScheduler


def _stats_files(mu: np.ndarray, cov: np.ndarray):
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)
    mu_path = temp_path / "mu.npy"
    cov_path = temp_path / "cov.npy"
    np.save(mu_path, {1000.0: mu, 0.0: mu})
    np.save(cov_path, {1000.0: cov, 0.0: cov})
    return temp_dir, str(mu_path), str(cov_path)


def _scalar_stats_files():
    mu = np.array([[0.25], [0.50]], dtype=np.float32)
    cov = np.zeros((2, 2, 1), dtype=np.float32)
    cov[0, 0, 0] = 4.0
    cov[1, 1, 0] = 9.0
    cov[0, 1, 0] = 3.0
    cov[1, 0, 0] = 3.0
    return _stats_files(mu, cov)


def _channel_stats_files():
    mu = np.array(
        [
            [0.10, 0.20, 0.30],
            [0.40, 0.50, 0.60],
        ],
        dtype=np.float32,
    )
    cov = np.zeros((2, 2, 3), dtype=np.float32)
    cov[0, 0] = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    cov[1, 1] = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    cov[0, 1] = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    cov[1, 0] = cov[0, 1]
    return _stats_files(mu, cov)


def _scheduler(mu_path: str, cov_path: str, scalar: bool = True):
    scheduler = FlowMatchEulerQDriftScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        mu_dict_path=mu_path,
        cov_dict_path=cov_path,
        bias_scale=0.0,
        qdrift_scale=1.0,
        qdrift_scalar=scalar,
    )
    scheduler.set_timesteps(3)
    return scheduler


class FlowMatchQDriftScalarBroadcastTest(unittest.TestCase):
    def test_scalar_statistics_broadcast_to_packed_flux_layout(self):
        stats_dir, mu_path, cov_path = _scalar_stats_files()
        try:
            scheduler = _scheduler(mu_path, cov_path, scalar=True)
            model_output = torch.zeros(2, 5, 3)
            mu, cov = scheduler.get_conditional_statistics(scheduler.timesteps[0])
            mu_q, mu_e, var_q, var_e, cov_qe = scheduler._broadcast_statistics(
                mu, cov, model_output, model_output.device
            )

            self.assertEqual(tuple(mu_q.shape), (2, 5, 3))
            self.assertTrue(torch.allclose(mu_q, torch.full_like(model_output, 0.25)))
            self.assertTrue(torch.allclose(mu_e, torch.full_like(model_output, 0.50)))
            self.assertTrue(torch.allclose(var_q, torch.full_like(model_output, 4.0)))
            self.assertTrue(torch.allclose(var_e, torch.full_like(model_output, 9.0)))
            self.assertTrue(torch.allclose(cov_qe, torch.full_like(model_output, 3.0)))

            sigma_q = torch.sqrt(var_q)
            sigma_e = torch.sqrt(var_e)
            rho = cov_qe / (sigma_q * sigma_e + 1e-8)
            sigma_cond_sq = (sigma_e * torch.sqrt(1 - rho**2 + 1e-8)) ** 2
            self.assertTrue(torch.allclose(sigma_cond_sq, torch.full_like(model_output, 6.75)))
        finally:
            stats_dir.cleanup()

    def test_channelwise_statistics_preserve_image_and_packed_channel_axes(self):
        stats_dir, mu_path, cov_path = _channel_stats_files()
        try:
            scheduler = _scheduler(mu_path, cov_path, scalar=False)
            mu, cov = scheduler.get_conditional_statistics(scheduler.timesteps[0])

            image_output = torch.zeros(2, 3, 4, 4)
            _, _, _, image_var_e, _ = scheduler._broadcast_statistics(mu, cov, image_output, image_output.device)
            self.assertTrue(torch.allclose(image_var_e[:, 0], torch.full((2, 4, 4), 5.0)))
            self.assertTrue(torch.allclose(image_var_e[:, 1], torch.full((2, 4, 4), 6.0)))
            self.assertTrue(torch.allclose(image_var_e[:, 2], torch.full((2, 4, 4), 7.0)))

            packed_output = torch.zeros(2, 5, 3)
            _, _, _, packed_var_e, _ = scheduler._broadcast_statistics(mu, cov, packed_output, packed_output.device)
            self.assertTrue(torch.allclose(packed_var_e[..., 0], torch.full((2, 5), 5.0)))
            self.assertTrue(torch.allclose(packed_var_e[..., 1], torch.full((2, 5), 6.0)))
            self.assertTrue(torch.allclose(packed_var_e[..., 2], torch.full((2, 5), 7.0)))
        finally:
            stats_dir.cleanup()

    def test_actual_step_matches_scalar_analytic_correction_for_batch_two(self):
        stats_dir, mu_path, cov_path = _scalar_stats_files()
        try:
            scheduler = _scheduler(mu_path, cov_path, scalar=True)
            sample = torch.zeros(2, 5, 3)
            model_output = torch.full_like(sample, 0.2)
            timestep = scheduler.timesteps[0]

            sigma = scheduler.sigmas[0].to(dtype=torch.float32)
            sigma_next = scheduler.sigmas[1].to(dtype=torch.float32)
            dt = sigma_next - sigma
            conditional_variance = torch.tensor(6.75, dtype=torch.float32)
            correction_factor = conditional_variance * torch.abs(dt) / (2.0 * sigma + 1e-12)
            expected = sample + model_output * dt * (1.0 + correction_factor)

            actual = scheduler.step(model_output, timestep, sample).prev_sample

            self.assertEqual(tuple(actual.shape), (2, 5, 3))
            self.assertTrue(torch.isfinite(actual).all())
            self.assertTrue(torch.allclose(actual, expected.expand_as(actual), atol=1e-6, rtol=1e-6))
        finally:
            stats_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
