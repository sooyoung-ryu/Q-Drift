# Copyright 2025 TSAIL Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from bisect import bisect_left
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_utils import SchedulerOutput
from diffusers.utils.torch_utils import randn_tensor


class DPMSolverQDriftScheduler(torch.nn.Module):
    """
    DPMSolverMultistepScheduler + Q-Drift (marginal-preserving) correction.

    Design:
      1) Conditional bias removal on the *raw* UNet output:  u_corr = u_quant - E[error | u_quant]
      2) Compute conditional variance Var(error | u_quant) = sigma_cond^2
      3) Propagate that variance into the solver's converted model output m (epsilon or x0_pred) via linear maps
      4) For each DPM-Solver step, compute injected one-step variance in y=x/alpha space:
           Var(Δy) = Σ_j (C_j/alpha_t)^2 Var(Δm_j)
         Match it to the λ-parameterized marginal-preserving SDE diffusion:
           Var(Δy) = 2 * beta_tilde_lambda * ∫_{λ_s}^{λ_t} σ̄(λ)^2 dλ
                   = beta_tilde_lambda * (sigma_bar_s^2 - sigma_bar_t^2)
         => beta_tilde_lambda = Var(Δy) / (sigma_bar_s^2 - sigma_bar_t^2)
      5) Paired drift => scale the deterministic drift increment in y-space:
           y_t = y_s + a * (y_t^{base} - y_s),  a = 1 + qdrift_scale * beta_tilde_lambda.

    Assumptions:
      - thresholding=False (convert_model_output must stay linear in model_output)
      - mu/cov are collected in raw UNet output space, with error := (quant - fp16)
      - multistep quantization errors are approximately independent across timesteps

    Notes:
      - For algorithm_type in {"sde-dpmsolver", "sde-dpmsolver++"}, we default to bias removal only
        because those methods already add explicit diffusion.
    """

    def __init__(
        self,
        base_scheduler: DPMSolverMultistepScheduler,
        mu_dict_path: Optional[str] = None,
        cov_dict_path: Optional[str] = None,
        log_dir: Optional[str] = None,
        bias_scale: float = 1.0,
        qdrift_scale: float = 1.0,
        qdrift_scalar: bool = False,
    ):
        super().__init__()
        self.s = base_scheduler
        # Diffusers' `DiffusionPipeline.device` infers device from the first `torch.nn.Module`
        # component. Since this wrapper is a `torch.nn.Module` (while the underlying scheduler
        # is not), we need a reliable device anchor that is moved by `.to(device)` even before
        # any scheduler tensors (e.g., `sigmas`) are materialized.
        self.register_buffer("_pipeline_anchor", torch.empty(0, dtype=torch.float32), persistent=False)

        self.bias_scale = float(bias_scale)
        self.qdrift_scale = float(qdrift_scale)
        # Ablation: collapse the per-channel conditional variance to one scalar per step.
        self.qdrift_scalar = bool(qdrift_scalar)

        self.use_qdrift = (mu_dict_path is not None) and (cov_dict_path is not None)
        self.enable_logging = False
        self.log_file = None

        self.mu_dict = {}
        self.cov_dict = {}
        self._qdrift_available_timesteps: List[float] = []

        if self.use_qdrift:
            mu_raw = np.load(mu_dict_path, allow_pickle=True).item()
            cov_raw = np.load(cov_dict_path, allow_pickle=True).item()
            self.mu_dict = {float(k): v for k, v in mu_raw.items()}
            self.cov_dict = {float(k): v for k, v in cov_raw.items()}
            self._qdrift_available_timesteps = sorted(self.mu_dict.keys())

            if log_dir is not None:
                os.makedirs(log_dir, exist_ok=True)
                self.log_file = os.path.join(log_dir, "qdrift_log.jsonl")
                with open(self.log_file, "w", encoding="utf-8"):
                    pass
                self.enable_logging = True

        # Keep parallel buffers for the conditional variance of *converted* model outputs.
        # Align with base_scheduler.model_outputs shifting order.
        self.model_output_vars: List[Optional[torch.Tensor]] = [None] * int(self.s.config.solver_order)

    def __getattr__(self, name: str):
        # Delegate scheduler API/attrs to the wrapped diffusers scheduler.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.s, name)

    @property
    def dtype(self) -> torch.dtype:
        return self._pipeline_anchor.dtype

    @property
    def device(self) -> torch.device:
        return self._pipeline_anchor.device

    def to(self, *args, **kwargs):
        device = kwargs.get("device", None)
        if len(args) == 1 and not isinstance(args[0], torch.dtype):
            device = args[0]
        elif len(args) >= 2:
            device = args[0]

        # Move underlying scheduler tensors to device, but do NOT propagate dtype casting:
        # schedulers typically keep internal buffers in fp32 for stability, and diffusers'
        # native schedulers are not `torch.nn.Module`s (so pipelines wouldn't cast them).
        if device is not None and hasattr(self.s, "to"):
            try:
                self.s.to(device=torch.device(device))
            except TypeError:
                self.s.to(torch.device(device))

        # Only move this wrapper (anchor buffer) to device; keep dtype unchanged.
        if device is None:
            return super().to(*args, **kwargs)
        return super().to(device=device)

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "DPMSolverQDriftScheduler":
        mu_dict_path = kwargs.pop("mu_dict_path", None)
        cov_dict_path = kwargs.pop("cov_dict_path", None)
        log_dir = kwargs.pop("log_dir", None)
        bias_scale = kwargs.pop("bias_scale", 1.0)
        qdrift_scale = kwargs.pop("qdrift_scale", 1.0)
        qdrift_scalar = kwargs.pop("qdrift_scalar", False)
        base = DPMSolverMultistepScheduler.from_pretrained(*args, **kwargs)
        return cls(
            base_scheduler=base,
            mu_dict_path=mu_dict_path,
            cov_dict_path=cov_dict_path,
            log_dir=log_dir,
            bias_scale=bias_scale,
            qdrift_scale=qdrift_scale,
            qdrift_scalar=qdrift_scalar,
        )

    @classmethod
    def from_config(cls, config, **kwargs) -> "DPMSolverQDriftScheduler":
        mu_dict_path = kwargs.pop("mu_dict_path", None)
        cov_dict_path = kwargs.pop("cov_dict_path", None)
        log_dir = kwargs.pop("log_dir", None)
        bias_scale = kwargs.pop("bias_scale", 1.0)
        qdrift_scale = kwargs.pop("qdrift_scale", 1.0)
        qdrift_scalar = kwargs.pop("qdrift_scalar", False)
        base = DPMSolverMultistepScheduler.from_config(config, **kwargs)
        return cls(
            base_scheduler=base,
            mu_dict_path=mu_dict_path,
            cov_dict_path=cov_dict_path,
            log_dir=log_dir,
            bias_scale=bias_scale,
            qdrift_scale=qdrift_scale,
            qdrift_scalar=qdrift_scalar,
        )

    def set_timesteps(self, *args, **kwargs):
        out = self.s.set_timesteps(*args, **kwargs)
        self.model_output_vars = [None] * int(self.s.config.solver_order)
        return out

    def set_begin_index(self, *args, **kwargs):
        out = self.s.set_begin_index(*args, **kwargs)
        self.model_output_vars = [None] * int(self.s.config.solver_order)
        return out

    # ---------------------------
    # Stats helpers (same as EulerQDriftScheduler)
    # ---------------------------

    def get_conditional_statistics(
        self, timestep: Union[int, float, torch.Tensor], num_channels: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_qdrift:
            mu = torch.zeros((2, num_channels), dtype=torch.float32)
            cov = torch.eye(2, dtype=torch.float32).unsqueeze(-1).repeat(1, 1, num_channels)
            return mu, cov

        t_val = float(timestep.item()) if isinstance(timestep, torch.Tensor) else float(timestep)

        if t_val in self.mu_dict:
            mu = torch.from_numpy(self.mu_dict[t_val]).float()
            cov = torch.from_numpy(self.cov_dict[t_val]).float()
            return mu, cov

        available = self._qdrift_available_timesteps
        if len(available) == 0:
            mu = torch.zeros((2, num_channels), dtype=torch.float32)
            cov = torch.eye(2, dtype=torch.float32).unsqueeze(-1).repeat(1, 1, num_channels)
            return mu, cov

        if t_val <= available[0]:
            t0 = t1 = available[0]
            w = 0.0
        elif t_val >= available[-1]:
            t0 = t1 = available[-1]
            w = 0.0
        else:
            idx = bisect_left(available, t_val)
            t0 = available[idx - 1]
            t1 = available[idx]
            denom = (t1 - t0) if (t1 - t0) != 0 else 1.0
            w = float((t_val - t0) / denom)

        mu0 = self.mu_dict[t0]
        mu1 = self.mu_dict[t1]
        cov0 = self.cov_dict[t0]
        cov1 = self.cov_dict[t1]

        mu_np = (1.0 - w) * mu0 + w * mu1
        cov_np = (1.0 - w) * cov0 + w * cov1
        mu = torch.from_numpy(mu_np).float()
        cov = torch.from_numpy(cov_np).float()
        return mu, cov

    def _broadcast_statistics(
        self,
        mu: torch.Tensor,
        cov: torch.Tensor,
        num_channels: int,
        view_shape: List[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = mu.to(device=device, dtype=torch.float32)
        cov = cov.to(device=device, dtype=torch.float32)

        # mu: (2, C)
        if mu.ndim == 2 and mu.shape[0] == 2 and mu.shape[1] in {1, num_channels}:
            mu_quant = mu[0].view([1, mu.shape[1]] + [1] * (len(view_shape) - 2)).expand(view_shape)
            mu_error = mu[1].view([1, mu.shape[1]] + [1] * (len(view_shape) - 2)).expand(view_shape)
        else:
            raise ValueError(f"Unsupported mu shape: {tuple(mu.shape)}")

        # cov: (2,2,C)
        if cov.ndim == 3 and cov.shape[0] == 2 and cov.shape[1] == 2 and cov.shape[2] in {1, num_channels}:
            var_quant = cov[0, 0].view([1, cov.shape[2]] + [1] * (len(view_shape) - 2)).expand(view_shape)
            var_error = cov[1, 1].view([1, cov.shape[2]] + [1] * (len(view_shape) - 2)).expand(view_shape)
            cov_quant_error = cov[0, 1].view([1, cov.shape[2]] + [1] * (len(view_shape) - 2)).expand(view_shape)
        else:
            raise ValueError(f"Unsupported cov shape: {tuple(cov.shape)}")

        return mu_quant, mu_error, var_quant, var_error, cov_quant_error

    # ---------------------------
    # Variance propagation through convert_model_output (linear part only)
    # ---------------------------

    def _converted_output_error_scale(self, sample: torch.Tensor, step_index: int) -> torch.Tensor:
        """
        Return B such that converted output m = ... + B * (raw UNet output u).
        Then Var(Δm) = B^2 Var(Δu).
        """
        alg = self.s.config.algorithm_type
        pred = self.s.config.prediction_type

        sigma = self.s.sigmas[step_index].to(device=sample.device, dtype=torch.float32)
        alpha, sigma_t = self.s._sigma_to_alpha_sigma_t(sigma)

        spatial_dims = sample.dim() - 2
        view = [1, 1] + [1] * spatial_dims

        def as_b(x: torch.Tensor) -> torch.Tensor:
            return x.view(view)

        if alg in ["dpmsolver++", "sde-dpmsolver++"]:
            if pred == "epsilon":
                return as_b(-(sigma_t / (alpha + 1e-20)))
            if pred == "sample":
                return as_b(torch.ones_like(sigma))
            if pred == "v_prediction":
                return as_b(-sigma_t)
            if pred == "flow_prediction":
                return as_b(-sigma)
            raise ValueError(f"Unsupported prediction_type for {alg}: {pred}")

        if alg in ["dpmsolver", "sde-dpmsolver"]:
            if pred == "epsilon":
                return as_b(torch.ones_like(sigma))
            if pred == "sample":
                return as_b(-(alpha / (sigma_t + 1e-20)))
            if pred == "v_prediction":
                return as_b(alpha)
            raise ValueError(f"Unsupported prediction_type for {alg}: {pred}")

        raise ValueError(f"Unsupported algorithm_type: {alg}")

    # ---------------------------
    # Q-Drift scale computation (variance matching in λ)
    # ---------------------------

    def _qdrift_scale_from_coeffs(
        self,
        coeffs_x: List[torch.Tensor],
        vars_m: List[Optional[torch.Tensor]],
        *,
        alpha_t: torch.Tensor,
        sigma_bar_s: torch.Tensor,
        sigma_bar_t: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the marginal-preserving Q-Drift scale factor for a DPM-Solver step.

        In y=x/alpha space, match the injected implicit variance Var(Δy) to the exact
        diffusion variance of the λ-parameterized marginal-preserving family:

          Var_diff = 2 * beta_tilde_lambda * ∫_{λ_s}^{λ_t} σ̄(λ)^2 dλ.

        With σ̄(λ)=exp(-λ) and σ̄_s, σ̄_t as the scheduler's endpoint noise levels, we have:

          2 * ∫ σ̄(λ)^2 dλ = σ̄_s^2 - σ̄_t^2,

        so beta_tilde_lambda = Var(Δy) / (σ̄_s^2 - σ̄_t^2).
        """
        spatial_dims = sample.dim() - 2
        view = [1, 1] + [1] * spatial_dims

        alpha_t_b = alpha_t.view(view)
        sigma_bar_s_b = sigma_bar_s.view(view)
        sigma_bar_t_b = sigma_bar_t.view(view)

        var_dx = None
        for C, V in zip(coeffs_x, vars_m):
            if V is None:
                continue
            term = (C.view(view) ** 2) * V
            var_dx = term if var_dx is None else (var_dx + term)

        if var_dx is None:
            return torch.ones(view, device=sample.device, dtype=torch.float32)

        var_dy = var_dx / (alpha_t_b**2 + 1e-20)
        denom = (sigma_bar_s_b**2 - sigma_bar_t_b**2).abs().clamp(min=1e-20)
        beta_tilde = var_dy / denom
        return 1.0 + self.qdrift_scale * beta_tilde

    def _post_scale_delta_y(
        self,
        *,
        sample: torch.Tensor,
        base_prev: torch.Tensor,
        alpha_s: torch.Tensor,
        alpha_t: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply Q-Drift as a post-scaling of the deterministic increment in y=x/alpha space:
          y_t = y_s + a * (y_t^{base} - y_s),  x_t = alpha_t * y_t.

        This directly implements Eq. (app:dpm:qdrift_update) in `paper/Q-Drift.tex`, and preserves the
        dpmsolver++ cancellation structure (where y_t^{base} - y_s reduces to a pure ε-quadrature sum).
        """
        spatial_dims = sample.dim() - 2
        view = [1, 1] + [1] * spatial_dims

        alpha_s_b = alpha_s.view(view)
        alpha_t_b = alpha_t.view(view)

        y_s = sample / (alpha_s_b + 1e-20)
        y_base = base_prev / (alpha_t_b + 1e-20)
        y_t = y_s + a * (y_base - y_s)
        return alpha_t_b * y_t

    # ---------------------------
    # Main step (wrap the official scheduler)
    # ---------------------------

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        if self.s.num_inference_steps is None:
            raise ValueError("Run set_timesteps() on the base scheduler first.")

        if self.use_qdrift and getattr(self.s.config, "thresholding", False):
            raise ValueError("Q-Drift requires thresholding=False (convert_model_output must be linear).")

        if self.s.step_index is None:
            self.s._init_step_index(timestep)

        lower_order_final = (self.s.step_index == len(self.s.timesteps) - 1) and (
            self.s.config.euler_at_final
            or (self.s.config.lower_order_final and len(self.s.timesteps) < 15)
            or self.s.config.final_sigmas_type == "zero"
        )
        lower_order_second = (
            (self.s.step_index == len(self.s.timesteps) - 2) and self.s.config.lower_order_final and len(self.s.timesteps) < 15
        )

        # Match diffusers' behavior: convert_model_output() is applied on the incoming dtype
        # (often fp16/bf16), and only then `sample` is upcast to fp32 for the solver update.
        sample_in = sample
        sample_f = sample.to(torch.float32)

        u = model_output.to(torch.float32)

        # ---- (1) Conditional bias removal on raw output + conditional variance
        u_corr = u
        V_u = None

        if self.use_qdrift:
            spatial_dims = u.dim() - 2
            view_shape = [1, u.shape[1]] + [1] * spatial_dims

            mu, cov = self.get_conditional_statistics(timestep, u.shape[1])
            mu_q, mu_e, var_q, var_e, cov_qe = self._broadcast_statistics(
                mu=mu, cov=cov, num_channels=u.shape[1], view_shape=view_shape, device=u.device
            )

            sigma_q = torch.sqrt(var_q.clamp(min=1e-8))
            sigma_e = torch.sqrt(var_e.clamp(min=1e-8))
            rho = cov_qe / (sigma_q * sigma_e + 1e-8)
            rho = torch.clamp(rho, -0.9999, 0.9999)

            mu_cond = mu_e + rho * (sigma_e / (sigma_q + 1e-8)) * (u - mu_q)
            sigma_cond = sigma_e * torch.sqrt(1 - rho**2 + 1e-8)

            u_corr = u - self.bias_scale * mu_cond
            V_u = sigma_cond**2
            if self.qdrift_scalar:
                V_u = V_u.mean(dim=tuple(range(1, V_u.dim())), keepdim=True)

        # ---- (2) Convert corrected output to solver's needed output
        # Cast back to input dtype for parity with the base scheduler when Q-Drift is disabled.
        u_corr_in = u_corr.to(dtype=model_output.dtype)
        m = self.s.convert_model_output(u_corr_in, sample=sample_in)
        target_dtype = m.dtype

        # ---- (3) Store converted output and its conditional variance (propagated)
        for i in range(int(self.s.config.solver_order) - 1):
            self.s.model_outputs[i] = self.s.model_outputs[i + 1]
            self.model_output_vars[i] = self.model_output_vars[i + 1]
        self.s.model_outputs[-1] = m

        if self.use_qdrift and V_u is not None:
            B = self._converted_output_error_scale(sample_f, int(self.s.step_index))
            V_m = (B**2) * V_u
        else:
            V_m = None
        self.model_output_vars[-1] = V_m

        # ---- noise for SDE variants (same as base)
        if self.s.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"] and variance_noise is None:
            noise = randn_tensor(m.shape, generator=generator, device=m.device, dtype=torch.float32)
        elif self.s.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
            noise = variance_noise.to(device=m.device, dtype=torch.float32)
        else:
            noise = None

        # ---- (4) Do the update with Q-Drift post-scaling of the deterministic Δy increment
        if int(self.s.config.solver_order) == 1 or self.s.lower_order_nums < 1 or lower_order_final:
            prev = self._first_order_update(sample_f, noise=noise)
        elif int(self.s.config.solver_order) == 2 or self.s.lower_order_nums < 2 or lower_order_second:
            prev = self._second_order_update(sample_f, noise=noise)
        else:
            prev = self._third_order_update(sample_f, noise=noise)

        if self.s.lower_order_nums < int(self.s.config.solver_order):
            self.s.lower_order_nums += 1

        prev = prev.to(target_dtype)
        self.s._step_index += 1

        if self.use_qdrift and self.enable_logging:
            t_val = float(timestep.item()) if isinstance(timestep, torch.Tensor) else float(timestep)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "timestep": t_val,
                            "step_index": int(self.s.step_index - 1),
                            "bias_scale": self.bias_scale,
                            "qdrift_scale": self.qdrift_scale,
                            "u_mean": float(u.mean().item()),
                            "u_corr_mean": float(u_corr.mean().item()),
                            "m_mean": float(m.mean().item()),
                            "V_u_mean": float(V_u.mean().item()) if V_u is not None else None,
                            "V_m_mean": float(V_m.mean().item()) if V_m is not None else None,
                        }
                    )
                    + "\n"
                )

        if not return_dict:
            return (prev,)
        return SchedulerOutput(prev_sample=prev)

    # ---------------------------
    # Q-Drift scaled DPM-Solver updates (match diffusers formulas exactly)
    # ---------------------------

    def _first_order_update(self, sample: torch.Tensor, noise: Optional[torch.Tensor]) -> torch.Tensor:
        i = int(self.s.step_index)
        sigma_bar_t = self.s.sigmas[i + 1].to(device=sample.device, dtype=torch.float32)
        sigma_bar_s = self.s.sigmas[i].to(device=sample.device, dtype=torch.float32)

        alpha_t, sigma_t = self.s._sigma_to_alpha_sigma_t(sigma_bar_t)
        alpha_s, sigma_s = self.s._sigma_to_alpha_sigma_t(sigma_bar_s)
        bar_sigma_t = sigma_t / (alpha_t + 1e-20)
        bar_sigma_s = sigma_s / (alpha_s + 1e-20)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
        h = lambda_t - lambda_s

        m0 = self.s.model_outputs[-1]
        V0 = self.model_output_vars[-1]

        alg = self.s.config.algorithm_type

        if alg in ["sde-dpmsolver", "sde-dpmsolver++"]:
            if noise is None:
                raise ValueError("SDE DPM-Solver requires `noise`.")
            if alg == "sde-dpmsolver++":
                return (
                    (sigma_t / sigma_s * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * m0
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
            return (
                (alpha_t / alpha_s) * sample
                - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * m0
                + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
            )

        if alg == "dpmsolver++":
            A = sigma_t / sigma_s
            C0 = -(alpha_t * (torch.exp(-h) - 1.0))
        elif alg == "dpmsolver":
            A = alpha_t / alpha_s
            C0 = -(sigma_t * (torch.exp(h) - 1.0))
        else:
            raise ValueError(f"Unsupported algorithm_type: {alg}")

        base_prev = A * sample + C0 * m0
        if (not self.use_qdrift) or (V0 is None):
            return base_prev

        a = self._qdrift_scale_from_coeffs(
            coeffs_x=[C0],
            vars_m=[V0],
            alpha_t=alpha_t,
            sigma_bar_s=bar_sigma_s,
            sigma_bar_t=bar_sigma_t,
            sample=sample,
        )
        return self._post_scale_delta_y(sample=sample, base_prev=base_prev, alpha_s=alpha_s, alpha_t=alpha_t, a=a)

    def _second_order_update(self, sample: torch.Tensor, noise: Optional[torch.Tensor]) -> torch.Tensor:
        i = int(self.s.step_index)
        sigma_bar_t = self.s.sigmas[i + 1].to(device=sample.device, dtype=torch.float32)
        sigma_bar_s0 = self.s.sigmas[i].to(device=sample.device, dtype=torch.float32)
        sigma_bar_s1 = self.s.sigmas[i - 1].to(device=sample.device, dtype=torch.float32)

        alpha_t, sigma_t = self.s._sigma_to_alpha_sigma_t(sigma_bar_t)
        alpha_s0, sigma_s0 = self.s._sigma_to_alpha_sigma_t(sigma_bar_s0)
        alpha_s1, sigma_s1 = self.s._sigma_to_alpha_sigma_t(sigma_bar_s1)
        bar_sigma_t = sigma_t / (alpha_t + 1e-20)
        bar_sigma_s0 = sigma_s0 / (alpha_s0 + 1e-20)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

        h = lambda_t - lambda_s0
        h0 = lambda_s0 - lambda_s1
        r0 = h0 / (h + 1e-20)

        m0 = self.s.model_outputs[-1]
        m1 = self.s.model_outputs[-2]
        V0 = self.model_output_vars[-1]
        V1 = self.model_output_vars[-2]

        alg = self.s.config.algorithm_type

        if alg in ["sde-dpmsolver", "sde-dpmsolver++"]:
            if noise is None:
                raise ValueError("SDE DPM-Solver requires `noise`.")
            D0 = m0
            D1 = (1.0 / r0) * (m0 - m1)
            if alg == "sde-dpmsolver++":
                if self.s.config.solver_type == "midpoint":
                    return (
                        (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                        + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                        + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1
                        + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                    )
                return (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )

            if self.s.config.solver_type == "midpoint":
                return (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * (torch.exp(h) - 1.0)) * D1
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
            return (
                (alpha_t / alpha_s0) * sample
                - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                - 2.0 * (sigma_t * ((torch.exp(h) - 1.0) / (h + 1e-20) - 1.0)) * D1
                + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
            )

        # Deterministic: compute the base update exactly as diffusers does (D0/D1 form),
        # then apply Q-Drift as a post-scaling in y=x/alpha space.
        if alg == "dpmsolver++":
            D0 = m0
            D1 = (1.0 / (r0 + 1e-20)) * (m0 - m1)
            A = sigma_t / sigma_s0
            c = alpha_t * (torch.exp(-h) - 1.0)  # negative
            if self.s.config.solver_type == "midpoint":
                base_prev = (A * sample) - (alpha_t * (torch.exp(-h) - 1.0)) * D0 - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
                C0 = -c * (1.0 + 0.5 / (r0 + 1e-20))
                C1 = +c * (0.5 / (r0 + 1e-20))
            else:  # heun
                base_prev = (A * sample) - (alpha_t * (torch.exp(-h) - 1.0)) * D0 + (alpha_t * ((torch.exp(-h) - 1.0) / (h + 1e-20) + 1.0)) * D1
                c0 = -c
                c1 = alpha_t * ((torch.exp(-h) - 1.0) / (h + 1e-20) + 1.0)
                C0 = c0 + c1 * (1.0 / (r0 + 1e-20))
                C1 = -c1 * (1.0 / (r0 + 1e-20))
        elif alg == "dpmsolver":
            D0 = m0
            D1 = (1.0 / (r0 + 1e-20)) * (m0 - m1)
            A = alpha_t / alpha_s0
            c = sigma_t * (torch.exp(h) - 1.0)
            if self.s.config.solver_type == "midpoint":
                base_prev = (A * sample) - (sigma_t * (torch.exp(h) - 1.0)) * D0 - 0.5 * (sigma_t * (torch.exp(h) - 1.0)) * D1
                C0 = -c * (1.0 + 0.5 / (r0 + 1e-20))
                C1 = +c * (0.5 / (r0 + 1e-20))
            else:  # heun
                base_prev = (A * sample) - (sigma_t * (torch.exp(h) - 1.0)) * D0 - (sigma_t * ((torch.exp(h) - 1.0) / (h + 1e-20) - 1.0)) * D1
                c0 = -c
                c1 = -(sigma_t * ((torch.exp(h) - 1.0) / (h + 1e-20) - 1.0))
                C0 = c0 + c1 * (1.0 / (r0 + 1e-20))
                C1 = -c1 * (1.0 / (r0 + 1e-20))
        else:
            raise ValueError(f"Unsupported algorithm_type: {alg}")
        if (not self.use_qdrift) or (V0 is None) or (V1 is None):
            return base_prev

        a = self._qdrift_scale_from_coeffs(
            coeffs_x=[C0, C1],
            vars_m=[V0, V1],
            alpha_t=alpha_t,
            sigma_bar_s=bar_sigma_s0,
            sigma_bar_t=bar_sigma_t,
            sample=sample,
        )
        return self._post_scale_delta_y(sample=sample, base_prev=base_prev, alpha_s=alpha_s0, alpha_t=alpha_t, a=a)

    def _third_order_update(self, sample: torch.Tensor, noise: Optional[torch.Tensor]) -> torch.Tensor:
        i = int(self.s.step_index)
        sigma_bar_t = self.s.sigmas[i + 1].to(device=sample.device, dtype=torch.float32)
        sigma_bar_s0 = self.s.sigmas[i].to(device=sample.device, dtype=torch.float32)
        sigma_bar_s1 = self.s.sigmas[i - 1].to(device=sample.device, dtype=torch.float32)
        sigma_bar_s2 = self.s.sigmas[i - 2].to(device=sample.device, dtype=torch.float32)

        alpha_t, sigma_t = self.s._sigma_to_alpha_sigma_t(sigma_bar_t)
        alpha_s0, sigma_s0 = self.s._sigma_to_alpha_sigma_t(sigma_bar_s0)
        alpha_s1, sigma_s1 = self.s._sigma_to_alpha_sigma_t(sigma_bar_s1)
        alpha_s2, sigma_s2 = self.s._sigma_to_alpha_sigma_t(sigma_bar_s2)
        bar_sigma_t = sigma_t / (alpha_t + 1e-20)
        bar_sigma_s0 = sigma_s0 / (alpha_s0 + 1e-20)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)
        lambda_s2 = torch.log(alpha_s2) - torch.log(sigma_s2)

        h = lambda_t - lambda_s0
        h0 = lambda_s0 - lambda_s1
        h1 = lambda_s1 - lambda_s2
        r0 = h0 / (h + 1e-20)
        r1 = h1 / (h + 1e-20)

        m0 = self.s.model_outputs[-1]
        m1 = self.s.model_outputs[-2]
        m2 = self.s.model_outputs[-3]
        V0 = self.model_output_vars[-1]
        V1 = self.model_output_vars[-2]
        V2 = self.model_output_vars[-3]

        alg = self.s.config.algorithm_type

        if alg in ["sde-dpmsolver", "sde-dpmsolver++"]:
            if noise is None:
                raise ValueError("SDE DPM-Solver requires `noise`.")
            if alg != "sde-dpmsolver++":
                raise ValueError("diffusers does not define a 3rd-order update for sde-dpmsolver.")
            D0 = m0
            D1_0 = (1.0 / r0) * (m0 - m1)
            D1_1 = (1.0 / r1) * (m1 - m2)
            D1 = D1_0 + (r0 / (r0 + r1 + 1e-20)) * (D1_0 - D1_1)
            D2 = (1.0 / (r0 + r1 + 1e-20)) * (D1_0 - D1_1)
            return (
                (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                + (alpha_t * (1.0 - torch.exp(-2.0 * h))) * D0
                + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                + (alpha_t * ((1.0 - torch.exp(-2.0 * h) - 2.0 * h) / (2.0 * h) ** 2 - 0.5)) * D2
                + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            )

        # Deterministic: compute the base update exactly as diffusers does (D0/D1/D2 form),
        # and separately compute the implied linear coefficients C0/C1/C2 for variance matching.
        D0 = m0
        D1_0 = (1.0 / (r0 + 1e-20)) * (m0 - m1)
        D1_1 = (1.0 / (r1 + 1e-20)) * (m1 - m2)
        D1 = D1_0 + (r0 / (r0 + r1 + 1e-20)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1 + 1e-20)) * (D1_0 - D1_1)

        a0 = 1.0 / (r0 + 1e-20)
        a1 = 1.0 / (r1 + 1e-20)
        k = r0 / (r0 + r1 + 1e-20)
        d = 1.0 / (r0 + r1 + 1e-20)

        # D1 weights
        w1_0 = (1.0 + k) * a0
        w1_1 = -((1.0 + k) * a0 + k * a1)
        w1_2 = k * a1

        # D2 weights
        w2_0 = d * a0
        w2_1 = -d * (a0 + a1)
        w2_2 = d * a1

        if alg == "dpmsolver++":
            A = sigma_t / sigma_s0
            c0 = alpha_t * (torch.exp(-h) - 1.0)
            c1 = alpha_t * ((torch.exp(-h) - 1.0) / (h + 1e-20) + 1.0)
            c2 = alpha_t * ((torch.exp(-h) - 1.0 + h) / ((h + 1e-20) ** 2) - 0.5)
            base_prev = (A * sample) - (alpha_t * (torch.exp(-h) - 1.0)) * D0 + (alpha_t * ((torch.exp(-h) - 1.0) / (h + 1e-20) + 1.0)) * D1 - (alpha_t * ((torch.exp(-h) - 1.0 + h) / ((h + 1e-20) ** 2) - 0.5)) * D2
            C0 = (-c0) + c1 * w1_0 + (-c2) * w2_0
            C1 = (0.0) + c1 * w1_1 + (-c2) * w2_1
            C2 = (0.0) + c1 * w1_2 + (-c2) * w2_2
        elif alg == "dpmsolver":
            A = alpha_t / alpha_s0
            c0 = sigma_t * (torch.exp(h) - 1.0)
            c1 = sigma_t * ((torch.exp(h) - 1.0) / (h + 1e-20) - 1.0)
            c2 = sigma_t * ((torch.exp(h) - 1.0 - h) / ((h + 1e-20) ** 2) - 0.5)
            base_prev = (A * sample) - (sigma_t * (torch.exp(h) - 1.0)) * D0 - (sigma_t * ((torch.exp(h) - 1.0) / (h + 1e-20) - 1.0)) * D1 - (sigma_t * ((torch.exp(h) - 1.0 - h) / ((h + 1e-20) ** 2) - 0.5)) * D2
            C0 = (-c0) + (-c1) * w1_0 + (-c2) * w2_0
            C1 = (0.0) + (-c1) * w1_1 + (-c2) * w2_1
            C2 = (0.0) + (-c1) * w1_2 + (-c2) * w2_2
        else:
            raise ValueError(f"Unsupported algorithm_type: {alg}")

        if (not self.use_qdrift) or (V0 is None) or (V1 is None) or (V2 is None):
            return base_prev

        a = self._qdrift_scale_from_coeffs(
            coeffs_x=[C0, C1, C2],
            vars_m=[V0, V1, V2],
            alpha_t=alpha_t,
            sigma_bar_s=bar_sigma_s0,
            sigma_bar_t=bar_sigma_t,
            sample=sample,
        )
        return self._post_scale_delta_y(sample=sample, base_prev=base_prev, alpha_s=alpha_s0, alpha_t=alpha_t, a=a)
