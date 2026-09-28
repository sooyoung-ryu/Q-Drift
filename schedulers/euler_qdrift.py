# Copyright 2025 Katherine Crowson and The HuggingFace Team. All rights reserved.
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
import math
import os
from bisect import bisect_left
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, is_scipy_available, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin


if is_scipy_available():
    import scipy.stats

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->EulerDiscrete
class EulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None
    
    
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


# Copied from diffusers.schedulers.scheduling_ddim.rescale_zero_terminal_snr
def rescale_zero_terminal_snr(betas):
    """
    Rescales betas to have zero terminal SNR Based on https://huggingface.co/papers/2305.08891 (Algorithm 1)


    Args:
        betas (`torch.Tensor`):
            the betas that the scheduler is being initialized with.

    Returns:
        `torch.Tensor`: rescaled betas with zero terminal SNR
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


class EulerQDriftScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler with integrated Q-drift correction for quantized SDXL models.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear` or `scaled_linear`.
        trained_betas (`np.ndarray`, *optional*):
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        interpolation_type(`str`, defaults to `"linear"`, *optional*):
            The interpolation type to compute intermediate sigmas for the scheduler denoising steps. Should be on of
            `"linear"` or `"log_linear"`.
        use_karras_sigmas (`bool`, *optional*, defaults to `False`):
            Whether to use Karras sigmas for step sizes in the noise schedule during the sampling process. If `True`,
            the sigmas are determined according to a sequence of noise levels {σi}.
        use_exponential_sigmas (`bool`, *optional*, defaults to `False`):
            Whether to use exponential sigmas for step sizes in the noise schedule during the sampling process.
        use_beta_sigmas (`bool`, *optional*, defaults to `False`):
            Whether to use beta sigmas for step sizes in the noise schedule during the sampling process. Refer to [Beta
            Sampling is All You Need](https://huggingface.co/papers/2407.12173) for more information.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps, as required by some model families.
        rescale_betas_zero_snr (`bool`, defaults to `False`):
            Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
            dark samples instead of limiting it to samples with medium brightness. Loosely related to
            [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).
        final_sigmas_type (`str`, defaults to `"zero"`):
            The final `sigma` value for the noise schedule during the sampling process. If `"sigma_min"`, the final
            sigma is the same as the last sigma in the training schedule. If `zero`, the final sigma is set to 0.
        mu_dict_path (`str`, *optional*):
            Path to the stored per-timestep Gaussian means used for Q-drift correction.
        cov_dict_path (`str`, *optional*):
            Path to the stored per-timestep Gaussian covariance matrices for Q-drift correction.
        log_dir (`str`, *optional*):
            Directory to store detailed Q-drift logs for debugging/analysis.
        bias_scale (`float`, defaults to 0.0):
            Scaling factor applied to the conditional mean before subtracting the Q-drift bias.
        qdrift_scale (`float`, defaults to 1.0):
            Scaling factor applied to the drift compensation term (dt correction).
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "epsilon",
        interpolation_type: str = "linear",
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        rho: float = 7.0,
        timestep_spacing: str = "linspace",
        timestep_type: str = "discrete",  # can be "discrete" or "continuous"
        steps_offset: int = 0,
        rescale_betas_zero_snr: bool = False,
        final_sigmas_type: str = "zero",  # can be "zero" or "sigma_min"
        mu_dict_path: Optional[str] = None,
        cov_dict_path: Optional[str] = None,
        log_dir: Optional[str] = None,
        bias_scale: float = 1.0,
        bias_stochastic: bool = False,
        qdrift_scalar: bool = False,
        qdrift_unconditional: bool = False,
        qdrift_scale: float = 1.0,
    ):
        if self.config.use_beta_sigmas and not is_scipy_available():
            raise ImportError("Make sure to install scipy if you want to use beta sigmas.")
        if sum([self.config.use_beta_sigmas, self.config.use_exponential_sigmas, self.config.use_karras_sigmas]) > 1:
            raise ValueError(
                "Only one of `config.use_beta_sigmas`, `config.use_exponential_sigmas`, `config.use_karras_sigmas` can be used."
            )
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        if rescale_betas_zero_snr:
            # Close to 0 without being 0 so first sigma is not inf
            # FP16 smallest positive subnormal works well here
            self.alphas_cumprod[-1] = 2**-24

        sigmas = (((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5).flip(0)
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=float)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        # setable values
        self.num_inference_steps = None

        # TODO: Support the full EDM scalings for all prediction types and timestep types
        if timestep_type == "continuous" and prediction_type == "v_prediction":
            self.timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas])
        else:
            self.timesteps = timesteps

        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self.is_scale_input_called = False
        self.use_karras_sigmas = use_karras_sigmas
        self.use_exponential_sigmas = use_exponential_sigmas
        self.use_beta_sigmas = use_beta_sigmas

        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

        # Q-Drift parameters/statistics
        self.bias_scale = bias_scale
        # Stochastic dual denoising (D^2-DPM S-D^2): subtract a draw from the
        # conditional Gaussian instead of its mean, which returns the variance that
        # mean subtraction removes. Makes the update random on a deterministic sampler.
        self.bias_stochastic = bias_stochastic
        # Ablation: collapse the per-channel conditional variance to one scalar per
        # step by averaging over channels, so the correction is channel-agnostic.
        self.qdrift_scalar = qdrift_scalar
        # Ablation: use the unconditional Var(delta_eps) in place of the conditional
        # variance E[Var(delta_eps | eps_hat)], i.e. drop the (1 - rho^2) factor.
        self.qdrift_unconditional = qdrift_unconditional
        self.qdrift_scale = qdrift_scale
        self.use_qdrift = mu_dict_path is not None and cov_dict_path is not None
        if self.use_qdrift:
            mu_raw = np.load(mu_dict_path, allow_pickle=True).item()
            cov_raw = np.load(cov_dict_path, allow_pickle=True).item()
            self.mu_dict = {float(k): v for k, v in mu_raw.items()}
            self.cov_dict = {float(k): v for k, v in cov_raw.items()}
            self._qdrift_available_timesteps = sorted(self.mu_dict.keys())
            print("Q-Drift enabled with statistics from:")
            print(f"  mu_dict: {mu_dict_path}")
            print(f"  cov_dict: {cov_dict_path}")
            print(f"  Timesteps: {self._qdrift_available_timesteps}")

            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                self.log_file = os.path.join(log_dir, "qdrift_log.jsonl")
                with open(self.log_file, "w"):
                    pass
                self.enable_logging = True
                print(f"Q-Drift logging enabled: {self.log_file}")
            else:
                self.enable_logging = False
        else:
            self.mu_dict = {}
            self.cov_dict = {}
            self._qdrift_available_timesteps = []
            self.enable_logging = False
            print("Q-Drift disabled (no statistics provided)")

    @property
    def init_noise_sigma(self):
        # standard deviation of the initial noise distribution
        max_sigma = max(self.sigmas) if isinstance(self.sigmas, list) else self.sigmas.max()
        if self.config.timestep_spacing in ["linspace", "trailing"]:
            return max_sigma

        return (max_sigma**2 + 1) ** 0.5

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep. Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.

        Args:
            sample (`torch.Tensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index].to(device=sample.device, dtype=sample.dtype)
        sample = sample / ((sigma**2 + 1) ** 0.5)

        self.is_scale_input_called = True
        return sample

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary timesteps schedule. If `None`, timesteps will be generated
                based on the `timestep_spacing` attribute. If `timesteps` is passed, `num_inference_steps` and `sigmas`
                must be `None`, and `timestep_spacing` attribute will be ignored.
            sigmas (`List[float]`, *optional*):
                Custom sigmas used to support arbitrary timesteps schedule schedule. If `None`, timesteps and sigmas
                will be generated based on the relevant scheduler attributes. If `sigmas` is passed,
                `num_inference_steps` and `timesteps` must be `None`, and the timesteps will be generated based on the
                custom sigmas schedule.
        """

        if timesteps is not None and sigmas is not None:
            raise ValueError("Only one of `timesteps` or `sigmas` should be set.")
        if num_inference_steps is None and timesteps is None and sigmas is None:
            raise ValueError("Must pass exactly one of `num_inference_steps` or `timesteps` or `sigmas.")
        if num_inference_steps is not None and (timesteps is not None or sigmas is not None):
            raise ValueError("Can only pass one of `num_inference_steps` or `timesteps` or `sigmas`.")
        if timesteps is not None and self.config.use_karras_sigmas:
            raise ValueError("Cannot set `timesteps` with `config.use_karras_sigmas = True`.")
        if timesteps is not None and self.config.use_exponential_sigmas:
            raise ValueError("Cannot set `timesteps` with `config.use_exponential_sigmas = True`.")
        if timesteps is not None and self.config.use_beta_sigmas:
            raise ValueError("Cannot set `timesteps` with `config.use_beta_sigmas = True`.")
        if (
            timesteps is not None
            and self.config.timestep_type == "continuous"
            and self.config.prediction_type == "v_prediction"
        ):
            raise ValueError(
                "Cannot set `timesteps` with `config.timestep_type = 'continuous'` and `config.prediction_type = 'v_prediction'`."
            )

        if num_inference_steps is None:
            num_inference_steps = len(timesteps) if timesteps is not None else len(sigmas) - 1
        self.num_inference_steps = num_inference_steps

        if sigmas is not None:
            log_sigmas = np.log(np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5))
            sigmas = np.array(sigmas).astype(np.float32)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas[:-1]])

        else:
            if timesteps is not None:
                timesteps = np.array(timesteps).astype(np.float32)
            else:
                # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://huggingface.co/papers/2305.08891
                if self.config.timestep_spacing == "linspace":
                    timesteps = np.linspace(
                        0, self.config.num_train_timesteps - 1, num_inference_steps, dtype=np.float32
                    )[::-1].copy()
                elif self.config.timestep_spacing == "leading":
                    step_ratio = self.config.num_train_timesteps // self.num_inference_steps
                    # creates integer timesteps by multiplying by ratio
                    # casting to int to avoid issues when num_inference_step is power of 3
                    timesteps = (
                        (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.float32)
                    )
                    timesteps += self.config.steps_offset
                elif self.config.timestep_spacing == "trailing":
                    step_ratio = self.config.num_train_timesteps / self.num_inference_steps
                    # creates integer timesteps by multiplying by ratio
                    # casting to int to avoid issues when num_inference_step is power of 3
                    timesteps = (
                        (np.arange(self.config.num_train_timesteps, 0, -step_ratio)).round().copy().astype(np.float32)
                    )
                    timesteps -= 1
                else:
                    raise ValueError(
                        f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
                    )

            sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
            log_sigmas = np.log(sigmas)
            if self.config.interpolation_type == "linear":
                sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
            elif self.config.interpolation_type == "log_linear":
                sigmas = torch.linspace(np.log(sigmas[-1]), np.log(sigmas[0]), num_inference_steps + 1).exp().numpy()
            else:
                raise ValueError(
                    f"{self.config.interpolation_type} is not implemented. Please specify interpolation_type to either"
                    " 'linear' or 'log_linear'"
                )

            if self.config.use_karras_sigmas:
                sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=self.num_inference_steps)
                timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])

            elif self.config.use_exponential_sigmas:
                sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
                timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])

            elif self.config.use_beta_sigmas:
                sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
                timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])

            if self.config.final_sigmas_type == "sigma_min":
                sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
            elif self.config.final_sigmas_type == "zero":
                sigma_last = 0
            else:
                raise ValueError(
                    f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
                )

            sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

        # TODO: Support the full EDM scalings for all prediction types and timestep types
        if self.config.timestep_type == "continuous" and self.config.prediction_type == "v_prediction":
            self.timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas[:-1]]).to(device=device)
        else:
            self.timesteps = torch.from_numpy(timesteps.astype(np.float32)).to(device=device)

        self._step_index = None
        self._begin_index = None
        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    def _sigma_to_t(self, sigma, log_sigmas):
        # get log sigma
        log_sigma = np.log(np.maximum(sigma, 1e-10))

        # get distribution
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # get sigmas range
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1

        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # interpolate sigmas
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)

        # transform interpolation to time range
        t = (1 - w) * low_idx + w * high_idx
        t = t.reshape(sigma.shape)
        return t

    # Copied from https://github.com/crowsonkb/k-diffusion/blob/686dbad0f39640ea25c8a8c6a6e56bb40eacefa2/k_diffusion/sampling.py#L17
    def _convert_to_karras(self, in_sigmas: torch.Tensor, num_inference_steps) -> torch.Tensor:
        """Constructs the noise schedule of Karras et al. (2022)."""

        # Hack to make sure that other schedulers which copy this function don't break
        # TODO: Add this logic to the other schedulers
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        rho = float(getattr(self.config, "rho", 7.0))
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    # Copied from https://github.com/crowsonkb/k-diffusion/blob/686dbad0f39640ea25c8a8c6a6e56bb40eacefa2/k_diffusion/sampling.py#L26
    def _convert_to_exponential(self, in_sigmas: torch.Tensor, num_inference_steps: int) -> torch.Tensor:
        """Constructs an exponential noise schedule."""

        # Hack to make sure that other schedulers which copy this function don't break
        # TODO: Add this logic to the other schedulers
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        sigmas = np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), num_inference_steps))
        return sigmas

    def _convert_to_beta(
        self, in_sigmas: torch.Tensor, num_inference_steps: int, alpha: float = 0.6, beta: float = 0.6
    ) -> torch.Tensor:
        """From "Beta Sampling is All You Need" [arXiv:2407.12173] (Lee et. al, 2024)"""

        # Hack to make sure that other schedulers which copy this function don't break
        # TODO: Add this logic to the other schedulers
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        sigmas = np.array(
            [
                sigma_min + (ppf * (sigma_max - sigma_min))
                for ppf in [
                    scipy.stats.beta.ppf(timestep, alpha, beta)
                    for timestep in 1 - np.linspace(0, 1, num_inference_steps)
                ]
            ]
        )
        return sigmas

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def get_conditional_statistics(
        self, timestep: Union[int, float, torch.Tensor], num_channels: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return timestep-specific Gaussian statistics (mean + covariance) for Q-drift correction.
        """
        if not self.use_qdrift:
            mu = torch.zeros((2, num_channels), dtype=torch.float32)
            cov = torch.eye(2, dtype=torch.float32).unsqueeze(-1).repeat(1, 1, num_channels)
            return mu, cov

        t_val = float(timestep.item()) if isinstance(timestep, torch.Tensor) else float(timestep)

        # Exact match fast path (works when sampling uses the same timestep grid as collection).
        if t_val in self.mu_dict:
            mu = torch.from_numpy(self.mu_dict[t_val]).float()
            cov = torch.from_numpy(self.cov_dict[t_val]).float()
            return mu, cov

        available = getattr(self, "_qdrift_available_timesteps", None) or sorted(self.mu_dict.keys())
        if len(available) == 0:
            mu = torch.zeros((2, num_channels), dtype=torch.float32)
            cov = torch.eye(2, dtype=torch.float32).unsqueeze(-1).repeat(1, 1, num_channels)
            return mu, cov

        # Clamp outside range.
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
        """
        Broadcast per-channel Q-drift statistics to match `model_output` arithmetic.

        Expected formats:
          - mu: (2, C) or (2, 1)
          - cov: (2, 2, C) or (2, 2, 1)
        Returns broadcasted tensors shaped for arithmetic with `model_output`.
        """

        mu = mu.to(device=device, dtype=torch.float32)
        cov = cov.to(device=device, dtype=torch.float32)

        if mu.ndim == 2 and mu.shape[0] == 2 and mu.shape[1] in {1, num_channels}:
            mu_quant = mu[0].view([1, mu.shape[1]] + [1] * (len(view_shape) - 2))
            mu_error = mu[1].view([1, mu.shape[1]] + [1] * (len(view_shape) - 2))
            mu_quant = mu_quant.expand(view_shape)
            mu_error = mu_error.expand(view_shape)
        elif mu.ndim == 1 and mu.shape[0] == 2:
            raise ValueError(
                "Tensor-wise stats (mu shape (2,)) are no longer supported here; "
                "re-run gaussian_modeling to produce channel-wise stats (mu shape (2,C))."
            )
        else:
            raise ValueError(f"Unsupported mu shape for Q-drift stats: {tuple(mu.shape)}")

        if cov.ndim == 3 and cov.shape[0] == 2 and cov.shape[1] == 2 and cov.shape[2] in {1, num_channels}:
            var_quant = cov[0, 0].view([1, cov.shape[2]] + [1] * (len(view_shape) - 2))
            var_error = cov[1, 1].view([1, cov.shape[2]] + [1] * (len(view_shape) - 2))
            cov_quant_error = cov[0, 1].view([1, cov.shape[2]] + [1] * (len(view_shape) - 2))
            var_quant = var_quant.expand(view_shape)
            var_error = var_error.expand(view_shape)
            cov_quant_error = cov_quant_error.expand(view_shape)
        elif cov.ndim == 2 and tuple(cov.shape) == (2, 2):
            raise ValueError(
                "Tensor-wise stats (cov shape (2,2)) are no longer supported here; "
                "re-run gaussian_modeling to produce channel-wise stats (cov shape (2,2,C))."
            )
        else:
            raise ValueError(f"Unsupported cov shape for Q-drift stats: {tuple(cov.shape)}")

        return mu_quant, mu_error, var_quant, var_error, cov_quant_error

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[EulerDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        original_dtype = model_output.dtype
        sample = sample.to(torch.float32)
        model_output = model_output.to(torch.float32)

        sigma = self.sigmas[self.step_index].to(device=sample.device, dtype=torch.float32)

        gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            noise = randn_tensor(
                model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
            )
            eps = noise * s_noise
            sample = sample + eps * (sigma_hat**2 - sigma**2).clamp(min=0.0) ** 0.5

        corrected_output = model_output
        mu_cond = sigma_cond = correction_factor = dt_corrected = None
        if self.use_qdrift:
            # ERROR DEFINITION (critical):
            #   error = quant_output - fp16_output
            #   fp16_output = quant_output - error
            # so E[fp16 | quant] = quant - E[error | quant]
            spatial_dims = model_output.dim() - 2
            view_shape = [1, model_output.shape[1]] + [1] * spatial_dims
            mu, cov = self.get_conditional_statistics(timestep, model_output.shape[1])
            mu_quant, mu_error, var_quant, var_error, cov_quant_error = self._broadcast_statistics(
                mu=mu,
                cov=cov,
                num_channels=model_output.shape[1],
                view_shape=view_shape,
                device=model_output.device,
            )

            sigma_quant = torch.sqrt(var_quant.clamp(min=1e-8))
            sigma_error = torch.sqrt(var_error.clamp(min=1e-8))
            rho = cov_quant_error / (sigma_quant * sigma_error + 1e-8)
            rho = torch.clamp(rho, -0.9999, 0.9999)

            mu_cond = mu_error + rho * (sigma_error / (sigma_quant + 1e-8)) * (model_output - mu_quant)
            sigma_cond = sigma_error * torch.sqrt(1 - rho**2 + 1e-8)

            if self.bias_stochastic:
                z = randn_tensor(
                    model_output.shape,
                    dtype=model_output.dtype,
                    device=model_output.device,
                    generator=generator,
                )
                delta_draw = mu_cond + sigma_cond * z
            else:
                delta_draw = mu_cond
            corrected_output = model_output - self.bias_scale * delta_draw

        # 1. compute predicted original sample (x_0) from the corrected output
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for backwards compatibility
        if self.config.prediction_type == "original_sample" or self.config.prediction_type == "sample":
            pred_original_sample = corrected_output
        elif self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma_hat * corrected_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = corrected_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        derivative = (sample - pred_original_sample) / sigma_hat

        sigma_next = self.sigmas[self.step_index + 1].to(device=sample.device, dtype=torch.float32)
        dt = sigma_next - sigma_hat

        if self.use_qdrift:
            # NOTE: with `bias_stochastic=True` (D^2-DPM S-D^2) the conditional variance has
            # already been injected into the prediction above, so this drift rescaling would
            # count it a second time. Faithful S-D^2 runs therefore use qdrift_scale=0.0,
            # which makes dt_corrected == dt.
            V_sigma = sigma_error**2 if self.qdrift_unconditional else sigma_cond**2
            if self.qdrift_scalar:
                V_sigma = V_sigma.mean().expand_as(V_sigma)
            correction_factor = V_sigma * torch.abs(dt) / (2.0 * sigma_hat + 1e-12)
            dt_corrected = dt * (1 + self.qdrift_scale * correction_factor)
            prev_sample = sample + derivative * dt_corrected
        else:
            prev_sample = sample + derivative * dt

        if self.use_qdrift and self.enable_logging:
            timestep_val = float(timestep.item()) if isinstance(timestep, torch.Tensor) else float(timestep)
            log_data = {
                "timestep": timestep_val,
                "step_index": int(self.step_index),
                "bias_scale": float(self.bias_scale),
                "bias_stochastic": bool(self.bias_stochastic),
                "qdrift_scalar": bool(self.qdrift_scalar),
                "qdrift_unconditional": bool(self.qdrift_unconditional),
                "qdrift_scale": float(self.qdrift_scale),
                "sigma": float(sigma.item()),
                "sigma_hat": float(sigma_hat.item()),
                "sigma_next": float(sigma_next.item()),
                "mu_quant": float(mu_quant.mean().item()),
                "mu_error": float(mu_error.mean().item()),
                "var_quant": float(var_quant.mean().item()),
                "var_error": float(var_error.mean().item()),
                "cov_quant_error": float(cov_quant_error.mean().item()),
                "sigma_quant": float(sigma_quant.mean().item()),
                "sigma_error": float(sigma_error.mean().item()),
                "rho": float(rho.mean().item()),
                "mu_cond": float(mu_cond.mean().item()),
                "sigma_cond": float(sigma_cond.mean().item()),
                "model_output_raw": float(model_output.mean().item()),
                "corrected_output": float(corrected_output.mean().item()),
                "derivative": float(derivative.mean().item()),
                "dt": float(dt.item()),
                "correction_factor": float(correction_factor.mean().item()),
                "dt_corrected": float(dt_corrected.mean().item()),
                "sample": float(sample.mean().item()),
                "pred_original_sample": float(pred_original_sample.mean().item()),
                "prev_sample": float(prev_sample.mean().item()),
            }
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_data) + "\n")

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(original_dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (
                prev_sample,
                pred_original_sample,
            )

        return EulerDiscreteSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples

    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if (
            isinstance(timesteps, int)
            or isinstance(timesteps, torch.IntTensor)
            or isinstance(timesteps, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.get_velocity()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if sample.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)
            timesteps = timesteps.to(sample.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(sample.device)
            timesteps = timesteps.to(sample.device)

        step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        alphas_cumprod = self.alphas_cumprod.to(sample)
        sqrt_alpha_prod = alphas_cumprod[step_indices] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[step_indices]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    def __len__(self):
        return self.config.num_train_timesteps
