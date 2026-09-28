# -*- coding: utf-8 -*-
"""Search-based uantization calibrator module."""

import gc
import json
import os
import re
import shutil
import typing as tp
from abc import ABC, abstractmethod
from dataclasses import _MISSING_TYPE, MISSING
from pathlib import Path

import psutil
import torch
import torch.nn as nn
import torch.utils.hooks

from ..data.cache import TensorCache, TensorsCache
from ..data.common import TensorType
from ..data.utils.reshape import ReshapeFn
from ..data.utils.shape import infer_view_shape
from ..quantizer.processor import Quantizer
from ..utils import tools
from ..utils.hooks import Hook
from .config import SearchBasedCalibConfig, SearchBasedCalibGranularity, SearchBasedCalibObjective

__all__ = ["SearchBasedCalibrator"]


def _reshape_w_for_wgts(w: torch.Tensor, w_view_shape: torch.Size) -> torch.Tensor:
    # (#g0, gs0, #g1, gs1, ...)
    w = w.view(w_view_shape)
    # (#g0, gs0, #g1, gs1, ...) -> (#g0, ..., gs1, ..., gs0)
    w = w.permute(*range(0, len(w_view_shape), 2), *range(3, len(w_view_shape), 2), 1)
    # (#g0, ..., gs0, gs1, ...) -> (#g0, ..., gs1 * gs2 * ..., gs0)
    return w.reshape(*w_view_shape[::2], -1, w_view_shape[1])


def _reshape_x_for_wgts(x: torch.Tensor, w_view_shape: torch.Size) -> torch.Tensor:
    # x is unfolded already
    num_samples = x.shape[0]
    # (1, n, #g1, gs1, ...)
    x = x.view(1, num_samples, *w_view_shape[2:])
    # (1, n, #g1, gs1, ...) -> (1, #g1, ..., n, gs1, ...)
    x = x.permute(*range(0, len(w_view_shape), 2), *range(1, len(w_view_shape), 2))
    return x.reshape(1, *w_view_shape[2::2], num_samples, -1)


def _reshape_x_for_ipts(x: torch.Tensor, x_view_shape: torch.Size) -> torch.Tensor:
    # x is original tensor without unfolding
    # (#g0, gs0, #g1, gs1, ...)
    x = x.view(x_view_shape)
    # (#g0, gs0, #g1, gs1, ...) -> (#g0, #g1, ..., gs0, gs2, ..., gs1)
    x = x.permute(*range(0, len(x_view_shape), 2), 1, *range(5, len(x_view_shape), 2), 3)
    # (#g0, #g1, ..., gs0, gs2, ..., gs1) -> (#g0, #g1, ..., gs0 * gs2 * ..., gs1)
    return x.reshape(*x_view_shape[::2], -1, x_view_shape[3])


def _reshape_w_for_ipts(w: torch.Tensor, x_view_shape: torch.Size) -> torch.Tensor:
    return w.transpose(0, 1).reshape(1, x_view_shape[2], *([1] * (w.ndim - 2)), x_view_shape[3], -1)


_CANDIDATE = tp.TypeVar("_CANDIDATE")
_CONFIG = tp.TypeVar("_CONFIG", bound=SearchBasedCalibConfig)


class SearchBasedCalibrator(ABC, tp.Generic[_CONFIG, _CANDIDATE]):
    """The base class for search-based calibration."""

    config: _CONFIG
    candidate: _CANDIDATE

    def __init__(
        self,
        tensor_type: TensorType,
        config: _CONFIG,
        w_quantizer: Quantizer | None,
        x_quantizer: Quantizer | None,
        y_quantizer: Quantizer | None,
        develop_dtype: torch.dtype,
    ) -> None:
        """Initialize the search-based calibrator.

        Args:
            tensor_type (`TensorType`):
                The tensor type.
            config (`_CONFIG`):
                The calibration configuration.
            w_quantizer (`Quantizer` or `None`):
                The w quantizer for x-w computation.
            x_quantizer (`Quantizer` or `None`):
                The x quantizer for x-w or y-x computation.
            y_quantizer (`Quantizer` or `None`):
                The y quantizer for y-x computation.
            develop_dtype (`torch.dtype`):
                The development data type.
        """
        self.tensor_type = tensor_type
        self.config = config
        self.objective = self.config.objective
        self.granularity = self.config.granularity
        self.opts_device = None
        self.develop_dtype = develop_dtype
        self.w_quantizer = w_quantizer
        self.x_quantizer = x_quantizer
        self.y_quantizer = y_quantizer
        self.needs_w_quant = self.w_quantizer is not None and self.w_quantizer.is_enabled()
        self.needs_x_quant = self.x_quantizer is not None and self.x_quantizer.is_enabled()
        self.needs_y_quant = self.y_quantizer is not None and self.y_quantizer.is_enabled()
        self.needs_x_quant_for_wgts = self.allows_x_quant_for_wgts and self.needs_x_quant
        self.needs_w_quant_for_wgts = self.allows_w_quant_for_wgts and self.needs_w_quant
        self.needs_x_quant_for_ipts = self.allows_x_quant_for_ipts and self.needs_x_quant
        self.needs_w_quant_for_ipts = self.allows_w_quant_for_ipts and self.needs_w_quant
        self.needs_x_quant_for_opts = self.allows_x_quant_for_opts and self.needs_x_quant
        self.needs_y_quant_for_opts = self.allows_y_quant_for_opts and self.needs_y_quant
        self.needs_w_quant_for_opts = self.allows_w_quant_for_opts and self.needs_w_quant
        if self.tensor_type == TensorType.Weights:
            self.quantizer = self.w_quantizer
            self.needs_quant = self.needs_w_quant
        elif self.tensor_type == TensorType.Inputs:
            self.quantizer = self.x_quantizer
            self.needs_quant = self.needs_x_quant
        elif self.tensor_type == TensorType.Outputs:
            self.quantizer = self.y_quantizer
            self.needs_quant = self.needs_y_quant
        else:
            raise ValueError(f"unknown tensor type: {self.tensor_type}")
        self.num_iters = getattr(self.config, "num_iters", 1)
        self.logger = tools.logging.getLogger(f"{__name__}.{self.__class__.__name__.replace('Agent', '')}")

    @property
    @abstractmethod
    def population_size(self) -> int:
        """Get the population size."""
        ...

    @property
    def allows_x_quant_for_wgts(self) -> bool:
        """Whether the calibrator allows input quantization when tensor_type is Weights."""
        return False

    @property
    def allows_w_quant_for_wgts(self) -> bool:
        """Whether the calibrator allows weight quantization when tensor_type is Weights."""
        return True

    @property
    def allows_x_quant_for_ipts(self) -> bool:
        """Whether the calibrator allows input quantization when tensor_type is Inputs."""
        return True

    @property
    def allows_w_quant_for_ipts(self) -> bool:
        """Whether the calibrator allows weight quantization when tensor_type is Inputs."""
        return False

    @property
    def allows_x_quant_for_opts(self) -> bool:
        """Whether the calibrator allows x quantization when tensor_type is Outputs."""
        return True

    @property
    def allows_y_quant_for_opts(self) -> bool:
        """Whether the calibrator allows y quantization when tensor_type is Outputs."""
        return True

    @property
    def allows_w_quant_for_opts(self) -> bool:
        """Whether the calibrator allows weight quantization when tensor_type is Outputs."""
        return False

    @property
    def needs_to_pre_reshape_x_for_wgts(self) -> bool:
        """Whether the calibrator needs to pre-reshape the inputs for weight quantization calibration."""
        return not self.needs_x_quant_for_wgts and self.config.pre_reshape

    @property
    def needs_to_pre_reshape_w_for_ipts(self) -> bool:
        """Whether the calibrator needs to pre-reshape the weights for input quantization calibration."""
        return not self.needs_w_quant_for_ipts and self.config.pre_reshape

    def _reset(self, **kwargs) -> None:
        pass

    def reset(self, **kwargs) -> None:
        """Reset the calibrator."""
        self.iter = 0
        self.candidate_id = 0
        self._reset(**kwargs)
        self._state_dict: list[tuple[nn.Parameter, torch.Tensor]] = []
        self._hooks: list[Hook | torch.utils.hooks.RemovableHandle] = []

    def is_done(self) -> bool:
        """Check if the calibration is done."""
        return self.iter >= self.num_iters

    def is_last_iter(self) -> bool:
        """Check if the current iteration is the last one."""
        return self.iter == self.num_iters - 1

    def is_last_candidate_in_iter(self) -> bool:
        """Check if the current candidate is the last one in the current iteration."""
        return self.candidate_id == self.population_size - 1

    @abstractmethod
    def get_best(self) -> _CANDIDATE:
        """Get the best candidate.

        Returns:
            `_CANDIDATE`:
                The best candidate.
        """
        ...

    @abstractmethod
    def _ask(self) -> _CANDIDATE:
        """Ask for the next candidate.

        Returns:
            `_CANDIDATE`:
                The next candidate.
        """
        ...

    @abstractmethod
    def _tell(self, error: list[torch.Tensor]) -> None:
        """Tell the error of the last candidate and update the best candidate.

        Args:
            error (`list[torch.Tensor]`):
                The error of the last candidate.
        """
        ...

    def ask(self) -> _CANDIDATE:
        """Ask for the next candidate.

        Returns:
            `_CANDIDATE`:
                The next candidate.
        """
        self.candidate = self._ask()
        return self.candidate

    def tell(self, error: list[torch.Tensor]) -> None:
        """Tell the error of the last candidate and update the best candidate.

        Args:
            error (`list[torch.Tensor]`):
                The error of the last candidate.
        """
        # Subclasses should set this to True when the best candidate is updated.
        self._last_tell_updated_best = False
        self._tell(error)
        self.candidate_id += 1
        if self.candidate_id >= self.population_size:
            self.iter += 1
            self.candidate_id = 0

    def _get_output_pair_dump_context(self) -> tuple[Path, Path, Path, Path, torch.dtype] | None:
        dump_root = os.environ.get("DEEPCOMPRESSOR_CALIB_OUTPUTPAIR_DUMP_DIR", "").strip()
        if not dump_root:
            return None
        ctx = os.environ.get("DEEPCOMPRESSOR_CALIB_CONTEXT", "").strip()
        if not ctx:
            ctx = f"{type(self).__name__}_{id(self)}"
        if os.environ.get("DEEPCOMPRESSOR_CALIB_OUTPUTPAIR_INSTANCE_CONTEXT", "").strip().lower() not in {
            "",
            "0",
            "false",
            "no",
        }:
            inst = os.environ.get("DEEPCOMPRESSOR_CALIB_CONTEXT_INSTANCE", "").strip()
            if inst:
                ctx = f"{ctx}--{inst}"
        ctx = re.sub(r"[^a-zA-Z0-9_.-]+", "_", ctx)
        root = Path(dump_root).expanduser().resolve()
        ctx_dir = root / ctx
        fp_dir = ctx_dir / "fp"
        tmp_dir = ctx_dir / "_tmp"
        best_dir = ctx_dir / "best"
        cand_dir = ctx_dir / "candidates"
        dtype_name = os.environ.get("DEEPCOMPRESSOR_CALIB_OUTPUTPAIR_DTYPE", "fp16").strip().lower()
        if dtype_name in {"fp16", "float16", "half"}:
            dump_dtype = torch.float16
        elif dtype_name in {"bf16", "bfloat16"}:
            dump_dtype = torch.bfloat16
        elif dtype_name in {"fp32", "float32"}:
            dump_dtype = torch.float32
        else:
            dump_dtype = torch.float16

        fp_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        cand_dir.mkdir(parents=True, exist_ok=True)
        ctx_dir.mkdir(parents=True, exist_ok=True)
        return fp_dir, tmp_dir, best_dir, cand_dir, dump_dtype

    def _output_pair_dump_requires_layer(self, error: list[torch.Tensor]) -> bool:
        require_layer = os.environ.get("DEEPCOMPRESSOR_CALIB_OUTPUTPAIR_REQUIRE_LAYER", "1").strip().lower() not in {
            "",
            "0",
            "false",
            "no",
        }
        if not require_layer:
            return True
        return all(isinstance(e, torch.Tensor) and e.numel() == 1 for e in error)

    @staticmethod
    def _maybe_extract_timestep(eval_kwargs: dict[str, tp.Any]) -> torch.Tensor | int | float | None:
        if not isinstance(eval_kwargs, dict):
            return None
        ts = eval_kwargs.get("timestep", None)
        if torch.is_tensor(ts):
            return ts.detach().cpu()
        if isinstance(ts, (int, float)):
            return ts
        return None

    def _dump_output_pair_fp_batch(
        self, *, fp_dir: Path, batch_idx: int, eval_kwargs: dict[str, tp.Any], y_fp: torch.Tensor, dump_dtype: torch.dtype
    ) -> None:
        if not self._env_truthy("DEEPCOMPRESSOR_CALIB_OUTPUTPAIR_SAVE_TENSORS", "0"):
            return
        out_path = fp_dir / f"batch_{int(batch_idx):05d}.pt"
        if out_path.exists():
            return
        payload = {
            "batch_idx": int(batch_idx),
            "timestep": self._maybe_extract_timestep(eval_kwargs),
            "y_fp": y_fp.detach().to(dtype=dump_dtype).cpu(),
        }
        torch.save(payload, str(out_path))

    def _dump_output_pair_tmp_batch(
        self,
        *,
        tmp_dir: Path,
        batch_idx: int,
        eval_kwargs: dict[str, tp.Any],
        y_q: torch.Tensor,
        dump_dtype: torch.dtype,
    ) -> None:
        if not self._env_truthy("DEEPCOMPRESSOR_CALIB_OUTPUTPAIR_SAVE_TENSORS", "0"):
            return
        out_path = tmp_dir / f"batch_{int(batch_idx):05d}.pt"
        payload = {
            "batch_idx": int(batch_idx),
            "timestep": self._maybe_extract_timestep(eval_kwargs),
            "y_q": y_q.detach().to(dtype=dump_dtype).cpu(),
        }
        torch.save(payload, str(out_path))

    @staticmethod
    def _env_truthy(name: str, default: str = "") -> bool:
        v = os.environ.get(name, default).strip().lower()
        return v not in {"", "0", "false", "no"}

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)).strip())
        except Exception:
            return int(default)

    @staticmethod
    def _get_cache_timesteps_full(cache: TensorsCache | None) -> torch.Tensor | None:
        if cache is None:
            return None
        batches = getattr(cache, "calib_timesteps_batches", None)
        if batches is None:
            return None
        if not isinstance(batches, list) or len(batches) == 0:
            return None
        ts = [b.detach().cpu().view(-1) for b in batches if torch.is_tensor(b)]
        if len(ts) == 0:
            return None
        return torch.cat(ts, dim=0)

    @staticmethod
    def _get_cache_guidances_full(cache: TensorsCache | None) -> torch.Tensor | None:
        if cache is None:
            return None
        batches = getattr(cache, "calib_guidances_batches", None)
        if batches is None:
            return None
        if not isinstance(batches, list) or len(batches) == 0:
            return None
        gs = [b.detach().cpu().view(-1) for b in batches if torch.is_tensor(b)]
        if len(gs) == 0:
            return None
        return torch.cat(gs, dim=0)

    @staticmethod
    def _diff_to_bck(diff: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Returns:
          x: [B, C, K] float32
          C: channel count
        """
        if diff.ndim == 4:  # [B,C,H,W]
            b, c, h, w = diff.shape
            return diff.reshape(b, c, -1), c
        if diff.ndim == 3:  # assume [B,L,C]
            b, l, c = diff.shape
            x = diff.permute(0, 2, 1).reshape(b, c, -1)
            return x, c
        if diff.ndim == 2:  # [B,C]
            b, c = diff.shape
            return diff.reshape(b, c, 1), c
        # Fallback: treat last dim as channel, flatten others.
        b = diff.shape[0]
        c = diff.shape[-1]
        x = diff.reshape(b, -1, c).permute(0, 2, 1).contiguous()
        return x, c

    @classmethod
    def _accumulate_sse_n_per_timestep_per_channel(
        cls,
        *,
        sse_by_t: dict[int, torch.Tensor] | dict[int, dict[int, torch.Tensor]],
        n_by_t: dict[int, float] | dict[int, dict[int, float]],
        diff: torch.Tensor,
        timesteps: torch.Tensor,
        guidances: torch.Tensor | None = None,
    ) -> None:
        diff = diff.to(torch.float32)
        x, c = cls._diff_to_bck(diff)  # [B,C,K]
        t = timesteps
        if t.ndim == 0:
            t = t.view(1).expand(x.shape[0])
        if t.ndim != 1:
            t = t.view(-1)[: x.shape[0]]
        if t.shape[0] != x.shape[0]:
            t = t[: x.shape[0]]

        g = guidances
        if g is not None:
            if g.ndim == 0:
                g = g.view(1).expand(x.shape[0])
            if g.ndim != 1:
                g = g.view(-1)[: x.shape[0]]
            if g.shape[0] != x.shape[0]:
                g = g[: x.shape[0]]

        if g is None:
            sse_flat: dict[int, torch.Tensor] = tp.cast(dict[int, torch.Tensor], sse_by_t)
            n_flat: dict[int, float] = tp.cast(dict[int, float], n_by_t)
            for tt in t.unique():
                mask = t == tt
                if not torch.any(mask):
                    continue
                xm = x[mask]
                sse = (xm * xm).sum(dim=(0, 2)).detach().cpu()  # [C]
                n = float(xm.numel() / max(1, int(c)))
                tt_i = int(tt.item())
                if tt_i not in sse_flat:
                    sse_flat[tt_i] = sse
                else:
                    sse_flat[tt_i] = sse_flat[tt_i] + sse
                n_flat[tt_i] = float(n_flat.get(tt_i, 0.0) + n)
            return

        sse_g: dict[int, dict[int, torch.Tensor]] = tp.cast(dict[int, dict[int, torch.Tensor]], sse_by_t)
        n_g: dict[int, dict[int, float]] = tp.cast(dict[int, dict[int, float]], n_by_t)
        assert g is not None
        for gg in g.unique():
            mask_g = g == gg
            if not torch.any(mask_g):
                continue
            gg_i = int(gg.item())
            sse_g.setdefault(gg_i, {})
            n_g.setdefault(gg_i, {})
            xg = x[mask_g]
            tg = t[mask_g]
            for tt in tg.unique():
                mask = tg == tt
                if not torch.any(mask):
                    continue
                xm = xg[mask]
                sse = (xm * xm).sum(dim=(0, 2)).detach().cpu()  # [C]
                n = float(xm.numel() / max(1, int(c)))
                tt_i = int(tt.item())
                if tt_i not in sse_g[gg_i]:
                    sse_g[gg_i][tt_i] = sse
                else:
                    sse_g[gg_i][tt_i] = sse_g[gg_i][tt_i] + sse
                n_g[gg_i][tt_i] = float(n_g[gg_i].get(tt_i, 0.0) + n)

    def _dump_output_pair_candidate_batch(
        self,
        *,
        cand_dir: Path,
        iter_idx: int,
        candidate_id: int,
        batch_idx: int,
        eval_kwargs: dict[str, tp.Any],
        y_q: torch.Tensor,
        dump_dtype: torch.dtype,
    ) -> None:
        max_candidates = self._env_int("DEEPCOMPRESSOR_CALIB_OUTPUTPAIR_MAX_CANDIDATES", -1)
        max_batches = self._env_int("DEEPCOMPRESSOR_CALIB_OUTPUTPAIR_MAX_BATCHES", -1)
        if max_candidates >= 0 and candidate_id >= max_candidates:
            return
        if max_batches >= 0 and batch_idx >= max_batches:
            return
        out_dir = cand_dir / f"iter_{int(iter_idx):03d}" / f"cand_{int(candidate_id):05d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"batch_{int(batch_idx):05d}.pt"
        payload = {
            "iter": int(iter_idx),
            "candidate_id": int(candidate_id),
            "batch_idx": int(batch_idx),
            "timestep": self._maybe_extract_timestep(eval_kwargs),
            "y_q": y_q.detach().to(dtype=dump_dtype).cpu(),
        }
        torch.save(payload, str(out_path))

    def _dump_output_pair_candidate_meta(
        self,
        *,
        cand_dir: Path,
        iter_idx: int,
        candidate_id: int,
        eval_kwargs: dict[str, tp.Any],
        error: list[torch.Tensor],
        candidate_meta: dict[str, tp.Any],
    ) -> None:
        max_candidates = self._env_int("DEEPCOMPRESSOR_CALIB_OUTPUTPAIR_MAX_CANDIDATES", -1)
        if max_candidates >= 0 and candidate_id >= max_candidates:
            return
        out_dir = cand_dir / f"iter_{int(iter_idx):03d}" / f"cand_{int(candidate_id):05d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "context": os.environ.get("DEEPCOMPRESSOR_CALIB_CONTEXT", ""),
            "tensor_type": str(self.tensor_type),
            "objective": str(self.objective),
            "granularity": str(self.granularity),
            "timestep": self._maybe_extract_timestep(eval_kwargs),
            "error": [e.detach().cpu() for e in error],
            **candidate_meta,
        }
        torch.save(meta, str(out_dir / "meta.pt"))

    def _finalize_output_pair_candidate(
        self,
        *,
        tmp_dir: Path,
        best_dir: Path,
        fp_dir: Path,
        eval_kwargs: dict[str, tp.Any],
        error: list[torch.Tensor],
        candidate_meta: dict[str, tp.Any],
    ) -> None:
        meta = {
            "context": os.environ.get("DEEPCOMPRESSOR_CALIB_CONTEXT", ""),
            "tensor_type": str(self.tensor_type),
            "objective": str(self.objective),
            "granularity": str(self.granularity),
            "timestep": self._maybe_extract_timestep(eval_kwargs),
            "error": [e.detach().cpu() for e in error],
            **candidate_meta,
        }
        torch.save(meta, str(tmp_dir / "meta.pt"))

        updated_best = bool(getattr(self, "_last_tell_updated_best", False))
        can_keep_best = self._output_pair_dump_requires_layer(error)

        if updated_best and can_keep_best:
            try:
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                tmp_dir.rename(best_dir)
            finally:
                tmp_dir.mkdir(parents=True, exist_ok=True)
            # Write a small pointer file.
            with open(best_dir.parent / "best_path.txt", "w", encoding="utf-8") as f:
                f.write(str(best_dir) + "\n")
                f.write(str(fp_dir) + "\n")
        else:
            # Best-only mode: ensure temp artifacts do not remain on disk.
            for p in tmp_dir.glob("*.pt"):
                try:
                    p.unlink()
                except Exception:
                    pass
            for p in tmp_dir.glob("*.meta.json"):
                try:
                    p.unlink()
                except Exception:
                    pass
            try:
                (tmp_dir / "meta.pt").unlink()
            except Exception:
                pass

    def _parse_ipts(self, ipts: TensorsCache | None, set_device: bool = False) -> TensorsCache | None:
        if ipts is None:
            return None
        if set_device:
            # Always honor `outputs_device` (default: cpu) to avoid storing all precomputed fp outputs on GPU,
            # which can easily OOM for long-sequence diffusion models like FLUX.
            self.opts_device = self.config.outputs_device
        if self.objective == SearchBasedCalibObjective.ProductsError:
            batch_size = self.config.element_batch_size
            calib_size = self.config.element_size
        elif self.objective == SearchBasedCalibObjective.OutputsError:
            batch_size = self.config.sample_batch_size
            calib_size = self.config.sample_size
        else:
            assert self.objective == SearchBasedCalibObjective.TensorError
            batch_size = -1
            calib_size = -1
        prev_size = len(ipts.front().data)
        parsed_ipts = TensorsCache(
            {
                key: ipt.repartition(
                    max_batch_size=batch_size,
                    max_size=calib_size,
                    standardize=self.objective == SearchBasedCalibObjective.ProductsError,
                    reshape=self.tensor_type == TensorType.Weights,
                )
                for key, ipt in ipts.items()
            }
        )
        # Preserve optional calibration metadata (e.g., timesteps) attached by cache loaders.
        if hasattr(ipts, "calib_timesteps_batches"):
            setattr(parsed_ipts, "calib_timesteps_batches", getattr(ipts, "calib_timesteps_batches"))
        if hasattr(ipts, "calib_guidances_batches"):
            setattr(parsed_ipts, "calib_guidances_batches", getattr(ipts, "calib_guidances_batches"))
        curr_size = len(parsed_ipts.front().data)
        assert all(len(ipt.data) == curr_size for ipt in parsed_ipts.values())
        return parsed_ipts

    def _parse_args(  # noqa: C901
        self,
        x_wgts: list[nn.Parameter] | None,
        y_wgts: list[nn.Parameter] | None,
        x_acts: TensorsCache | None,
        y_acts: TensorsCache | None,
        eval_inputs: TensorsCache | None,
        eval_module: nn.Module | None,
        x_mods: list[nn.Module] | None,
        y_mods: list[nn.Module] | None,
        orig_x_wgts: list[tuple[nn.Parameter, torch.Tensor]] | None,
        orig_y_wgts: list[tuple[nn.Parameter, torch.Tensor]] | None,
        orig_x_acts: TensorsCache | None,
        orig_y_acts: TensorsCache | None,
        orig_eval_inputs: TensorsCache | None,
    ) -> tuple[
        list[torch.Tensor | nn.Parameter] | None,  # x_wgts
        list[torch.Tensor | nn.Parameter] | None,  # y_wgts
        TensorsCache | None,  # x_acts
        TensorsCache | None,  # y_acts
        TensorsCache | None,  # eval_inputs
        nn.Module | None,  # eval_module
        list[nn.Module] | None,  # x_mods
        list[nn.Module] | None,  # y_mods
        list[tuple[nn.Parameter, torch.Tensor]] | None,  # orig_x_wgts
        list[tuple[nn.Parameter, torch.Tensor]] | None,  # orig_y_wgts
        TensorCache | None,  # orig_x_acts
        TensorCache | None,  # orig_y_acts
        TensorCache | None,  # orig_eval_inputs
    ]:
        # region Check the types of the arguments
        if x_wgts is not None:
            assert isinstance(x_wgts, (tuple, list)), "x_wgts should be a list"
            assert all(isinstance(w, nn.Parameter) for w in x_wgts), "wgts should be a list of nn.Parameter"
        if y_wgts is not None:
            assert isinstance(y_wgts, (tuple, list)), "y_wgts should be a list"
            assert all(isinstance(w, nn.Parameter) for w in y_wgts), "wgts should be a list of nn.Parameter"
        if x_acts is not None:
            assert isinstance(x_acts, TensorsCache), "x_acts should be a TensorsCache"
        if y_acts is not None:
            assert isinstance(y_acts, TensorsCache), "y_acts should be a TensorsCache"
        if eval_inputs is not None:
            assert isinstance(eval_inputs, TensorsCache), "eval_inputs should be a TensorsCache"
        if x_mods is not None:
            assert isinstance(x_mods, (tuple, list)), "x_mods should be a list"
        if y_mods is not None:
            assert isinstance(y_mods, (tuple, list)), "y_mods should be a list"
        if orig_x_wgts is not None:
            assert isinstance(orig_x_wgts, (tuple, list)), "orig_x_wgts should be a list"
            assert all(isinstance(p, nn.Parameter) and isinstance(w, torch.Tensor) for p, w in orig_x_wgts), (
                "orig_x_wgts should be a list of tuples of nn.Parameter and torch.Tensor"
            )
            if x_wgts is not None:
                assert len(orig_x_wgts) >= len(x_wgts), "orig_wgts should have at least as mtp.Any elements as wgts"
                assert all(p is w for (p, _), w in zip(orig_x_wgts, x_wgts, strict=False)), (
                    "the parameters in orig_wgts should be in wgts in the same order"
                )
        if orig_y_wgts is not None:
            assert isinstance(orig_y_wgts, (tuple, list)), "orig_y_wgts should be a list"
            assert all(isinstance(p, nn.Parameter) and isinstance(w, torch.Tensor) for p, w in orig_y_wgts), (
                "orig_y_wgts should be a list of tuples of nn.Parameter and torch.Tensor"
            )
            if y_wgts is not None:
                assert len(orig_y_wgts) >= len(y_wgts), "orig_wgts should have at least as mtp.Any elements as wgts"
                assert all(p is w for (p, _), w in zip(orig_y_wgts, y_wgts, strict=False)), (
                    "the parameters in orig_wgts should be in wgts in the same order"
                )
        if orig_x_acts is not None:
            assert isinstance(orig_x_acts, TensorsCache), "orig_x_acts should be a TensorsCache"
        if orig_y_acts is not None:
            assert isinstance(orig_y_acts, TensorsCache), "orig_y_acts should be a TensorsCache"
        if orig_eval_inputs is not None:
            assert isinstance(orig_eval_inputs, TensorsCache), "orig_eval_inputs should be a TensorsCache"
        # endregion
        self.objective = self.config.objective
        self.granularity = self.config.granularity
        if self.tensor_type == TensorType.Outputs:
            # ! currently only support OutputsError and Layer granularity for Outputs
            self.objective = SearchBasedCalibObjective.OutputsError
            self.granularity = SearchBasedCalibGranularity.Layer
        if self.objective == SearchBasedCalibObjective.TensorError:
            if x_wgts is not None:
                x_wgts = [w.detach().data for w in x_wgts]
            if y_wgts is not None:
                y_wgts = [w.detach().data for w in y_wgts]
            if self.tensor_type == TensorType.Weights:
                assert x_wgts is not None, "wgts should not be None when tensor_type is Weights"
            elif self.tensor_type == TensorType.Inputs:
                assert x_acts is not None, "mod_ipts should not be None when tensor_type is Inputs"
                eval_inputs, orig_eval_inputs = x_acts, orig_x_acts
            else:  # self.tensor_type == TensorType.Outputs
                assert y_acts is not None, "opts should not be None when tensor_type is Outputs"
                eval_inputs, orig_eval_inputs = y_acts, orig_y_acts
            eval_module = None
        elif self.objective == SearchBasedCalibObjective.ProductsError:
            assert self.tensor_type in (
                TensorType.Weights,
                TensorType.Inputs,
            ), "tensor_type should be Weights or Inputs when objective is ProductsError"
            assert x_wgts is not None, "wgts should not be None when objective is ProductsError"
            x_wgts = [w.detach().data for w in x_wgts]
            if y_wgts is not None:
                y_wgts = [w.detach().data for w in y_wgts]
            x_acts = x_acts or eval_inputs
            orig_x_acts = orig_x_acts or orig_eval_inputs
            assert x_acts is not None, "x_acts should not be None when objective is ProductsError"
            eval_inputs, orig_eval_inputs = x_acts, orig_x_acts
        elif self.objective == SearchBasedCalibObjective.OutputsError:
            assert eval_inputs is not None, "eval_inputs should not be None when objective is OutputsError"
            assert eval_module is not None, "eval_module should not be None when OutputsError"
            if (
                isinstance(eval_module, (nn.Linear, nn.Conv2d))
                and self.granularity.value < SearchBasedCalibGranularity.Layer.value
                and self.tensor_type != TensorType.Outputs
            ):
                self.objective = SearchBasedCalibObjective.ProductsError
                x_wgts = [w.detach().data for w in x_wgts]
                if y_wgts is not None:
                    y_wgts = [w.detach().data for w in y_wgts]
                x_acts = x_acts or eval_inputs
                orig_x_acts = orig_x_acts or orig_eval_inputs
                assert x_acts is not None, "x_acts should not be None when objective is ProductsError"
                eval_inputs, orig_eval_inputs = x_acts, orig_x_acts
            else:
                self.objective = SearchBasedCalibObjective.OutputsError
                self.granularity = SearchBasedCalibGranularity.Layer
        else:
            raise ValueError(f"unknown objective: {self.objective}")
        self.logger.debug(
            f"+ tensor_type: {self.tensor_type}, objective: {self.objective}, granularity: {self.granularity}"
        )
        return (
            x_wgts,
            y_wgts,
            x_acts,
            y_acts,
            self._parse_ipts(eval_inputs, set_device=True),
            eval_module,
            x_mods,
            y_mods,
            orig_x_wgts,
            orig_y_wgts,
            orig_x_acts,
            orig_y_acts,
            self._parse_ipts(orig_eval_inputs),
        )

    # region Reshape functions for computing products
    def _reshape_w_for_wgts_centric_partial_products(self, w: torch.Tensor, *, view_shape: torch.Size) -> torch.Tensor:
        return _reshape_w_for_wgts(w, view_shape)

    def _reshape_x_for_wgts_centric_partial_products(
        self, x: torch.Tensor, *, view_shape: torch.Size, fn: ReshapeFn
    ) -> torch.Tensor:
        return _reshape_x_for_wgts(fn(x), view_shape)

    def _reshape_w_for_ipts_centric_partial_products(self, w: torch.Tensor, *, view_shape: torch.Size) -> torch.Tensor:
        return _reshape_w_for_ipts(w, view_shape)

    def _reshape_x_for_ipts_centric_partial_products(
        self, x: torch.Tensor, *, view_shape: torch.Size, fn: ReshapeFn = None
    ) -> torch.Tensor:
        return _reshape_x_for_ipts(x, view_shape)

    def _reshape_w_for_full_products(self, w: torch.Tensor, *, view_shape: torch.Size = None) -> torch.Tensor:
        return w.view(w.shape[0], -1).T

    def _reshape_x_for_full_products(
        self, x: torch.Tensor, *, fn: ReshapeFn, view_shape: torch.Size = None
    ) -> torch.Tensor:
        return fn(x).view(x.shape[0], -1)

    # endregion

    @abstractmethod
    def _process_x_in_xw(self, x: torch.Tensor, channels_dim: int | _MISSING_TYPE = MISSING) -> torch.Tensor: ...

    @abstractmethod
    def _process_w_in_xw(self, w: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def _process_y_in_yx(self, y: torch.Tensor, channels_dim: int | _MISSING_TYPE = MISSING) -> torch.Tensor: ...

    @abstractmethod
    def _process_x_in_yx(self, x: torch.Tensor, channels_dim: int | _MISSING_TYPE = MISSING) -> torch.Tensor: ...

    @abstractmethod
    def _process_xw_in_yx(self, w: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def _process_yw_in_yx(self, w: torch.Tensor) -> torch.Tensor: ...

    def _recover_mod(self) -> None:
        for p, w in self._state_dict:
            p.data = w
        self._state_dict.clear()
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _process_wgts_centric_mod(
        self, wgts: list[nn.Parameter], mods: list[nn.Module], update_state_dict: bool = True, **kwargs
    ) -> None:
        if self.needs_w_quant_for_wgts:
            for w in wgts:
                if update_state_dict:
                    self._state_dict.append((w, w.data))
                w.data = self._process_w_in_xw(w.data)
        if self.needs_x_quant_for_wgts:
            self._hooks.append(self.x_quantizer.as_hook(func=self._process_x_in_xw, is_output=False).register(mods))

    def _process_ipts_centric_mod(
        self, wgts: list[nn.Parameter], mods: list[nn.Module], update_state_dict: bool = True, **kwargs
    ) -> None:
        if self.needs_w_quant_for_ipts:
            for w in wgts:
                if update_state_dict:
                    self._state_dict.append((w, w.data))
                w.data = self._process_w_in_xw(w.data)
        if self.needs_x_quant_for_ipts:
            self._hooks.append(self.x_quantizer.as_hook(self._process_x_in_xw, is_output=False).register(mods))

    def _process_opts_centric_mod(
        self,
        x_wgts: list[nn.Parameter],
        y_wgts: list[nn.Parameter],
        x_mods: list[nn.Module],
        y_mods: list[nn.Module],
        update_state_dict: bool = True,
        **kwargs,
    ) -> None:
        if self.needs_w_quant_for_opts:
            for w in x_wgts:
                if update_state_dict:
                    self._state_dict.append((w, w.data))
                w.data = self._process_xw_in_yx(w.detach().data)
            for w in y_wgts:
                if update_state_dict:
                    self._state_dict.append((w, w.data))
                w.data = self._process_yw_in_yx(w.detach().data)
        if self.needs_x_quant_for_opts:
            self._hooks.append(self.x_quantizer.as_hook(self._process_x_in_yx, is_output=True).register(x_mods))
        if self.needs_y_quant_for_opts:
            self._hooks.append(self.y_quantizer.as_hook(self._process_y_in_yx, is_output=True).register(y_mods))

    def calibrate(
        self,
        x_wgts: list[nn.Parameter] | None = None,
        y_wgts: list[nn.Parameter] | None = None,
        x_acts: TensorsCache | None = None,
        y_acts: TensorsCache | None = None,
        x_mods: list[nn.Module] | None = None,
        y_mods: list[nn.Module] | None = None,
        eval_inputs: TensorsCache | None = None,
        eval_module: nn.Module | None = None,
        eval_kwargs: dict[str, tp.Any] | None = None,
        orig_x_wgts: list[tuple[nn.Parameter, torch.Tensor]] | None = None,
        orig_y_wgts: list[tuple[nn.Parameter, torch.Tensor]] | None = None,
        orig_x_acts: TensorsCache | None = None,
        orig_y_acts: TensorsCache | None = None,
        orig_eval_inputs: TensorsCache | None = None,
        **kwargs,
    ) -> _CANDIDATE:
        """Calibrate the quantization parameters.

        Args:
            x_wgts (`list[nn.Parameter]` or `None`, *optional*, defaults to `None`):
                The weights in x-w computation, or weights that generates x for y-x computation.
            y_wgts (`list[nn.Parameter]` or `None`, *optional*, defaults to `None`):
                The weights that generates y for y-x computation.
            x_acts (`TensorsCache` or `None`, *optional*, defaults to `None`):
                The x activations. It should be x for x-w or y-x computation.
            y_acts (`TensorsCache` or `None`, *optional*, defaults to `None`):
                The y activations. It should be y for y-x computation.
            eval_inputs (`TensorsCache` or `None`, *optional*, defaults to `None`):
                The inputs of evaluation module `eval_module`.
            eval_module (`nn.Module` or `None`, *optional*, defaults to `None`):
                The module used for evaluation.
            x_mods (`list[nn.Module]` or `None`, *optional*, defaults to `None`):
                The modules for x activation quantization.
                It should be the modules that take in x for x-w computation,
                or the modules that generates x for y-x computation.
            y_mods (`list[nn.Module]` or `None`, *optional*, defaults to `None`):
                The modules for y activation quantization.
                It should be the modules that generates y for y-x computation.
            orig_x_wgts (`list[tuple[nn.Parameter, torch.Tensor]]` or `None`, *optional*, defaults to `None`):
                The original weights for `x_mods`.
            orig_y_wgts (`list[tuple[nn.Parameter, torch.Tensor]]` or `None`, *optional*, defaults to `None`):
                The original weights for `y_mods`.
            orig_x_acts (`TensorsCache` or `None`, *optional*, defaults to `None`):
                The original x activations `x_acts`.
            orig_y_acts (`TensorsCache` or `None`, *optional*, defaults to `None`):
                The original y activations `y_acts`.
            orig_eval_inputs (`TensorsCache` or `None`, *optional*, defaults to `None`):
                The original inputs of evaluation module `eval_inputs`.
            eval_kwargs (`dict[str, tp.Any]` or `None`, *optional*, defaults to `None`):
                The keyword arguments for evaluation module `eval_module`.

        Returns:
            `_CANDIDATE`:
                The best candidate.
        """
        tools.logging.Formatter.indent_inc()
        if self.w_quantizer is not None and self.w_quantizer.is_enabled():
            self.logger.debug(f"+ w: {self.w_quantizer.config.quant_dtype}")
        else:
            self.logger.debug("+ w: None")
        if self.x_quantizer is not None and self.x_quantizer.is_enabled():
            self.logger.debug(f"+ x: {self.x_quantizer.config.quant_dtype}")
        else:
            self.logger.debug("+ x: None")
        if self.y_quantizer is not None and self.y_quantizer.is_enabled():
            self.logger.debug(f"+ y: {self.y_quantizer.config.quant_dtype}")
        else:
            self.logger.debug("+ y: None")
        (
            x_wgts,
            y_wgts,
            x_acts,
            y_acts,
            eval_inputs,
            eval_module,
            x_mods,
            y_mods,
            orig_x_wgts,
            orig_y_wgts,
            orig_x_acts,
            orig_y_acts,
            orig_eval_inputs,
        ) = self._parse_args(
            x_wgts,
            y_wgts,
            x_acts,
            y_acts,
            eval_inputs,
            eval_module,
            x_mods,
            y_mods,
            orig_x_wgts,
            orig_y_wgts,
            orig_x_acts,
            orig_y_acts,
            orig_eval_inputs,
        )
        eval_kwargs = eval_kwargs or {}
        self.logger.debug(f"+ finished parsing calibration arguments, ram usage: {psutil.virtual_memory().percent}")
        self.reset(
            x_wgts=x_wgts,
            y_wgts=y_wgts,
            x_acts=x_acts,
            y_acts=y_acts,
            eval_inputs=eval_inputs,
            eval_module=eval_module,
            x_mods=x_mods,
            y_mods=y_mods,
            orig_x_wgts=orig_x_wgts,
            orig_y_wgts=orig_y_wgts,
            orig_x_acts=orig_x_acts,
            orig_y_acts=orig_y_acts,
            orig_eval_inputs=orig_eval_inputs,
            eval_kwargs=eval_kwargs,
            **kwargs,
        )
        self.logger.debug(f"+ finished resetting calibrator, ram usage: {psutil.virtual_memory().percent}")
        gc.collect()
        torch.cuda.empty_cache()
        if self.tensor_type == TensorType.Weights:
            result = self._calibrate_wgts(
                x_wgts, eval_inputs, eval_module, x_mods, orig_x_wgts, orig_eval_inputs, eval_kwargs, **kwargs
            )
        elif self.tensor_type == TensorType.Inputs:
            result = self._calibrate_ipts(
                x_wgts, eval_inputs, eval_module, x_mods, orig_x_wgts, orig_eval_inputs, eval_kwargs, **kwargs
            )
        else:
            result = self._calibrate_opts(
                x_wgts,
                y_wgts,
                eval_inputs,
                eval_module,
                x_mods,
                y_mods,
                orig_x_wgts,
                orig_y_wgts,
                orig_eval_inputs,
                eval_kwargs,
                **kwargs,
            )
        tools.logging.Formatter.indent_dec()
        return result

    def _calibrate_wgts(  # noqa: C901
        self,
        wgts: list[torch.Tensor | nn.Parameter],
        ipts: TensorsCache | None,
        eval_module: nn.Module | None,
        mods: list[nn.Module] | None,
        orig_wgts: list[tuple[nn.Parameter, torch.Tensor]] | None,
        orig_ipts: TensorsCache | None,
        eval_kwargs: dict[str, tp.Any],
        **kwargs,
    ) -> tp.Any:
        # region Step 1: Calculate the baseline
        if self.objective == SearchBasedCalibObjective.TensorError:
            if orig_wgts is None:
                orig_wgts = [(None, w.detach().data) for w in wgts]
            assert all(w.shape[1:] == wgts[0].shape[1:] for w in wgts)
            assert all(w.shape[1:] == wgts[0].shape[1:] for _, w in orig_wgts)
            orig_opts = None
            w_view_shapes = [infer_view_shape(w.shape, self.w_quantizer.config.largest_group_shape) for w in wgts]
        elif self.objective == SearchBasedCalibObjective.ProductsError:
            if orig_wgts is None:
                orig_wgts = [(None, w.detach().data) for w in wgts]
            assert len(orig_wgts) == len(wgts)
            assert all(w.shape[1:] == wgts[0].shape[1:] for w in wgts)
            assert all(w.shape[1:] == wgts[0].shape[1:] for _, w in orig_wgts)
            w_view_shapes = [infer_view_shape(w.shape, self.w_quantizer.config.largest_group_shape) for w in wgts]
            if self.granularity != SearchBasedCalibGranularity.Layer:
                _reshape_x = self._reshape_x_for_wgts_centric_partial_products
                _reshape_w = self._reshape_w_for_wgts_centric_partial_products
            else:
                _reshape_x = self._reshape_x_for_full_products
                _reshape_w = self._reshape_w_for_full_products
            assert isinstance(ipts, TensorsCache), "ipts should not be None for ProductsError"
            if orig_ipts is None:
                orig_ipts = ipts
            same_ipts = orig_ipts is ipts
            orig_ipts = TensorsCache(
                {
                    key: TensorCache(
                        [_reshape_x(x, view_shape=w_view_shapes[0], fn=ipt.reshape) for x in ipt.data],
                        **ipt.get_factory_kwargs(channels_dim=1, reshape=ReshapeFn()),
                    )
                    for key, ipt in orig_ipts.items()
                },
            )
            orig_opts: dict[tuple[int, ...], torch.Tensor] = {}
            for j, (_, w) in enumerate(orig_wgts):
                w = _reshape_w(w, view_shape=w_view_shapes[j])
                for s, ipt in enumerate(orig_ipts):
                    for i, x in enumerate(ipt.data):
                        x = x.to(device=w.device, non_blocking=True)
                        y = torch.matmul(x, w)
                        y = y.view(*y.shape[:-2], y.shape[-2] * y.shape[-1])
                        orig_opts[(i, s, j)] = y.to(device=self.opts_device or y.device, non_blocking=True)
            if self.needs_to_pre_reshape_x_for_wgts:
                if same_ipts:
                    ipts = orig_ipts
                else:
                    ipts = TensorsCache(
                        {
                            key: TensorCache(
                                [_reshape_x(x, view_shape=w_view_shapes[0], fn=ipt.reshape) for x in ipt.data],
                                **ipt.get_factory_kwargs(channels_dim=1, reshape=ReshapeFn()),
                            )
                            for key, ipt in ipts.items()
                        }
                    )
            del orig_wgts, orig_ipts, same_ipts
        elif self.objective == SearchBasedCalibObjective.OutputsError:
            w_view_shapes, _state_dict = [], []
            dump_ctx = self._get_output_pair_dump_context()
            if dump_ctx is not None:
                fp_dir, tmp_dir, best_dir, cand_dir, dump_dtype = dump_ctx
            if orig_wgts is not None:
                _state_dict = [(p, p.data) for p, _ in orig_wgts]
                for p, w in orig_wgts:
                    p.data = w.to(device=p.data.device)
            if orig_ipts is None:
                orig_ipts = ipts
            assert isinstance(orig_ipts, TensorsCache), "orig_ipts should not be None for OutputsError"
            orig_opts: dict[tuple[int, ...], torch.Tensor] = {}
            for i in range(len(orig_ipts.front().data)):
                ipt = orig_ipts.extract(i, eval_kwargs)
                y = eval_module(*ipt.args, **ipt.kwargs)
                y = y[0] if not isinstance(y, torch.Tensor) else y
                assert isinstance(y, torch.Tensor), "eval_mod should return a tensor"
                orig_opts[(i,)] = y.to(device=self.opts_device or y.device, non_blocking=True)
                if dump_ctx is not None:
                    self._dump_output_pair_fp_batch(
                        fp_dir=fp_dir, batch_idx=i, eval_kwargs=eval_kwargs, y_fp=y, dump_dtype=dump_dtype
                    )
                del ipt, y
            for p, s in _state_dict:
                p.data = s
            del orig_wgts, orig_ipts, _state_dict
        else:
            raise ValueError(f"Unknown objective {self.objective}")
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.debug(f"+ finished calculating the original outputs, ram usage: {psutil.virtual_memory().percent}")
        # endregion
        while not self.is_done():
            self.ask()
            e: list[torch.Tensor] = []
            # region Step 2: Calculate the errors
            if self.objective == SearchBasedCalibObjective.TensorError:
                assert isinstance(orig_wgts, (tuple, list))
                for w, (_, orig_w), w_view_shape in zip(wgts, orig_wgts, w_view_shapes, strict=True):
                    e_w = self._process_w_in_xw(w).sub_(orig_w)
                    if self.granularity == SearchBasedCalibGranularity.Group:
                        e_w = e_w.view(w_view_shape).abs_().pow_(self.config.degree)
                        e_w = e_w.sum(dim=tuple(range(1, len(w_view_shape), 2))).view(w_view_shape[::2])
                    elif self.granularity == SearchBasedCalibGranularity.ChannelGroup:
                        e_w = e_w.view(*w_view_shape[:4], -1).abs_().pow_(self.config.degree)
                        e_w = e_w.sum(dim=(0, 1, 3, 4)).view(w_view_shape[2])
                    elif self.granularity == SearchBasedCalibGranularity.Layer:
                        e_w = e_w.abs_().pow_(self.config.degree).sum().view(-1)
                    else:
                        raise ValueError(f"Unknown granularity {self.granularity}")
                    e.append(e_w)
            elif self.objective == SearchBasedCalibObjective.ProductsError:
                e = [None] * len(wgts)
                for j, w in enumerate(wgts):
                    w = _reshape_w(self._process_w_in_xw(w), view_shape=w_view_shapes[j])
                    for s, ipt in enumerate(ipts):
                        for i, x in enumerate(ipt.data):
                            x = x.to(device=w.device, non_blocking=True)
                            if not self.needs_to_pre_reshape_x_for_wgts:
                                x = self._process_x_in_xw(x, channels_dim=ipt.channels_dim)
                                x = _reshape_x(x, view_shape=w_view_shapes[j], fn=ipt.reshape)
                            y = torch.matmul(x, w)
                            y = y.view(*y.shape[:-2], y.shape[-2] * y.shape[-1])
                            y = y.sub_(orig_opts[(i, s, j)].to(device=w.device, non_blocking=True))
                            if self.granularity == SearchBasedCalibGranularity.Group:
                                y = y.to(self.develop_dtype).pow_(self.config.degree).sum(dim=-1)
                            elif self.granularity == SearchBasedCalibGranularity.ChannelGroup:
                                y = y.view(y.shape[0], y.shape[1], -1)
                                y = y.to(self.develop_dtype).pow_(self.config.degree).sum(dim=(0, 2))
                            elif self.granularity == SearchBasedCalibGranularity.Layer:
                                y = y.to(self.develop_dtype).pow_(self.config.degree).sum().view(-1)
                            else:
                                raise ValueError(f"Unknown granularity {self.granularity}")
                            if e[j] is None:
                                e[j] = y
                            else:
                                e[j].add_(y)
            elif self.objective == SearchBasedCalibObjective.OutputsError:
                self._process_wgts_centric_mod(wgts=wgts, mods=mods, **kwargs)
                e = [None]
                dump_ctx = self._get_output_pair_dump_context()
                save_all = self._env_truthy("DEEPCOMPRESSOR_CALIB_OUTPUTPAIR_SAVE_ALL", "0")
                if dump_ctx is not None:
                    fp_dir, tmp_dir, best_dir, cand_dir, dump_dtype = dump_ctx
                    candidate_meta = {
                        "iter": int(self.iter),
                        "candidate_id": int(self.candidate_id),
                        "population_size": int(self.population_size),
                    }
                    # Per-timestep per-channel SSE/N for this candidate.
                    sse_by_t: dict[int, torch.Tensor] | dict[int, dict[int, torch.Tensor]] = {}
                    n_by_t: dict[int, float] | dict[int, dict[int, float]] = {}
                    ts_full = self._get_cache_timesteps_full(ipts)
                    gs_full = self._get_cache_guidances_full(ipts)
                    if ts_full is None or gs_full is None or int(gs_full.numel()) != int(ts_full.numel()):
                        gs_full = None
                    offset = 0
                for i in range(len(ipts.front().data)):
                    ipt = ipts.extract(i, eval_kwargs)
                    y_out = eval_module(*ipt.args, **ipt.kwargs)
                    y_out = y_out[0] if not isinstance(y_out, torch.Tensor) else y_out
                    assert isinstance(y_out, torch.Tensor), "eval_mod should return a tensor"
                    if dump_ctx is not None:
                        self._dump_output_pair_tmp_batch(
                            tmp_dir=tmp_dir, batch_idx=i, eval_kwargs=eval_kwargs, y_q=y_out, dump_dtype=dump_dtype
                        )
                        if save_all:
                            self._dump_output_pair_candidate_batch(
                                cand_dir=cand_dir,
                                iter_idx=int(self.iter),
                                candidate_id=int(self.candidate_id),
                                batch_idx=i,
                                eval_kwargs=eval_kwargs,
                                y_q=y_out,
                                dump_dtype=dump_dtype,
                            )
                        if ts_full is not None:
                            bs = int(y_out.shape[0])
                            t_batch = ts_full[offset : offset + bs].to(device=y_out.device)
                            g_batch = gs_full[offset : offset + bs].to(device=y_out.device) if gs_full is not None else None
                            offset += bs
                            y_fp = orig_opts[(i,)].to(device=y_out.device, non_blocking=True)
                            diff = y_out - y_fp
                            self._accumulate_sse_n_per_timestep_per_channel(
                                sse_by_t=sse_by_t, n_by_t=n_by_t, diff=diff, timesteps=t_batch, guidances=g_batch
                            )
                    y = (y_out - orig_opts[(i,)].to(device=y_out.device, non_blocking=True)).to(self.develop_dtype)
                    y = y.pow_(self.config.degree).sum().view(-1)
                    if e[0] is None:
                        e[0] = y
                    else:
                        e[0].add_(y)
                    del ipt, y, y_out
                self._recover_mod()
            else:
                raise ValueError(f"Unknown objective {self.objective}")
            # endregion
            self.tell(e)
            if self.objective == SearchBasedCalibObjective.OutputsError and dump_ctx is not None:
                if ts_full is not None:
                    sse_obj = sse_by_t
                    n_obj = n_by_t
                    mse_obj: dict[tp.Any, tp.Any] = {}
                    if sse_obj:
                        first_v = next(iter(sse_obj.values()))
                        if isinstance(first_v, dict):
                            for gg_i, sse_t in tp.cast(dict[int, dict[int, torch.Tensor]], sse_obj).items():
                                n_t = tp.cast(dict[int, dict[int, float]], n_obj).get(int(gg_i), {})
                                mse_obj[int(gg_i)] = {t: (sse_t[t] / (float(n_t.get(t, 0.0)) + 1e-12)) for t in sse_t}
                        else:
                            mse_obj = {
                                t: (tp.cast(dict[int, torch.Tensor], sse_obj)[t] / (float(tp.cast(dict[int, float], n_obj)[t]) + 1e-12))
                                for t in tp.cast(dict[int, torch.Tensor], sse_obj).keys()
                            }
                    stats = {
                        "context": os.environ.get("DEEPCOMPRESSOR_CALIB_CONTEXT", ""),
                        "instance": os.environ.get("DEEPCOMPRESSOR_CALIB_CONTEXT_INSTANCE", ""),
                        "iter": int(self.iter),
                        "candidate_id": int(self.candidate_id),
                        "SSE": sse_by_t,
                        "N": n_by_t,
                        "MSE": mse_obj,
                        "bucket_by_guidance": bool(gs_full is not None),
                    }
                    torch.save(stats, str(tmp_dir / "sse_n.pt"))
                if save_all:
                    self._dump_output_pair_candidate_meta(
                        cand_dir=cand_dir,
                        iter_idx=int(self.iter),
                        candidate_id=int(self.candidate_id),
                        eval_kwargs=eval_kwargs,
                        error=e,
                        candidate_meta=candidate_meta,
                    )
                self._finalize_output_pair_candidate(
                    tmp_dir=tmp_dir,
                    best_dir=best_dir,
                    fp_dir=fp_dir,
                    eval_kwargs=eval_kwargs,
                    error=e,
                    candidate_meta=candidate_meta,
                )
        return self.get_best()

    def _calibrate_ipts(  # noqa: C901
        self,
        wgts: list[torch.Tensor | nn.Parameter],
        ipts: TensorsCache,
        eval_module: nn.Module | None,
        mods: list[nn.Module] | None,
        orig_wgts: list[tuple[nn.Parameter, torch.Tensor]] | None,
        orig_ipts: TensorsCache | None,
        eval_kwargs: dict[str, tp.Any],
        **kwargs,
    ) -> tp.Any:
        if orig_ipts is None:
            orig_ipts = ipts
        assert ipts.num_tensors == orig_ipts.num_tensors
        assert all(
            x.shape == orig_x.shape
            for ipt, orig_ipt in zip(ipts, orig_ipts, strict=True)
            for x, orig_x in zip(ipt.data, orig_ipt.data, strict=True)
        )
        # region Step 1: Calculate the outputs
        if self.objective == SearchBasedCalibObjective.TensorError:
            assert all(x.shape == ipt.data[0].shape for ipt in ipts for x in ipt.data)
            orig_opts = None
            x_view_shapes = [
                infer_view_shape(
                    ipt.data[0].view(-1, *ipt.data[0].shape[ipt.channels_dim :]).shape,
                    self.x_quantizer.config.largest_group_shape,
                    skip_first_dim=True,
                )
                for ipt in ipts
            ]
            del orig_wgts
        elif self.objective == SearchBasedCalibObjective.ProductsError:
            assert all(ipt.channels_dim == 1 for ipt in ipts)
            assert all(ipt.channels_dim == 1 for ipt in orig_ipts)
            assert all(x.shape[1:] == ipts.front().data[0].shape[1:] for ipt in ipts for x in ipt.data)
            if orig_wgts is None:
                orig_wgts = [(None, w.detach().data) for w in wgts]
            assert len(orig_wgts) == len(wgts)
            if self.granularity != SearchBasedCalibGranularity.Layer:
                _reshape_x = self._reshape_x_for_ipts_centric_partial_products
                _reshape_w = self._reshape_w_for_ipts_centric_partial_products
            else:
                _reshape_x = self._reshape_x_for_full_products
                _reshape_w = self._reshape_w_for_full_products
            x_view_shapes = [
                infer_view_shape(ipt.data[0].shape, self.x_quantizer.config.largest_group_shape, skip_first_dim=True)
                for ipt in ipts
            ]
            orig_opts: dict[tuple[int, ...], torch.Tensor] = {}
            for j, (_, w) in enumerate(orig_wgts):
                w = _reshape_w(w, view_shape=x_view_shapes[0])
                for s, ipt in enumerate(orig_ipts):
                    for i, x in enumerate(ipt.data):
                        x = x.to(device=w.device, non_blocking=True)
                        x = _reshape_x(x, view_shape=x_view_shapes[s], fn=ipt.reshape)
                        y = torch.matmul(x, w)
                        y = y.view(*y.shape[:-2], y.shape[-2] * y.shape[-1])
                        orig_opts[(i, s, j)] = y.to(device=self.opts_device or y.device, non_blocking=True)
            if self.needs_to_pre_reshape_w_for_ipts:
                for j, w in enumerate(wgts):
                    wgts[j] = _reshape_w(w, view_shape=x_view_shapes[0])
            del orig_wgts, orig_ipts
        elif self.objective == SearchBasedCalibObjective.OutputsError:
            x_view_shapes, _state_dict = [], []
            dump_ctx = self._get_output_pair_dump_context()
            if dump_ctx is not None:
                fp_dir, tmp_dir, best_dir, cand_dir, dump_dtype = dump_ctx
            if orig_wgts is not None:
                _state_dict = [(p, p.data) for p, _ in orig_wgts]
                for p, w in orig_wgts:
                    p.data = w.to(device=p.data.device)
            orig_opts: dict[tuple[int, ...], torch.Tensor] = {}
            for i in range(len(orig_ipts.front().data)):
                ipt = orig_ipts.extract(i, eval_kwargs)
                y = eval_module(*ipt.args, **ipt.kwargs)
                y = y[0] if not isinstance(y, torch.Tensor) else y
                assert isinstance(y, torch.Tensor), "eval_mod should return a tensor"
                orig_opts[(i,)] = y.to(device=self.opts_device or y.device, non_blocking=True)
                if dump_ctx is not None:
                    self._dump_output_pair_fp_batch(
                        fp_dir=fp_dir, batch_idx=i, eval_kwargs=eval_kwargs, y_fp=y, dump_dtype=dump_dtype
                    )
                del ipt, y
            for p, s in _state_dict:
                p.data = s
            del orig_wgts, orig_ipts, _state_dict
        else:
            raise ValueError(f"Unknown objective {self.objective}")
        gc.collect()
        torch.cuda.empty_cache()
        # endregion
        while not self.is_done():
            self.ask()
            e: list[torch.Tensor] = []
            # region Step 2: Calculate the outputs errors
            if self.objective == SearchBasedCalibObjective.TensorError:
                e = [None] * len(ipts)
                for s, (ipt, x_view_shape) in enumerate(zip(ipts, x_view_shapes, strict=True)):
                    for x in ipt.data:
                        e_x = self._process_x_in_xw(x, channels_dim=ipt.channels_dim).sub_(x)
                        if self.granularity == SearchBasedCalibGranularity.Group:
                            e_x = e_x.view(x_view_shape).abs_().pow_(self.config.degree)
                            e_x = e_x.sum(dim=tuple(range(1, len(x_view_shape), 2)))
                        if self.granularity == SearchBasedCalibGranularity.ChannelGroup:
                            e_x = e_x.view(*x_view_shape[:4], -1).abs_().pow_(self.config.degree)
                            e_x = e_x.sum(dim=(0, 1, 3, 4)).view(x_view_shape[2])
                        elif self.granularity == SearchBasedCalibGranularity.Layer:
                            e_x = e_x.abs_().pow_(self.config.degree).sum().view(-1)
                        else:
                            raise ValueError(f"Unknown granularity {self.granularity}")
                        if e[s] is None:
                            e[s] = e_x
                        else:
                            e[s].add_(e_x)
            elif self.objective == SearchBasedCalibObjective.ProductsError:
                e = [None] * len(ipts)
                for j, w in enumerate(wgts):
                    if not self.needs_to_pre_reshape_w_for_ipts:
                        w = self._process_w_in_xw(w)
                        w = _reshape_w(w, view_shape=x_view_shapes[0])
                    for s, ipt in enumerate(ipts):
                        for i, x in enumerate(ipt.data):
                            x = x.to(device=w.device, non_blocking=True)
                            x = self._process_x_in_xw(x, channels_dim=ipt.channels_dim)
                            x = _reshape_x(x, view_shape=x_view_shapes[s], fn=ipt.reshape)
                            y = torch.matmul(x, w)
                            y = y.view(*y.shape[:-2], y.shape[-2] * y.shape[-1])
                            y = y.sub_(orig_opts[(i, s, j)].to(device=w.device, non_blocking=True))
                            if self.granularity == SearchBasedCalibGranularity.Group:
                                y = y.to(self.develop_dtype).pow_(self.config.degree).sum(dim=-1)
                            elif self.granularity == SearchBasedCalibGranularity.ChannelGroup:
                                y = y.view(y.shape[0], y.shape[1], -1)
                                y = y.to(self.develop_dtype).pow_(self.config.degree).sum(dim=(0, 2))
                            elif self.granularity == SearchBasedCalibGranularity.Layer:
                                y = y.to(self.develop_dtype).pow_(self.config.degree).sum().view(-1)
                            else:
                                raise ValueError(f"Unknown granularity {self.granularity}")
                            if e[s] is None:
                                e[s] = y
                            else:
                                e[s].add_(y)
            elif self.objective == SearchBasedCalibObjective.OutputsError:
                self._process_ipts_centric_mod(wgts=wgts, mods=mods, **kwargs)
                e = [None]
                dump_ctx = self._get_output_pair_dump_context()
                save_all = self._env_truthy("DEEPCOMPRESSOR_CALIB_OUTPUTPAIR_SAVE_ALL", "0")
                if dump_ctx is not None:
                    fp_dir, tmp_dir, best_dir, cand_dir, dump_dtype = dump_ctx
                    candidate_meta = {
                        "iter": int(self.iter),
                        "candidate_id": int(self.candidate_id),
                        "population_size": int(self.population_size),
                    }
                    sse_by_t: dict[int, torch.Tensor] | dict[int, dict[int, torch.Tensor]] = {}
                    n_by_t: dict[int, float] | dict[int, dict[int, float]] = {}
                    ts_full = self._get_cache_timesteps_full(ipts)
                    gs_full = self._get_cache_guidances_full(ipts)
                    if ts_full is None or gs_full is None or int(gs_full.numel()) != int(ts_full.numel()):
                        gs_full = None
                    offset = 0
                for i in range(len(ipts.front().data)):
                    ipt = ipts.extract(i, eval_kwargs)
                    y_out = eval_module(*ipt.args, **ipt.kwargs)
                    y_out = y_out[0] if not isinstance(y_out, torch.Tensor) else y_out
                    assert isinstance(y_out, torch.Tensor), "eval_mod should return a tensor"
                    if dump_ctx is not None:
                        self._dump_output_pair_tmp_batch(
                            tmp_dir=tmp_dir, batch_idx=i, eval_kwargs=eval_kwargs, y_q=y_out, dump_dtype=dump_dtype
                        )
                        if save_all:
                            self._dump_output_pair_candidate_batch(
                                cand_dir=cand_dir,
                                iter_idx=int(self.iter),
                                candidate_id=int(self.candidate_id),
                                batch_idx=i,
                                eval_kwargs=eval_kwargs,
                                y_q=y_out,
                                dump_dtype=dump_dtype,
                            )
                        if ts_full is not None:
                            bs = int(y_out.shape[0])
                            t_batch = ts_full[offset : offset + bs].to(device=y_out.device)
                            g_batch = gs_full[offset : offset + bs].to(device=y_out.device) if gs_full is not None else None
                            offset += bs
                            y_fp = orig_opts[(i,)].to(device=y_out.device, non_blocking=True)
                            diff = y_out - y_fp
                            self._accumulate_sse_n_per_timestep_per_channel(
                                sse_by_t=sse_by_t, n_by_t=n_by_t, diff=diff, timesteps=t_batch, guidances=g_batch
                            )
                    y = (y_out - orig_opts[(i,)].to(device=y_out.device, non_blocking=True)).to(self.develop_dtype)
                    y = y.pow_(self.config.degree).sum().view(-1)
                    if e[0] is None:
                        e[0] = y
                    else:
                        e[0].add_(y)
                    del ipt, y, y_out
                self._recover_mod()
            else:
                raise ValueError(f"Unknown objective {self.objective}")
            # endregion
            self.tell(e)
            if self.objective == SearchBasedCalibObjective.OutputsError and dump_ctx is not None:
                if ts_full is not None:
                    sse_obj = sse_by_t
                    n_obj = n_by_t
                    mse_obj: dict[tp.Any, tp.Any] = {}
                    if sse_obj:
                        first_v = next(iter(sse_obj.values()))
                        if isinstance(first_v, dict):
                            for gg_i, sse_t in tp.cast(dict[int, dict[int, torch.Tensor]], sse_obj).items():
                                n_t = tp.cast(dict[int, dict[int, float]], n_obj).get(int(gg_i), {})
                                mse_obj[int(gg_i)] = {t: (sse_t[t] / (float(n_t.get(t, 0.0)) + 1e-12)) for t in sse_t}
                        else:
                            mse_obj = {
                                t: (tp.cast(dict[int, torch.Tensor], sse_obj)[t] / (float(tp.cast(dict[int, float], n_obj)[t]) + 1e-12))
                                for t in tp.cast(dict[int, torch.Tensor], sse_obj).keys()
                            }
                    stats = {
                        "context": os.environ.get("DEEPCOMPRESSOR_CALIB_CONTEXT", ""),
                        "instance": os.environ.get("DEEPCOMPRESSOR_CALIB_CONTEXT_INSTANCE", ""),
                        "iter": int(self.iter),
                        "candidate_id": int(self.candidate_id),
                        "SSE": sse_by_t,
                        "N": n_by_t,
                        "MSE": mse_obj,
                        "bucket_by_guidance": bool(gs_full is not None),
                    }
                    torch.save(stats, str(tmp_dir / "sse_n.pt"))
                if save_all:
                    self._dump_output_pair_candidate_meta(
                        cand_dir=cand_dir,
                        iter_idx=int(self.iter),
                        candidate_id=int(self.candidate_id),
                        eval_kwargs=eval_kwargs,
                        error=e,
                        candidate_meta=candidate_meta,
                    )
                self._finalize_output_pair_candidate(
                    tmp_dir=tmp_dir,
                    best_dir=best_dir,
                    fp_dir=fp_dir,
                    eval_kwargs=eval_kwargs,
                    error=e,
                    candidate_meta=candidate_meta,
                )
        return self.get_best()

    def _calibrate_opts(  # noqa: C901
        self,
        x_wgts: list[torch.Tensor | nn.Parameter],
        y_wgts: list[torch.Tensor | nn.Parameter],
        eval_inputs: TensorsCache | None,
        eval_module: nn.Module | None,
        x_mods: list[nn.Module] | None,
        y_mods: list[nn.Module] | None,
        orig_x_wgts: list[tuple[nn.Parameter, torch.Tensor]] | None,
        orig_y_wgts: list[tuple[nn.Parameter, torch.Tensor]] | None,
        orig_eval_inputs: TensorsCache | None,
        eval_kwargs: dict[str, tp.Any],
        **kwargs,
    ) -> tp.Any:
        # region Step 1: Calculate the outputs
        if self.objective == SearchBasedCalibObjective.OutputsError:
            assert eval_inputs is not None, "eval_inputs should not be None when objective is OutputsError"
            dump_ctx = self._get_output_pair_dump_context()
            if dump_ctx is not None:
                fp_dir, tmp_dir, best_dir, cand_dir, dump_dtype = dump_ctx
            if orig_eval_inputs is None:
                orig_eval_inputs = eval_inputs
            assert eval_inputs.num_tensors == orig_eval_inputs.num_tensors
            assert all(
                x.shape == orig_x.shape
                for key, ipt in eval_inputs.items()
                for x, orig_x in zip(ipt.data, orig_eval_inputs[key].data, strict=True)
            )
            _x_state_dict, _y_state_dict = [], []
            if orig_x_wgts is not None:
                _x_state_dict = [(p, p.data) for p, _ in orig_x_wgts]
                for p, w in orig_x_wgts:
                    p.data = w.to(device=p.data.device)
            if orig_y_wgts is not None:
                _y_state_dict = [(p, p.data) for p, _ in orig_y_wgts]
                for p, w in orig_y_wgts:
                    p.data = w.to(device=p.data.device)
            orig_opts: dict[tuple[int, ...], torch.Tensor] = {}
            for i in range(len(orig_eval_inputs.front().data)):
                ipt = orig_eval_inputs.extract(i, eval_kwargs)
                y = eval_module(*ipt.args, **ipt.kwargs)
                y = y[0] if not isinstance(y, torch.Tensor) else y
                assert isinstance(y, torch.Tensor), "eval_mod should return a tensor"
                orig_opts[(i,)] = y.to(device=self.opts_device or y.device, non_blocking=True)
                if dump_ctx is not None:
                    self._dump_output_pair_fp_batch(
                        fp_dir=fp_dir, batch_idx=i, eval_kwargs=eval_kwargs, y_fp=y, dump_dtype=dump_dtype
                    )
                del ipt, y
            for p, s in _x_state_dict:
                p.data = s
            for p, s in _y_state_dict:
                p.data = s
            del orig_x_wgts, orig_y_wgts, orig_eval_inputs, _x_state_dict, _y_state_dict
        else:
            raise ValueError(f"Unknown objective {self.objective}")
        gc.collect()
        torch.cuda.empty_cache()
        # endregion
        while not self.is_done():
            self.ask()
            e: list[torch.Tensor] = []
            # region Step 2: Calculate the outputs errors
            if self.objective == SearchBasedCalibObjective.OutputsError:
                self._process_opts_centric_mod(
                    x_wgts=x_wgts,
                    y_wgts=y_wgts,
                    x_mods=x_mods,
                    y_mods=y_mods,
                    **kwargs,
                )
                e = [None]
                dump_ctx = self._get_output_pair_dump_context()
                save_all = self._env_truthy("DEEPCOMPRESSOR_CALIB_OUTPUTPAIR_SAVE_ALL", "0")
                if dump_ctx is not None:
                    fp_dir, tmp_dir, best_dir, cand_dir, dump_dtype = dump_ctx
                    candidate_meta = {
                        "iter": int(self.iter),
                        "candidate_id": int(self.candidate_id),
                        "population_size": int(self.population_size),
                    }
                    sse_by_t: dict[int, torch.Tensor] | dict[int, dict[int, torch.Tensor]] = {}
                    n_by_t: dict[int, float] | dict[int, dict[int, float]] = {}
                    ts_full = self._get_cache_timesteps_full(eval_inputs)
                    gs_full = self._get_cache_guidances_full(eval_inputs)
                    if ts_full is None or gs_full is None or int(gs_full.numel()) != int(ts_full.numel()):
                        gs_full = None
                    offset = 0
                for i in range(len(eval_inputs.front().data)):
                    ipt = eval_inputs.extract(i, eval_kwargs)
                    y_out = eval_module(*ipt.args, **ipt.kwargs)
                    y_out = y_out[0] if not isinstance(y_out, torch.Tensor) else y_out
                    assert isinstance(y_out, torch.Tensor), "eval_mod should return a tensor"
                    if dump_ctx is not None:
                        self._dump_output_pair_tmp_batch(
                            tmp_dir=tmp_dir, batch_idx=i, eval_kwargs=eval_kwargs, y_q=y_out, dump_dtype=dump_dtype
                        )
                        if save_all:
                            self._dump_output_pair_candidate_batch(
                                cand_dir=cand_dir,
                                iter_idx=int(self.iter),
                                candidate_id=int(self.candidate_id),
                                batch_idx=i,
                                eval_kwargs=eval_kwargs,
                                y_q=y_out,
                                dump_dtype=dump_dtype,
                            )
                        if ts_full is not None:
                            bs = int(y_out.shape[0])
                            t_batch = ts_full[offset : offset + bs].to(device=y_out.device)
                            g_batch = gs_full[offset : offset + bs].to(device=y_out.device) if gs_full is not None else None
                            offset += bs
                            y_fp = orig_opts[(i,)].to(device=y_out.device, non_blocking=True)
                            diff = y_out - y_fp
                            self._accumulate_sse_n_per_timestep_per_channel(
                                sse_by_t=sse_by_t, n_by_t=n_by_t, diff=diff, timesteps=t_batch, guidances=g_batch
                            )
                    y = (y_out - orig_opts[(i,)].to(device=y_out.device, non_blocking=True)).to(self.develop_dtype)
                    y = y.pow_(self.config.degree).sum().view(-1)
                    if e[0] is None:
                        e[0] = y
                    else:
                        e[0].add_(y)
                    del ipt, y, y_out
                self._recover_mod()
            else:
                raise ValueError(f"Unknown objective {self.objective}")
            # endregion
            self.tell(e)
            if self.objective == SearchBasedCalibObjective.OutputsError and dump_ctx is not None:
                if ts_full is not None:
                    sse_obj = sse_by_t
                    n_obj = n_by_t
                    mse_obj: dict[tp.Any, tp.Any] = {}
                    if sse_obj:
                        first_v = next(iter(sse_obj.values()))
                        if isinstance(first_v, dict):
                            for gg_i, sse_t in tp.cast(dict[int, dict[int, torch.Tensor]], sse_obj).items():
                                n_t = tp.cast(dict[int, dict[int, float]], n_obj).get(int(gg_i), {})
                                mse_obj[int(gg_i)] = {t: (sse_t[t] / (float(n_t.get(t, 0.0)) + 1e-12)) for t in sse_t}
                        else:
                            mse_obj = {
                                t: (tp.cast(dict[int, torch.Tensor], sse_obj)[t] / (float(tp.cast(dict[int, float], n_obj)[t]) + 1e-12))
                                for t in tp.cast(dict[int, torch.Tensor], sse_obj).keys()
                            }
                    stats = {
                        "context": os.environ.get("DEEPCOMPRESSOR_CALIB_CONTEXT", ""),
                        "instance": os.environ.get("DEEPCOMPRESSOR_CALIB_CONTEXT_INSTANCE", ""),
                        "iter": int(self.iter),
                        "candidate_id": int(self.candidate_id),
                        "SSE": sse_by_t,
                        "N": n_by_t,
                        "MSE": mse_obj,
                        "bucket_by_guidance": bool(gs_full is not None),
                    }
                    torch.save(stats, str(tmp_dir / "sse_n.pt"))
                if save_all:
                    self._dump_output_pair_candidate_meta(
                        cand_dir=cand_dir,
                        iter_idx=int(self.iter),
                        candidate_id=int(self.candidate_id),
                        eval_kwargs=eval_kwargs,
                        error=e,
                        candidate_meta=candidate_meta,
                    )
                self._finalize_output_pair_candidate(
                    tmp_dir=tmp_dir,
                    best_dir=best_dir,
                    fp_dir=fp_dir,
                    eval_kwargs=eval_kwargs,
                    error=e,
                    candidate_meta=candidate_meta,
                )
        return self.get_best()
