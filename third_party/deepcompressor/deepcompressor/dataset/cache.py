# -*- coding: utf-8 -*-
"""Caching calibration dataset."""

import functools
import gc
import json
import os
import re
import typing as tp
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import MISSING
from pathlib import Path

import psutil
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.hooks
from tqdm import tqdm

from ..data.cache import IOTensorsCache, ModuleForwardInput, TensorCache
from ..data.utils.reshape import ConvInputReshapeFn, ConvOutputReshapedFn, LinearReshapeFn
from ..utils import tools
from ..utils.common import tree_copy_with_ref, tree_map
from ..utils.hooks import EarlyStopException, EarlyStopHook, Hook
from .action import CacheAction

__all__ = ["BaseCalibCacheLoader"]


class BaseCalibCacheLoader(ABC):
    """Base class for caching calibration dataset."""

    dataset: torch.utils.data.Dataset
    batch_size: int

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int):
        """Initialize the dataset.

        Args:
            dataset (`torch.utils.data.Dataset`):
                Calibration dataset.
            batch_size (`int`):
                Batch size.
        """
        self.dataset = dataset
        self.batch_size = batch_size

    @property
    def num_samples(self) -> int:
        """Number of samples in the dataset."""
        return len(self.dataset)

    @abstractmethod
    def iter_samples(self, *args, **kwargs) -> tp.Generator[ModuleForwardInput, None, None]:
        """Iterate over model input samples."""
        ...

    def _init_cache(self, name: str, module: nn.Module) -> IOTensorsCache:
        """Initialize activation cache.

        Args:
            name (`str`):
                Module name.
            module (`nn.Module`):
                Module.

        Returns:
            `IOTensorsCache`:
                Tensors cache for inputs and outputs.
        """
        if isinstance(module, (nn.Linear,)):
            return IOTensorsCache(
                inputs=TensorCache(channels_dim=-1, reshape=LinearReshapeFn()),
                outputs=TensorCache(channels_dim=-1, reshape=LinearReshapeFn()),
            )
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            assert module.padding_mode == "zeros", f"Padding mode {module.padding_mode} is not supported"
            if isinstance(module.padding, str):
                if module.padding == "valid":
                    padding = (0,) * len(module.kernel_size)
                elif module.padding == "same":
                    padding = tuple(reversed(tuple(t for t in module._reversed_padding_repeated_twice[::2])))
            else:
                padding = tuple(module.padding)
            return IOTensorsCache(
                inputs=TensorCache(
                    channels_dim=1,
                    reshape=ConvInputReshapeFn(module.kernel_size, padding, module.stride, module.dilation),
                ),
                outputs=TensorCache(channels_dim=1, reshape=ConvOutputReshapedFn()),
            )
        else:
            raise NotImplementedError(f"Module {module.__class__.__name__} is not supported")

    def _convert_layer_inputs(
        self, m: nn.Module, args: tuple[tp.Any, ...], kwargs: dict[str, tp.Any], save_all: bool = False
    ) -> ModuleForwardInput:
        """Convert layer inputs to module forward input.

        Args:
            m (`nn.Module`):
                Layer.
            args (`tuple[Any, ...]`):
                Layer input arguments.
            kwargs (`dict[str, Any]`):
                Layer input keyword arguments.
            save_all (`bool`, *optional*, defaults to `False`):
                Whether to save all inputs.

        Returns:
            `ModuleForwardInput`:
                Module forward input.
        """
        x = args[0].detach().cpu() if save_all else MISSING
        return ModuleForwardInput(args=[x, *args[1:]], kwargs=kwargs)

    def _convert_layer_outputs(self, m: nn.Module, outputs: tp.Any) -> dict[str | int, tp.Any]:
        """Convert layer outputs to dictionary for updating the next layer inputs.

        Args:
            m (`nn.Module`):
                Layer.
            outputs (`Any`):
                Layer outputs.

        Returns:
            `dict[str | int, Any]`:
                Dictionary for updating the next layer inputs.
        """
        if not isinstance(outputs, torch.Tensor):
            outputs = outputs[0]
        assert isinstance(outputs, torch.Tensor), f"Invalid outputs type: {type(outputs)}"
        return {0: outputs.detach().cpu()}

    def _layer_forward_pre_hook(
        self,
        m: nn.Module,
        args: tuple[torch.Tensor, ...],
        kwargs: dict[str, tp.Any],
        cache: list[ModuleForwardInput],
        save_all: bool = False,
    ) -> None:
        inputs = self._convert_layer_inputs(m, args, kwargs, save_all=save_all)
        if len(cache) > 0:
            inputs.args = tree_copy_with_ref(inputs.args, cache[0].args)
            inputs.kwargs = tree_copy_with_ref(inputs.kwargs, cache[0].kwargs)
        else:
            inputs.args = tree_map(lambda x: x, inputs.args)
            inputs.kwargs = tree_map(lambda x: x, inputs.kwargs)
        cache.append(inputs)

    @torch.inference_mode()
    def _iter_layer_activations(  # noqa: C901
        self,
        model: nn.Module,
        *args,
        action: CacheAction,
        layers: tp.Sequence[nn.Module] | None = None,
        needs_inputs_fn: tp.Callable[[str, nn.Module], bool] | bool | None = True,
        needs_outputs_fn: tp.Callable[[str, nn.Module], bool] | bool | None = None,
        recomputes: list[bool] | None = None,
        use_prev_layer_outputs: list[bool] | None = None,
        early_stop_module: nn.Module | None = None,
        clear_after_yield: bool = True,
        **kwargs,
    ) -> tp.Generator[
        tuple[
            str,
            tuple[
                nn.Module,
                dict[str, IOTensorsCache],
                list[ModuleForwardInput],
            ],
        ],
        None,
        None,
    ]:
        """Iterate over model activations in layers.

        Args:
            model (`nn.Module`):
                Model.
            action (`CacheAction`):
                Action for caching activations.
            layers (`Sequence[nn.Module]` or `None`, *optional*, defaults to `None`):
                Layers to cache activations. If `None`, cache all layers.
            needs_inputs_fn (`Callable[[str, nn.Module], bool]` or `bool` or `None`, *optional*, defaults to `True`):
                Function for determining whether to cache inputs for a module given its name and itself.
            needs_outputs_fn (`Callable[[str, nn.Module], bool]` or `bool` or `None`, *optional*, defaults to `None`):
                Function for determining whether to cache outputs for a module given its name and itself.
            recomputes (`list[bool]` or `bool` or `None`, *optional*, defaults to `None`):
                Whether to recompute the activations for each layer.
            use_prev_layer_outputs (`list[bool]` or `bool` or `None`, *optional*, defaults to `None`):
                Whether to use the previous layer outputs as inputs for the current layer.
            early_stop_module (`nn.Module` or `None`, *optional*, defaults to `None`):
                Module for early stopping.
            clear_after_yield (`bool`, *optional*, defaults to `True`):
                Whether to clear the cache after yielding the activations.
            *args: Arguments for ``iter_samples``.
            **kwargs: Keyword arguments for ``iter_samples``.

        Yields:
            Generator[
                tuple[str, tuple[nn.Module, dict[str, IOTensorsCache], list[ModuleForwardInput]]],
                None,
                None
            ]:
                Generator of tuple of
                    - layer name
                    - a tuple of
                        - layer itself
                        - inputs and outputs cache of each module in the layer
                        - layer input arguments
        """
        if needs_outputs_fn is None:
            needs_outputs_fn = lambda name, module: False  # noqa: E731
        elif isinstance(needs_outputs_fn, bool):
            if needs_outputs_fn:
                needs_outputs_fn = lambda name, module: True  # noqa: E731
            else:
                needs_outputs_fn = lambda name, module: False  # noqa: E731
        if needs_inputs_fn is None:
            needs_inputs_fn = lambda name, module: False  # noqa: E731
        elif isinstance(needs_inputs_fn, bool):
            if needs_inputs_fn:
                needs_inputs_fn = lambda name, module: True  # noqa: E731
            else:
                needs_inputs_fn = lambda name, module: False  # noqa: E731

        dump_force_outputs = os.environ.get("DEEPCOMPRESSOR_CALIB_DUMP_FORCE_OUTPUTS", "").strip().lower() not in {
            "",
            "0",
            "false",
            "no",
        }
        if layers is None:
            recomputes = [True]
            use_prev_layer_outputs = [False]
        else:
            assert isinstance(layers, (nn.Sequential, nn.ModuleList, list, tuple))
            if recomputes is None:
                recomputes = [False] * len(layers)
            elif isinstance(recomputes, bool):
                recomputes = [recomputes] * len(layers)
            if use_prev_layer_outputs is None:
                use_prev_layer_outputs = [True] * len(layers)
            elif isinstance(use_prev_layer_outputs, bool):
                use_prev_layer_outputs = [use_prev_layer_outputs] * len(layers)
            use_prev_layer_outputs[0] = False
            assert len(recomputes) == len(use_prev_layer_outputs) == len(layers)
        cache: dict[str, dict[str, IOTensorsCache]] = {}
        module_names: dict[str, list[str]] = {"": []}
        named_layers: OrderedDict[str, nn.Module] = {"": model}
        # region we first collect infomations for yield modules
        forward_cache: dict[str, list[ModuleForwardInput]] = {}
        info_hooks: list[Hook] = []
        forward_hooks: list[torch.utils.hooks.RemovableHandle] = []
        hook_args: dict[str, list[tuple[str, nn.Module, bool, bool]]] = {}
        layer_name = ""
        for module_name, module in model.named_modules():
            if layers is not None and module_name and module in layers:
                layer_name = module_name
                assert layer_name not in module_names
                named_layers[layer_name] = module
                module_names[layer_name] = []
                forward_cache[layer_name] = []
            if layers is None or (layer_name and module_name.startswith(layer_name)):
                # we only cache modules in the layer
                needs_inputs = needs_inputs_fn(module_name, module)
                needs_outputs = needs_outputs_fn(module_name, module)
                if dump_force_outputs and needs_inputs:
                    needs_outputs = True
                if needs_inputs or needs_outputs:
                    module_names[layer_name].append(module_name)
                    cache.setdefault(layer_name, {})[module_name] = self._init_cache(module_name, module)
                    hook_args.setdefault(layer_name, []).append((module_name, module, needs_inputs, needs_outputs))
                    info_hooks.extend(
                        action.register(
                            name=module_name,
                            module=module,
                            cache=cache[layer_name][module_name],
                            info_mode=True,
                            needs_inputs=needs_inputs,
                            needs_outputs=needs_outputs,
                        )
                    )
        if len(cache) == 0:
            return
        if layers is not None:
            module_names.pop("")
            named_layers.pop("")
            assert layer_name, "No layer in the given layers is found in the model"
            assert "" not in cache, "The model should not have empty layer name"
            ordered_named_layers: OrderedDict[str, nn.Module] = OrderedDict()
            for layer in layers:
                for name, module in named_layers.items():
                    if module is layer:
                        ordered_named_layers[name] = module
                        break
            assert len(ordered_named_layers) == len(named_layers)
            assert len(ordered_named_layers) == len(layers)
            named_layers = ordered_named_layers
            del ordered_named_layers
            for layer_idx, (layer_name, layer) in enumerate(named_layers.items()):
                forward_hooks.append(
                    layer.register_forward_pre_hook(
                        functools.partial(
                            self._layer_forward_pre_hook,
                            cache=forward_cache[layer_name],
                            save_all=not recomputes[layer_idx] and not use_prev_layer_outputs[layer_idx],
                        ),
                        with_kwargs=True,
                    )
                )
        else:
            assert len(named_layers) == 1 and "" in named_layers
            assert len(module_names) == 1 and "" in module_names
            assert len(cache) == 1 and "" in cache
        # endregion
        with tools.logging.redirect_tqdm():
            # region we then collect cache information by running the model with all samples
            if early_stop_module is not None:
                forward_hooks.append(early_stop_module.register_forward_hook(EarlyStopHook()))
            with torch.inference_mode():
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dump_root = os.environ.get("DEEPCOMPRESSOR_CALIB_DUMP_DIR", "").strip()
                save_timesteps = os.environ.get("DEEPCOMPRESSOR_CALIB_TIMESTEPS", "").strip().lower() not in {
                    "",
                    "0",
                    "false",
                    "no",
                }
                if not save_timesteps:
                    save_timesteps = dump_root and os.environ.get(
                        "DEEPCOMPRESSOR_CALIB_DUMP_SAVE_TIMESTEPS", "1"
                    ).strip().lower() not in {"", "0", "false", "no"}
                save_guidances = os.environ.get("DEEPCOMPRESSOR_CALIB_GUIDANCES", "").strip().lower() not in {
                    "",
                    "0",
                    "false",
                    "no",
                }
                if not save_guidances:
                    # If output-pair dumping is enabled, guidance bucketing is generally useful and cheap.
                    save_guidances = (
                        os.environ.get("DEEPCOMPRESSOR_CALIB_OUTPUTPAIR_DUMP_DIR", "").strip() != ""
                        and os.environ.get("DEEPCOMPRESSOR_CALIB_OUTPUTPAIR_SAVE_GUIDANCES", "1").strip().lower()
                        not in {"", "0", "false", "no"}
                    )
                calib_timesteps_batches: list[torch.Tensor] = []
                calib_guidances_batches: list[torch.Tensor] = []
                tbar = tqdm(
                    desc="collecting acts info",
                    leave=False,
                    total=self.num_samples,
                    unit="samples",
                    dynamic_ncols=True,
                )
                num_samples = 0
                for sample in self.iter_samples(*args, **kwargs):
                    num_samples += self.batch_size
                    sample = sample.to(device=device)
                    if save_timesteps:
                        ts = sample.kwargs.get("timestep", None)
                        if torch.is_tensor(ts):
                            calib_timesteps_batches.append(ts.detach().cpu().view(-1))
                        elif isinstance(ts, (int, float)):
                            # Some diffusion loaders pass scalar Python timesteps; normalize to a 1D tensor so
                            # downstream code can bucket per-sample errors without re-running the model.
                            # Best-effort batch size inference.
                            bs = None
                            for a in getattr(sample, "args", []) or []:
                                if torch.is_tensor(a) and a.ndim >= 1:
                                    bs = int(a.shape[0])
                                    break
                            if bs is None:
                                for v in (getattr(sample, "kwargs", {}) or {}).values():
                                    if torch.is_tensor(v) and v.ndim >= 1:
                                        bs = int(v.shape[0])
                                        break
                            if bs is None:
                                bs = int(self.batch_size)
                            calib_timesteps_batches.append(
                                torch.full((bs,), int(ts), dtype=torch.long, device="cpu")
                            )
                    if save_guidances:
                        g = getattr(sample, "guidance", None)
                        if torch.is_tensor(g):
                            calib_guidances_batches.append(g.detach().cpu().view(-1).to(dtype=torch.long))
                        elif isinstance(g, (int, float)):
                            bs = None
                            for a in getattr(sample, "args", []) or []:
                                if torch.is_tensor(a) and a.ndim >= 1:
                                    bs = int(a.shape[0])
                                    break
                            if bs is None:
                                for v in (getattr(sample, "kwargs", {}) or {}).values():
                                    if torch.is_tensor(v) and v.ndim >= 1:
                                        bs = int(v.shape[0])
                                        break
                            if bs is None:
                                bs = int(self.batch_size)
                            calib_guidances_batches.append(
                                torch.full((bs,), int(g), dtype=torch.long, device="cpu")
                            )
                    try:
                        model(*sample.args, **sample.kwargs)
                    except EarlyStopException:
                        pass
                    tbar.update(self.batch_size)
                    tbar.set_postfix({"ram usage": psutil.virtual_memory().percent})
                    if psutil.virtual_memory().percent > 90:
                        raise RuntimeError("memory usage > 90%%, aborting")
                del dump_root
                for layer_cache in cache.values():
                    for module_cache in layer_cache.values():
                        module_cache.set_num_samples(num_samples)
            for hook in forward_hooks:
                hook.remove()
            for hook in info_hooks:
                hook.remove()
            del info_hooks, forward_hooks
            # endregion
            for layer_idx, (layer_name, layer) in enumerate(named_layers.items()):
                # region we first register hooks for caching activations
                layer_hooks: list[Hook] = []
                for module_name, module, needs_inputs, needs_outputs in hook_args[layer_name]:
                    layer_hooks.extend(
                        action.register(
                            name=module_name,
                            module=module,
                            cache=cache[layer_name][module_name],
                            info_mode=False,
                            needs_inputs=needs_inputs,
                            needs_outputs=needs_outputs,
                        )
                    )
                hook_args.pop(layer_name)
                # endregion
                if recomputes[layer_idx]:
                    if layers is None:
                        if early_stop_module is not None:
                            layer_hooks.append(EarlyStopHook().register(early_stop_module))
                    else:
                        layer_hooks.append(EarlyStopHook().register(layer))
                    tbar = tqdm(
                        desc=f"collecting acts in {layer_name}",
                        leave=False,
                        total=self.num_samples,
                        unit="samples",
                        dynamic_ncols=True,
                    )
                    for sample in self.iter_samples(*args, **kwargs):
                        sample = sample.to(device=device)
                        try:
                            model(*sample.args, **sample.kwargs)
                        except EarlyStopException:
                            pass
                        tbar.update(self.batch_size)
                        tbar.set_postfix({"ram usage": psutil.virtual_memory().percent})
                        if psutil.virtual_memory().percent > 90:
                            raise RuntimeError("memory usage > 90%%, aborting")
                        gc.collect()
                else:
                    # region we then forward the layer to collect activations
                    device = next(layer.parameters()).device
                    layer_outputs: list[tp.Any] = []
                    tbar = tqdm(
                        forward_cache[layer_name],
                        desc=f"collecting acts in {layer_name}",
                        leave=False,
                        unit="batches",
                        dynamic_ncols=True,
                    )
                    if not use_prev_layer_outputs[layer_idx]:
                        prev_layer_outputs: list[dict[str | int, tp.Any]] = [None] * len(tbar)
                    for i, inputs in enumerate(tbar):
                        inputs = inputs.update(prev_layer_outputs[i]).to(device=device)
                        outputs = layer(*inputs.args, **inputs.kwargs)
                        layer_outputs.append(self._convert_layer_outputs(layer, outputs))
                        tbar.set_postfix({"ram usage": psutil.virtual_memory().percent})
                        if psutil.virtual_memory().percent > 90:
                            raise RuntimeError("memory usage > 90%%, aborting")
                    prev_layer_outputs = layer_outputs
                    del inputs, outputs, layer_outputs
                    if (layer_idx == len(named_layers) - 1) or not use_prev_layer_outputs[layer_idx + 1]:
                        del prev_layer_outputs
                    # endregion
                for hook in layer_hooks:
                    hook.remove()
                del layer_hooks
                layer_inputs = forward_cache.pop(layer_name, [])
                if not recomputes[layer_idx] and not use_prev_layer_outputs[layer_idx]:
                    layer_inputs = [
                        self._convert_layer_inputs(layer, inputs.args, inputs.kwargs) for inputs in layer_inputs
                    ]
                gc.collect()
                torch.cuda.empty_cache()

                # Attach calibration timesteps (if available) to every cached TensorsCache so downstream
                # calibration code can bucket errors by timestep without re-running the model.
                if save_timesteps and calib_timesteps_batches:
                    for io_cache in cache[layer_name].values():
                        if io_cache.inputs is not None:
                            setattr(io_cache.inputs, "calib_timesteps_batches", calib_timesteps_batches)
                        if io_cache.outputs is not None:
                            setattr(io_cache.outputs, "calib_timesteps_batches", calib_timesteps_batches)
                if save_guidances and calib_guidances_batches:
                    for io_cache in cache[layer_name].values():
                        if io_cache.inputs is not None:
                            setattr(io_cache.inputs, "calib_guidances_batches", calib_guidances_batches)
                        if io_cache.outputs is not None:
                            setattr(io_cache.outputs, "calib_guidances_batches", calib_guidances_batches)

                # Optional: dump cached activations (inputs/outputs) to disk for debugging / alternative analyses.
                # This is intentionally env-var gated to avoid I/O overhead by default.
                dump_root = os.environ.get("DEEPCOMPRESSOR_CALIB_DUMP_DIR", "").strip()
                if dump_root:
                    try:
                        dump_every = int(os.environ.get("DEEPCOMPRESSOR_CALIB_DUMP_EVERY_N_LAYERS", "1"))
                    except Exception:
                        dump_every = 1
                    try:
                        dump_limit = int(os.environ.get("DEEPCOMPRESSOR_CALIB_DUMP_LAYER_LIMIT", "-1"))
                    except Exception:
                        dump_limit = -1
                    dump_io = os.environ.get("DEEPCOMPRESSOR_CALIB_DUMP_IO", "both").strip().lower()
                    wants_inputs = dump_io in {"both", "input", "inputs", "all"}
                    wants_outputs = dump_io in {"both", "output", "outputs", "all"}

                    if dump_every < 1:
                        dump_every = 1
                    should_dump = (layer_idx % dump_every == 0) and (dump_limit < 0 or layer_idx < dump_limit)
                    if should_dump:
                        logger = tools.logging.getLogger(__name__)

                        def _safe_name(name: str) -> str:
                            name = name or "root"
                            return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)

                        out_dir = Path(dump_root)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        layer_path = out_dir / f"{layer_idx:04d}_{_safe_name(layer_name)}.pt"

                        payload: dict[str, tp.Any] = {
                            "layer_idx": int(layer_idx),
                            "layer_name": str(layer_name),
                            "num_layer_inputs": int(len(layer_inputs)),
                            "calib_timesteps_batches": calib_timesteps_batches if save_timesteps else None,
                            "calib_guidances_batches": calib_guidances_batches if save_guidances else None,
                            "modules": {},
                        }

                        def _dump_tensors_cache(tc) -> dict[str | int, dict[str, tp.Any]]:
                            out: dict[str | int, dict[str, tp.Any]] = {}
                            if tc is None:
                                return out
                            for k, tensor_cache in tc.items():
                                tensors = [t.detach().cpu() for t in tensor_cache.data]
                                channels_dim = tensor_cache.channels_dim
                                if channels_dim is not None:
                                    try:
                                        channels_dim = int(channels_dim)
                                    except Exception:
                                        channels_dim = None
                                out[k] = {
                                    "tensors": tensors,
                                    "channels_dim": channels_dim,
                                    "num_total": int(tensor_cache.num_total),
                                    "num_cached": int(tensor_cache.num_cached),
                                    "num_samples": int(tensor_cache.num_samples),
                                    "orig_device": str(tensor_cache.orig_device),
                                }
                            return out

                        seen: dict[int, str] = {}
                        for module_name, io_cache in cache[layer_name].items():
                            cache_id = id(io_cache)
                            if cache_id in seen:
                                payload["modules"][module_name] = {"alias_of": seen[cache_id]}
                                continue
                            seen[cache_id] = module_name
                            entry: dict[str, tp.Any] = {}
                            if wants_inputs:
                                entry["inputs"] = _dump_tensors_cache(io_cache.inputs)
                            if wants_outputs:
                                entry["outputs"] = _dump_tensors_cache(io_cache.outputs)
                            if entry:
                                payload["modules"][module_name] = entry

                        torch.save(payload, str(layer_path))
                        meta_path = str(layer_path) + ".meta.json"
                        meta = {
                            "layer_idx": int(layer_idx),
                            "layer_name": str(layer_name),
                            "dump_io": dump_io,
                            "num_modules": int(len(payload["modules"])),
                            "path": str(layer_path),
                        }
                        with open(meta_path, "w", encoding="utf-8") as f:
                            json.dump(meta, f, indent=2)
                        logger.info(f"- Dumped calib activations: {layer_path}")
                    del wants_inputs, wants_outputs
                del dump_root

                yield layer_name, (layer, cache[layer_name], layer_inputs)
                # region clear layer cache
                if clear_after_yield:
                    for module_cache in cache[layer_name].values():
                        module_cache.clear()
                cache.pop(layer_name)
                del layer_inputs
                gc.collect()
                torch.cuda.empty_cache()
                # endregion

    @abstractmethod
    def iter_layer_activations(  # noqa: C901
        self,
        model: nn.Module,
        *args,
        action: CacheAction,
        needs_inputs_fn: tp.Callable[[str, nn.Module], bool] | bool | None = True,
        needs_outputs_fn: tp.Callable[[str, nn.Module], bool] | bool | None = None,
        **kwargs,
    ) -> tp.Generator[
        tuple[
            str,
            tuple[
                nn.Module,
                dict[str, IOTensorsCache],
                list[ModuleForwardInput],
            ],
        ],
        None,
        None,
    ]:
        """Iterate over model activations in layers.

        Args:
            model (`nn.Module`):
                Model.
            action (`CacheAction`):
                Action for caching activations.
            needs_inputs_fn (`Callable[[str, nn.Module], bool]` or `bool` or `None`, *optional*, defaults to `True`):
                Function for determining whether to cache inputs for a module given its name and itself.
            needs_outputs_fn (`Callable[[str, nn.Module], bool]` or `bool` or `None`, *optional*, defaults to `None`):
                Function for determining whether to cache outputs for a module given its name and itself.
            *args: Arguments for ``iter_samples``.
            **kwargs: Keyword arguments for ``iter_samples``.

        Yields:
            Generator[
                tuple[str, tuple[nn.Module, dict[str, IOTensorsCache], list[ModuleForwardInput]]],
                None,
                None
            ]:
                Generator of tuple of
                    - layer name
                    - a tuple of
                        - layer itself
                        - inputs and outputs cache of each module in the layer
                        - layer input arguments
        """
        ...
