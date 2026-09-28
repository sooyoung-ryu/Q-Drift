# -*- coding: utf-8 -*-
"""Deepcompressor Extension."""

import os

from torch.utils.cpp_extension import load

__all__ = ["_C"]

dirpath = os.path.dirname(__file__)

_verbose_env = os.environ.get("DEEPCOMPRESSOR_EXT_VERBOSE", "0").strip().lower()
_verbose = _verbose_env not in ("", "0", "false", "no", "off")
_build_directory = os.environ.get("DEEPCOMPRESSOR_EXT_BUILD_DIR", "").strip() or None

if _verbose:
    print(
        "[DeepCompressor] Loading C++/CUDA extension (this may take a while on first run). "
        f"build_directory={_build_directory or '<torch default>'}",
        flush=True,
    )

_C = load(
    name="deepcompressor_C",
    sources=[f"{dirpath}/pybind.cpp", f"{dirpath}/quantize/quantize.cu"],
    extra_cflags=["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_HALF2_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--ptxas-options=--allow-expensive-optimizations=true",
        "--threads=8",
    ],
    build_directory=_build_directory,
    verbose=_verbose,
)
