from __future__ import annotations

import typing as _typing

# Python <3.11 compatibility: DeepCompressor uses `typing.Self` throughout.
if not hasattr(_typing, "Self"):
    from typing_extensions import Self as _Self

    setattr(_typing, "Self", _Self)

from .version import __version__  # noqa: E402,F401
