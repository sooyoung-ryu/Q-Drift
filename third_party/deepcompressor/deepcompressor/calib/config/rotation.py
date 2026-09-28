# -*- coding: utf-8 -*-
"""Quantization Rotation configuration."""

import os
import typing as tp
from dataclasses import dataclass, field

import omniconfig
from omniconfig import configclass

__all__ = ["QuantRotationConfig"]


@configclass
@dataclass
class QuantRotationConfig:
    """Configuration for rotation quantization.

    Args:
        name (`str`):
            The name of the rotation quantization configuration. If `path` is provided, this is required.
            Otherwise, it is set to "random" if `random` is `True`, and "hadamard" otherwise.
        path (`str`, *optional*, default=`""`):
            The path to the rotation matrix. If provided, `name` must be set.
        random (`bool`, *optional*, default=`False`):
            Whether to use random hadamard sample as rotation matrix.
        transforms (`list[str]`, *optional*, default=`[]`):
            The module keys using explicit hadamard transform.
    """

    name: str = ""
    path: str = ""
    random: bool = False
    transforms: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.transforms = sorted(set(self.transforms or []))
        if self.path and os.path.exists(self.path):
            assert self.name, "The name of the rotation quantization configuration must be provided."
            self.random = False
        else:
            self.path = ""
            self.name = "random" if self.random else "hadamard"

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """Get the directory names of the rotation quantization configuration.

        Returns:
            list[str]: The directory names of the rotation quantization configuration.
        """
        name = self.name
        if self.transforms:
            name += f".[{'+'.join(self.transforms)}]"
        return [f"{prefix}.{name}" if prefix else name]

    @classmethod
    def update_get_arguments(
        cls: type["QuantRotationConfig"],
        *,
        overwrites: dict[str, tp.Callable[[omniconfig.Arguments], None] | None] | None = None,
        defaults: dict[str, tp.Any] | None = None,
    ) -> tuple[dict[str, tp.Callable[[omniconfig.Arguments], None] | None], dict[str, tp.Any]]:
        """Get the arguments for the rotation quantization configuration."""
        overwrites = overwrites or {}
        defaults = defaults or {}

        collect_fn = omniconfig.ADD_PREFIX_BOOL_FIELDS("transform", **defaults)

        def add_transforms_argument(parser):
            collect_fn(parser)
            parser.add_argument("--transforms", nargs="+", default=[], help="The keys of the modules to transform.")

        overwrites.setdefault("transforms", add_transforms_argument)
        return overwrites, defaults

    @classmethod
    def update_from_dict(
        cls: type["QuantRotationConfig"], *, parsed_args: dict[str, tp.Any], overwrites: dict[str, tp.Any]
    ) -> tuple[dict[str, tp.Any], dict[str, tp.Any]]:
        """Create a rotation quantization configuration from the parsed arguments."""
        parsed_args.setdefault("transforms", []).extend(omniconfig.COLLECT_PREFIX_BOOL_FIELDS(parsed_args, "transform"))
        return parsed_args, overwrites
