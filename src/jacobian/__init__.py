"""Jacobian package metadata.

The supported native mathematical API lives under :mod:`jacobian.math`.
"""

from importlib.metadata import PackageNotFoundError, version

__all__: list[str] = []

try:
    __version__ = version("jacobian")
except PackageNotFoundError:  # pragma: no cover - only an unpackaged source tree
    __version__ = "0+unknown"
