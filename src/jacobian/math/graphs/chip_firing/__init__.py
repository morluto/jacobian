"""Chip firing operations."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math.graphs.chip_firing.operations import (
        abel_jacobi,
        canonical_divisor,
        critical_group,
        degree,
        fire_vector,
        firing,
        laplacian,
        parallel_step,
        q_reduced,
        reduced_laplacian,
        stabilize,
        verify_laplacian,
        verify_reduced_laplacian,
    )

__all__ = [
    "abel_jacobi",
    "canonical_divisor",
    "critical_group",
    "degree",
    "fire_vector",
    "firing",
    "laplacian",
    "parallel_step",
    "q_reduced",
    "reduced_laplacian",
    "stabilize",
    "verify_laplacian",
    "verify_reduced_laplacian",
]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.operations"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
