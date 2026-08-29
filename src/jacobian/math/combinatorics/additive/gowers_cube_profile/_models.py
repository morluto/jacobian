"""Typed contracts for the Gowers cube count operation."""

from jacobian._models import StrictModel


class GowersCubeRequest(StrictModel):
    """Request the Gowers cube profile."""

    modulus: int
    subset: tuple[int, ...]
    order: int


class GowersCubeResult(StrictModel):
    """The exact Gowers cube count profile."""

    modulus: int
    subset: tuple[int, ...]
    order: int
    cube_count: int
    normalized_count: int


__all__ = [
    "GowersCubeRequest",
    "GowersCubeResult",
]
