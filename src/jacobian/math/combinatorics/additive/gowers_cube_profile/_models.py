"""Typed contracts for the Gowers cube count operation."""

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_GOWERS_CUBE_ORDER = 12
MAX_GOWERS_CUBE_VERTEX_CHECKS = 2_000_000


def gowers_cube_work(modulus: int, order: int) -> int:
    return int(modulus ** (order + 1) * (1 << order) * order)


class GowersCubeRequest(StrictModel):
    """Request the Gowers cube profile."""

    modulus: int = Field(gt=0)
    subset: tuple[int, ...]
    order: int = Field(ge=1, le=MAX_GOWERS_CUBE_ORDER)

    @model_validator(mode="after")
    def require_canonical_bounded_cube(self) -> Self:
        if len(self.subset) != len(set(self.subset)) or any(
            not 0 <= value < self.modulus for value in self.subset
        ):
            raise PydanticCustomError(
                "gowers_cube.canonical_subset",
                "subset must contain distinct canonical residues modulo modulus",
            )
        if (
            self.subset
            and gowers_cube_work(self.modulus, self.order)
            > MAX_GOWERS_CUBE_VERTEX_CHECKS
        ):
            raise PydanticCustomError(
                "gowers_cube.work_exceeded",
                "Gowers cube enumeration exceeds the 2000000-vertex-check bound",
            )
        return self


class GowersCubeResult(StrictModel):
    """The exact Gowers cube count profile."""

    modulus: int
    subset: tuple[int, ...]
    order: int
    cube_count: int
    normalized_count: CanonicalRational


__all__ = [
    "MAX_GOWERS_CUBE_ORDER",
    "MAX_GOWERS_CUBE_VERTEX_CHECKS",
    "GowersCubeRequest",
    "GowersCubeResult",
    "gowers_cube_work",
]
