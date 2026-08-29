"""Typed contracts for homogeneous progression set system construction."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_N = 1000


class HomogeneousProgressionRequest(StrictModel):
    """Request to construct the homogeneous progression set system on [n]."""

    n: int = Field(ge=0, le=MAX_N)

    @model_validator(mode="after")
    def validate_n(self) -> Self:
        if self.n > MAX_N:
            raise PydanticCustomError(
                "discrepancy.n_too_large",
                f"n must not exceed {MAX_N}",
            )
        return self


class HomogeneousProgressionResult(StrictModel):
    """The homogeneous progression set system on [n]."""

    n: int
    ground_set_size: int
    sets: tuple[tuple[int, ...], ...]


__all__ = [
    "MAX_N",
    "HomogeneousProgressionRequest",
    "HomogeneousProgressionResult",
]
