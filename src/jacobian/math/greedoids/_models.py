"""Typed wire contracts for greedoid operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.greedoids.values import FiniteFeasibleSetSystem

MAX_GROUND_SIZE = 64
"""Schema-visible cap on ground-set cardinality for greedoid requests."""

MAX_FEASIBLE_COUNT = 4096
"""Schema-visible cap on feasible-row count for greedoid requests."""


def require_bounded_carrier(system: FiniteFeasibleSetSystem) -> None:
    """Bound the greedoid execution envelope before any kernel expands.

    The shared carrier is structural only; these operation-owned ceilings
    control the ordered-pair and family-scan work of the greedoid kernels.
    """

    if len(system.ground) > MAX_GROUND_SIZE:
        raise ValueError(
            f"ground size exceeds the bounded budget of {MAX_GROUND_SIZE} elements"
        )
    if len(system.feasible) > MAX_FEASIBLE_COUNT:
        raise ValueError(
            f"feasible-set count exceeds the bounded budget of "
            f"{MAX_FEASIBLE_COUNT} rows"
        )


class RecognizeRequest(StrictModel):
    """Recognize a feasible-set family as a greedoid."""

    system: FiniteFeasibleSetSystem

    @model_validator(mode="after")
    def require_bounded_system(self) -> Self:
        require_bounded_carrier(self.system)
        return self


class RecognizeResult(StrictModel):
    """``GREEDOID`` with rank/bases, or ``NOT_A_GREEDOID`` with the first obstruction."""

    status: str
    obstruction: str | None = None
    larger_set: tuple[int, ...] | None = None
    smaller_set: tuple[int, ...] | None = None
    feasible_set: tuple[int, ...] | None = None
    rank: int | None = None
    bases: tuple[tuple[int, ...], ...] = ()
    ground_size: int | None = None

    @model_validator(mode="after")
    def bind_status(self) -> Self:
        if self.status not in ("GREEDOID", "NOT_A_GREEDOID"):
            raise ValueError("status must be GREEDOID or NOT_A_GREEDOID")
        if self.status == "GREEDOID":
            if self.obstruction is not None:
                raise ValueError("a GREEDOID result has no obstruction")
        else:
            if self.obstruction is None:
                raise ValueError("a NOT_A_GREEDOID result must name an obstruction")
        return self


class RankRequest(StrictModel):
    """Compute greedoid rank for an optional ground subset."""

    system: FiniteFeasibleSetSystem
    subset: tuple[int, ...] | None = Field(default=None)

    @model_validator(mode="after")
    def require_valid_subset(self) -> Self:
        require_bounded_carrier(self.system)
        if self.subset is not None:
            n = len(self.system.ground)
            if len(set(self.subset)) != len(self.subset):
                raise ValueError("subset must not contain duplicates")
            if any(not 0 <= i < n for i in self.subset):
                raise ValueError("subset indices must be in range")
        return self


class RankResult(StrictModel):
    """The greedoid rank of the supplied subset."""

    rank: int = Field(ge=0)
    subset: tuple[int, ...] | None = Field(default=None)


class BasesRequest(StrictModel):
    """Compute the maximal feasible subsets (bases)."""

    system: FiniteFeasibleSetSystem
    subset: tuple[int, ...] | None = Field(default=None)

    @model_validator(mode="after")
    def require_valid_subset(self) -> Self:
        require_bounded_carrier(self.system)
        if self.subset is not None:
            n = len(self.system.ground)
            if len(set(self.subset)) != len(self.subset):
                raise ValueError("subset must not contain duplicates")
            if any(not 0 <= i < n for i in self.subset):
                raise ValueError("subset indices must be in range")
        return self


class BasesResult(StrictModel):
    """The basis family and common rank."""

    rank: int = Field(ge=0)
    bases: tuple[tuple[int, ...], ...]


class BasicWordProfileRequest(StrictModel):
    """Profile a candidate basic word."""

    system: FiniteFeasibleSetSystem
    word: tuple[int, ...] = Field(default=())

    @model_validator(mode="after")
    def require_bounded_system(self) -> Self:
        require_bounded_carrier(self.system)
        return self


class BasicWordProfileResult(StrictModel):
    """Whether the word is a basic word, with first obstruction if not."""

    status: str
    obstruction: str | None = None
    prefix_index: int | None = None
    prefix_set: tuple[int, ...] | None = None
    prefix_length: int | None = None
    is_full: bool | None = None
    rank: int | None = None

    @model_validator(mode="after")
    def bind_status(self) -> Self:
        if self.status not in ("BASIC_WORD", "NOT_A_BASIC_WORD"):
            raise ValueError("status must be BASIC_WORD or NOT_A_BASIC_WORD")
        return self


class ConvexGeometryRequest(StrictModel):
    """Compute the complementary closed-set family of a full-support antimatroid."""

    system: FiniteFeasibleSetSystem

    @model_validator(mode="after")
    def require_bounded_system(self) -> Self:
        require_bounded_carrier(self.system)
        return self


class ConvexGeometryResult(StrictModel):
    """The closed-set family and the feasible->closed complement map.

    ``complement_map`` is an ordered list of ``(feasible, closed)`` pairs so
    the wire representation stays JSON-safe.
    """

    closed_family: tuple[tuple[int, ...], ...]
    complement_map: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]


__all__ = [
    "BasesRequest",
    "BasesResult",
    "BasicWordProfileRequest",
    "BasicWordProfileResult",
    "ConvexGeometryRequest",
    "ConvexGeometryResult",
    "RankRequest",
    "RankResult",
    "RecognizeRequest",
    "RecognizeResult",
]
