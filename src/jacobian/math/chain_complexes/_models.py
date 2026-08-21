"""Typed wire contracts for based finite chain complex operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel


class MatrixEntry(StrictModel):
    """One entry of a sparse differential matrix."""

    row: int = Field(ge=0)
    col: int = Field(ge=0)
    value: str = Field(min_length=1, max_length=256)


class ChainComplex(StrictModel):
    """A bounded homological chain complex over a prime field GF(p).

    The complex has groups C_{n_min}, ..., C_{n_max} with differentials
    d_n : C_n -> C_{n-1}. Each group has a finite basis (dimension).
    """

    prime: int = Field(ge=2, le=10_000)
    min_degree: int = Field(ge=-10, le=10)
    max_degree: int = Field(ge=-10, le=10)
    dimensions: tuple[int, ...] = Field(min_length=1, max_length=21)
    differentials: tuple[tuple[MatrixEntry, ...], ...] = Field(
        min_length=0, max_length=21
    )

    @model_validator(mode="after")
    def require_valid_complex(self) -> Self:
        if self.max_degree < self.min_degree:
            raise ValueError("max_degree must be >= min_degree")
        expected_length = self.max_degree - self.min_degree + 1
        if len(self.dimensions) != expected_length:
            raise ValueError("dimensions must cover the degree range")
        if any(d < 0 for d in self.dimensions):
            raise ValueError("dimensions must be non-negative")
        if self.differentials:
            if len(self.differentials) != expected_length - 1:
                raise ValueError("differentials must cover the degree gaps")
            for i, diff in enumerate(self.differentials):
                source_dim = self.dimensions[i]
                target_dim = self.dimensions[i + 1]
                for entry in diff:
                    if entry.row >= target_dim:
                        raise ValueError("differential entry row exceeds target dimension")
                    if entry.col >= source_dim:
                        raise ValueError("differential entry col exceeds source dimension")
        return self


class HomologyRequest(StrictModel):
    """Compute the homology of a chain complex."""

    complex: ChainComplex


class HomologyGroup(StrictModel):
    """One homology group Betti number."""

    degree: int
    betti: int = Field(ge=0)
    dimension: int = Field(ge=0)
    boundary_rank: int = Field(ge=0)
    cycle_rank: int = Field(ge=0)


class HomologyResult(StrictModel):
    """Homology groups of a chain complex."""

    groups: tuple[HomologyGroup, ...] = Field(min_length=1)
    prime: int = Field(ge=2, le=10_000)
    min_degree: int = Field(ge=-10, le=10)
    max_degree: int = Field(ge=-10, le=10)

    @model_validator(mode="after")
    def require_consistent_groups(self) -> Self:
        if len(self.groups) != self.max_degree - self.min_degree + 1:
            raise ValueError("homology groups must cover the degree range")
        return self


class MappingConeRequest(StrictModel):
    """Compute the mapping cone of a chain map f: C -> D."""

    source: ChainComplex
    target: ChainComplex
    chain_map: tuple[tuple[MatrixEntry, ...], ...] = Field(min_length=0, max_length=21)


class MappingConeResult(StrictModel):
    """The mapping cone complex of a chain map."""

    cone: ChainComplex


__all__ = [
    "ChainComplex",
    "HomologyGroup",
    "HomologyRequest",
    "HomologyResult",
    "MappingConeRequest",
    "MappingConeResult",
    "MatrixEntry",
]
