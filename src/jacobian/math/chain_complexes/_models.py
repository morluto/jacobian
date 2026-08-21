"""Typed wire contracts for chain complex operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.chain_complexes.values import (
    ChainComplexValue,
    CoefficientField,
    HomologyResult,
    MappingConeResult,
    TensorProductResult,
)


class ConstructChainComplexRequest(StrictModel):
    """Construct a chain complex from differential matrices."""

    coefficient_field: CoefficientField = CoefficientField.RATIONAL
    prime: int | None = Field(default=None, ge=2)
    basis_sizes: tuple[int, ...] = Field(min_length=1)
    differential_matrices: tuple[tuple[tuple[str, ...], ...], ...]

    @model_validator(mode="after")
    def require_consistent_dimensions(self) -> Self:
        if len(self.basis_sizes) != len(self.differential_matrices) + 1:
            raise ValueError(
                "need one more basis size than differential matrices"
            )
        return self


class VerifyDifferentialRequest(StrictModel):
    """Verify that d^2 = 0 for a chain complex."""

    complex: ChainComplexValue


class VerifyChainMapRequest(StrictModel):
    """Verify that a chain map commutes with differentials."""

    source: ChainComplexValue
    target: ChainComplexValue
    map_matrices: tuple[tuple[tuple[str, ...], ...], ...]


class ComputeHomologyRequest(StrictModel):
    """Compute homology of a chain complex."""

    complex: ChainComplexValue


class MappingConeRequest(StrictModel):
    """Compute the mapping cone of a chain map."""

    source: ChainComplexValue
    target: ChainComplexValue
    map_matrices: tuple[tuple[tuple[str, ...], ...], ...]


class TensorProductRequest(StrictModel):
    """Compute the tensor product of two chain complexes."""

    left: ChainComplexValue
    right: ChainComplexValue


__all__ = [
    "ComputeHomologyRequest",
    "ConstructChainComplexRequest",
    "MappingConeRequest",
    "TensorProductRequest",
    "VerifyChainMapRequest",
    "VerifyDifferentialRequest",
]
