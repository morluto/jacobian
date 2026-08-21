"""Provider-independent exact values for finite based chain complexes."""

from __future__ import annotations

from enum import StrEnum
from fractions import Fraction

from pydantic import Field

from jacobian._models import StrictModel

MAX_CHAIN_DEGREE = 32
MAX_BASIS_SIZE = 64
MAX_MATRIX_CELLS = 4096


class CoefficientField(StrEnum):
    RATIONAL = "QQ"
    PRIME_FIELD = "GF_p"


class ChainComplexValue(StrictModel):
    """A finite based chain complex over an exact field."""

    coefficient_field: CoefficientField
    prime: int | None = Field(default=None, ge=2)
    degree_min: int = Field(ge=-MAX_CHAIN_DEGREE, le=MAX_CHAIN_DEGREE)
    degree_max: int = Field(ge=-MAX_CHAIN_DEGREE, le=MAX_CHAIN_DEGREE)
    basis_sizes: tuple[int, ...]
    differential_matrices: tuple[tuple[tuple[str, ...], ...], ...]


class HomologyGroupValue(StrictModel):
    """One homology group of a chain complex."""

    degree: int
    cycle_rank: int = Field(ge=0)
    boundary_rank: int = Field(ge=0)
    betti_number: int = Field(ge=0)


class HomologyResult(StrictModel):
    """Homology of a chain complex."""

    homology_groups: tuple[HomologyGroupValue, ...]
    coefficient_field: CoefficientField


class MappingConeResult(StrictModel):
    """The mapping cone of a chain map."""

    cone_basis_sizes: tuple[int, ...]
    cone_differential_matrices: tuple[tuple[tuple[str, ...], ...], ...]
    source_degree_min: int
    target_degree_min: int


class TensorProductResult(StrictModel):
    """The tensor product of two chain complexes."""

    tensor_basis_sizes: tuple[int, ...]
    tensor_differential_matrices: tuple[tuple[tuple[str, ...], ...], ...]


__all__ = [
    "ChainComplexValue",
    "CoefficientField",
    "HomologyGroupValue",
    "HomologyResult",
    "MappingConeResult",
    "TensorProductResult",
    "MAX_BASIS_SIZE",
    "MAX_CHAIN_DEGREE",
    "MAX_MATRIX_CELLS",
]


class VerificationResult(StrictModel):
    """Result of verifying a chain complex property."""

    is_valid: bool
    detail: str


__all__.append("VerificationResult")
