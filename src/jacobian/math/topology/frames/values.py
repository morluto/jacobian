"""Canonical finite vector-family values used by frame operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.number_theory.number_fields import GaussianRational

MAX_VECTOR_CELLS = 524_288
MAX_DIM = MAX_VECTOR_CELLS
MAX_COMPLEX_FRAME_CELLS = 4_096
MAX_COMPLEX_BASIS_COUNT = 16
MAX_COMPLEX_BASIS_PAIRS = 120
MAX_COMPLEX_PROFILE_CELLS = 16_384
MAX_COMPLEX_INNER_PRODUCT_WORK = 2_000_000
MAX_COMPLEX_COMPONENT_DIGITS = 128
MAX_CYCLOTOMIC_ORDER = 24
MAX_CYCLOTOMIC_FRAME_CELLS = 1_024
MAX_CYCLOTOMIC_COMPONENT_DIGITS = 256
_MAX_VECTOR_ENTRY = (1 << 53) - 1


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"frames.{reason}", message)


class VectorFamily(StrictModel):
    """A bounded family of integer vectors in Euclidean coordinate space.

    Spanning and nonzero-vector requirements are operation-specific admission
    decisions made by the native kernels.
    """

    dimension: int = Field(ge=0, le=MAX_DIM)
    vectors: tuple[tuple[int, ...], ...] = Field(
        max_length=MAX_VECTOR_CELLS,
        description=(
            "Ordered vectors with len(vectors) * dimension <= "
            f"{MAX_VECTOR_CELLS} materialized cells."
        ),
    )

    @model_validator(mode="after")
    def require_rectangular_family(self) -> Self:
        dimension = self.dimension
        if any(len(vector) != dimension for vector in self.vectors):
            raise _validation_error(
                "vector_dimension_mismatch", "all vectors must have equal dimension"
            )
        if len(self.vectors) * dimension > MAX_VECTOR_CELLS:
            raise _validation_error(
                "vector_cell_budget",
                "vector family exceeds the materialized-cell budget",
            )
        if any(
            abs(entry) > _MAX_VECTOR_ENTRY
            for vector in self.vectors
            for entry in vector
        ):
            raise _validation_error(
                "vector_entry_out_of_range", "vector entries must be bounded"
            )
        return self


class ComplexFrame(StrictModel):
    """A bounded finite family of exact complex representatives.

    Complex-frame operations that describe bases or projective configurations
    use nonzero representatives and normalize their Hermitian overlaps
    conceptually; unit coordinate norms are not a construction requirement.
    """

    dimension: int = Field(ge=1, le=MAX_DIM)
    vectors: tuple[tuple[GaussianRational, ...], ...] = Field(
        max_length=MAX_VECTOR_CELLS,
        description=(
            "Ordered complex vectors with len(vectors) * dimension <= "
            f"{MAX_COMPLEX_FRAME_CELLS} materialized cells and exactly "
            "dimension entries per vector."
        ),
    )

    @model_validator(mode="after")
    def require_rectangular_family(self) -> Self:
        if any(len(vector) != self.dimension for vector in self.vectors):
            raise _validation_error(
                "complex_vector_dimension_mismatch",
                "all complex vectors must have equal dimension",
            )
        if len(self.vectors) * self.dimension > MAX_COMPLEX_FRAME_CELLS:
            raise _validation_error(
                "complex_vector_cell_budget",
                "complex vector family exceeds the materialized-cell budget",
            )
        return self


def euler_phi(order: int) -> int:
    """Return Euler's totient with integer-only arithmetic."""

    result, remaining, prime = order, order, 2
    while prime * prime <= remaining:
        if remaining % prime == 0:
            while remaining % prime == 0:
                remaining //= prime
            result -= result // prime
        prime += 1 if prime == 2 else 2
    if remaining > 1:
        result -= result // remaining
    return result


class CyclotomicScalar(StrictModel):
    """One exact element of Q(zeta_order) in the power basis.

    Coefficients are ascending powers of the distinguished primitive root;
    the representative is reduced modulo the order's cyclotomic polynomial
    by the consuming kernels, and equality is coefficient-wise.
    """

    order: int = Field(ge=1, le=MAX_CYCLOTOMIC_ORDER)
    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        description="Ascending-power coefficients of length phi(order).",
    )

    @model_validator(mode="after")
    def require_cyclotomic_shape(self) -> Self:
        if len(self.coefficients) != euler_phi(self.order):
            raise _validation_error(
                "cyclotomic_scalar_degree",
                "scalar coefficients must have length phi(order)",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, *, order: int, coefficients: tuple[CanonicalRational, ...]
    ) -> Self:
        return cls.model_construct(order=order, coefficients=coefficients)


class CyclotomicFrame(StrictModel):
    """A bounded family of exact cyclotomic representatives.

    Projective status uses normalized Hermitian overlaps, so representatives
    need not have unit coordinate norm; the zero vector is rejected per
    vector by the consuming kernels, not by structural parsing.
    """

    order: int = Field(ge=1, le=MAX_CYCLOTOMIC_ORDER)
    dimension: int = Field(ge=1, le=MAX_DIM)
    vectors: tuple[tuple[CyclotomicScalar, ...], ...] = Field(
        max_length=MAX_CYCLOTOMIC_FRAME_CELLS,
        description=(
            "Ordered cyclotomic vectors with len(vectors) * dimension <= "
            "MAX_CYCLOTOMIC_FRAME_CELLS entries sharing one cyclotomic order."
        ),
    )

    @model_validator(mode="after")
    def require_rectangular_family(self) -> Self:
        if any(len(vector) != self.dimension for vector in self.vectors):
            raise _validation_error(
                "cyclotomic_vector_dimension_mismatch",
                "all cyclotomic vectors must have equal dimension",
            )
        if any(scalar.order != self.order for vector in self.vectors for scalar in vector):
            raise _validation_error(
                "cyclotomic_order_mismatch",
                "every scalar must share the frame cyclotomic order",
            )
        if len(self.vectors) * self.dimension > MAX_CYCLOTOMIC_FRAME_CELLS:
            raise _validation_error(
                "cyclotomic_vector_cell_budget",
                "cyclotomic vector family exceeds the materialized-cell budget",
            )
        return self


__all__ = [
    "MAX_COMPLEX_BASIS_COUNT",
    "MAX_COMPLEX_BASIS_PAIRS",
    "MAX_COMPLEX_COMPONENT_DIGITS",
    "MAX_COMPLEX_FRAME_CELLS",
    "MAX_COMPLEX_INNER_PRODUCT_WORK",
    "MAX_COMPLEX_PROFILE_CELLS",
    "MAX_CYCLOTOMIC_FRAME_CELLS",
    "MAX_CYCLOTOMIC_ORDER",
    "MAX_DIM",
    "MAX_VECTOR_CELLS",
    "ComplexFrame",
    "CyclotomicFrame",
    "CyclotomicScalar",
    "VectorFamily",
    "euler_phi",
]
