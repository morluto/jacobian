"""Source-bound quotient spaces of finite-dimensional prime-field spaces."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.matrices.finite_fields._bounds import (
    MAX_PRIME_FIELD_ELIMINATION_WORK,
    MAX_PRIME_FIELD_FLINT_PRIME,
    MAX_PRIME_FIELD_MATRIX_AXIS,
    MAX_PRIME_FIELD_MATRIX_CELLS,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix


def _error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"prime_field_quotient.{reason}", message)


class PrimeFieldSubspace(StrictModel):
    """A finitely generated subspace bound to GF(p)^ambient_dimension.

    Rows are generators, so dependent and empty generator families are valid.
    The declared ambient dimension preserves the zero-subspace axis.
    """

    prime: StrictInt = Field(ge=2, le=MAX_PRIME_FIELD_FLINT_PRIME)
    ambient_dimension: StrictInt = Field(ge=0, le=MAX_PRIME_FIELD_MATRIX_AXIS)
    generators: tuple[tuple[StrictInt, ...], ...] = ()

    @model_validator(mode="after")
    def require_canonical_generators(self) -> Self:
        if len(self.generators) > MAX_PRIME_FIELD_MATRIX_AXIS:
            raise _error("generator_count", "subspace generator count exceeds 1024")
        if len(self.generators) * self.ambient_dimension > MAX_PRIME_FIELD_MATRIX_CELLS:
            raise _error("generator_cells", "subspace generators exceed the cell bound")
        if any(len(vector) != self.ambient_dimension for vector in self.generators):
            raise _error(
                "generator_axis", "every generator must match the ambient axis"
            )
        if any(
            not 0 <= entry < self.prime
            for vector in self.generators
            for entry in vector
        ):
            raise _error(
                "generator_residue", "entries must be canonical residues in GF(p)"
            )
        return self


class PrimeFieldQuotientSpace(StrictModel):
    """The quotient V/W with a retained basis and its canonical projection.

    ``projection`` multiplies ambient column coordinates to produce quotient
    coordinates in ``quotient_basis``. Its kernel is the retained denominator
    subspace, making this a reusable target value rather than detached vectors.
    """

    source: PrimeFieldSubspace
    quotient_basis: tuple[tuple[StrictInt, ...], ...]
    projection: PrimeFieldMatrix

    @model_validator(mode="after")
    def require_structural_axes(self) -> Self:
        quotient_dimension = len(self.quotient_basis)
        ambient_dimension = self.source.ambient_dimension
        if quotient_dimension > ambient_dimension:
            raise _error(
                "quotient_dimension", "quotient dimension exceeds its ambient axis"
            )
        if any(len(vector) != ambient_dimension for vector in self.quotient_basis):
            raise _error(
                "basis_axis", "quotient representatives must use ambient coordinates"
            )
        if any(
            not 0 <= entry < self.source.prime
            for vector in self.quotient_basis
            for entry in vector
        ):
            raise _error(
                "basis_residue", "quotient basis entries must be canonical residues"
            )
        if (
            self.projection.prime != self.source.prime
            or self.projection.columns != ambient_dimension
            or len(self.projection.entries) != quotient_dimension
        ):
            raise _error(
                "projection_axes", "projection axes must match the bound quotient"
            )
        proof_cells = (
            len(self.source.generators) * ambient_dimension
            + 2 * quotient_dimension * ambient_dimension
            + quotient_dimension
        )
        if proof_cells > MAX_PRIME_FIELD_MATRIX_CELLS:
            raise _error(
                "proof_cells", "quotient proof data exceeds the exact cell bound"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: PrimeFieldSubspace,
        quotient_basis: tuple[tuple[int, ...], ...],
        projection: PrimeFieldMatrix,
    ) -> Self:
        return cls.model_construct(
            source=source, quotient_basis=quotient_basis, projection=projection
        )


class PrimeFieldQuotientRequest(StrictModel):
    subspace: PrimeFieldSubspace

    @model_validator(mode="after")
    def admit_quotient_work_and_output(self) -> Self:
        dimension = self.subspace.ambient_dimension
        generators = len(self.subspace.generators)
        if dimension + generators > MAX_PRIME_FIELD_MATRIX_AXIS:
            raise _error(
                "intermediate_axis",
                "quotient elimination admits ambient dimension plus generator count at most 1024",
            )
        if 2 * dimension * dimension > MAX_PRIME_FIELD_MATRIX_CELLS:
            raise _error(
                "result_cells",
                "quotient basis and projection together exceed the exact result cell bound",
            )
        # Retain the source rows, quotient basis, projection, and the worst-case
        # quotient-coordinate vector for source-bound projection results.
        aggregate_cells = generators * dimension + 2 * dimension * dimension + dimension
        if aggregate_cells > MAX_PRIME_FIELD_MATRIX_CELLS:
            raise _error(
                "aggregate_cells",
                "source generators, quotient basis, and projection exceed the aggregate cell bound",
            )
        elimination_work = (
            dimension * generators * min(dimension, generators)
            + dimension * (dimension + generators) * dimension
            + dimension**3
        )
        if elimination_work > MAX_PRIME_FIELD_ELIMINATION_WORK:
            raise _error(
                "elimination_work",
                "quotient basis and projection exceed the exact elimination work bound",
            )
        return self


class PrimeFieldVectorProjectionRequest(StrictModel):
    quotient: PrimeFieldQuotientSpace
    vector: tuple[StrictInt, ...]

    @model_validator(mode="after")
    def require_vector_axis(self) -> Self:
        if len(self.vector) != self.quotient.source.ambient_dimension:
            raise _error("vector_axis", "vector must match the quotient ambient axis")
        if any(not 0 <= entry < self.quotient.source.prime for entry in self.vector):
            raise _error(
                "vector_residue", "vector entries must be canonical residues in GF(p)"
            )
        dimension = self.quotient.source.ambient_dimension
        generator_count = len(self.quotient.source.generators)
        quotient_dimension = len(self.quotient.quotient_basis)
        validation_work = (
            dimension * generator_count * min(dimension, generator_count)
            + dimension * quotient_dimension * min(dimension, quotient_dimension)
            + quotient_dimension * dimension * generator_count
            + quotient_dimension * dimension * quotient_dimension
        )
        if validation_work > MAX_PRIME_FIELD_ELIMINATION_WORK:
            raise _error(
                "projection_validation_work",
                "quotient proof exceeds the elimination work bound",
            )
        return self


class PrimeFieldQuotientVector(StrictModel):
    """One quotient vector with its exact quotient parent retained."""

    quotient: PrimeFieldQuotientSpace
    coordinates: tuple[StrictInt, ...]

    @model_validator(mode="after")
    def require_quotient_axis(self) -> Self:
        if len(self.coordinates) != len(self.quotient.quotient_basis):
            raise _error("coordinate_axis", "coordinates must match the quotient basis")
        if any(
            not 0 <= entry < self.quotient.source.prime for entry in self.coordinates
        ):
            raise _error(
                "coordinate_residue", "coordinates must be canonical residues in GF(p)"
            )
        return self

    @classmethod
    def _from_kernel(
        cls, quotient: PrimeFieldQuotientSpace, coordinates: tuple[int, ...]
    ) -> Self:
        return cls.model_construct(quotient=quotient, coordinates=coordinates)


__all__ = [
    "PrimeFieldQuotientRequest",
    "PrimeFieldQuotientSpace",
    "PrimeFieldQuotientVector",
    "PrimeFieldSubspace",
    "PrimeFieldVectorProjectionRequest",
]
