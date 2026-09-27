"""Source-bound quotient spaces of finite-dimensional prime-field spaces."""

from __future__ import annotations

from typing import Any, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.matrices.finite_fields._bounds import (
    MAX_PRIME_FIELD_ELIMINATION_WORK,
    MAX_PRIME_FIELD_FLINT_PRIME,
    MAX_PRIME_FIELD_MATRIX_AXIS,
    MAX_PRIME_FIELD_MATRIX_CELLS,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix


def _error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"prime_field_quotient.{reason}", message)


def _require_canonical_generators(
    prime: object,
    ambient_dimension: object,
    generators: object,
) -> None:
    """Admit one generator family against its declared field and ambient axis.

    Both the wire model and the native operations call this directly so an
    unchecked ``model_copy`` or ``model_construct`` carrier cannot skip the
    structural invariants that Pydantic established for parsed input.
    """
    if type(ambient_dimension) is not int or not (
        0 <= ambient_dimension <= MAX_PRIME_FIELD_MATRIX_AXIS
    ):
        raise _error(
            "subspace_axis",
            "subspace ambient dimension must be a bounded nonnegative integer",
        )
    if type(prime) is not int or not 2 <= prime <= MAX_PRIME_FIELD_FLINT_PRIME:
        raise _error(
            "subspace_prime", "subspace prime must be an integer in [2, 2147483647]"
        )
    if type(generators) is not tuple or any(
        type(vector) is not tuple for vector in generators
    ):
        raise _error(
            "generator_type", "subspace generators must be a tuple of coordinate tuples"
        )
    if len(generators) > MAX_PRIME_FIELD_MATRIX_AXIS:
        raise _error("generator_count", "subspace generator count exceeds 1024")
    if len(generators) * ambient_dimension > MAX_PRIME_FIELD_MATRIX_CELLS:
        raise _error("generator_cells", "subspace generators exceed the cell bound")
    if any(len(vector) != ambient_dimension for vector in generators):
        raise _error("generator_axis", "every generator must match the ambient axis")
    if any(
        type(entry) is not int or not 0 <= entry < prime
        for vector in generators
        for entry in vector
    ):
        raise _error("generator_residue", "entries must be canonical residues in GF(p)")


def _require_quotient_envelope(ambient_dimension: int, generator_count: int) -> None:
    """Admit the bounded work, intermediate, and output envelope for V/W."""
    if ambient_dimension + generator_count > MAX_PRIME_FIELD_MATRIX_AXIS:
        raise _error(
            "intermediate_axis",
            "quotient elimination admits ambient dimension plus generator count at most 1024",
        )
    if 2 * ambient_dimension * ambient_dimension > MAX_PRIME_FIELD_MATRIX_CELLS:
        raise _error(
            "result_cells",
            "quotient basis and projection together exceed the exact result cell bound",
        )
    # Retain the source rows, quotient basis, projection, and the worst-case
    # quotient-coordinate vector for source-bound projection results.
    aggregate_cells = (
        generator_count * ambient_dimension
        + 2 * ambient_dimension * ambient_dimension
        + ambient_dimension
    )
    if aggregate_cells > MAX_PRIME_FIELD_MATRIX_CELLS:
        raise _error(
            "aggregate_cells",
            "source generators, quotient basis, and projection exceed the aggregate cell bound",
        )
    elimination_work = (
        ambient_dimension * generator_count * min(ambient_dimension, generator_count)
        + ambient_dimension * (ambient_dimension + generator_count) * ambient_dimension
        + ambient_dimension**3
    )
    if elimination_work > MAX_PRIME_FIELD_ELIMINATION_WORK:
        raise _error(
            "elimination_work",
            "quotient basis and projection exceed the exact elimination work bound",
        )


def _require_quotient_structure(
    source: PrimeFieldSubspace,
    quotient_basis: object,
    projection: object,
) -> None:
    """Admit the structural field, axis, and residue invariants of a quotient."""
    if type(quotient_basis) is not tuple or any(
        type(vector) is not tuple for vector in quotient_basis
    ):
        raise _error(
            "basis_type", "quotient basis must be a tuple of coordinate tuples"
        )
    ambient_dimension = source.ambient_dimension
    prime = source.prime
    quotient_dimension = len(quotient_basis)
    if quotient_dimension > ambient_dimension:
        raise _error(
            "quotient_dimension", "quotient dimension exceeds its ambient axis"
        )
    if any(len(vector) != ambient_dimension for vector in quotient_basis):
        raise _error(
            "basis_axis", "quotient representatives must use ambient coordinates"
        )
    if any(
        type(entry) is not int or not 0 <= entry < prime
        for vector in quotient_basis
        for entry in vector
    ):
        raise _error(
            "basis_residue", "quotient basis entries must be canonical residues"
        )
    if type(projection) is not PrimeFieldMatrix:
        raise _error(
            "projection_type", "projection must be a canonical PrimeFieldMatrix value"
        )
    if (
        projection.prime != prime
        or projection.columns != ambient_dimension
        or len(projection.entries) != quotient_dimension
    ):
        raise _error(
            "projection_axes", "projection field and axes must match the bound quotient"
        )
    proof_cells = (
        len(source.generators) * ambient_dimension
        + 2 * quotient_dimension * ambient_dimension
        + quotient_dimension
    )
    if proof_cells > MAX_PRIME_FIELD_MATRIX_CELLS:
        raise _error("proof_cells", "quotient proof data exceeds the exact cell bound")


class PrimeFieldSubspace(StrictModel):
    """A finitely generated subspace bound to GF(p)^ambient_dimension.

    Rows are generators, so dependent and empty generator families are valid.
    The declared ambient dimension preserves the zero-subspace axis.
    """

    prime: StrictInt = Field(ge=2, le=MAX_PRIME_FIELD_FLINT_PRIME)
    ambient_dimension: StrictInt = Field(ge=0, le=MAX_PRIME_FIELD_MATRIX_AXIS)
    generators: tuple[tuple[StrictInt, ...], ...] = ()

    @model_validator(mode="before")
    @classmethod
    def preflight_generator_size(cls, data: Any) -> Any:
        """Reject oversized raw generator containers before Pydantic conversion.

        A caller can otherwise hand in far more than 1024 rows or one overlong
        row and make Pydantic materialize and type-check every nested
        ``StrictInt`` before the axis and cell limits run.
        """
        raw_generators = (
            data.get("generators")
            if isinstance(data, dict)
            else getattr(data, "generators", None)
        )
        if isinstance(raw_generators, (list, tuple)):
            if len(raw_generators) > MAX_PRIME_FIELD_MATRIX_AXIS:
                raise _error("generator_count", "subspace generator count exceeds 1024")
            materialized_cells = 0
            for vector in raw_generators:
                if not isinstance(vector, (list, tuple)):
                    break
                if len(vector) > MAX_PRIME_FIELD_MATRIX_AXIS:
                    raise _error(
                        "generator_axis", "every generator must match the ambient axis"
                    )
                materialized_cells += len(vector)
                if materialized_cells > MAX_PRIME_FIELD_MATRIX_CELLS:
                    raise _error(
                        "generator_cells", "subspace generators exceed the cell bound"
                    )
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_canonical_generators(self) -> Self:
        _require_canonical_generators(
            self.prime, self.ambient_dimension, self.generators
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
        _require_quotient_structure(self.source, self.quotient_basis, self.projection)
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
        _require_quotient_envelope(
            self.subspace.ambient_dimension, len(self.subspace.generators)
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
]
