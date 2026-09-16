"""Typed contracts for exact integer relation lattices of configurations.

The grading-independent leaf published here binds one bounded integer matrix
``A`` in ``ZZ^(d x n)`` to the complete kernel lattice ``ker_Z(A)``. It reuses
the canonical ``IntegerLattice`` value and the maintained Smith and Hermite
normal-form kernels rather than introducing a second integer-module carrier.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, WithJsonSchema, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.math.lattices._models import IntegerLattice
from jacobian.math.matrices.values import IntegerMatrix

MAX_RELATION_LATTICE_DIMENSION = 12
MAX_RELATION_LATTICE_INPUT_DIGITS = 8
MAX_RELATION_LATTICE_BASIS_ENTRIES = (
    MAX_RELATION_LATTICE_DIMENSION * MAX_RELATION_LATTICE_DIMENSION
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable error owned by the affine-semigroup contracts."""

    return PydanticCustomError(f"affine_semigroup.{reason}", message)


def _configuration_json_schema() -> JsonSchemaValue:
    """Publish the relation-lattice axis bound on the shared integer matrix."""

    schema = IntegerMatrix.model_json_schema()
    properties = schema["properties"]
    for field_name in ("row_count", "column_count"):
        properties[field_name].update(
            minimum=1,
            maximum=MAX_RELATION_LATTICE_DIMENSION,
        )
    schema["description"] = (
        "One nonempty integer generator configuration A in ZZ^(d x n) with "
        f"1 <= d, n <= {MAX_RELATION_LATTICE_DIMENSION} and at most "
        f"{MAX_RELATION_LATTICE_INPUT_DIGITS} decimal digits per scalar."
    )
    return schema


class RelationLatticeRequest(StrictModel):
    """Reconstruct the complete integer relation lattice ``ker_Z(A)``."""

    configuration: Annotated[
        IntegerMatrix,
        WithJsonSchema(_configuration_json_schema()),
    ] = Field(
        description=(
            "Bounded integer generator configuration A whose columns are the "
            "generators and whose rows are the ambient coordinates. The leaf "
            "computes the full integer kernel lattice, not one solution."
        )
    )

    @model_validator(mode="after")
    def require_admitted_envelope(self) -> Self:
        rows = self.configuration.row_count
        columns = self.configuration.column_count
        if not (
            1 <= rows <= MAX_RELATION_LATTICE_DIMENSION
            and 1 <= columns <= MAX_RELATION_LATTICE_DIMENSION
        ):
            raise _validation_error(
                "budget_exceeded",
                "relation-lattice configuration axes are limited to "
                f"{MAX_RELATION_LATTICE_DIMENSION} rows and columns",
            )
        limit = 10**MAX_RELATION_LATTICE_INPUT_DIGITS
        if any(
            abs(int(value)) >= limit
            for row in self.configuration.entries
            for value in row
        ):
            raise _validation_error(
                "budget_exceeded",
                "relation-lattice configuration scalars are limited to "
                f"{MAX_RELATION_LATTICE_INPUT_DIGITS} decimal digits",
            )
        return self


class RelationLatticeResult(StrictModel):
    """Canonical integer kernel lattice of one retained configuration.

    The row-Hermite basis ``relation_basis`` has ``nullity`` rows in ``ZZ^n``
    and satisfies ``A relation_basis^T = 0`` exactly. The same lattice is
    also exposed through the canonical ``IntegerLattice`` value so that
    downstream lattice operations compose unchanged.
    """

    configuration: IntegerMatrix
    relation_lattice: IntegerLattice
    relation_basis: IntegerMatrix
    hnf_transformation: IntegerMatrix
    rank: int = Field(ge=0, le=MAX_RELATION_LATTICE_DIMENSION)
    nullity: int = Field(ge=0, le=MAX_RELATION_LATTICE_DIMENSION)
    smith_invariant_factors: tuple[ExactInteger, ...] = Field(
        max_length=MAX_RELATION_LATTICE_DIMENSION
    )
    smith_rank: int = Field(ge=0, le=MAX_RELATION_LATTICE_DIMENSION)
    saturated_basis: IntegerMatrix
    saturation_inclusion_transform: IntegerMatrix
    saturation_index: ExactInteger = Field(ge=1)
    is_saturated: bool
    convention: Literal["ROW_HERMITE_NORMAL_FORM_OF_INTEGER_KERNEL"] = (
        "ROW_HERMITE_NORMAL_FORM_OF_INTEGER_KERNEL"
    )
    relation: Literal["CONFIGURATION_TIMES_RELATION_BASIS_TRANSPOSE_IS_ZERO"] = (
        "CONFIGURATION_TIMES_RELATION_BASIS_TRANSPOSE_IS_ZERO"
    )

    @model_validator(mode="after")
    def require_structural_consistency(self) -> Self:
        columns = self.configuration.column_count
        if self.rank + self.nullity != columns:
            raise _validation_error(
                "relation_lattice_rank_identity",
                "rank + nullity must equal the configuration generator count",
            )
        if (
            self.relation_basis.row_count != self.nullity
            or self.relation_basis.column_count != columns
        ):
            raise _validation_error(
                "relation_lattice_basis_shape",
                "relation basis must have nullity rows and one column per generator",
            )
        if (
            self.relation_lattice.ambient_dimension != columns
            or self.relation_lattice.basis != self.relation_basis
        ):
            raise _validation_error(
                "relation_lattice_binding",
                "the canonical lattice value must retain the relation basis",
            )
        if self.smith_rank != self.rank:
            raise _validation_error(
                "relation_lattice_smith_rank",
                "configuration rank must match the Smith rank",
            )
        if self.saturation_index != 1 or not self.is_saturated:
            raise _validation_error(
                "relation_lattice_saturation_identity",
                "an integer kernel lattice is always saturated in ZZ^n",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        configuration: IntegerMatrix,
        relation_lattice: IntegerLattice,
        relation_basis: IntegerMatrix,
        hnf_transformation: IntegerMatrix,
        rank: int,
        nullity: int,
        smith_invariant_factors: tuple[int, ...],
        smith_rank: int,
        saturated_basis: IntegerMatrix,
        saturation_inclusion_transform: IntegerMatrix,
        saturation_index: int,
    ) -> Self:
        """Build the result after the admitted kernel established every value."""

        return cls.model_construct(
            configuration=configuration,
            relation_lattice=relation_lattice,
            relation_basis=relation_basis,
            hnf_transformation=hnf_transformation,
            rank=rank,
            nullity=nullity,
            smith_invariant_factors=tuple(
                int(value) for value in smith_invariant_factors
            ),
            smith_rank=smith_rank,
            saturated_basis=saturated_basis,
            saturation_inclusion_transform=saturation_inclusion_transform,
            saturation_index=int(saturation_index),
            is_saturated=saturation_index == 1,
        )


__all__ = [
    "MAX_RELATION_LATTICE_BASIS_ENTRIES",
    "MAX_RELATION_LATTICE_DIMENSION",
    "MAX_RELATION_LATTICE_INPUT_DIGITS",
    "RelationLatticeRequest",
    "RelationLatticeResult",
]
