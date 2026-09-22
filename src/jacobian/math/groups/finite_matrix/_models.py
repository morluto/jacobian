"""Canonical prime-field GL/SL values and natural-action contracts."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.math.groups.actions._models import FinitePermutationAction
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix

MAX_LINEAR_GROUP_PRIME = 1_000_003
MAX_LINEAR_GROUP_DIMENSION = 12
MAX_LINEAR_GROUP_GENERATORS = 2 * (MAX_LINEAR_GROUP_DIMENSION - 1) + 1
MAX_LINEAR_GROUP_GENERATOR_CELLS = 4096
MAX_NATURAL_VECTOR_ACTION_SIZE = 50
MAX_LINEAR_GROUP_ORDER_DIGITS = 2048

GeneratorFamily = Annotated[
    tuple[PrimeFieldMatrix, ...],
    Field(min_length=1, max_length=MAX_LINEAR_GROUP_GENERATORS),
]
VectorAxis = Annotated[
    tuple[tuple[StrictInt, ...], ...],
    Field(min_length=1, max_length=MAX_NATURAL_VECTOR_ACTION_SIZE),
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"finite_matrix_group.{reason}", message)


class PrimeFieldLinearGroupRequest(StrictModel):
    """The standard coordinate space ``GF(p)^n`` for a named full group."""

    prime: StrictInt = Field(
        ge=2,
        le=MAX_LINEAR_GROUP_PRIME,
        description="Prime characteristic p of the canonical field GF(p).",
    )
    dimension: StrictInt = Field(
        ge=1,
        le=MAX_LINEAR_GROUP_DIMENSION,
        description="Dimension n of the standard labelled coordinate space GF(p)^n.",
    )


class _PrimeFieldLinearGroup(StrictModel):
    prime: StrictInt = Field(ge=2, le=MAX_LINEAR_GROUP_PRIME)
    dimension: StrictInt = Field(ge=1, le=MAX_LINEAR_GROUP_DIMENSION)
    order: ExactInteger
    generators: GeneratorFamily

    @model_validator(mode="after")
    def require_structural_group_context(self) -> Self:
        if self.order < 1:
            raise _validation_error(
                "order_not_positive", "group order must be positive"
            )
        for generator in self.generators:
            if generator.prime != self.prime:
                raise _validation_error(
                    "generator_field_mismatch",
                    "every generator must use the group's canonical prime field",
                )
            if len(generator.entries) != self.dimension or (
                generator.columns != self.dimension
            ):
                raise _validation_error(
                    "generator_shape_mismatch",
                    "every generator must be an n by n matrix on the retained space",
                )
        return self


class PrimeFieldGeneralLinearGroup(_PrimeFieldLinearGroup):
    """The full ``GL(n,p)`` with a canonical complete generator family."""


class PrimeFieldSpecialLinearGroup(_PrimeFieldLinearGroup):
    """The determinant-one group ``SL(n,p)`` with complete generators."""

    ambient_general_linear_order: ExactInteger
    determinant_index: ExactInteger

    @model_validator(mode="after")
    def require_structural_index(self) -> Self:
        if self.determinant_index < 1 or self.ambient_general_linear_order < 1:
            raise _validation_error(
                "index_not_positive",
                "ambient order and determinant index must be positive",
            )
        return self


class GeneralLinearNaturalActionRequest(StrictModel):
    group: PrimeFieldGeneralLinearGroup


class SpecialLinearNaturalActionRequest(StrictModel):
    group: PrimeFieldSpecialLinearGroup


class PrimeFieldGeneralLinearNaturalAction(StrictModel):
    """The source-bound natural GL action on nonzero coordinate vectors."""

    group: PrimeFieldGeneralLinearGroup
    vectors: VectorAxis
    action: FinitePermutationAction

    @model_validator(mode="after")
    def require_structural_axis(self) -> Self:
        _require_action_axis(self.group, self.vectors, self.action)
        return self


class PrimeFieldSpecialLinearNaturalAction(StrictModel):
    """The source-bound natural SL action on nonzero coordinate vectors."""

    group: PrimeFieldSpecialLinearGroup
    vectors: VectorAxis
    action: FinitePermutationAction

    @model_validator(mode="after")
    def require_structural_axis(self) -> Self:
        _require_action_axis(self.group, self.vectors, self.action)
        return self


def _require_action_axis(
    group: _PrimeFieldLinearGroup,
    vectors: tuple[tuple[int, ...], ...],
    action: FinitePermutationAction,
) -> None:
    if any(len(vector) != group.dimension for vector in vectors):
        raise _validation_error(
            "vector_dimension_mismatch",
            "every natural-action vector must use the retained coordinate axis",
        )
    if any(
        coordinate < 0 or coordinate >= group.prime
        for vector in vectors
        for coordinate in vector
    ):
        raise _validation_error(
            "vector_residue_out_of_range",
            "natural-action vectors must use canonical prime-field residues",
        )
    labels = tuple(
        "[" + ",".join(str(value) for value in vector) + "]" for vector in vectors
    )
    if action.domain != labels:
        raise _validation_error(
            "action_axis_mismatch",
            "permutation-action labels must equal the retained vector axis",
        )
    if len(action.generators) != len(group.generators):
        raise _validation_error(
            "action_generator_count_mismatch",
            "the natural action must retain one permutation per matrix generator",
        )


__all__ = [
    "MAX_LINEAR_GROUP_DIMENSION",
    "MAX_LINEAR_GROUP_GENERATORS",
    "MAX_LINEAR_GROUP_GENERATOR_CELLS",
    "MAX_LINEAR_GROUP_ORDER_DIGITS",
    "MAX_LINEAR_GROUP_PRIME",
    "MAX_NATURAL_VECTOR_ACTION_SIZE",
    "GeneralLinearNaturalActionRequest",
    "PrimeFieldGeneralLinearGroup",
    "PrimeFieldGeneralLinearNaturalAction",
    "PrimeFieldLinearGroupRequest",
    "PrimeFieldSpecialLinearGroup",
    "PrimeFieldSpecialLinearNaturalAction",
    "SpecialLinearNaturalActionRequest",
]
