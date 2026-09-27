"""Canonical prime- and extension-field GL/SL values and action contracts."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.finite_fields.values import (
    Axis,
    AxisBoundMatrix,
    FiniteFieldPresentation,
    ProjectivePoint,
)
from jacobian.math.groups.actions._models import FinitePermutationAction
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix

MAX_LINEAR_GROUP_PRIME = 1_000_003
MAX_LINEAR_GROUP_DIMENSION = 12
MAX_LINEAR_GROUP_GENERATORS = 2 * (MAX_LINEAR_GROUP_DIMENSION - 1) + 1
MAX_LINEAR_GROUP_GENERATOR_CELLS = 4096
MAX_NATURAL_VECTOR_ACTION_SIZE = 50
MAX_LINEAR_GROUP_ORDER_DIGITS = 2048
MAX_EXTENSION_FIELD_LINEAR_GROUP_GENERATORS = (
    2 * (MAX_LINEAR_GROUP_DIMENSION - 1) * 16 + 1
)
MAX_PROJECTIVE_POINT_ACTION_SIZE = 50

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


def _require_extension_space(
    presentation: FiniteFieldPresentation, vector_axis: Axis
) -> None:
    if presentation.degree < 2:
        raise _validation_error(
            "extension_field_required",
            "extension-field matrix groups require presentation degree at least two",
        )
    if not 1 <= len(vector_axis.labels) <= MAX_LINEAR_GROUP_DIMENSION:
        raise _validation_error(
            "dimension_bound",
            f"the vector-space dimension must be between 1 and {MAX_LINEAR_GROUP_DIMENSION}",
        )


class ExtensionFieldLinearGroupRequest(StrictModel):
    """An extension field and labelled coordinate space for a full matrix group."""

    presentation: FiniteFieldPresentation
    vector_axis: Axis = Field(
        description=(
            "Ordered coordinate axis; extension-field matrix-group constructors "
            f"support dimensions 1 through {MAX_LINEAR_GROUP_DIMENSION}."
        )
    )

    @model_validator(mode="after")
    def require_admitted_extension_space(self) -> Self:
        _require_extension_space(self.presentation, self.vector_axis)
        return self


ExtensionGeneratorFamily = Annotated[
    tuple[AxisBoundMatrix, ...],
    Field(min_length=1, max_length=MAX_EXTENSION_FIELD_LINEAR_GROUP_GENERATORS),
]


class _ExtensionFieldLinearGroup(StrictModel):
    """A named full linear group over one exact extension-field coordinate axis."""

    presentation: FiniteFieldPresentation
    vector_axis: Axis
    order: ExactInteger
    generators: ExtensionGeneratorFamily

    @model_validator(mode="before")
    @classmethod
    def require_raw_generator_envelope(cls, data: object) -> object:
        data = canonicalize_json_containers(data)
        if not isinstance(data, dict):
            return data
        axis = data.get("vector_axis")
        labels = axis.get("labels") if isinstance(axis, dict) else None
        generators = data.get("generators")
        if not isinstance(labels, (list, tuple)) or not isinstance(
            generators, (list, tuple)
        ):
            return data
        dimension = len(labels)
        if dimension > MAX_LINEAR_GROUP_DIMENSION:
            raise _validation_error(
                "dimension_bound",
                "the vector-space dimension exceeds its supported bound",
            )
        if len(generators) > MAX_EXTENSION_FIELD_LINEAR_GROUP_GENERATORS:
            raise _validation_error(
                "generator_count_bound",
                "the canonical generator family exceeds its structural bound",
            )
        if dimension and len(generators) * dimension * dimension > (
            MAX_LINEAR_GROUP_GENERATOR_CELLS
        ):
            raise _validation_error(
                "generator_cells_exceeded",
                "the generator matrices exceed the aggregate cell bound",
            )
        for generator in generators:
            entries = generator.get("entries") if isinstance(generator, dict) else None
            if not isinstance(entries, (list, tuple)):
                continue
            if len(entries) > dimension or any(
                isinstance(row, (list, tuple)) and len(row) > dimension
                for row in entries
            ):
                raise _validation_error(
                    "generator_shape_mismatch",
                    "a generator matrix exceeds the retained square dimension",
                )
        return data

    @model_validator(mode="after")
    def require_structural_group_context(self) -> Self:
        _require_extension_space(self.presentation, self.vector_axis)
        if self.order < 1:
            raise _validation_error(
                "order_not_positive", "group order must be positive"
            )
        dimension = len(self.vector_axis.labels)
        cells = 0
        for generator in self.generators:
            if (
                generator.presentation != self.presentation
                or generator.row_axis != self.vector_axis
                or generator.column_axis != self.vector_axis
            ):
                raise _validation_error(
                    "generator_context_mismatch",
                    "every generator must use the group's field and coordinate axes",
                )
            cells += dimension * dimension
        if cells > MAX_LINEAR_GROUP_GENERATOR_CELLS:
            raise _validation_error(
                "generator_cells_exceeded",
                "the generator matrices exceed the aggregate cell bound",
            )
        return self


class ExtensionFieldGeneralLinearGroup(_ExtensionFieldLinearGroup):
    """The complete GL(V) over a presented extension field."""


class ExtensionFieldSpecialLinearGroup(_ExtensionFieldLinearGroup):
    """The determinant-one subgroup SL(V) over a presented extension field."""

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


class ExtensionFieldGeneralLinearProjectiveActionRequest(StrictModel):
    group: ExtensionFieldGeneralLinearGroup


class ExtensionFieldSpecialLinearProjectiveActionRequest(StrictModel):
    group: ExtensionFieldSpecialLinearGroup


def _require_projective_action_axis(
    presentation: FiniteFieldPresentation,
    vector_axis: Axis,
    points: tuple[ProjectivePoint, ...],
    action: FinitePermutationAction,
    generators: tuple[AxisBoundMatrix, ...],
) -> None:
    if not 1 <= len(points) <= MAX_PROJECTIVE_POINT_ACTION_SIZE:
        raise _validation_error(
            "projective_point_axis_bound",
            "the projective point axis must contain between 1 and 50 points",
        )
    if any(
        point.presentation != presentation or point.axis != vector_axis
        for point in points
    ):
        raise _validation_error(
            "projective_point_context_mismatch",
            "projective points must use the group's field presentation and vector axis",
        )
    labels = tuple(
        "["
        + ";".join(
            ",".join(map(str, coordinate.coordinates))
            for coordinate in point.coordinates
        )
        + "]"
        for point in points
    )
    if action.domain != labels:
        raise _validation_error(
            "projective_action_axis_mismatch",
            "the permutation action domain must equal the canonical projective-point labels",
        )
    if len(action.generators) != len(generators):
        raise _validation_error(
            "projective_action_generator_count_mismatch",
            "the projective action must retain one permutation per matrix generator",
        )


class ExtensionFieldGeneralLinearProjectiveAction(StrictModel):
    group: ExtensionFieldGeneralLinearGroup
    points: tuple[ProjectivePoint, ...] = Field(
        min_length=1, max_length=MAX_PROJECTIVE_POINT_ACTION_SIZE
    )
    action: FinitePermutationAction

    @model_validator(mode="after")
    def require_projective_axis(self) -> Self:
        _require_projective_action_axis(
            self.group.presentation,
            self.group.vector_axis,
            self.points,
            self.action,
            self.group.generators,
        )
        return self


class ExtensionFieldSpecialLinearProjectiveAction(StrictModel):
    group: ExtensionFieldSpecialLinearGroup
    points: tuple[ProjectivePoint, ...] = Field(
        min_length=1, max_length=MAX_PROJECTIVE_POINT_ACTION_SIZE
    )
    action: FinitePermutationAction

    @model_validator(mode="after")
    def require_projective_axis(self) -> Self:
        _require_projective_action_axis(
            self.group.presentation,
            self.group.vector_axis,
            self.points,
            self.action,
            self.group.generators,
        )
        return self


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
    "ExtensionFieldGeneralLinearGroup",
    "ExtensionFieldGeneralLinearProjectiveAction",
    "ExtensionFieldGeneralLinearProjectiveActionRequest",
    "ExtensionFieldLinearGroupRequest",
    "ExtensionFieldSpecialLinearGroup",
    "ExtensionFieldSpecialLinearProjectiveAction",
    "ExtensionFieldSpecialLinearProjectiveActionRequest",
    "GeneralLinearNaturalActionRequest",
    "PrimeFieldGeneralLinearGroup",
    "PrimeFieldGeneralLinearNaturalAction",
    "PrimeFieldLinearGroupRequest",
    "PrimeFieldSpecialLinearGroup",
    "PrimeFieldSpecialLinearNaturalAction",
    "SpecialLinearNaturalActionRequest",
]
