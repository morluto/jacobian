"""Typed polynomial quadratic forms over finite fields of characteristic two."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math._labels import OpaqueLabel
from jacobian.math.finite_fields.values import (
    FiniteFieldElement,
    FiniteFieldPresentation,
)

MAX_FINITE_QUADRATIC_FORM_AXIS = 256
MAX_FINITE_QUADRATIC_FORM_TERMS = 4096


def _error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"quadratic_form.characteristic_two.{reason}", message)


class FiniteFieldQuadraticCrossTerm(StrictModel):
    """A nonzero coefficient of x_left*x_right, with left < right."""

    left: int = Field(ge=0)
    right: int = Field(ge=0)
    coefficient: FiniteFieldElement

    @model_validator(mode="after")
    def require_nonzero_ordered_term(self) -> Self:
        if self.left >= self.right:
            raise _error("cross_term_order", "cross terms require left < right")
        if self.coefficient.is_zero:
            raise _error("zero_cross_term", "zero cross terms must be omitted")
        return self


class FiniteFieldQuadraticForm(StrictModel):
    """Polynomial form Q(x)=sum a_i*x_i^2+sum c_ij*x_i*x_j over one field.

    Diagonal square coefficients are retained: in characteristic two their
    contribution to the polar form vanishes, but they remain part of Q.
    """

    domain: Literal["FINITE_FIELD"] = "FINITE_FIELD"
    field: FiniteFieldPresentation
    axis: tuple[OpaqueLabel, ...]
    diagonal_coefficients: tuple[FiniteFieldElement, ...]
    cross_terms: tuple[FiniteFieldQuadraticCrossTerm, ...] = ()

    @model_validator(mode="after")
    def require_canonical_polynomial(self) -> Self:
        if len(self.axis) > MAX_FINITE_QUADRATIC_FORM_AXIS:
            raise _error("axis_bound", "finite-field quadratic-form axis is too large")
        if len(set(self.axis)) != len(self.axis):
            raise _error("axis_unique", "quadratic-form axis labels must be unique")
        if len(self.diagonal_coefficients) != len(self.axis):
            raise _error("diagonal_length", "diagonal coefficients must match the axis")
        if (
            len(self.diagonal_coefficients) + len(self.cross_terms)
            > MAX_FINITE_QUADRATIC_FORM_TERMS
        ):
            raise _error(
                "term_bound", "finite-field quadratic-form support is too large"
            )
        coefficients = (
            *self.diagonal_coefficients,
            *(term.coefficient for term in self.cross_terms),
        )
        if any(value.presentation != self.field for value in coefficients):
            raise _error(
                "coefficient_parent", "all coefficients must use the declared field"
            )
        positions = tuple((term.left, term.right) for term in self.cross_terms)
        if any(right >= len(self.axis) for _, right in positions):
            raise _error("cross_term_range", "cross-term indices must lie on the axis")
        if positions != tuple(sorted(positions)) or len(set(positions)) != len(
            positions
        ):
            raise _error("cross_terms_order", "cross terms must be unique and ordered")
        return self


class FiniteFieldQuadraticVector(StrictModel):
    """A field-valued vector on the form's explicitly ordered axis."""

    field: FiniteFieldPresentation
    axis: tuple[OpaqueLabel, ...]
    coordinates: tuple[FiniteFieldElement, ...]

    @model_validator(mode="after")
    def require_canonical_vector(self) -> Self:
        if len(self.axis) > MAX_FINITE_QUADRATIC_FORM_AXIS:
            raise _error("axis_bound", "finite-field vector axis is too large")
        if len(set(self.axis)) != len(self.axis):
            raise _error("axis_unique", "vector axis labels must be unique")
        if len(self.coordinates) != len(self.axis):
            raise _error("coordinate_length", "coordinates must match the axis")
        if any(value.presentation != self.field for value in self.coordinates):
            raise _error(
                "coordinate_parent", "all coordinates must use the declared field"
            )
        return self


class FiniteFieldQuadraticEvaluationRequest(StrictModel):
    form: FiniteFieldQuadraticForm
    vector: FiniteFieldQuadraticVector

    @model_validator(mode="after")
    def require_matching_context(self) -> Self:
        if self.vector.field != self.form.field or self.vector.axis != self.form.axis:
            raise _error(
                "evaluation_context", "vector field and axis must match the form"
            )
        return self


class FiniteFieldQuadraticPairingRequest(StrictModel):
    form: FiniteFieldQuadraticForm
    left: FiniteFieldQuadraticVector
    right: FiniteFieldQuadraticVector

    @model_validator(mode="after")
    def require_matching_context(self) -> Self:
        if any(
            vector.field != self.form.field or vector.axis != self.form.axis
            for vector in (self.left, self.right)
        ):
            raise _error(
                "pairing_context", "both vector fields and axes must match the form"
            )
        return self


class FiniteFieldQuadraticEvaluationResult(StrictModel):
    form: FiniteFieldQuadraticForm
    vector: FiniteFieldQuadraticVector
    value: FiniteFieldElement

    @model_validator(mode="after")
    def retain_source_parent(self) -> Self:
        if (
            self.vector.field != self.form.field
            or self.vector.axis != self.form.axis
            or self.value.presentation != self.form.field
        ):
            raise _error(
                "evaluation_result_context",
                "result values must retain the form context",
            )
        return self


class FiniteFieldQuadraticPairingResult(StrictModel):
    form: FiniteFieldQuadraticForm
    left: FiniteFieldQuadraticVector
    right: FiniteFieldQuadraticVector
    value: FiniteFieldElement

    @model_validator(mode="after")
    def retain_source_parent(self) -> Self:
        if (
            any(
                vector.field != self.form.field or vector.axis != self.form.axis
                for vector in (self.left, self.right)
            )
            or self.value.presentation != self.form.field
        ):
            raise _error(
                "pairing_result_context", "result values must retain the form context"
            )
        return self
