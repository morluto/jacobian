"""Canonical quadratic-polynomial values and vectors over ``Z/mZ``."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import DecimalIntegerEncoding
from jacobian._models import StrictModel
from jacobian.math._labels import OpaqueLabel
from jacobian.math.number_theory.quadratic_forms.integral._models import (
    IntegralQuadraticForm,
)

MAX_MODULAR_QUADRATIC_FORM_AXIS = 128
MAX_MODULAR_QUADRATIC_FORM_TERMS = 2_048
MAX_MODULAR_QUADRATIC_FORM_INTEGER_DIGITS = 256
MAX_MODULAR_QUADRATIC_RESULT_DIGITS = 1_000_000

ModularFormInteger = Annotated[
    int,
    DecimalIntegerEncoding(max_digits=MAX_MODULAR_QUADRATIC_FORM_INTEGER_DIGITS),
]


def _error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"quadratic_form.modular.{reason}", message)


class ModularQuadraticCrossTerm(StrictModel):
    """A nonzero canonical residue coefficient of one mixed monomial."""

    left: int = Field(ge=0)
    right: int = Field(ge=0)
    coefficient: ModularFormInteger

    @model_validator(mode="after")
    def ordered_nonzero(self) -> Self:
        if self.left >= self.right:
            raise _error("cross_term_order", "mixed terms require left < right")
        if self.coefficient == 0:
            raise _error("zero_cross_term", "zero mixed residues must be omitted")
        return self


class ModularQuadraticPolynomial(StrictModel):
    """The polynomial ``sum a_i*x_i^2 + sum c_ij*x_i*x_j`` over ``Z/mZ``.

    Coefficients use the canonical integer representatives ``0 <= c < m``.
    The modulus is part of the value's parent; the ordered coordinate axis is
    part of its source and target semantics. This value describes a polynomial
    presentation, not a unique representative of every function ``(Z/mZ)^n ->
    Z/mZ``.
    """

    domain: Literal["Z_MOD_N"] = "Z_MOD_N"
    modulus: ModularFormInteger = Field(ge=1)
    axis: tuple[OpaqueLabel, ...] = Field(max_length=MAX_MODULAR_QUADRATIC_FORM_AXIS)
    diagonal_residues: tuple[ModularFormInteger, ...] = Field(
        max_length=MAX_MODULAR_QUADRATIC_FORM_AXIS
    )
    cross_terms: tuple[ModularQuadraticCrossTerm, ...] = Field(
        max_length=MAX_MODULAR_QUADRATIC_FORM_TERMS
    )

    @model_validator(mode="after")
    def canonical_polynomial(self) -> Self:
        if len(self.axis) > MAX_MODULAR_QUADRATIC_FORM_AXIS:
            raise _error("axis_bound", "modular form axes are limited to 128 labels")
        if any(
            not isinstance(label, str) or not label or len(label) > 128
            for label in self.axis
        ):
            raise _error(
                "axis_label",
                "axis labels must be nonempty strings of at most 128 characters",
            )
        if len(set(self.axis)) != len(self.axis):
            raise _error("axis_unique", "modular form axis labels must be unique")
        if len(self.diagonal_residues) != len(self.axis):
            raise _error("diagonal_length", "one diagonal residue is required per axis")
        if any(not 0 <= value < self.modulus for value in self.diagonal_residues):
            raise _error(
                "coefficient_residue", "diagonal coefficients must lie in [0,m)"
            )
        positions = tuple((term.left, term.right) for term in self.cross_terms)
        if any(right >= len(self.axis) for _, right in positions):
            raise _error("cross_term_range", "mixed-term indices must lie on the axis")
        if positions != tuple(sorted(set(positions))):
            raise _error("cross_term_order", "mixed terms must be unique and ordered")
        if any(not 0 < term.coefficient < self.modulus for term in self.cross_terms):
            raise _error("coefficient_residue", "mixed residues must lie in (0,m)")
        if len(self.axis) + len(self.cross_terms) > MAX_MODULAR_QUADRATIC_FORM_TERMS:
            raise _error("term_bound", "modular polynomial support exceeds its bound")
        return self


class ModularCoordinateVector(StrictModel):
    """A coordinate vector in one explicitly presented ``(Z/mZ)^n``."""

    modulus: ModularFormInteger = Field(ge=1)
    axis: tuple[OpaqueLabel, ...] = Field(max_length=MAX_MODULAR_QUADRATIC_FORM_AXIS)
    coordinates: tuple[ModularFormInteger, ...] = Field(
        max_length=MAX_MODULAR_QUADRATIC_FORM_AXIS
    )

    @model_validator(mode="after")
    def canonical_coordinates(self) -> Self:
        if any(
            not isinstance(label, str) or not label or len(label) > 128
            for label in self.axis
        ):
            raise _error(
                "axis_label",
                "axis labels must be nonempty strings of at most 128 characters",
            )
        if len(self.axis) != len(self.coordinates):
            raise _error("vector_length", "vector coordinates must match its axis")
        if len(set(self.axis)) != len(self.axis):
            raise _error("axis_unique", "vector axis labels must be unique")
        if any(not 0 <= value < self.modulus for value in self.coordinates):
            raise _error("vector_residue", "coordinates must lie in [0,m)")
        return self


class ModularInteger(StrictModel):
    """One residue in the parent ring ``Z/mZ``."""

    modulus: ModularFormInteger = Field(ge=1)
    residue: ModularFormInteger

    @model_validator(mode="after")
    def canonical_residue(self) -> Self:
        if not 0 <= self.residue < self.modulus:
            raise _error("scalar_residue", "residue must lie in [0,m)")
        return self


class ModularReductionRequest(StrictModel):
    """Reduce one integral polynomial's coefficients into a finite residue ring."""

    form: IntegralQuadraticForm
    modulus: ModularFormInteger = Field(
        ge=1,
        description=(
            "Positive modulus m (at most 256 decimal digits); the target ring is "
            "the explicitly retained quotient Z/mZ. Modulus 1 is the zero ring."
        ),
    )


class ModularEvaluationRequest(StrictModel):
    """Evaluate a modular polynomial on a vector in the same parent and axis."""

    polynomial: ModularQuadraticPolynomial
    vector: ModularCoordinateVector

    @model_validator(mode="after")
    def require_target_binding(self) -> Self:
        if self.polynomial.modulus != self.vector.modulus:
            raise _error("parent_mismatch", "polynomial and vector moduli must agree")
        if self.polynomial.axis != self.vector.axis:
            raise _error("axis_mismatch", "polynomial and vector axes must agree")
        return self


__all__ = [
    "MAX_MODULAR_QUADRATIC_FORM_AXIS",
    "MAX_MODULAR_QUADRATIC_FORM_INTEGER_DIGITS",
    "MAX_MODULAR_QUADRATIC_FORM_TERMS",
    "MAX_MODULAR_QUADRATIC_RESULT_DIGITS",
    "ModularCoordinateVector",
    "ModularEvaluationRequest",
    "ModularFormInteger",
    "ModularInteger",
    "ModularQuadraticCrossTerm",
    "ModularQuadraticPolynomial",
    "ModularReductionRequest",
]
