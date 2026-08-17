"""Domain-owned request/result models for exact polynomial interpolation."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational

MAX_POINTS = 32


def _component_digits(value: CanonicalRational) -> int:
    """Max decimal digit count of the numerator or denominator of a canonical rational."""

    return max(
        len(value.num.lstrip("-")),
        len(value.den.lstrip("-")),
    )


class RationalPoint(ContractModel):
    """One (x, y) data point for interpolation."""

    x: CanonicalRational
    y: CanonicalRational


class NewtonInterpolationRequest(ContractModel):
    """Newton-form interpolation through given data points."""

    points: tuple[RationalPoint, ...] = Field(min_length=1, max_length=MAX_POINTS)

    @model_validator(mode="after")
    def require_unique_x(self) -> Self:
        xs = [p.x.as_fraction() for p in self.points]
        if len(set(xs)) != len(xs):
            raise ValueError("interpolation x-values must be distinct")
        return self

    @model_validator(mode="after")
    def require_bounded_divided_differences(self) -> Self:
        """Reject inputs whose divided differences would overflow the canonical bound.

        Every divided difference of order ``j`` is a rational function of the
        inputs in which no single variable appears with degree greater than the
        number of points ``n``.  When each input component is bounded to ``B``
        decimal digits, every divided difference therefore fits within ``n * B``
        digits.  Requiring ``n * B <= MAX_CANONICAL_RATIONAL_DIGITS`` guarantees
        every divided difference, and hence every coefficient of the
        interpolating polynomial, is representable as a canonical rational.
        """

        n = len(self.points)
        per_point_bound = MAX_CANONICAL_RATIONAL_DIGITS // n
        for point in self.points:
            for component in (point.x, point.y):
                if _component_digits(component) > per_point_bound:
                    raise ValueError(
                        "interpolation input exceeds the divided-difference bound"
                    )
        return self


class MultipointEvaluationRequest(ContractModel):
    """Evaluate a polynomial at multiple points simultaneously."""

    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_POINTS + 1
    )
    evaluation_points: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_POINTS
    )

    @model_validator(mode="after")
    def require_bounded_evaluation_results(self) -> Self:
        """Reject inputs whose Horner evaluation results would overflow the canonical bound.

        Horner evaluation of a degree ``d`` polynomial is a sum of at most
        ``d + 1`` products, each the product of one coefficient and a power of
        the evaluation point of degree at most ``d``.  When every coefficient
        and evaluation point is bounded to ``B`` decimal digits, each
        evaluation result therefore fits within ``(d + 1) * B`` digits.
        Requiring ``(d + 1) * B <= MAX_CANONICAL_RATIONAL_DIGITS`` (where
        ``d + 1`` is the coefficient count, which dominates the point count)
        guarantees every result value is representable as a canonical rational.
        """

        degree_plus_one = len(self.coefficients)
        per_operand_bound = MAX_CANONICAL_RATIONAL_DIGITS // degree_plus_one
        for component in (*self.coefficients, *self.evaluation_points):
            if _component_digits(component) > per_operand_bound:
                raise ValueError("evaluation operand exceeds the Horner-result bound")
        return self


class NewtonInterpolationResult(ContractModel):
    """The interpolating polynomial in coefficient form [a_0, ..., a_n]."""

    coefficients: tuple[CanonicalRational, ...] = Field(min_length=1)
    divided_differences: tuple[CanonicalRational, ...] = Field(min_length=1)
    method: Literal["NEWTON_DIVIDED_DIFFERENCE"] = "NEWTON_DIVIDED_DIFFERENCE"


class MultipointEvaluationResult(ContractModel):
    """Polynomial values at the evaluation points."""

    values: tuple[CanonicalRational, ...] = Field(min_length=1)
    method: Literal["HORNER_EVALUATION"] = "HORNER_EVALUATION"


__all__ = [
    "MultipointEvaluationRequest",
    "MultipointEvaluationResult",
    "NewtonInterpolationRequest",
    "NewtonInterpolationResult",
    "RationalPoint",
]
