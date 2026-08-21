"""Typed exact contracts for polynomial interpolation operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian._models import StrictModel
from jacobian.math.polynomial_interpolation_ops._kernel import (
    divided_difference_coefficients,
    evaluate_newton_form,
)

MAX_POINTS = 32
_MAX_RATIONAL_DIGITS = 256


def _require_distinct(nodes: tuple[CanonicalRational, ...]) -> None:
    if len({node.as_integer_ratio() for node in nodes}) != len(nodes):
        raise ValueError("interpolation nodes must be pairwise distinct")


def _require_bounded(values: tuple[CanonicalRational, ...], label: str) -> None:
    for value in values:
        require_bounded_rational(
            value,
            max_digits=_MAX_RATIONAL_DIGITS,
            label=label,
        )


class InterpolationSamples(StrictModel):
    """One bounded graph of a rational-valued function on distinct nodes."""

    nodes: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_POINTS,
    )
    values: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_POINTS,
    )

    @model_validator(mode="after")
    def require_well_formed_samples(self) -> Self:
        if len(self.nodes) != len(self.values):
            raise ValueError("nodes and values must have the same length")
        _require_distinct(self.nodes)
        _require_bounded(self.nodes, "interpolation node")
        _require_bounded(self.values, "interpolation value")
        for coefficient in divided_difference_coefficients(self.nodes, self.values):
            require_bounded_rational(
                CanonicalRational.from_fraction(coefficient),
                max_digits=MAX_CANONICAL_RATIONAL_DIGITS,
                label="derived Newton coefficient",
            )
        return self


class DividedDifferencesRequest(StrictModel):
    samples: InterpolationSamples


class NewtonFormRequest(StrictModel):
    samples: InterpolationSamples


class NewtonForm(StrictModel):
    """A directly evaluable Newton-basis polynomial over ``QQ``."""

    nodes: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_POINTS,
    )
    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_POINTS,
    )

    @model_validator(mode="after")
    def require_basis_shape(self) -> Self:
        if len(self.nodes) != len(self.coefficients):
            raise ValueError("Newton nodes and coefficients must have the same length")
        _require_distinct(self.nodes)
        _require_bounded(self.nodes, "Newton node")
        return self


class NewtonEvaluateRequest(StrictModel):
    newton_form: NewtonForm
    evaluation_point: CanonicalRational

    @model_validator(mode="after")
    def require_bounded_point(self) -> Self:
        require_bounded_rational(
            self.evaluation_point,
            max_digits=_MAX_RATIONAL_DIGITS,
            label="evaluation point",
        )
        require_bounded_rational(
            CanonicalRational.from_fraction(
                evaluate_newton_form(
                    self.newton_form.nodes,
                    self.newton_form.coefficients,
                    self.evaluation_point,
                )
            ),
            max_digits=MAX_CANONICAL_RATIONAL_DIGITS,
            label="derived Newton evaluation",
        )
        return self


class DividedDifferencesResult(StrictModel):
    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_POINTS,
    )
    method: str = "NEWTON_DIVIDED_DIFFERENCES"


NewtonFormResult = NewtonForm


class NewtonEvaluateResult(StrictModel):
    result: CanonicalRational
    method: str = "NEWTON_HORNER"


__all__ = [
    "DividedDifferencesRequest",
    "DividedDifferencesResult",
    "InterpolationSamples",
    "NewtonEvaluateRequest",
    "NewtonEvaluateResult",
    "NewtonForm",
    "NewtonFormRequest",
    "NewtonFormResult",
]
