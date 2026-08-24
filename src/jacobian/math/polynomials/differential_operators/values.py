"""Canonical constant-coefficient differential operators over ``QQ``."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_TERMS,
    MAX_POLYNOMIAL_VARIABLES,
    PolynomialVariable,
)

MAX_DIFFERENTIAL_ORDER = (1 << 53) - 1

DifferentialOrder = Annotated[
    StrictInt,
    Field(ge=0, le=MAX_DIFFERENTIAL_ORDER),
]


class DifferentialOperatorTerm(StrictModel):
    """One nonzero rational multiple of a partial-derivative multi-index."""

    coefficient: CanonicalRational
    orders: tuple[DifferentialOrder, ...] = Field(
        min_length=1,
        max_length=MAX_POLYNOMIAL_VARIABLES,
        description=(
            "Derivative orders on the operator's complete ordered variable axis. "
            "For variables (x, y), orders (2, 1) denotes partial_x^2 partial_y. "
            "Each order stays inside the strict-JSON interoperable integer range."
        ),
        examples=[(2, 1)],
    )

    @model_validator(mode="after")
    def require_nonzero_term(self) -> Self:
        if self.coefficient.as_fraction() == 0:
            raise ValueError("zero differential-operator terms must be omitted")
        return self


class ConstantCoefficientDifferentialOperator(StrictModel):
    """A sparse element of ``QQ[partial_1, ..., partial_n]``.

    The declared variable order identifies each derivative slot. Terms are
    already combined and listed in descending lexicographic order of their
    multi-indices. The empty term tuple is the zero operator on that axis.
    """

    differential_operator_schema_version: Literal["1"] = "1"
    domain: Literal["QQ"] = "QQ"
    variables: tuple[PolynomialVariable, ...] = Field(
        min_length=1,
        max_length=MAX_POLYNOMIAL_VARIABLES,
        description=(
            "Complete ordered polynomial-variable axis; derivative-order slots "
            "are interpreted in exactly this order."
        ),
        examples=[("x", "y")],
    )
    terms: tuple[DifferentialOperatorTerm, ...] = Field(
        default=(),
        max_length=MAX_POLYNOMIAL_TERMS,
        description=(
            "Nonzero terms in descending lexicographic order of their derivative "
            "multi-indices. Equal multi-indices must already be combined."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_operator(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("differential-operator variables must be unique")
        if any(len(term.orders) != len(self.variables) for term in self.terms):
            raise ValueError(
                "every derivative multi-index must match the declared variable order"
            )
        orders = tuple(term.orders for term in self.terms)
        if len(set(orders)) != len(orders):
            raise ValueError("derivative multi-indices must be unique")
        if orders != tuple(sorted(orders, reverse=True)):
            raise ValueError(
                "differential-operator terms must use descending lexicographic order"
            )
        return self


__all__ = [
    "ConstantCoefficientDifferentialOperator",
    "DifferentialOperatorTerm",
]
