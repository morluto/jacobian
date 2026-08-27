"""Typed contracts for rational polynomial multiplication."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.math.polynomials._models import _validation_error
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_TERMS,
    RationalPolynomial,
    require_polynomial_budget,
)

MAX_MULTIPLY_TERMS = 1024
MAX_MULTIPLY_DEGREE = 1000
MAX_MULTIPLY_RESULT_TERMS = MAX_POLYNOMIAL_TERMS


class RationalPolynomialMultiplyRequest(StrictModel):
    """Two rational polynomials in the same variable ring for exact multiplication."""

    left: RationalPolynomial
    right: RationalPolynomial

    @model_validator(mode="after")
    def require_matching_rings_and_budget(self) -> Self:
        if self.left.variables != self.right.variables:
            raise _validation_error("polynomials must use the same ordered variables")
        require_polynomial_budget(
            self.left,
            maximum_terms=MAX_MULTIPLY_TERMS,
            maximum_exponent=MAX_MULTIPLY_DEGREE,
        )
        require_polynomial_budget(
            self.right,
            maximum_terms=MAX_MULTIPLY_TERMS,
            maximum_exponent=MAX_MULTIPLY_DEGREE,
        )
        product_term_work = len(self.left.polynomial.terms) * len(
            self.right.polynomial.terms
        )
        if product_term_work > MAX_MULTIPLY_RESULT_TERMS:
            raise _validation_error(
                "the polynomial product may exceed the canonical term limit"
            )
        return self


__all__ = ["MAX_MULTIPLY_RESULT_TERMS", "RationalPolynomialMultiplyRequest"]
