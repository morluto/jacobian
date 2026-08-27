"""Typed contracts for rational polynomial multiplication."""

from __future__ import annotations

from pydantic import Field, model_validator
from typing import Self

from jacobian._models import StrictModel
from jacobian.math.polynomials.values import RationalPolynomial
from jacobian.math.polynomials._models import _validation_error, require_polynomial_budget

MAX_MULTIPLY_TERMS = 1024
MAX_MULTIPLY_DEGREE = 1000


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
        return self


__all__ = ["RationalPolynomialMultiplyRequest"]
