"""Typed wire contracts for polynomial map operations."""

from __future__ import annotations

from typing import Any, Self

import sympy
from pydantic import Field, model_validator
from sympy.polys.polyerrors import CoercionFailed

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalRational


class RationalPolynomialExpr(ContractModel):
    """A rational polynomial as a SymPy-compatible string expression.

    The polynomial is given as a string like "x**2 + 2*y" that sympy can parse.
    Variables are named in the expression string itself.
    """

    expression: str = Field(
        min_length=1,
        max_length=2000,
        description="Polynomial expression with rational coefficients.",
    )

    @model_validator(mode="after")
    def require_polynomial(self) -> Self:
        _require_polynomial_expression(self.expression)
        return self


def _require_polynomial_expression(raw: str) -> Any:
    try:
        expression = sympy.sympify(raw)
    except (sympy.SympifyError, TypeError, SyntaxError) as exc:
        raise ValueError("polynomial expression must be a polynomial") from exc
    symbols = tuple(expression.free_symbols)
    if symbols:
        if not expression.is_polynomial(*symbols):
            raise ValueError("polynomial expression must be a polynomial")
        try:
            sympy.Poly(expression, *symbols, domain=sympy.QQ)
        except (CoercionFailed, sympy.PolynomialError, TypeError, ValueError) as exc:
            raise ValueError(
                "polynomial expression must have rational coefficients"
            ) from exc
    elif not expression.is_rational:
        raise ValueError("polynomial expression must be a polynomial")
    return expression


class VariablePoint(ContractModel):
    """A rational point: ordered variable names and their rational values."""

    variables: tuple[str, ...] = Field(min_length=1, max_length=20)
    values: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def require_matching_lengths(self) -> Self:
        if len(self.variables) != len(self.values):
            raise ValueError("variables and values must have the same length")
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("variable names must be unique")
        return self


class EvalRequest(ContractModel):
    """Evaluate a polynomial at a rational point."""

    polynomial: RationalPolynomialExpr
    point: VariablePoint

    @model_validator(mode="after")
    def require_complete_rational_evaluation(self) -> Self:
        expression = _require_polynomial_expression(self.polynomial.expression)
        free = {str(symbol) for symbol in expression.free_symbols}
        given = set(self.point.variables)
        if not free <= given:
            raise ValueError("evaluation point must cover every free variable")
        return self


class EvalResult(ContractModel):
    """The rational value of the polynomial at the point."""

    value: str


class JacobianRequest(ContractModel):
    """Compute the Jacobian matrix of a polynomial map."""

    input_variables: tuple[str, ...] = Field(min_length=1, max_length=20)
    output_polynomials: tuple[RationalPolynomialExpr, ...] = Field(
        min_length=1, max_length=20
    )


class JacobianResult(ContractModel):
    """The Jacobian matrix as a flat list of entries (row-major order)."""

    n_inputs: int = Field(ge=1)
    n_outputs: int = Field(ge=1)
    entries: tuple[str, ...]


class CompositionRequest(ContractModel):
    """Compose outer(f(g(x)))."""

    outer: RationalPolynomialExpr
    inner: RationalPolynomialExpr
    inner_variable: str
    outer_variable: str


class CompositionResult(ContractModel):
    """The composed polynomial expression."""

    expression: str


__all__ = [
    "CompositionRequest",
    "CompositionResult",
    "EvalRequest",
    "EvalResult",
    "JacobianRequest",
    "JacobianResult",
    "RationalPolynomialExpr",
    "VariablePoint",
]
