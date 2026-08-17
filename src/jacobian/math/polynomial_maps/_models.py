"""Typed wire contracts for polynomial map operations."""

from __future__ import annotations

from typing import Any, Self

import sympy
from pydantic import Field, model_validator
from sympy.polys.polyerrors import CoercionFailed

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel


class RationalPolynomialExpr(StrictModel):
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


class VariablePoint(StrictModel):
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


class EvalRequest(StrictModel):
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


class EvalResult(StrictModel):
    """The rational value of the polynomial at the point."""

    value: str


class JacobianRequest(StrictModel):
    """Compute the Jacobian matrix of a polynomial map."""

    input_variables: tuple[str, ...] = Field(min_length=1, max_length=20)
    output_polynomials: tuple[RationalPolynomialExpr, ...] = Field(
        min_length=1, max_length=20
    )

    @model_validator(mode="after")
    def require_unique_and_complete_variables(self) -> Self:
        if len(set(self.input_variables)) != len(self.input_variables):
            raise ValueError("input variables must be unique")
        declared = set(self.input_variables)
        import sympy

        for poly in self.output_polynomials:
            expression = sympy.sympify(poly.expression)
            free = {str(s) for s in expression.free_symbols}
            undeclared = free - declared
            if undeclared:
                raise ValueError(
                    f"output polynomial references undeclared variables: {undeclared}"
                )
        return self


class JacobianResult(StrictModel):
    """The Jacobian matrix as a flat list of entries (row-major order)."""

    n_inputs: int = Field(ge=1)
    n_outputs: int = Field(ge=1)
    entries: tuple[str, ...]


class CompositionRequest(StrictModel):
    """Compose outer(f(g(x)))."""

    outer: RationalPolynomialExpr
    inner: RationalPolynomialExpr
    inner_variable: str
    outer_variable: str

    @model_validator(mode="after")
    def require_valid_composition(self) -> Self:
        import sympy

        outer_expr = sympy.sympify(self.outer.expression)
        outer_free = {str(s) for s in outer_expr.free_symbols}
        if self.outer_variable not in outer_free:
            raise ValueError(
                f"outer variable '{self.outer_variable}' must appear in the outer polynomial"
            )

        inner_expr = sympy.sympify(self.inner.expression)
        inner_free = {str(s) for s in inner_expr.free_symbols}
        if self.inner_variable not in inner_free:
            raise ValueError(
                f"inner variable '{self.inner_variable}' must appear in the inner polynomial"
            )

        return self


class CompositionResult(StrictModel):
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
