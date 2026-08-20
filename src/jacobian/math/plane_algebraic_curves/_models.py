"""Typed wire contracts for plane algebraic curve operations."""

from __future__ import annotations

import keyword
from typing import Literal, Self

import sympy
from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_VARS = 3
MAX_COEFF = 4096
HOMOGENIZING_COORDINATE = "z"


def _require_valid_variables(variables: tuple[str, ...]) -> None:
    """Reject duplicate, empty, reserved, or non-identifier variable names."""
    seen: set[str] = set()
    for name in variables:
        if not name or not name.isidentifier():
            raise ValueError("variable names must be valid identifiers")
        if keyword.iskeyword(name):
            raise ValueError("variable names must not be Python keywords")
        if name in seen:
            raise ValueError("variable names must be unique")
        seen.add(name)


def _parse_polynomial(raw: str, variables: tuple[str, ...]) -> sympy.Basic:
    """Parse *raw* as a sympy polynomial over *variables* with rational coefficients.

    Converts parse failures and non-polynomial expressions into ValueError so
    that a request the model accepts always yields a typed domain result.
    Coefficients must be rational numbers (integers or rationals over QQ);
    transcendental or symbolic constants such as pi are rejected.
    """
    var_symbols = sympy.symbols(variables)
    var_map = dict(zip(variables, var_symbols, strict=True))
    try:
        expression = sympy.sympify(raw, locals=var_map)
    except (sympy.SympifyError, TypeError, SyntaxError) as exc:
        raise ValueError("polynomial must be a valid expression") from exc
    free = {str(symbol) for symbol in expression.free_symbols}
    undeclared = free - set(variables)
    if undeclared:
        raise ValueError(
            f"polynomial references undeclared variables: {sorted(undeclared)}"
        )
    if not expression.is_polynomial(*var_symbols):
        raise ValueError("polynomial expression must be a polynomial")
    # Validate coefficients are rational (over QQ)
    try:
        poly = sympy.Poly(expression, *var_symbols, domain=sympy.QQ)
    except sympy.CoercionFailed as exc:
        raise ValueError("polynomial coefficients must be rational") from exc
    if poly.domain != sympy.QQ:
        raise ValueError(
            f"polynomial coefficients must be rational, got domain {poly.domain}"
        )
    return expression


def _require_polynomial(raw: str, variables: tuple[str, ...]) -> sympy.Basic:
    _require_valid_variables(variables)
    return _parse_polynomial(raw, variables)


class AffineCurveRequest(StrictModel):
    """An affine plane curve f(x, y) = 0."""

    variables: tuple[str, ...] = Field(min_length=2, max_length=2)
    polynomial: str = Field(min_length=1, max_length=MAX_COEFF)

    @model_validator(mode="after")
    def require_valid_polynomial(self) -> Self:
        _require_polynomial(self.polynomial, self.variables)
        return self


class ProjectiveClosureRequest(StrictModel):
    """Compute the projective closure of an affine curve."""

    variables: tuple[str, ...] = Field(min_length=2, max_length=2)
    polynomial: str = Field(min_length=1, max_length=MAX_COEFF)

    @model_validator(mode="after")
    def require_valid_polynomial(self) -> Self:
        _require_valid_variables(self.variables)
        if HOMOGENIZING_COORDINATE in self.variables:
            raise ValueError(
                f"variable names must not include the reserved homogenizing "
                f"coordinate '{HOMOGENIZING_COORDINATE}'"
            )
        _parse_polynomial(self.polynomial, self.variables)
        return self


class AffineChartRequest(StrictModel):
    """Extract an affine chart from a projective curve."""

    variables: tuple[str, ...] = Field(min_length=3, max_length=3)
    polynomial: str = Field(min_length=1, max_length=MAX_COEFF)
    chart_variable: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_valid_chart(self) -> Self:
        _require_polynomial(self.polynomial, self.variables)
        if self.chart_variable not in self.variables:
            raise ValueError("chart_variable must be one of the projective variables")
        # Validate that the polynomial is homogeneous (all terms have the same total degree)
        var_symbols = sympy.symbols(self.variables)
        poly = sympy.Poly(
            _parse_polynomial(self.polynomial, self.variables),
            *var_symbols,
            domain=sympy.QQ,
        )
        if not poly.is_homogeneous:
            raise ValueError("projective polynomial must be homogeneous")
        return self


# Results


class AffineCurveResult(StrictModel):
    is_valid: bool
    degree: int = Field(ge=0)
    method: Literal["SYMPY_CURVE_CHECK"] = "SYMPY_CURVE_CHECK"


class ProjectiveClosureResult(StrictModel):
    polynomial: str
    variables: tuple[str, ...]
    method: Literal["HOMOGENIZATION"] = "HOMOGENIZATION"


class AffineChartResult(StrictModel):
    polynomial: str
    variables: tuple[str, ...]
    method: Literal["DEHOMOGENIZATION"] = "DEHOMOGENIZATION"


__all__ = [
    "MAX_COEFF",
    "MAX_VARS",
    "AffineChartRequest",
    "AffineChartResult",
    "AffineCurveRequest",
    "AffineCurveResult",
    "ProjectiveClosureRequest",
    "ProjectiveClosureResult",
]
