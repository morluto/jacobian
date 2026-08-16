"""Typed wire contracts for polynomial map operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalRational


class RationalPolynomialExpr(ContractModel):
    """A rational polynomial as a SymPy-compatible string expression.

    The polynomial is given as a string like "x**2 + 2*y" that sympy can parse.
    Variables are named in the expression string itself.
    """

    expression: str = Field(min_length=1, max_length=2000)


class VariablePoint(ContractModel):
    """A rational point: ordered variable names and their rational values."""

    variables: tuple[str, ...] = Field(min_length=1, max_length=20)
    values: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def require_matching_lengths(self) -> Self:
        if len(self.variables) != len(self.values):
            raise ValueError("variables and values must have the same length")
        return self


class EvalRequest(ContractModel):
    """Evaluate a polynomial at a rational point."""

    polynomial: RationalPolynomialExpr
    point: VariablePoint


class EvalResult(ContractModel):
    """The rational value of the polynomial at the point."""

    value: str


class JacobianRequest(ContractModel):
    """Compute the Jacobian matrix of a polynomial map."""

    input_variables: tuple[str, ...] = Field(min_length=1, max_length=20)
    output_polynomials: tuple[RationalPolynomialExpr, ...] = Field(min_length=1, max_length=20)


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
    "RationalPolynomialExpr",
    "VariablePoint",
    "EvalRequest",
    "EvalResult",
    "JacobianRequest",
    "JacobianResult",
    "CompositionRequest",
    "CompositionResult",
]
