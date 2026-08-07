"""Isolated strict worker for typed SymPy polynomial normalization."""

from __future__ import annotations

import sys
from typing import Any

import sympy
from pydantic import ValidationError
from sympy import QQ, Poly

from jacobian.canonical import canonicalize_json, loads_strict_json
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.polynomial_expressions import (
    PolynomialAddExpression,
    PolynomialExpressionArtifact,
    PolynomialExpressionNode,
    PolynomialMultiplyExpression,
    PolynomialNegateExpression,
    PolynomialPowerExpression,
    PolynomialRationalExpression,
    PolynomialVariableExpression,
)
from jacobian.contracts.polynomials import (
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)
from jacobian.provider_runtime import (
    SYMPY_POLYNOMIAL_WORKER_PROTOCOL,
    SYMPY_VERSION,
)


def _sympy_expression(
    expression: PolynomialExpressionNode,
    symbols: dict[str, Any],
) -> Any:
    if isinstance(expression, PolynomialRationalExpression):
        return sympy.Rational(expression.value.as_fraction())
    if isinstance(expression, PolynomialVariableExpression):
        return symbols[expression.name]
    if isinstance(expression, PolynomialAddExpression):
        return sympy.Add(
            *(_sympy_expression(operand, symbols) for operand in expression.operands)
        )
    if isinstance(expression, PolynomialMultiplyExpression):
        return sympy.Mul(
            *(_sympy_expression(operand, symbols) for operand in expression.operands)
        )
    if isinstance(expression, PolynomialNegateExpression):
        return -_sympy_expression(expression.operand, symbols)
    if isinstance(expression, PolynomialPowerExpression):
        return sympy.Pow(
            _sympy_expression(expression.base, symbols),
            expression.exponent,
        )
    raise TypeError("unsupported typed polynomial expression")


def _wire_polynomial(polynomial: Poly) -> SparseRationalPolynomial:
    return SparseRationalPolynomial(
        terms=tuple(
            RationalPolynomialTerm(
                coefficient=CanonicalRational(
                    num=str(coefficient.numerator),
                    den=str(coefficient.denominator),
                ),
                exponents=tuple(int(exponent) for exponent in exponents),
            )
            for exponents, coefficient in polynomial.terms()
            if coefficient != 0
        )
    )


def _run(payload: object) -> dict[str, Any]:
    if (
        not isinstance(payload, dict)
        or set(payload) != {"protocol", "expression"}
        or payload.get("protocol") != SYMPY_POLYNOMIAL_WORKER_PROTOCOL
    ):
        raise ValueError("invalid worker request")
    if sympy.__version__ != SYMPY_VERSION:
        raise RuntimeError("unexpected SymPy version")
    expression = PolynomialExpressionArtifact.model_validate(payload["expression"])
    generators = tuple(sympy.Symbol(name) for name in expression.variables)
    symbol_table = dict(zip(expression.variables, generators, strict=True))
    polynomial = Poly(
        _sympy_expression(expression.expression, symbol_table),
        *generators,
        domain=QQ,
    )
    normalized = _wire_polynomial(polynomial)
    return {
        "protocol": SYMPY_POLYNOMIAL_WORKER_PROTOCOL,
        "status": "NORMALIZATION_PRODUCED",
        "backend_version": SYMPY_VERSION,
        "normalized": normalized.model_dump(mode="json"),
    }


def main() -> int:
    try:
        request = loads_strict_json(sys.stdin.buffer.read())
        response = _run(request)
    except (
        ArithmeticError,
        KeyError,
        RuntimeError,
        TypeError,
        ValidationError,
        ValueError,
    ):
        sys.stderr.write("typed polynomial normalization failed\n")
        return 1
    sys.stdout.buffer.write(canonicalize_json(response) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
