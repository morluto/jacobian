"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/analysis/approximation/test_approximation.py``. The preamble is carried over verbatim so the moved tests
resolve every name they use.
"""

from __future__ import annotations

import json
from fractions import Fraction
from typing import TypedDict

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.analysis.approximation._models import (
    LagrangeBasisRequest,
    LagrangeBasisResult,
    LagrangeInterpolationData,
    LagrangeInterpolationRequest,
    LagrangeInterpolationResult,
    RationalNodeSet,
)
from jacobian.math.analysis.approximation.operations import (
    lagrange_basis,
    lagrange_interpolate,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def compute_lagrange_basis(request: LagrangeBasisRequest) -> LagrangeBasisResult:
    return lagrange_basis(request.nodes)


def compute_lagrange_interpolation(
    request: LagrangeInterpolationRequest,
) -> LagrangeInterpolationResult:
    return LagrangeInterpolationResult(
        source=LagrangeInterpolationData(nodes=request.nodes, values=request.values),
        polynomial=lagrange_interpolate(request.nodes.nodes, request.values),
    )


def _node(num: str, den: str = "1") -> RationalWire:
    return {"num": num, "den": den}


def _canonical_node(node: RationalWire) -> CanonicalRational:
    return CanonicalRational.model_validate_json(json.dumps(node))


def _node_set(*nodes: RationalWire) -> RationalNodeSet:
    return RationalNodeSet(nodes=tuple(_canonical_node(node) for node in nodes))


def _canonical_values(*values: RationalWire) -> tuple[CanonicalRational, ...]:
    return tuple(_canonical_node(value) for value in values)


def _polynomial_terms(
    polynomial: RationalPolynomial,
) -> dict[tuple[int, ...], Fraction]:
    return {
        term.exponents: term.coefficient.as_fraction()
        for term in polynomial.polynomial.terms
    }


def _assert_validation_code(
    exc_info: pytest.ExceptionInfo[ValidationError], code: str
) -> None:
    assert any(error["type"] == code for error in exc_info.value.errors())


def _canonical(num: int, den: int = 1) -> CanonicalRational:
    return CanonicalRational(num=num, den=den)


def _dense_polynomial(coeffs: list[Fraction]) -> RationalPolynomial:
    terms = tuple(
        RationalPolynomialTerm(
            coefficient=CanonicalRational.from_fraction(c),
            exponents=(exp,),
        )
        for exp, c in sorted(enumerate(coeffs), key=lambda pair: pair[0], reverse=True)
        if c != 0
    )
    return RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(terms=terms),
    )


def test_interpolant_is_published_alongside_basis() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    ids = {tool.operation_id for tool in BUILTIN_TOOLS}
    assert "approximation.lagrange.interpolate.compute" in ids
    assert "approximation.lagrange.basis.compute" in ids


class RationalWire(TypedDict):
    num: str
    den: str
