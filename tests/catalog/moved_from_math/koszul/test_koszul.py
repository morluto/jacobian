"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/koszul/test_koszul.py``. The preamble is carried over verbatim so the moved tests
resolve every name they use.
"""

from __future__ import annotations

import json
from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.koszul import KoszulComplexValue, koszul_complex
from jacobian.math.koszul._models import KoszulComplexRequest
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

OPERATION_ID = "koszul.complex.construct.compute"
_XY = ("x", "y")


def _poly(
    variables: tuple[str, ...], terms: dict[tuple[int, ...], Fraction]
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(coefficient),
                    exponents=exponents,
                )
                for exponents, coefficient in sorted(terms.items(), reverse=True)
            )
        ),
    )


def _monomial(
    variables: tuple[str, ...], exponents: tuple[int, ...], value: int = 1
) -> RationalPolynomial:
    return _poly(variables, {exponents: Fraction(value)})


def _terms(polynomial: RationalPolynomial) -> dict[tuple[int, ...], Fraction]:
    return {
        term.exponents: Fraction(term.coefficient.num, term.coefficient.den)
        for term in polynomial.polynomial.terms
    }


def _dense_cells(matrix) -> dict[tuple[int, int], dict[tuple[int, ...], Fraction]]:
    return {
        (entry.row, entry.column): _terms(entry.polynomial) for entry in matrix.entries
    }


def _replay_differential_squares(value: KoszulComplexValue) -> None:
    """Independently compose consecutive differentials and require zero."""

    cells = tuple(_dense_cells(matrix) for matrix in value.differentials)
    for degree in range(2, len(value.sequence) + 1):
        outer = cells[degree - 2]
        inner = cells[degree - 1]
        middle_columns = {column for (_, column) in inner} | {row for (row, _) in inner}
        for target_row in range(value.basis_sizes[degree - 2]):
            for source_column in range(value.basis_sizes[degree]):
                composed: dict[tuple[int, ...], Fraction] = {}
                for middle in middle_columns:
                    left = outer.get((target_row, middle))
                    right = inner.get((middle, source_column))
                    if left is None or right is None:
                        continue
                    for right_exponents, right_coefficient in right.items():
                        for left_exponents, left_coefficient in left.items():
                            exponents = tuple(
                                a + b
                                for a, b in zip(
                                    left_exponents, right_exponents, strict=True
                                )
                            )
                            total = composed.get(exponents, Fraction(0)) + (
                                left_coefficient * right_coefficient
                            )
                            if total:
                                composed[exponents] = total
                            else:
                                composed.pop(exponents, None)
                assert composed == {}, (degree, target_row, source_column)


def _catalog_tool():
    return next(tool for tool in BUILTIN_TOOLS if tool.operation_id == OPERATION_ID)


def _wire_payload(
    variables: tuple[str, ...], sequence: tuple[RationalPolynomial, ...]
) -> str:
    return KoszulComplexRequest(
        variables=variables, sequence=sequence
    ).model_dump_json()


def test_catalog_example_executes() -> None:
    tool = _catalog_tool()
    expected = koszul_complex(_XY, (_X, _Y))
    assert len(tool.examples) == 1
    for example in tool.examples:
        request = tool.request_type.model_validate_json(
            json.dumps(example.input), strict=True
        )
        assert request.variables == _XY
        assert request.sequence == (_X, _Y)
        assert tool.run(request) == expected


_X = _monomial(_XY, (1, 0))
_Y = _monomial(_XY, (0, 1))
