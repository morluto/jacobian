"""Exact materialization of the normalized bivariate degree-(2,3) slice."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Literal

from jacobian.canonical import format_canonical_integer
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.nullstellensatz import (
    BoundedRationalPolynomial,
    BoundedRationalPolynomialTerm,
    JacobianDegreeChart,
    NamedBoundedRationalPolynomial,
    NormalizedJacobianDegreeSliceSystem,
)

VARIABLES = (
    "a20",
    "a11",
    "a02",
    "b20",
    "b11",
    "b02",
    "b30",
    "b21",
    "b12",
    "b03",
    "t",
)
QUADRATIC_TOP: tuple[Literal["a20", "a11", "a02"], ...] = (
    "a20",
    "a11",
    "a02",
)
CUBIC_TOP: tuple[Literal["b30", "b21", "b12", "b03"], ...] = (
    "b30",
    "b21",
    "b12",
    "b03",
)
GENERATOR_IDS = ("j30", "j21", "j12", "j03", "j20", "j11", "j02", "j10", "j01")


def _exponents(monomial: Mapping[str, int]) -> tuple[int, ...]:
    return tuple(monomial.get(variable, 0) for variable in VARIABLES)


def _polynomial(
    terms: tuple[tuple[int, Mapping[str, int]], ...],
) -> BoundedRationalPolynomial:
    combined: dict[tuple[int, ...], Fraction] = {}
    for coefficient, monomial in terms:
        exponent = _exponents(monomial)
        combined[exponent] = combined.get(exponent, Fraction(0)) + coefficient
    return BoundedRationalPolynomial(
        terms=tuple(
            BoundedRationalPolynomialTerm(
                coefficient=CanonicalRational(
                    num=format_canonical_integer(coefficient.numerator),
                    den=format_canonical_integer(coefficient.denominator),
                ),
                exponents=exponent,
            )
            for exponent, coefficient in sorted(combined.items(), reverse=True)
            if coefficient
        )
    )


def _base_generators() -> tuple[NamedBoundedRationalPolynomial, ...]:
    formulas: tuple[tuple[str, tuple[tuple[int, Mapping[str, int]], ...]], ...] = (
        ("j30", ((2, {"a20": 1, "b21": 1}), (-3, {"a11": 1, "b30": 1}))),
        (
            "j21",
            (
                (4, {"a20": 1, "b12": 1}),
                (-1, {"a11": 1, "b21": 1}),
                (-6, {"a02": 1, "b30": 1}),
            ),
        ),
        (
            "j12",
            (
                (6, {"a20": 1, "b03": 1}),
                (1, {"a11": 1, "b12": 1}),
                (-4, {"a02": 1, "b21": 1}),
            ),
        ),
        ("j03", ((3, {"a11": 1, "b03": 1}), (-2, {"a02": 1, "b12": 1}))),
        (
            "j20",
            (
                (1, {"b21": 1}),
                (2, {"a20": 1, "b11": 1}),
                (-2, {"a11": 1, "b20": 1}),
            ),
        ),
        (
            "j11",
            (
                (2, {"b12": 1}),
                (4, {"a20": 1, "b02": 1}),
                (-4, {"a02": 1, "b20": 1}),
            ),
        ),
        (
            "j02",
            (
                (3, {"b03": 1}),
                (2, {"a11": 1, "b02": 1}),
                (-2, {"a02": 1, "b11": 1}),
            ),
        ),
        ("j10", ((1, {"b11": 1}), (2, {"a20": 1}))),
        ("j01", ((2, {"b02": 1}), (1, {"a11": 1}))),
    )
    return tuple(
        NamedBoundedRationalPolynomial(
            polynomial_id=generator_id,
            polynomial=_polynomial(terms),
        )
        for generator_id, terms in formulas
    )


def materialize_degree_23_system() -> NormalizedJacobianDegreeSliceSystem:
    """Return the complete 12-chart exact-degree system over QQ."""

    base = _base_generators()
    charts = []
    for quadratic in QUADRATIC_TOP:
        for cubic in CUBIC_TOP:
            chart_id = f"{quadratic}-{cubic}"
            rabinowitsch = NamedBoundedRationalPolynomial(
                polynomial_id="rabinowitsch",
                polynomial=_polynomial(
                    (
                        (1, {"t": 1, quadratic: 1, cubic: 1}),
                        (-1, {}),
                    )
                ),
            )
            charts.append(
                JacobianDegreeChart(
                    chart_id=chart_id,
                    selected_quadratic_coefficient=quadratic,
                    selected_cubic_coefficient=cubic,
                    variables=VARIABLES,
                    generators=(*base, rabinowitsch),
                )
            )
    return NormalizedJacobianDegreeSliceSystem(charts=tuple(charts))


__all__ = [
    "CUBIC_TOP",
    "GENERATOR_IDS",
    "QUADRATIC_TOP",
    "VARIABLES",
    "materialize_degree_23_system",
]
