from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials.tropical import operations as tropical_operations
from jacobian.math.polynomials.tropical._tools import TOOLS
from jacobian.math.polynomials.tropical.operations import (
    tropical_polynomial_active_terms,
)
from jacobian.math.polynomials.tropical.values import (
    TropicalPolynomial,
    TropicalPolynomialTerm,
    TropicalScalar,
    TropicalSemiring,
    TropicalVector,
)


def _poly(convention: str, terms: tuple[tuple[tuple[int, ...], int], ...]):
    semiring = TropicalSemiring(convention=convention, base="QQ")  # type: ignore[arg-type]
    return TropicalPolynomial(
        semiring=semiring,
        variables=("x", "y"),
        terms=tuple(
            TropicalPolynomialTerm(
                exponents=exponents,
                coefficient=TropicalScalar(
                    semiring=semiring,
                    kind="FINITE",
                    value=CanonicalRational.from_integer_ratio(coefficient, 1),
                ),
            )
            for exponents, coefficient in terms
        ),
    )


def _point(poly, x: int | None, y: int | None) -> TropicalVector:
    infinity = (
        "POSITIVE_INFINITY"
        if poly.semiring.convention == "MIN_PLUS"
        else "NEGATIVE_INFINITY"
    )
    return TropicalVector(
        semiring=poly.semiring,
        axis=poly.variables,
        entries=tuple(
            TropicalScalar(
                semiring=poly.semiring,
                kind="FINITE" if coordinate is not None else infinity,
                value=(
                    CanonicalRational.from_integer_ratio(coordinate, 1)
                    if coordinate is not None
                    else None
                ),
            )
            for coordinate in (x, y)
        ),
    )


def _oracle(poly, point):
    candidates: list[tuple[int, Fraction]] = []
    for index, term in enumerate(poly.terms):
        value = Fraction(term.coefficient.value.num, term.coefficient.value.den)
        for exponent, coordinate in zip(term.exponents, point.entries, strict=True):
            if exponent and coordinate.kind != "FINITE":
                break
            if exponent:
                value += exponent * Fraction(coordinate.value.num, coordinate.value.den)
        else:
            candidates.append((index, value))
    if not candidates:
        return None, ()
    extremum = (
        min(value for _, value in candidates)
        if poly.semiring.convention == "MIN_PLUS"
        else max(value for _, value in candidates)
    )
    return extremum, tuple(index for index, value in candidates if value == extremum)


@pytest.mark.parametrize(
    ("convention", "terms", "coords"),
    [
        (
            "MIN_PLUS",
            (((0, 0), 0), ((0, 1), 2), ((1, 0), 0), ((1, 1), 9)),
            (0, 0),
        ),
        (
            "MAX_PLUS",
            (((0, 0), 0), ((0, 1), 2), ((1, 0), 0), ((1, 1), 9)),
            (0, 0),
        ),
        (
            "MIN_PLUS",
            (((0, 0), 3), ((0, 1), 2), ((1, 0), 0), ((1, 1), 9)),
            (2, -1),
        ),
        (
            "MAX_PLUS",
            (((0, 0), 3), ((0, 1), 2), ((1, 0), 0), ((1, 1), 9)),
            (2, -1),
        ),
    ],
)
def test_active_term_indices_match_independent_fraction_oracle(
    convention, terms, coords
):
    polynomial = _poly(convention, terms)
    point = _point(polynomial, *coords)
    expected_value, expected_indices = _oracle(polynomial, point)

    result = tropical_polynomial_active_terms(polynomial, point)

    assert result.polynomial == polynomial
    assert result.point == point
    assert tuple(row.index for row in result.active_terms) == expected_indices
    assert tuple(row.term for row in result.active_terms) == tuple(
        polynomial.terms[index] for index in expected_indices
    )
    assert tuple(row.value.value.as_fraction() for row in result.active_terms) == (
        (expected_value,) * len(expected_indices)
    )
    if expected_value is not None:
        assert result.value.value is not None
        assert result.value.value.as_fraction() == expected_value


def test_active_terms_skip_unfinite_monomials_and_handle_zero_polynomial():
    polynomial = _poly("MIN_PLUS", (((0, 0), 1), ((0, 1), -1), ((1, 0), 0)))
    point = _point(polynomial, 4, None)
    result = tropical_polynomial_active_terms(polynomial, point)
    assert tuple(term.index for term in result.active_terms) == (0,)
    assert result.value.value is not None
    assert result.value.value.as_fraction() == 1

    empty = _poly("MAX_PLUS", ())
    empty_result = tropical_polynomial_active_terms(empty, _point(empty, 0, 0))
    assert empty_result.value.kind == "NEGATIVE_INFINITY"
    assert empty_result.active_terms == ()


def test_active_term_operation_is_published_and_serialization_keeps_source_indices():
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "tropical.polynomial.active_terms.compute"
    )
    example = tool.examples[0]
    request = tool.request_type.model_validate_json(
        encode_strict_json(example.input), strict=True
    )
    result = tool.run(request)
    restored = type(result).model_validate_json(result.model_dump_json())
    assert tuple(row.index for row in restored.active_terms) == (0, 1)
    assert restored.polynomial == request.polynomial


def test_active_term_evaluation_growth_is_rejected_before_expansion(monkeypatch):
    polynomial = _poly("MIN_PLUS", (((100, 0), 0),))
    huge = 10**8_190
    point = TropicalVector(
        semiring=polynomial.semiring,
        axis=polynomial.variables,
        entries=(
            TropicalScalar(
                semiring=polynomial.semiring,
                kind="FINITE",
                value=CanonicalRational.from_integer_ratio(huge, 1),
            ),
            TropicalScalar(
                semiring=polynomial.semiring,
                kind="FINITE",
                value=CanonicalRational.from_integer_ratio(0, 1),
            ),
        ),
    )

    def arithmetic_must_not_start(*_args, **_kwargs):
        raise AssertionError("preflight must precede monomial expansion")

    monkeypatch.setattr(
        tropical_operations, "tropical_scalar_multiply", arithmetic_must_not_start
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        tropical_polynomial_active_terms(polynomial, point)
    assert error.value.errors()[0]["type"] == "tropical.active_term_scalar_growth"


def test_active_term_result_size_is_admitted_before_expansion(monkeypatch):
    semiring = TropicalSemiring(convention="MIN_PLUS", base="QQ")
    coefficient = TropicalScalar(
        semiring=semiring,
        kind="FINITE",
        value=CanonicalRational.from_integer_ratio(10**8_187, 1),
    )
    polynomial = TropicalPolynomial(
        semiring=semiring,
        variables=("x",),
        terms=tuple(
            TropicalPolynomialTerm(exponents=(exponent,), coefficient=coefficient)
            for exponent in range(512)
        ),
    )
    point = TropicalVector(
        semiring=semiring,
        axis=("x",),
        entries=(
            TropicalScalar(
                semiring=semiring,
                kind="FINITE",
                value=CanonicalRational.from_integer_ratio(0, 1),
            ),
        ),
    )

    def arithmetic_must_not_start(*_args, **_kwargs):
        raise AssertionError("output admission must precede monomial expansion")

    monkeypatch.setattr(
        tropical_operations, "tropical_scalar_multiply", arithmetic_must_not_start
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        tropical_polynomial_active_terms(polynomial, point)
    assert error.value.errors()[0]["type"] == "tropical.active_term_output_bytes"
