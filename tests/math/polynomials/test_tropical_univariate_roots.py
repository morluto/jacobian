from __future__ import annotations

from fractions import Fraction
from itertools import combinations

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.tropical._models import UnivariateRootsRequest
from jacobian.math.polynomials.tropical._tools import compute_univariate_roots
from jacobian.math.polynomials.tropical.values import (
    TropicalPolynomial,
    TropicalPolynomialTerm,
    TropicalScalar,
    TropicalSemiring,
)


def _poly(
    values: tuple[tuple[int, int | Fraction], ...],
    convention: str = "MIN_PLUS",
    base: str = "QQ",
) -> TropicalPolynomial:
    semiring = TropicalSemiring(convention=convention, base=base)  # type: ignore[arg-type]
    return TropicalPolynomial(
        semiring=semiring,
        variables=("x",),
        terms=tuple(
            TropicalPolynomialTerm(
                exponents=(exponent,),
                coefficient=TropicalScalar(
                    semiring=semiring,
                    kind="FINITE",
                    value=CanonicalRational.from_fraction(Fraction(coefficient)),
                ),
            )
            for exponent, coefficient in sorted(values)
        ),
    )


def _oracle(
    poly: TropicalPolynomial,
) -> list[tuple[Fraction, tuple[int, ...], int, int]]:
    """Brute-force all line crossings, then test exact signs on both sides."""
    lines = [
        (term.exponents[0], term.coefficient.value.as_fraction()) for term in poly.terms
    ]
    sign = 1 if poly.semiring.convention == "MIN_PLUS" else -1
    candidates = sorted(
        {Fraction(b2 - b1, m1 - m2) for (m1, b1), (m2, b2) in combinations(lines, 2)}
    )
    roots = []
    for index, value in enumerate(candidates):
        tied_value = min(sign * (m * value + b) for m, b in lines)
        tied = tuple(
            sorted(m for m, b in lines if sign * (m * value + b) == tied_value)
        )
        if len(tied) < 2:
            continue
        left_probe = (candidates[index - 1] + value) / 2 if index else value - 1
        right_probe = (
            (value + candidates[index + 1]) / 2
            if index + 1 < len(candidates)
            else value + 1
        )
        left_value = min(sign * (m * left_probe + b) for m, b in lines)
        right_value = min(sign * (m * right_probe + b) for m, b in lines)
        left_slope = next(
            m for m, b in lines if sign * (m * left_probe + b) == left_value
        )
        right_slope = next(
            m for m, b in lines if sign * (m * right_probe + b) == right_value
        )
        if left_slope != right_slope:
            roots.append((value, tied, left_slope, right_slope))
    return roots


@pytest.mark.parametrize("convention", ["MIN_PLUS", "MAX_PLUS"])
def test_roots_match_independent_pairwise_sign_oracle(convention: str) -> None:
    fixtures = (
        ((0, 0), (1, 1)),
        ((0, 2), (2, 0), (4, 3)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, Fraction(1, 3)), (1, Fraction(-2, 5)), (3, Fraction(7, 4))),
        ((0, 4), (1, -3), (2, 5), (4, -7), (6, 8)),
    )
    for fixture in fixtures:
        profile = compute_univariate_roots(
            UnivariateRootsRequest(polynomial=_poly(fixture, convention))
        )
        expected = _oracle(profile.source)
        actual = [
            (
                root.value.as_fraction(),
                root.active_exponents,
                root.left_slope,
                root.right_slope,
            )
            for root in profile.roots
        ]
        assert actual == expected
        assert len(profile.intervals) == len(profile.roots) + 1
        assert profile.intervals[0].lower is None
        assert profile.intervals[-1].upper is None
        for root in profile.roots:
            assert root.multiplicity == abs(root.right_slope - root.left_slope)


def test_three_way_tie_is_reported_even_when_middle_line_has_no_open_interval() -> None:
    profile = compute_univariate_roots(
        UnivariateRootsRequest(polynomial=_poly(((0, 0), (1, 0), (2, 0))))
    )

    assert len(profile.roots) == 1
    assert profile.roots[0].value.as_fraction() == 0
    assert profile.roots[0].active_exponents == (0, 1, 2)
    assert profile.roots[0].multiplicity == 2
    assert tuple(interval.active_exponent for interval in profile.intervals) == (2, 0)


def test_constant_monomial_and_zero_polynomial_are_explicit() -> None:
    constant = compute_univariate_roots(
        UnivariateRootsRequest(polynomial=_poly(((0, 7),)))
    )
    monomial = compute_univariate_roots(
        UnivariateRootsRequest(polynomial=_poly(((3, 7),)))
    )
    zero_poly = _poly(())
    zero = compute_univariate_roots(UnivariateRootsRequest(polynomial=zero_poly))

    assert constant.kind == monomial.kind == "FINITE_PROFILE"
    assert constant.roots == monomial.roots == ()
    assert (constant.intervals[0].active_exponent, constant.intervals[0].slope) == (
        0,
        0,
    )
    assert (monomial.intervals[0].active_exponent, monomial.intervals[0].slope) == (
        3,
        3,
    )
    assert (zero.kind, zero.intervals, zero.roots) == ("ZERO_POLYNOMIAL", (), ())


def test_nonunivariate_polynomial_is_rejected() -> None:
    poly = _poly(((0, 0),))
    multivariate = TropicalPolynomial(
        semiring=poly.semiring,
        variables=("x", "y"),
        terms=(
            TropicalPolynomialTerm(
                exponents=(0, 0), coefficient=poly.terms[0].coefficient
            ),
        ),
    )
    with pytest.raises(OperationDomainValidationError, match="univariate"):
        compute_univariate_roots(UnivariateRootsRequest(polynomial=multivariate))


def test_pairwise_work_is_admitted_before_crossover_arithmetic() -> None:
    poly = _poly(tuple((exponent, exponent * exponent) for exponent in range(363)))
    with pytest.raises(OperationResourceAdmissionError, match="crossover work"):
        compute_univariate_roots(UnivariateRootsRequest(polynomial=poly))


def test_result_byte_estimate_is_admitted_before_crossover_arithmetic() -> None:
    large = 10**3_999
    poly = _poly(tuple((exponent, large + exponent) for exponent in range(360)))
    with pytest.raises(OperationResourceAdmissionError, match="result-byte bound"):
        compute_univariate_roots(UnivariateRootsRequest(polynomial=poly))
