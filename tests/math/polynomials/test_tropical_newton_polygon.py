from __future__ import annotations

from fractions import Fraction
from itertools import product

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials.tropical._models import UnivariateNewtonPolygonRequest
from jacobian.math.polynomials.tropical._tools import compute_univariate_newton_polygon
from jacobian.math.polynomials.tropical.operations import (
    tropical_polynomial_univariate_newton_polygon,
)
from jacobian.math.polynomials.tropical.values import (
    TropicalPolynomial,
    TropicalPolynomialTerm,
    TropicalScalar,
    TropicalSemiring,
)


def _polynomial(convention: str, coefficients: tuple[int, ...]) -> TropicalPolynomial:
    semiring = TropicalSemiring(convention=convention, base="ZZ")  # type: ignore[arg-type]
    return TropicalPolynomial(
        semiring=semiring,
        variables=("x",),
        terms=tuple(
            TropicalPolynomialTerm(
                exponents=(exponent,),
                coefficient=TropicalScalar(
                    semiring=semiring,
                    kind="FINITE",
                    value=CanonicalRational.from_integer_ratio(value, 1),
                ),
            )
            for exponent, value in enumerate(coefficients)
        ),
    )


def _pair_oracle(
    coefficients: tuple[int, ...], convention: str
) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    """Brute-force supporting chords, independent of the monotone-chain kernel."""
    sign = 1 if convention == "MIN_PLUS" else -1
    result = []
    for left in range(len(coefficients)):
        for right in range(left + 1, len(coefficients)):
            dy = coefficients[right] - coefficients[left]
            supported = all(
                sign
                * (
                    (coefficients[index] - coefficients[left]) * (right - left)
                    - dy * (index - left)
                )
                >= 0
                for index in range(len(coefficients))
            )
            has_collinear_outside_chord = any(
                (coefficients[index] - coefficients[left]) * (right - left)
                == dy * (index - left)
                for index in (*range(left), *range(right + 1, len(coefficients)))
            )
            if supported and not has_collinear_outside_chord:
                face = tuple(
                    index
                    for index in range(left, right + 1)
                    if (coefficients[index] - coefficients[left]) * (right - left)
                    == dy * (index - left)
                )
                result.append((left, right, face))
    return tuple(result)


@pytest.mark.parametrize("convention", ["MIN_PLUS", "MAX_PLUS"])
def test_newton_hull_matches_independent_supporting_chord_oracle(
    convention: str,
) -> None:
    for coefficients in product(range(3), repeat=4):
        polynomial = _polynomial(convention, coefficients)
        profile = tropical_polynomial_univariate_newton_polygon(polynomial)
        actual = tuple(
            (
                profile.hull_vertices[edge.left_vertex_index].exponent,
                profile.hull_vertices[edge.right_vertex_index].exponent,
                edge.source_term_indices,
            )
            for edge in profile.edges
        )
        assert actual == _pair_oracle(coefficients, convention)
        assert all(
            edge.multiplicity
            == profile.hull_vertices[edge.right_vertex_index].exponent
            - profile.hull_vertices[edge.left_vertex_index].exponent
            for edge in profile.edges
        )


def test_newton_profile_retains_collinear_face_terms_and_exact_roots() -> None:
    polynomial = _polynomial("MIN_PLUS", (0, 0, 0, 0))
    profile = compute_univariate_newton_polygon(
        UnivariateNewtonPolygonRequest(polynomial=polynomial)
    ).profile
    assert tuple(vertex.source_term_index for vertex in profile.hull_vertices) == (
        0,
        3,
    )
    assert profile.edges[0].source_term_indices == (0, 1, 2, 3)
    assert profile.edges[0].slope == CanonicalRational.from_integer_ratio(0, 1)
    assert profile.edges[0].tropical_root == CanonicalRational.from_integer_ratio(0, 1)
    assert profile.edges[0].multiplicity == 3
    restored = type(profile).model_validate_json(profile.model_dump_json())
    assert restored == profile


def test_max_plus_uses_upper_hull_and_negative_slope_roots() -> None:
    profile = tropical_polynomial_univariate_newton_polygon(
        _polynomial("MAX_PLUS", (0, 2, 0))
    )
    assert tuple(vertex.source_term_index for vertex in profile.hull_vertices) == (
        0,
        1,
        2,
    )
    assert tuple(edge.tropical_root.as_fraction() for edge in profile.edges) == (
        Fraction(-2),
        Fraction(2),
    )


def test_newton_profile_retains_rational_coefficient_heights() -> None:
    semiring = TropicalSemiring(convention="MIN_PLUS", base="QQ")
    polynomial = TropicalPolynomial(
        semiring=semiring,
        variables=("x",),
        terms=tuple(
            TropicalPolynomialTerm(
                exponents=(exponent,),
                coefficient=TropicalScalar(
                    semiring=semiring,
                    kind="FINITE",
                    value=CanonicalRational.from_integer_ratio(numerator, denominator),
                ),
            )
            for exponent, numerator, denominator in ((0, 1, 3), (2, 7, 6))
        ),
    )

    profile = tropical_polynomial_univariate_newton_polygon(polynomial)

    assert profile.edges[0].slope.as_fraction() == Fraction(5, 12)
    assert profile.edges[0].tropical_root.as_fraction() == Fraction(-5, 12)
    assert profile.edges[0].multiplicity == 2


def test_newton_profile_rejects_output_growth_before_hull_construction() -> None:
    semiring = TropicalSemiring(convention="MIN_PLUS", base="ZZ")
    huge = CanonicalRational.from_integer_ratio(10**8_191, 1)
    coefficient = TropicalScalar(semiring=semiring, kind="FINITE", value=huge)
    polynomial = TropicalPolynomial(
        semiring=semiring,
        variables=("x",),
        terms=tuple(
            TropicalPolynomialTerm(exponents=(index,), coefficient=coefficient)
            for index in range(300)
        ),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        tropical_polynomial_univariate_newton_polygon(polynomial)
    assert error.value.errors()[0]["type"] == "tropical.newton_polygon_output_budget"
