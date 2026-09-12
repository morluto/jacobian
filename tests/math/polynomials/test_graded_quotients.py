from __future__ import annotations

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.graded.operations import (
    h_vector,
    hilbert_dimension,
    hilbert_function,
    hilbert_multiplicity,
    hilbert_polynomial,
    hilbert_series,
    initial_monomial_ideal,
    standard_monomials,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _ideal(*exponents: tuple[int, ...]) -> RationalPolynomialIdeal:
    variables = tuple("xy"[: len(exponents[0])])
    return RationalPolynomialIdeal(
        variables=variables,
        generators=tuple(
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num=1, den=1),
                            exponents=exponent,
                        ),
                    )
                ),
            )
            for exponent in exponents
        ),
    )


def test_initial_ideal_standard_monomials_and_hilbert_prefix_compose() -> None:
    initial = initial_monomial_ideal(_ideal((2, 0)))
    assert initial.initial_ideal.generators[0].polynomial.terms[0].exponents == (2, 0)
    degree_two = standard_monomials(initial.initial_ideal, 2)
    assert degree_two.monomials == ((1, 1), (0, 2))
    profile = hilbert_function(_ideal((2, 0)), max_degree=4)
    assert profile.values == (1, 2, 2, 2, 2)


def test_standard_monomials_reject_nonhomogeneous_initial_input() -> None:
    variables = ("x", "y")
    nonhomogeneous = RationalPolynomialIdeal(
        variables=variables,
        generators=(
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num=1, den=1),
                            exponents=(1, 0),
                        ),
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num=1, den=1),
                            exponents=(0, 0),
                        ),
                    )
                ),
            ),
        ),
    )
    with pytest.raises(OperationDomainValidationError, match="homogeneous"):
        initial_monomial_ideal(nonhomogeneous)


def test_hilbert_series_polynomial_dimension_multiplicity_and_h_vector() -> None:
    ideal = _ideal((2, 0))
    series = hilbert_series(ideal, prefix_degree=4)
    assert series.prefix == (1, 2, 2, 2, 2)
    assert series.denominator_exponent == 1
    assert series.ambient_denominator_exponent == 2
    assert h_vector(ideal).h_vector == (1, 1)
    assert hilbert_dimension(ideal).dimension == 1
    assert hilbert_multiplicity(ideal).multiplicity == 2
    polynomial = hilbert_polynomial(ideal).polynomial
    assert polynomial.variables == ("m",)
    assert polynomial.polynomial.terms[0].coefficient == CanonicalRational(num=2, den=1)
    assert polynomial.polynomial.terms[0].exponents == (0,)
    assert type(series).model_validate_json(series.model_dump_json()) == series


def test_zero_dimensional_macaulay_fixture_has_length_one_series() -> None:
    ideal = _ideal((1, 0), (0, 1))
    series = hilbert_series(ideal, prefix_degree=3)
    assert series.prefix == (1, 0, 0, 0)
    assert series.denominator_exponent == 0
    assert hilbert_dimension(ideal).dimension == 0
    assert hilbert_multiplicity(ideal).multiplicity == 1
    assert hilbert_polynomial(ideal).polynomial.polynomial.terms[0].exponents == (0,)


def test_hilbert_invariants_are_order_invariant_even_when_initial_generators_differ() -> None:
    variables = ("x", "y")
    ideal = RationalPolynomialIdeal(
        variables=variables,
        generators=(
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num=1, den=1),
                            exponents=(2, 0),
                        ),
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num=1, den=1),
                            exponents=(0, 2),
                        ),
                    )
                ),
            ),
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(
                    terms=(
                        RationalPolynomialTerm(
                            coefficient=CanonicalRational(num=1, den=1),
                            exponents=(1, 1),
                        ),
                    )
                ),
            ),
        ),
    )
    assert hilbert_series(ideal, "lex", prefix_degree=5).prefix == hilbert_series(
        ideal, "grevlex", prefix_degree=5
    ).prefix
