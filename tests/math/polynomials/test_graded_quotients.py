from __future__ import annotations

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.graded.operations import (
    hilbert_function,
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
