"""Canonical rational Laurent-polynomial multiplication."""

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationMatchRequest,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials._laurent import rational_laurent_multiply
from jacobian.math.polynomials.values import (
    RationalLaurentPolynomial,
    RationalLaurentPolynomialTerm,
)


def term(coefficient: int, *exponents: int) -> RationalLaurentPolynomialTerm:
    return RationalLaurentPolynomialTerm(
        coefficient=CanonicalRational(num=coefficient, den=1), exponents=exponents
    )


def test_signed_exponents_and_exact_cancellation() -> None:
    left = RationalLaurentPolynomial(variables=("x",), terms=(term(1, 1), term(1, -1)))
    right = RationalLaurentPolynomial(
        variables=("x",), terms=(term(1, 1), term(-1, -1))
    )

    product = rational_laurent_multiply(left, right)

    assert product.terms == (term(1, 2), term(-1, -2))
    assert (
        RationalLaurentPolynomial.model_validate_json(product.model_dump_json())
        == product
    )


def test_zero_preserves_parent_axis() -> None:
    zero = RationalLaurentPolynomial(variables=("x", "y"), terms=())
    monomial = RationalLaurentPolynomial(variables=("x", "y"), terms=(term(2, -3, 4),))

    assert rational_laurent_multiply(zero, monomial) == zero


def test_monomial_operand_shifts_and_scales_sparse_support() -> None:
    monomial = RationalLaurentPolynomial(variables=("x", "y"), terms=(term(2, -3, 4),))
    source = RationalLaurentPolynomial(
        variables=("x", "y"), terms=(term(3, 2, -1), term(-1, 0, -2))
    )

    result = rational_laurent_multiply(monomial, source)

    assert result.terms == (term(6, -1, 3), term(-2, -3, 2))


def test_axis_mismatch_rejects_before_convolution() -> None:
    left = RationalLaurentPolynomial(variables=("x",), terms=(term(1, 0),))
    right = RationalLaurentPolynomial(variables=("y",), terms=(term(1, 0),))

    with pytest.raises(OperationDomainValidationError, match="ordered variable axis"):
        rational_laurent_multiply(left, right)


def test_coefficient_growth_is_rejected_before_convolution() -> None:
    coefficient = 10**20_000
    left = RationalLaurentPolynomial(variables=("x",), terms=(term(coefficient, 0),))
    right = RationalLaurentPolynomial(variables=("x",), terms=(term(coefficient, 0),))
    with pytest.raises(OperationResourceAdmissionError):
        rational_laurent_multiply(left, right)


def test_catalog_discovers_laurent_multiplication_vocabulary() -> None:
    result = Catalog.open().match(
        OperationMatchRequest(need="rational Laurent polynomial multiplication")
    )

    assert (
        result.matches[0].operation_id == "polynomial.laurent.rational.multiply.compute"
    )
