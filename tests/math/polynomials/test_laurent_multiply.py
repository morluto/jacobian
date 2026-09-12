"""Canonical rational Laurent-polynomial multiplication."""

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
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


def test_exponent_growth_is_rejected_before_convolution() -> None:
    limit = 32_768
    left = RationalLaurentPolynomial(variables=("x",), terms=(term(1, limit),))
    right = RationalLaurentPolynomial(variables=("x",), terms=(term(1, 1),))
    with pytest.raises(OperationResourceAdmissionError, match="exponent"):
        rational_laurent_multiply(left, right)


def test_catalog_admits_exponent_growth_before_convolution_bound() -> None:
    operation = Catalog.open().operation("polynomial.laurent.rational.multiply.compute")
    assert operation is not None

    left = RationalLaurentPolynomial(
        variables=("x",),
        terms=tuple(term(1, exponent) for exponent in range(32_768, 30_719, -1)),
    )
    right = RationalLaurentPolynomial(
        variables=("x",), terms=(term(1, 1), term(1, 0))
    )
    request = operation.request_type(left=left, right=right)

    with pytest.raises(OperationResourceAdmissionError) as error:
        operation.run(request)

    assert error.value.errors()[0]["type"] == "polynomial.laurent.exponent_growth"


def test_opposite_boundary_exponents_multiply_to_one() -> None:
    left = RationalLaurentPolynomial(variables=("x",), terms=(term(1, 32_768),))
    right = RationalLaurentPolynomial(variables=("x",), terms=(term(1, -32_768),))
    result = rational_laurent_multiply(left, right)
    assert result == RationalLaurentPolynomial(variables=("x",), terms=(term(1, 0),))
    restored = RationalLaurentPolynomial.model_validate_json(result.model_dump_json())
    assert rational_laurent_multiply(restored, left) == left
