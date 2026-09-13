"""Catalog-owned admission for rational Laurent multiplication."""

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    RationalLaurentPolynomial,
    RationalLaurentPolynomialTerm,
)


def _term(coefficient: int, *exponents: int) -> RationalLaurentPolynomialTerm:
    return RationalLaurentPolynomialTerm(
        coefficient=CanonicalRational(num=coefficient, den=1), exponents=exponents
    )


def test_catalog_admits_exponent_growth_before_convolution_bound() -> None:
    operation = Catalog.open().operation("polynomial.laurent.rational.multiply.compute")
    assert operation is not None

    left_count = 65
    right_count = 64
    left = RationalLaurentPolynomial(
        variables=("x",),
        terms=tuple(
            _term(1, MAX_POLYNOMIAL_EXPONENT - index) for index in range(left_count)
        ),
    )
    right = RationalLaurentPolynomial(
        variables=("x",),
        terms=tuple(_term(1, index) for index in range(right_count - 1, -1, -1)),
    )
    request = operation.request_type(left=left, right=right)

    with pytest.raises(OperationResourceAdmissionError) as error:
        operation.run(request)

    assert error.value.errors()[0]["type"] == "polynomial.laurent.exponent_growth"
