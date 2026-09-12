"""Canonical rational Laurent-polynomial multiplication."""

import pytest

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationMatchRequest,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials import _laurent as laurent_module
from jacobian.math.polynomials._laurent import rational_laurent_multiply
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
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


def test_unit_monomial_shift_admits_carrier_height_coefficients() -> None:
    tall = 10 ** (MAX_CANONICAL_RATIONAL_DIGITS - 1)
    source = RationalLaurentPolynomial(variables=("x",), terms=(term(tall, 0),))
    unit = RationalLaurentPolynomial(variables=("x",), terms=(term(1, 3),))
    product = rational_laurent_multiply(unit, source)
    assert product.terms == (term(tall, 3),)
    assert rational_laurent_multiply(source, unit) == product


def test_non_unit_monomial_scale_admits_carrier_height_coefficients() -> None:
    tall = 10 ** (MAX_CANONICAL_RATIONAL_DIGITS - 1)
    source = RationalLaurentPolynomial(variables=("x",), terms=(term(tall, 0),))
    scale = RationalLaurentPolynomial(variables=("x",), terms=(term(2, 1),))
    product = rational_laurent_multiply(scale, source)
    assert product.terms == (term(2 * tall, 1),)
    assert rational_laurent_multiply(source, scale) == product


def test_two_term_integer_product_stays_inside_the_output_envelope() -> None:
    scale = 10**8191
    factor = RationalLaurentPolynomial(
        variables=("x",),
        terms=(term(scale, 1), term(scale, 0)),
    )
    product = rational_laurent_multiply(factor, factor)
    square = scale * scale
    assert product.terms == (
        term(square, 2),
        term(2 * square, 1),
        term(square, 0),
    )


def test_forged_operands_are_rejected_with_typed_domain_errors() -> None:
    forged = RationalLaurentPolynomial.model_construct(
        domain="QQ",
        variables=("x",),
        terms=(term(1, 0, 0), term(2, 0)),
    )
    unit = RationalLaurentPolynomial(variables=("x",), terms=(term(1, 1),))
    with pytest.raises(OperationDomainValidationError):
        rational_laurent_multiply(forged, unit)
    with pytest.raises(OperationDomainValidationError):
        rational_laurent_multiply(unit, object())  # type: ignore[arg-type]


def test_oversized_forged_terms_are_rejected_before_copy() -> None:
    unit = RationalLaurentPolynomial(variables=("x",), terms=(term(1, 0),))
    forged = RationalLaurentPolynomial.model_construct(
        domain="QQ",
        variables=("x",),
        terms=(term(1, 0),) * (MAX_POLYNOMIAL_TERMS + 1),
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        rational_laurent_multiply(forged, unit)
    assert exc_info.value.errors()[0]["type"] == "polynomial.laurent.term_bound"


def test_nonmonomial_coefficient_admission_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels: list[str] = []
    monkeypatch.setattr(
        laurent_module, "request_checkpoint", lambda label: labels.append(label)
    )
    left = RationalLaurentPolynomial(variables=("x",), terms=(term(1, 1), term(1, 0)))
    right = RationalLaurentPolynomial(variables=("x",), terms=(term(1, 1), term(-1, 0)))
    product = rational_laurent_multiply(left, right)
    assert product.terms == (term(1, 2), term(-1, 0))
    assert "during Laurent coefficient-height admission" in labels
    assert "during Laurent operand reconstruction" in labels


def test_monomial_multiplication_checkpoints_during_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels: list[str] = []
    monkeypatch.setattr(
        laurent_module, "request_checkpoint", lambda label: labels.append(label)
    )
    source = RationalLaurentPolynomial(
        variables=("x",), terms=tuple(term(1, index) for index in range(4, -1, -1))
    )
    unit = RationalLaurentPolynomial(variables=("x",), terms=(term(1, 1),))
    product = rational_laurent_multiply(unit, source)
    assert [item.exponents for item in product.terms] == [
        (5,),
        (4,),
        (3,),
        (2,),
        (1,),
    ]
    assert "during Laurent monomial height admission" in labels
    assert "during Laurent monomial multiplication" in labels
    assert "during Laurent result construction" in labels
    assert "during Laurent operand reconstruction" in labels


def test_exponent_growth_is_rejected_before_convolution() -> None:
    # Convolution work is 65 * 64 = 4160 > MAX_POLYNOMIAL_TERMS, so a guard
    # that still lived inside the product loop would raise convolution_bound
    # first. The extrema check must win with exponent_growth.
    left_count = 65
    right_count = 64
    assert left_count * right_count > MAX_POLYNOMIAL_TERMS
    left = RationalLaurentPolynomial(
        variables=("x",),
        terms=tuple(
            term(1, MAX_POLYNOMIAL_EXPONENT - index) for index in range(left_count)
        ),
    )
    right = RationalLaurentPolynomial(
        variables=("x",),
        terms=tuple(term(1, index) for index in range(right_count - 1, -1, -1)),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        rational_laurent_multiply(left, right)
    assert error.value.errors()[0]["type"] == "polynomial.laurent.exponent_growth"


def test_catalog_admits_exponent_growth_before_convolution_bound() -> None:
    operation = Catalog.open().operation("polynomial.laurent.rational.multiply.compute")
    assert operation is not None

    left_count = 65
    right_count = 64
    left = RationalLaurentPolynomial(
        variables=("x",),
        terms=tuple(
            term(1, MAX_POLYNOMIAL_EXPONENT - index) for index in range(left_count)
        ),
    )
    right = RationalLaurentPolynomial(
        variables=("x",),
        terms=tuple(term(1, index) for index in range(right_count - 1, -1, -1)),
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


def test_multiaxis_opposite_boundary_exponents_preserve_cancellation() -> None:
    left = RationalLaurentPolynomial(
        variables=("x", "y"), terms=(term(1, 32_768, -32_768),)
    )
    right = RationalLaurentPolynomial(
        variables=("x", "y"), terms=(term(1, -32_768, 32_768),)
    )

    assert rational_laurent_multiply(left, right) == RationalLaurentPolynomial(
        variables=("x", "y"), terms=(term(1, 0, 0),)
    )
