"""Canonical rational Laurent-polynomial multiplication."""

from fractions import Fraction

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


def test_shared_denominator_two_term_product_stays_inside_the_output_envelope() -> None:
    denominator = 10**8191 + 1
    coefficient = CanonicalRational(num=1, den=denominator)
    factor = RationalLaurentPolynomial(
        variables=("x",),
        terms=(
            RationalLaurentPolynomialTerm(coefficient=coefficient, exponents=(1,)),
            RationalLaurentPolynomialTerm(coefficient=coefficient, exponents=(0,)),
        ),
    )
    product = rational_laurent_multiply(factor, factor)
    square = CanonicalRational(num=1, den=denominator * denominator)
    doubled = CanonicalRational(num=2, den=denominator * denominator)
    assert product.terms == (
        RationalLaurentPolynomialTerm(coefficient=square, exponents=(2,)),
        RationalLaurentPolynomialTerm(coefficient=doubled, exponents=(1,)),
        RationalLaurentPolynomialTerm(coefficient=square, exponents=(0,)),
    )


def test_related_denominator_two_term_product_stays_inside_the_output_envelope() -> (
    None
):
    denominator = 10**8191 + 1
    left_coefficient = CanonicalRational(num=1, den=denominator)
    right_coefficient = CanonicalRational(num=1, den=2 * denominator)
    factor = RationalLaurentPolynomial(
        variables=("x",),
        terms=(
            RationalLaurentPolynomialTerm(coefficient=left_coefficient, exponents=(1,)),
            RationalLaurentPolynomialTerm(
                coefficient=right_coefficient, exponents=(0,)
            ),
        ),
    )
    product = rational_laurent_multiply(factor, factor)
    square = CanonicalRational(num=1, den=denominator * denominator)
    cross = CanonicalRational(num=1, den=denominator * denominator)
    constant = CanonicalRational(num=1, den=4 * denominator * denominator)
    assert product.terms == (
        RationalLaurentPolynomialTerm(coefficient=square, exponents=(2,)),
        RationalLaurentPolynomialTerm(coefficient=cross, exponents=(1,)),
        RationalLaurentPolynomialTerm(coefficient=constant, exponents=(0,)),
    )


def test_denominator_product_width_is_measured_exactly() -> None:
    """A representable denominator product at the canonical boundary is admitted.

    ``D * E`` with ``D = 10**16383`` and ``E = 10**16384`` has exactly 32,768
    digits, one less than the sum of the two factors' widths.
    """
    left_denominator = 10**16383
    right_denominator = 10**16384
    left = RationalLaurentPolynomial(
        variables=("x",),
        terms=(
            RationalLaurentPolynomialTerm(
                coefficient=CanonicalRational(num=1, den=left_denominator),
                exponents=(1,),
            ),
            RationalLaurentPolynomialTerm(
                coefficient=CanonicalRational(num=1, den=left_denominator),
                exponents=(0,),
            ),
        ),
    )
    right = RationalLaurentPolynomial(
        variables=("x",),
        terms=(
            RationalLaurentPolynomialTerm(
                coefficient=CanonicalRational(num=1, den=right_denominator),
                exponents=(1,),
            ),
            RationalLaurentPolynomialTerm(
                coefficient=CanonicalRational(num=1, den=right_denominator),
                exponents=(0,),
            ),
        ),
    )
    product = rational_laurent_multiply(left, right)
    assert len(product.terms) == 3
    expected_denominator = left_denominator * right_denominator
    assert product.terms[0].coefficient.den == expected_denominator
    assert expected_denominator == 10 ** (MAX_CANONICAL_RATIONAL_DIGITS - 1)


def test_scaled_numerator_width_is_measured_not_subtracted() -> None:
    """An oversized scaled numerator is refused during semantic admission.

    ``2401 * N1 * N2 + 4`` has 32,769 digits for 9-repdigit ``N1``, ``N2``, so
    the request must be refused before the convolution expands it.
    """
    left = RationalLaurentPolynomial(
        variables=("x",),
        terms=(
            RationalLaurentPolynomialTerm(
                coefficient=CanonicalRational(num=10**16382 - 1, den=2), exponents=(1,)
            ),
            RationalLaurentPolynomialTerm(
                coefficient=CanonicalRational(num=1, den=49), exponents=(0,)
            ),
        ),
    )
    right = RationalLaurentPolynomial(
        variables=("x",),
        terms=(
            RationalLaurentPolynomialTerm(
                coefficient=CanonicalRational(num=1, den=49), exponents=(1,)
            ),
            RationalLaurentPolynomialTerm(
                coefficient=CanonicalRational(num=10**16383 - 1, den=2), exponents=(0,)
            ),
        ),
    )
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        rational_laurent_multiply(left, right)
    assert exc_info.value.errors()[0]["type"] == "polynomial.laurent.coefficient_growth"


def test_rescaled_collision_numerator_is_rejected_before_convolution() -> None:
    scale = 10**9999
    denominator = scale + 1
    left = RationalLaurentPolynomial(
        variables=("x",),
        terms=(
            term(scale, 1),
            RationalLaurentPolynomialTerm(
                coefficient=CanonicalRational(num=1, den=denominator),
                exponents=(0,),
            ),
        ),
    )
    right = RationalLaurentPolynomial(
        variables=("x",),
        terms=(
            RationalLaurentPolynomialTerm(
                coefficient=CanonicalRational(num=1, den=denominator),
                exponents=(1,),
            ),
            term(scale, 0),
        ),
    )
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        rational_laurent_multiply(left, right)
    assert exc_info.value.errors()[0]["type"] == "polynomial.laurent.coefficient_growth"


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


def test_sparse_product_admits_large_operand_wide_lcm() -> None:
    """A product whose collision groups never combine a wide operand LCM.

    One factor has coprime 20,000-digit denominators on separate terms and the
    other has integer support with distinct pair sums, so no output coefficient
    ever needs both denominators. The operand-wide LCM is refused, but the
    per-collision-group bound admits the representable convolution.
    """

    left_denominator = 10**19999 + 1
    right_denominator = 10**19999 + 7
    left = RationalLaurentPolynomial(
        variables=("x",),
        terms=(
            RationalLaurentPolynomialTerm(
                coefficient=CanonicalRational(num=1, den=left_denominator),
                exponents=(1,),
            ),
            RationalLaurentPolynomialTerm(
                coefficient=CanonicalRational(num=1, den=right_denominator),
                exponents=(0,),
            ),
        ),
    )
    right = RationalLaurentPolynomial(
        variables=("x",),
        terms=(term(1, 2), term(1, 0)),
    )
    product = rational_laurent_multiply(left, right)
    assert tuple(item.exponents for item in product.terms) == ((3,), (2,), (1,), (0,))
    assert all(
        item.coefficient.den in (left_denominator, right_denominator)
        for item in product.terms
    )


def term_object(
    coefficient: CanonicalRational, *exponents: int
) -> RationalLaurentPolynomialTerm:
    return RationalLaurentPolynomialTerm(coefficient=coefficient, exponents=exponents)


def test_cross_cancelling_pair_coefficients_are_reduced_first() -> None:
    """A pair coefficient reduced to 1 is admitted despite wide denominators."""
    denominator = 10**19_999
    first = CanonicalRational(num=denominator, den=denominator + 1)
    second = CanonicalRational(num=denominator + 1, den=denominator)
    left = RationalLaurentPolynomial(
        variables=("x",), terms=(term_object(first, 1), term_object(first, 0))
    )
    right = RationalLaurentPolynomial(
        variables=("x",), terms=(term_object(second, 1), term_object(second, 0))
    )
    product = rational_laurent_multiply(left, right)
    assert tuple(item.exponents for item in product.terms) == ((2,), (1,), (0,))
    assert [item.coefficient.as_fraction() for item in product.terms] == [
        Fraction(1),
        Fraction(2),
        Fraction(1),
    ]


def test_capped_lcm_measures_the_exact_merge() -> None:
    """An exact LCM of exactly 32,768 digits is admitted, not refused."""
    largest = 10**32_767 - 1
    left_terms = (
        term_object(CanonicalRational(num=1, den=2 * largest), 1),
        term_object(CanonicalRational(num=1, den=3 * largest), 0),
    )
    right_terms = (
        term_object(CanonicalRational(num=1, den=1), 1),
        term_object(CanonicalRational(num=1, den=1), 0),
    )
    product = rational_laurent_multiply(
        RationalLaurentPolynomial(variables=("x",), terms=left_terms),
        RationalLaurentPolynomial(variables=("x",), terms=right_terms),
    )
    assert len(product.terms) == 3


def test_boundary_width_collision_sum_is_admitted() -> None:
    """A collision of two 32,768-digit summands is admitted when it fits."""
    huge = 10**32_767
    left = RationalLaurentPolynomial(
        variables=("x",), terms=(term(huge, 1), term(huge, 0))
    )
    right = RationalLaurentPolynomial(variables=("x",), terms=(term(1, 1), term(1, 0)))
    product = rational_laurent_multiply(left, right)
    assert [item.coefficient.as_fraction() for item in product.terms] == [
        Fraction(huge),
        Fraction(2 * huge),
        Fraction(huge),
    ]


def _rational_term(
    num: int, den: int, *exponents: int
) -> RationalLaurentPolynomialTerm:
    return RationalLaurentPolynomialTerm(
        coefficient=CanonicalRational(num=num, den=den), exponents=exponents
    )


def test_boundary_width_collision_cancels_by_sign() -> None:
    """(A*x + A)*(x - 1) = A*x^2 - A even though A is at the digit cap."""
    tall = 6 * 10 ** (MAX_CANONICAL_RATIONAL_DIGITS - 1)
    left = RationalLaurentPolynomial(
        variables=("x",), terms=(term(tall, 1), term(tall, 0))
    )
    right = RationalLaurentPolynomial(variables=("x",), terms=(term(1, 1), term(-1, 0)))
    product = rational_laurent_multiply(left, right)
    assert [(t.exponents, t.coefficient.num) for t in product.terms] == [
        ((2,), tall),
        ((0,), -tall),
    ]


def test_boundary_width_collision_sum_stays_within_the_cap() -> None:
    """(A*x + A)*(x + 1) has middle coefficient 2A within the digit cap."""
    tall = 10 ** (MAX_CANONICAL_RATIONAL_DIGITS - 1)
    left = RationalLaurentPolynomial(
        variables=("x",), terms=(term(tall, 1), term(tall, 0))
    )
    right = RationalLaurentPolynomial(variables=("x",), terms=(term(1, 1), term(1, 0)))
    product = rational_laurent_multiply(left, right)
    assert len(product.terms) == 3
    assert product.terms[1].coefficient.num == 2 * tall


def test_collision_group_lcm_reduction_below_the_cap_is_admitted() -> None:
    """A collision whose intermediate LCM is oversized but reduces is admitted."""
    scale = 4 * 10 ** (MAX_CANONICAL_RATIONAL_DIGITS - 2)
    left = RationalLaurentPolynomial(
        variables=("x",),
        terms=(
            _rational_term(1, 6 * scale, 1),
            _rational_term(1, 10 * scale, 0),
        ),
    )
    right = RationalLaurentPolynomial(variables=("x",), terms=(term(1, 1), term(1, 0)))
    product = rational_laurent_multiply(left, right)
    # The middle coefficient reduces to 1/(15 * 10^(cap-2)).
    assert product.terms[1].coefficient == CanonicalRational(
        num=1, den=15 * 10 ** (MAX_CANONICAL_RATIONAL_DIGITS - 2)
    )
