"""Canonical factor records compose unchanged across native and JSON boundaries."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials._models import PolynomialFactorizationResult
from jacobian.math.polynomials.operations import (
    polynomial_factorization,
    verify_polynomial_factorization,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

type Coefficients = tuple[int | Fraction, ...]


def _polynomial(coefficients: Coefficients) -> RationalPolynomial:
    """Build a polynomial from coefficients in increasing degree order."""
    return RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(value)),
                    exponents=(degree,),
                )
                for degree, value in reversed(tuple(enumerate(coefficients)))
                if value
            )
        ),
    )


def _coefficients(polynomial: RationalPolynomial) -> dict[int, Fraction]:
    return {
        term.exponents[0]: term.coefficient.as_fraction()
        for term in polynomial.polynomial.terms
    }


def _rebuild_product(result: PolynomialFactorizationResult) -> dict[int, Fraction]:
    """Check the defining identity with rational convolution, without factoring."""
    product = {0: result.coefficient.as_fraction()}
    for record in result.factors:
        factor = _coefficients(record.factor)
        for _ in range(record.multiplicity):
            expanded: dict[int, Fraction] = {}
            for left_degree, left in product.items():
                for right_degree, right in factor.items():
                    degree = left_degree + right_degree
                    expanded[degree] = expanded.get(degree, Fraction(0)) + left * right
            product = expanded
    return {degree: value for degree, value in product.items() if value}


@pytest.mark.parametrize(
    ("source_coefficients", "coefficient", "expected_factors"),
    (
        pytest.param((20, 12, 1), 1, (((2, 1), 1), ((10, 1), 1)), id="multi-digit"),
        pytest.param((6, -5, 1), 1, (((-3, 1), 1), ((-2, 1), 1)), id="signed"),
        pytest.param(
            (Fraction(1, 22), Fraction(13, 22), 1),
            1,
            (((Fraction(1, 2), 1), 1), ((Fraction(1, 11), 1), 1)),
            id="denominator-components",
        ),
        pytest.param(
            (Fraction(-20, 21), Fraction(-12, 7), Fraction(-3, 7)),
            Fraction(-3, 7),
            (((Fraction(2, 3), 1), 1), ((Fraction(10, 3), 1), 1)),
            id="rational-content-and-numerators",
        ),
        pytest.param(
            (400, 480, 184, 24, 1),
            1,
            (((2, 1), 2), ((10, 1), 2)),
            id="repeated-factors",
        ),
        pytest.param(
            (40, 44, 14, 1),
            1,
            (((10, 1), 1), ((2, 1), 2)),
            id="multiplicity-before-fingerprint",
        ),
        pytest.param(
            (10, 1, 10, 1),
            1,
            (((10, 1), 1), ((1, 0, 1), 1)),
            id="degree-before-fingerprint",
        ),
        pytest.param((), 0, (), id="zero"),
        pytest.param((1,), 1, (), id="unit"),
        pytest.param((-7,), -7, (), id="negative-constant"),
        pytest.param((Fraction(2, 11),), Fraction(2, 11), (), id="rational-constant"),
    ),
)
def test_factorization_has_one_canonical_order_through_json_and_verifier(
    source_coefficients: Coefficients,
    coefficient: int | Fraction,
    expected_factors: tuple[tuple[Coefficients, int], ...],
) -> None:
    source = _polynomial(source_coefficients)
    result = polynomial_factorization(source)

    assert _rebuild_product(result) == _coefficients(source)
    assert result.reconstructed == source
    assert verify_polynomial_factorization(result)
    assert PolynomialFactorizationResult.model_validate(result.model_dump()) == result

    decoded = PolynomialFactorizationResult.model_validate_json(
        result.model_dump_json()
    )
    assert decoded == result
    assert verify_polynomial_factorization(decoded)
    assert result.coefficient.as_fraction() == coefficient
    # The fingerprint compares exact integer numerator/denominator components,
    # independently of their decimal wire spelling or rational-value ordering.
    assert tuple((record.factor, record.multiplicity) for record in result.factors) == (
        tuple(
            (_polynomial(factor), multiplicity)
            for factor, multiplicity in expected_factors
        )
    )


def test_factorization_decoder_rejects_reversed_canonical_records() -> None:
    result = polynomial_factorization(_polynomial((20, 12, 1)))
    reversed_result = result.model_copy(
        update={"factors": tuple(reversed(result.factors))}
    )
    with pytest.raises(ValidationError) as error:
        PolynomialFactorizationResult.model_validate_json(
            reversed_result.model_dump_json()
        )
    assert error.value.errors()[0]["type"] == "polynomial.invariant"


@pytest.mark.parametrize("field", ("polynomial", "reconstructed"))
def test_factorization_verifier_rejects_changed_source_or_reconstruction(
    field: str,
) -> None:
    result = polynomial_factorization(_polynomial((20, 12, 1)))
    forged = result.model_copy(update={field: _polynomial((21, 12, 1))})
    decoded = PolynomialFactorizationResult.model_validate_json(
        forged.model_dump_json()
    )
    assert not verify_polynomial_factorization(decoded)


def _root_product(count: int) -> Coefficients:
    """Expand product(x-r), r=1..count by integer convolution."""
    coefficients = [1]
    for root in range(1, count + 1):
        expanded = [0] * (len(coefficients) + 1)
        for degree, value in enumerate(coefficients):
            expanded[degree] -= root * value
            expanded[degree + 1] += value
        coefficients = expanded
    return tuple(coefficients)


@pytest.mark.parametrize("count", (64, 65, 128))
def test_complete_factor_records_round_trip_beyond_the_old_count_limit(
    count: int,
) -> None:
    source = _polynomial(_root_product(count))
    result = polynomial_factorization(source)
    assert len(result.factors) == count
    assert result.coefficient.as_fraction() == 1
    assert tuple(record.factor for record in result.factors) == tuple(
        _polynomial((-root, 1)) for root in range(count, 0, -1)
    )
    assert all(record.multiplicity == 1 for record in result.factors)
    assert _rebuild_product(result) == _coefficients(source)
    assert result.reconstructed == source
    assert PolynomialFactorizationResult.model_validate(result.model_dump()) == result
    decoded = PolynomialFactorizationResult.model_validate_json(
        result.model_dump_json()
    )
    assert decoded == result
    assert verify_polynomial_factorization(decoded)


def test_factor_record_count_and_multiplicity_have_distinct_degree_bounds() -> None:
    source = _polynomial((0,) * 500 + (1,))
    result = polynomial_factorization(source)
    assert len(result.factors) == 1
    assert result.factors[0].multiplicity == 500
    assert result.factors[0].factor == _polynomial((0, 1))
    assert _rebuild_product(result) == _coefficients(source)
    assert (
        PolynomialFactorizationResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_factor_height_may_exceed_source_height_without_losing_canonical_output() -> (
    None
):
    p, q = 10**255, 10**255 + 1
    source = _polynomial((q, Fraction(1, p)))
    result = polynomial_factorization(source)
    assert result.coefficient.as_fraction() == Fraction(1, p)
    assert result.factors[0].factor == _polynomial((p * q, 1))
    assert _rebuild_product(result) == _coefficients(source)
    assert (
        PolynomialFactorizationResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_factor_decoder_keeps_the_admitted_degree_count_limit() -> None:
    result = polynomial_factorization(_polynomial((0, 1)))
    oversized = result.model_copy(update={"factors": result.factors * 501})
    with pytest.raises(ValidationError) as error:
        PolynomialFactorizationResult.model_validate_json(oversized.model_dump_json())
    assert error.value.errors()[0]["type"] == "too_long"
