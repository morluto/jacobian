"""Differentiation consumes its own canonical outputs and native primitives."""

from fractions import Fraction

import pytest

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials import (
    rational_polynomial_derivative as derivative,
)
from jacobian.math.polynomials import (
    rational_polynomial_integral as integral,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _polynomial(*terms: tuple[int, Fraction]) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("retained_x",),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(coefficient),
                    exponents=(exponent,),
                )
                for exponent, coefficient in terms
            )
        ),
    )


def _oracle(source: RationalPolynomial) -> dict[int, Fraction]:
    return {
        term.exponents[0] - 1: term.coefficient.as_fraction() * term.exponents[0]
        for term in source.polynomial.terms
        if term.exponents[0]
    }


def _check(source: RationalPolynomial) -> RationalPolynomial:
    result = derivative(source).derivative
    assert result.variables == source.variables
    assert {
        term.exponents[0]: term.coefficient.as_fraction()
        for term in result.polynomial.terms
    } == _oracle(source)
    assert RationalPolynomial.model_validate_json(result.model_dump_json()) == result
    return result


def test_derivative_consumes_serialized_expanded_coefficient() -> None:
    source = _polynomial((127, Fraction(10**256 - 1)))
    first = _check(source)
    second = _check(RationalPolynomial.model_validate_json(first.model_dump_json()))
    assert second == _polynomial((125, Fraction(16002 * (10**256 - 1))))


@pytest.mark.parametrize(
    "source",
    [
        _polynomial((127, Fraction(1))),
        _polynomial((126, Fraction(1, 10**256 - 1))),
        _polynomial((126, Fraction(1))),
        _polynomial((4, Fraction(-7, 9)), (0, Fraction(13, 5))),
        _polynomial(),
    ],
)
def test_derivative_consumes_serialized_native_integral(
    source: RationalPolynomial,
) -> None:
    primitive = integral(source).antiderivative
    restored = RationalPolynomial.model_validate_json(primitive.model_dump_json())
    assert _check(restored) == source


@pytest.mark.parametrize(
    "denominator",
    [1, 2, 10**MAX_CANONICAL_RATIONAL_DIGITS - 3],
    ids=["integer", "cancelled", "balanced"],
)
def test_full_width_surviving_coefficient_and_denominator_cancellation(
    denominator: int,
) -> None:
    numerator = 10**MAX_CANONICAL_RATIONAL_DIGITS - 1
    exponent = 2 if denominator == 2 else 1
    _check(_polynomial((exponent, Fraction(numerator, denominator))))


def test_maximum_exponent_and_discarded_wide_constant() -> None:
    _check(
        _polynomial(
            (MAX_POLYNOMIAL_EXPONENT, Fraction(1)),
            (0, Fraction(1, 10**MAX_CANONICAL_RATIONAL_DIGITS - 1)),
        )
    )


def test_dense_family_with_shared_denominator_remains_admitted() -> None:
    _check(_polynomial(*((e, Fraction(1, 10**256 - 1)) for e in range(127, -1, -1))))


def test_actual_output_overflow_is_typed_before_conversion() -> None:
    source = _polynomial((2, Fraction(10**MAX_CANONICAL_RATIONAL_DIGITS - 1)))
    with pytest.raises(OperationResourceAdmissionError) as error:
        derivative(source)
    assert error.value.errors()[0]["type"] == "polynomial.derivative_output_bound"


def test_dense_common_denominator_work_is_bounded() -> None:
    source = _polynomial(*((e, Fraction(1, 10**1000 + e)) for e in range(127, 0, -1)))
    with pytest.raises(OperationResourceAdmissionError) as error:
        derivative(source)
    assert error.value.errors()[0]["type"] == "polynomial.derivative_work_bound"


@pytest.mark.parametrize(
    "forgery", ["bool", "missing", "unreduced", "negative", "order", "domain"]
)
def test_native_forgery_cannot_bypass_derivative_admission(forgery: str) -> None:
    source = _polynomial((2, Fraction(1)), (1, Fraction(2)))
    term = source.polynomial.terms[0]
    if forgery == "bool":
        term = term.model_copy(update={"exponents": (True,)})
    elif forgery == "missing":
        term = RationalPolynomialTerm.model_construct(exponents=(2,))
    elif forgery == "unreduced":
        term = term.model_copy(
            update={"coefficient": CanonicalRational.model_construct(num=2, den=2)}
        )
    elif forgery == "negative":
        term = term.model_copy(
            update={"coefficient": CanonicalRational.model_construct(num=1, den=-1)}
        )
    if forgery == "domain":
        source = source.model_copy(update={"domain": "ZZ"})
    else:
        terms: tuple[RationalPolynomialTerm, ...] = (term, source.polynomial.terms[1])
        if forgery == "order":
            terms = tuple(reversed(terms))
        source = source.model_copy(
            update={"polynomial": SparseRationalPolynomial.model_construct(terms=terms)}
        )
    with pytest.raises(OperationDomainValidationError):
        derivative(source)
