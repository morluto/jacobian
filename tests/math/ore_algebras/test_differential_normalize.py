from fractions import Fraction

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.ore_algebras._models import DifferentialOreOperator
from jacobian.math.ore_algebras.operations import (
    differential_operator_normalize_polynomial_coefficients,
)
from jacobian.math.polynomials.values import RationalFunction


def _rf(terms: tuple[tuple[int | Fraction, int], ...]) -> RationalFunction:
    return RationalFunction.model_validate(
        {
            "domain": "QQ",
            "variables": ["x"],
            "numerator": {
                "terms": [
                    {
                        "coefficient": {
                            "num": Fraction(value).numerator,
                            "den": Fraction(value).denominator,
                        },
                        "exponents": [degree],
                    }
                    for value, degree in sorted(
                        terms, key=lambda pair: pair[1], reverse=True
                    )
                ]
            },
            "denominator": {
                "terms": [{"coefficient": {"num": 1, "den": 1}, "exponents": [0]}]
            },
        }
    )


def _operator(
    terms: tuple[tuple[int, RationalFunction], ...],
) -> DifferentialOreOperator:
    return DifferentialOreOperator.model_validate(
        {
            "variable": "x",
            "terms": [
                {"order": order, "coefficient": coefficient}
                for order, coefficient in terms
            ],
        }
    )


def _poly(operator: DifferentialOreOperator, order: int) -> dict[int, Fraction]:
    term = next(term for term in operator.terms if term.order == order)
    return {
        item.exponents[0]: item.coefficient.as_fraction()
        for item in term.coefficient.numerator.terms
    }


def test_normalization_extracts_exact_content_and_reconstructs_every_coefficient() -> (
    None
):
    source = _operator(
        (
            (0, _rf(((Fraction(4, 3), 0), (Fraction(2, 3), 1)))),
            (2, _rf(((Fraction(-2, 3), 0),))),
        )
    )

    result = differential_operator_normalize_polynomial_coefficients(source)

    assert result.scale.variables == ("x",)
    assert result.scale.numerator.terms[0].coefficient.as_fraction() == Fraction(-2, 3)
    assert _poly(result.normalized, 0) == {0: Fraction(-2), 1: Fraction(-1)}
    assert _poly(result.normalized, 2) == {0: Fraction(1)}
    for source_term in source.terms:
        actual = _poly(result.normalized, source_term.order)
        expected = {
            item.exponents[0]: item.coefficient.as_fraction()
            for item in source_term.coefficient.numerator.terms
        }
        assert {
            degree: value * Fraction(-2, 3) for degree, value in actual.items()
        } == expected


def test_normalization_is_positive_at_the_leading_ore_monomial() -> None:
    source = _operator(((0, _rf(((-3, 0),))), (1, _rf(((-1, 1),)))))

    result = differential_operator_normalize_polynomial_coefficients(source)

    assert result.scale.numerator.terms[0].coefficient.as_fraction() == -1
    assert _poly(result.normalized, 0) == {0: Fraction(3)}
    assert _poly(result.normalized, 1) == {1: Fraction(1)}


def test_zero_operator_normalizes_to_itself_with_unit_scale() -> None:
    zero = _operator(())

    result = differential_operator_normalize_polynomial_coefficients(zero)

    assert result.normalized == zero
    assert result.scale.variables == ("x",)
    assert result.scale.numerator.terms[0].coefficient.as_fraction() == 1


def test_normalization_rejects_nonpolynomial_rational_function_coefficients() -> None:
    rational = RationalFunction.model_validate(
        {
            "domain": "QQ",
            "variables": ["x"],
            "numerator": {
                "terms": [{"coefficient": {"num": 1, "den": 1}, "exponents": [0]}]
            },
            "denominator": {
                "terms": [{"coefficient": {"num": 1, "den": 1}, "exponents": [1]}]
            },
        }
    )
    with pytest.raises(OperationDomainValidationError, match="polynomial coefficients"):
        differential_operator_normalize_polynomial_coefficients(
            _operator(((0, rational),))
        )
