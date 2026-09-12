from __future__ import annotations

import json
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS, CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.differential_forms import (
    FormComponent,
    PolynomialDifferentialForm,
    wedge,
)
from jacobian.math.polynomials.differential_forms.values import (
    MAX_DIFFERENTIAL_FORM_EXPONENT,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_VARIABLES,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

R = CanonicalRational


def _poly(*terms: tuple[int, tuple[int, int]]) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("x", "y"),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=R.from_fraction(Fraction(coefficient)),
                    exponents=exponents,
                )
                for coefficient, exponents in terms
            )
        ),
    )


def _form(
    degree: int, *components: tuple[tuple[int, ...], RationalPolynomial]
) -> PolynomialDifferentialForm:
    return PolynomialDifferentialForm(
        variables=("x", "y"),
        degree=degree,
        components=tuple(
            FormComponent(indices=indices, coefficient=coefficient)
            for indices, coefficient in components
        ),
    )


def test_wedge_computes_permutation_sign_and_exact_product() -> None:
    dx = _form(1, ((0,), _poly((2, (1, 0)))))
    dy = _form(1, ((1,), _poly((3, (0, 1)))))
    result = wedge(dx, dy)
    assert result.degree == 2
    assert result.components[0].indices == (0, 1)
    assert result.components[0].coefficient.polynomial.terms[0].coefficient == R(
        num=6, den=1
    )
    assert wedge(dy, dx).components[0].coefficient.polynomial.terms[0].coefficient == R(
        num=-6, den=1
    )


def test_wedge_repeated_differentials_and_overdimension_are_zero() -> None:
    dx = _form(1, ((0,), _poly((1, (0, 0)))))
    assert wedge(dx, dx).components == ()
    two_form = _form(2, ((0, 1), _poly((1, (0, 0)))))
    assert wedge(two_form, dx).degree == 3
    assert wedge(two_form, dx).components == ()


def test_structural_zero_skips_irrelevant_exponent_admission() -> None:
    high = _form(
        1,
        ((0,), _poly((1, (MAX_DIFFERENTIAL_FORM_EXPONENT, 0)))),
    )
    assert wedge(high, high).components == ()


def test_wedge_reserves_output_support_before_convolution() -> None:
    left_terms = tuple((1, (exponent, 0)) for exponent in range(16, -1, -1))
    right_terms = tuple((1, (0, exponent)) for exponent in range(16, -1, -1))
    left = _form(1, ((0,), _poly(*left_terms)))
    right = _form(1, ((1,), _poly(*right_terms)))
    with pytest.raises(OperationResourceAdmissionError) as error:
        wedge(left, right)
    assert error.value.errors()[0]["type"] == "differential_form.wedge.output_budget"


def test_wedge_is_associative_and_serializable() -> None:
    dx = _form(1, ((0,), _poly((1, (1, 0)))))
    dy = _form(1, ((1,), _poly((1, (0, 1)))))
    scalar = _form(0, ((), _poly((2, (0, 0)))))
    left = wedge(wedge(scalar, dx), dy)
    right = wedge(scalar, wedge(dx, dy))
    assert left == right
    assert type(left).model_validate_json(left.model_dump_json()) == left


def test_form_rejects_unsorted_or_mismatched_components() -> None:
    with pytest.raises(ValidationError, match="component_order"):
        _form(
            1,
            ((1,), _poly((1, (0, 0)))),
            ((0,), _poly((1, (0, 0)))),
        )
    with pytest.raises(OperationDomainValidationError, match="identical"):
        wedge(
            _form(0, ((), _poly((1, (0, 0))))),
            PolynomialDifferentialForm(
                variables=("x", "z"),
                degree=0,
                components=(),
            ),
        )


def test_duplicate_differential_indices_are_rejected() -> None:
    with pytest.raises(ValidationError, match="component_basis"):
        _form(2, ((0, 0), _poly((1, (0, 0)))))


def test_overflowing_zero_form_degree_is_typed_admission() -> None:
    degree = 10**MAX_CANONICAL_INTEGER_DIGITS - 1
    zero = _form(degree)
    with pytest.raises(OperationResourceAdmissionError) as error:
        wedge(zero, zero)
    assert error.value.errors()[0]["type"] == "differential_form.wedge.degree_budget"


def test_serialized_differential_indices_have_a_schema_bound() -> None:
    variables = [f"x{index}" for index in range(MAX_POLYNOMIAL_VARIABLES)]
    payload = {
        "variables": variables,
        "degree": "1",
        "components": [
            {
                "indices": list(range(MAX_POLYNOMIAL_VARIABLES + 1)),
                "coefficient": {
                    "variables": variables,
                    "polynomial": {
                        "terms": [
                            {
                                "coefficient": {"num": "1", "den": "1"},
                                "exponents": [0] * MAX_POLYNOMIAL_VARIABLES,
                            }
                        ]
                    },
                },
            }
        ],
    }
    with pytest.raises(ValidationError, match="at most 8 items"):
        PolynomialDifferentialForm.model_validate_json(json.dumps(payload))


def test_high_coefficients_and_zero_degrees_compose() -> None:
    scalar = _form(0, ((), _poly((10**100, (0, 0)))))
    product = wedge(scalar, scalar)
    assert (
        product.components[0].coefficient.polynomial.terms[0].coefficient.num == 10**200
    )
    assert type(product).model_validate_json(product.model_dump_json()) == product
    zero = _form(16)
    squared = wedge(zero, zero)
    assert squared.degree == 32 and squared.components == ()
    assert wedge(squared, zero).degree == 48
