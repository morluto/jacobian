from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
)
from jacobian.math.polynomials.differential_forms import (
    FormComponent,
    PolynomialDifferentialForm,
    exterior_derivative,
    wedge,
)
from jacobian.math.polynomials.differential_forms.values import (
    MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

R = CanonicalRational


def _poly_on_axis(
    variables: tuple[str, ...],
    *terms: tuple[int | Fraction, tuple[int, ...]],
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
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


def _poly(*terms: tuple[int | Fraction, tuple[int, int]]) -> RationalPolynomial:
    return _poly_on_axis(("x", "y"), *terms)


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


@pytest.mark.parametrize("leading", (6, 7, 9))
@pytest.mark.parametrize("sign", (-1, 1))
@pytest.mark.parametrize("reciprocal", (False, True))
def test_exact_decimal_carrier_edge_survives_calculus(
    leading: int, sign: int, reciprocal: bool
) -> None:
    magnitude = leading * 10 ** (MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS - 1)
    value = Fraction(sign, magnitude) if reciprocal else Fraction(sign * magnitude)
    constant = _form(0, ((), _poly((value, (0, 0)))))
    decoded = PolynomialDifferentialForm.model_validate_json(constant.model_dump_json())
    assert exterior_derivative(decoded).components == ()
    linear = _form(0, ((), _poly((value, (1, 0)))))
    result = exterior_derivative(
        PolynomialDifferentialForm.model_validate_json(linear.model_dump_json())
    )
    expected = _form(1, ((0,), _poly((value, (0, 0)))))
    assert (
        PolynomialDifferentialForm.model_validate_json(result.model_dump_json())
        == expected
    )
    assert wedge(decoded, _form(0, ((), _poly((1, (0, 0)))))) == decoded


def test_wedge_reduced_sum_keeps_exact_decimal_denominator_edge() -> None:
    denominator = 7 * 10 ** (MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS - 1)
    # (x + 1) * (5x/D - 4/D) has middle coefficient 1/D. Both
    # source denominators are smaller; only the reduced sum reaches the edge.
    left = _form(0, ((), _poly((1, (1, 0)), (1, (0, 0)))))
    right = _form(
        0,
        (
            (),
            _poly(
                (Fraction(1, denominator // 5), (1, 0)),
                (Fraction(-2, denominator // 2), (0, 0)),
            ),
        ),
    )
    result = wedge(left, right)
    expected = _form(
        0,
        (
            (),
            _poly(
                (Fraction(5, denominator), (2, 0)),
                (Fraction(1, denominator), (1, 0)),
                (Fraction(-4, denominator), (0, 0)),
            ),
        ),
    )
    assert (
        PolynomialDifferentialForm.model_validate_json(result.model_dump_json())
        == expected
    )


@pytest.mark.parametrize("reciprocal", (False, True))
def test_native_exact_carrier_cutoff_still_rejects_one_digit_too_many(
    reciprocal: bool,
) -> None:
    value = R.model_construct(
        num=1 if reciprocal else 10**MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS,
        den=10**MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS if reciprocal else 1,
    )
    term = RationalPolynomialTerm.model_construct(coefficient=value, exponents=(0, 0))
    coefficient = RationalPolynomial.model_construct(
        variables=("x", "y"),
        polynomial=SparseRationalPolynomial.model_construct(terms=(term,)),
    )
    form = PolynomialDifferentialForm.model_construct(
        variables=("x", "y"),
        degree=0,
        components=(
            FormComponent.model_construct(indices=(), coefficient=coefficient),
        ),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        exterior_derivative(form)
    assert error.value.errors()[0]["type"] == "differential_form.coefficient_height"


def test_wedge_reduced_sum_rejects_actual_denominator_overflow() -> None:
    from jacobian.catalog.models import OperationResourceAdmissionError

    denominator = 2 * 10**MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS
    left = _form(0, ((), _poly((1, (1, 0)), (1, (0, 0)))))
    right = _form(
        0,
        (
            (),
            _poly(
                (Fraction(1, denominator // 5), (1, 0)),
                (Fraction(-2, denominator // 2), (0, 0)),
            ),
        ),
    )
    # Inputs fit the carrier; the middle coefficient's
    # denominator has one additional digit and really lies outside the carrier.
    with pytest.raises(OperationResourceAdmissionError) as error:
        wedge(left, right)
    assert (
        error.value.errors()[0]["type"] == "differential_form.wedge.coefficient_budget"
    )
