from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.differential_forms import (
    FormComponent,
    PolynomialDifferentialForm,
    wedge,
)
from jacobian.math.polynomials.values import (
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
