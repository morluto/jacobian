"""Exterior-calculus slice (#3723): d, contraction, pullback, Lie, primitives."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.differential_forms import (
    affine_homotopy_primitive,
    exterior_derivative,
    interior_product,
    lie_derivative,
    pullback,
    wedge,
)
from jacobian.math.polynomials.differential_forms.values import (
    FormComponent,
    PolynomialDifferentialForm,
    PolynomialMap,
    PolynomialVectorField,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

R = CanonicalRational


def _poly_on_axis(variables, *terms):
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


def _poly(*terms):
    return _poly_on_axis(("x", "y"), *terms)


def _form(variables, degree, *components):
    return PolynomialDifferentialForm(
        variables=variables,
        degree=degree,
        components=tuple(
            FormComponent(indices=indices, coefficient=coefficient)
            for indices, coefficient in components
        ),
    )


def _coeff(form, indices=(0, 1)):
    component = next(c for c in form.components if c.indices == indices)
    return {
        term.exponents: term.coefficient.as_fraction()
        for term in component.coefficient.polynomial.terms
    }


def _field(*components):
    return PolynomialVectorField(
        variables=("x", "y"),
        components=tuple(
            _poly(*terms) if terms else _poly_on_axis(("x", "y"))
            for terms in components
        ),
    )


def test_exterior_derivative_known_answer() -> None:
    # d(x dy) = dx wedge dy.
    result = exterior_derivative(_form(("x", "y"), 1, ((1,), _poly((1, (1, 0))))))
    assert result.degree == 2
    assert _coeff(result) == {(0, 0): Fraction(1)}


def test_exterior_derivative_squares_to_zero() -> None:
    alpha = _form(
        ("x", "y"),
        1,
        ((0,), _poly((1, (2, 0)), (3, (0, 1)))),
        ((1,), _poly((5, (1, 1)))),
    )
    assert exterior_derivative(exterior_derivative(alpha)).components == ()


def test_exterior_derivative_of_scalar_known_answer() -> None:
    # d(f g) = df*g + f*dg at the 0-form level: d(x^2 y) = 2xy dx + x^2 dy.
    from jacobian.math.polynomials.differential_forms.values import (
        PolynomialDifferentialForm as F,
    )

    scalar = F(
        variables=("x", "y"),
        degree=0,
        components=(FormComponent(indices=(), coefficient=_poly((1, (2, 1)))),),
    )
    result = exterior_derivative(scalar)
    assert _coeff(result, (0,)) == {(1, 1): Fraction(2)}
    assert _coeff(result, (1,)) == {(2, 0): Fraction(1)}


def test_contraction_known_answer() -> None:
    # i_{∂x}(dx ∧ dy) = dy.
    field = _field(((1, (0, 0)),), ())
    form = _form(("x", "y"), 2, ((0, 1), _poly((1, (0, 0)))))
    result = interior_product(field, form)
    assert result.degree == 1
    assert _coeff(result, (1,)) == {(0, 0): Fraction(1)}


def test_contraction_rejects_axis_mismatch() -> None:
    field = PolynomialVectorField(
        variables=("x",),
        components=(_poly_on_axis(("x",), (1, (0,))),),
    )
    form = _form(("x", "y"), 1, ((0,), _poly((1, (0, 0)))))
    with pytest.raises(OperationDomainValidationError, match="axis"):
        interior_product(field, form)


def test_pullback_known_answer() -> None:
    # phi(t) = (t^2, t^3); phi*(x dy - y dx) = t^4 dt.
    mapping = PolynomialMap(
        source_variables=("t",),
        target_variables=("x", "y"),
        images=(
            _poly_on_axis(("t",), (1, (2,))),
            _poly_on_axis(("t",), (1, (3,))),
        ),
    )
    form = _form(
        ("x", "y"),
        1,
        ((0,), _poly((-1, (0, 1)))),
        ((1,), _poly((1, (1, 0)))),
    )
    result = pullback(mapping, form)
    assert result.variables == ("t",)
    assert _coeff(result, (0,)) == {(4,): Fraction(1)}


def test_pullback_respects_wedge() -> None:
    # phi*(alpha wedge beta) = phi*alpha wedge phi*beta for
    # phi(s, t) = (s^2, t); both sides are the nonzero 2*s^3 ds wedge dt.
    mapping = PolynomialMap(
        source_variables=("s", "t"),
        target_variables=("x", "y"),
        images=(
            _poly_on_axis(("s", "t"), (1, (2, 0))),
            _poly_on_axis(("s", "t"), (1, (0, 1))),
        ),
    )
    alpha = _form(("x", "y"), 1, ((0,), _poly((1, (1, 0)))))
    beta = _form(("x", "y"), 1, ((1,), _poly((1, (0, 0)))))
    expected = _form(
        ("s", "t"),
        2,
        ((0, 1), _poly_on_axis(("s", "t"), (2, (3, 0)))),
    )
    pulled_back_wedge = pullback(mapping, wedge(alpha, beta))
    wedge_of_pullbacks = wedge(pullback(mapping, alpha), pullback(mapping, beta))
    assert pulled_back_wedge == expected
    assert wedge_of_pullbacks == expected


def test_lie_derivative_known_answer() -> None:
    # L_{∂x}(x dx) = dx.
    field = _field(((1, (0, 0)),), ())
    form = _form(("x", "y"), 1, ((0,), _poly((1, (1, 0)))))
    result = lie_derivative(field, form)
    assert _coeff(result, (0,)) == {(0, 0): Fraction(1)}


def test_primitive_round_trip() -> None:
    form = _form(("x", "y"), 2, ((0, 1), _poly((1, (0, 0)))))
    claim = affine_homotopy_primitive(form)
    assert claim.outcome == "CONSTRUCTED"
    assert claim.primitive is not None
    assert exterior_derivative(claim.primitive) == form


def test_primitive_rejects_non_closed() -> None:
    # d(x dy) = dx∧dy ≠ 0, so x dy has no primitive claim.
    form = _form(("x", "y"), 1, ((1,), _poly((1, (1, 0)))))
    claim = affine_homotopy_primitive(form)
    assert claim.outcome == "NOT_APPLICABLE"
    assert claim.primitive is None


def test_primitive_rejects_degree_zero() -> None:
    from jacobian.math.polynomials.differential_forms.values import (
        PolynomialDifferentialForm as F,
    )

    scalar = F(
        variables=("x",),
        degree=0,
        components=(
            FormComponent(indices=(), coefficient=_poly_on_axis(("x",), (1, (2,)))),
        ),
    )
    assert affine_homotopy_primitive(scalar).outcome == "NOT_APPLICABLE"
