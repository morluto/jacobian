"""Tests for canonical polynomial vector-calculus operations."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)
from jacobian.math.polynomials.vector_calculus._models import (
    CurlRequest,
    DirectionalDerivativeRequest,
    ScalarFieldRequest,
    VectorFieldRequest,
)
from jacobian.math.polynomials.vector_calculus._operations import (
    compute_curl,
    compute_directional_derivative,
    compute_divergence,
    compute_gradient,
    compute_laplacian,
)
from jacobian.math.polynomials.vector_calculus._tools import TOOLS


def _polynomial(
    variables: tuple[str, ...],
    terms: Mapping[tuple[int, ...], int | Fraction],
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(coefficient)),
                    exponents=exponents,
                )
                for exponents, coefficient in sorted(terms.items(), reverse=True)
                if coefficient
            )
        ),
    )


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "polynomial_field.scalar.gradient.compute",
        "polynomial_field.scalar.laplacian.compute",
        "polynomial_field.scalar.directional_derivative.compute",
        "polynomial_field.vector.divergence.compute",
        "polynomial_field.vector.curl.compute",
    }


def test_gradient_returns_composable_polynomials() -> None:
    source = _polynomial(("x", "y"), {(2, 0): 1, (0, 2): 1})
    result = compute_gradient(ScalarFieldRequest(polynomial=source))
    assert result.components == (
        _polynomial(("x", "y"), {(1, 0): 2}),
        _polynomial(("x", "y"), {(0, 1): 2}),
    )


def test_renaming_the_declared_axis_transports_the_gradient() -> None:
    original = compute_gradient(
        ScalarFieldRequest(polynomial=_polynomial(("x", "y"), {(2, 0): 1, (0, 1): 3}))
    )
    renamed = compute_gradient(
        ScalarFieldRequest(polynomial=_polynomial(("u", "v"), {(2, 0): 1, (0, 1): 3}))
    )

    assert tuple(component.polynomial for component in original.components) == tuple(
        component.polynomial for component in renamed.components
    )
    assert all(component.variables == ("x", "y") for component in original.components)
    assert all(component.variables == ("u", "v") for component in renamed.components)


def test_laplacian_returns_a_composable_polynomial() -> None:
    source = _polynomial(("x", "y"), {(3, 0): 1, (0, 3): 1})
    result = compute_laplacian(ScalarFieldRequest(polynomial=source))
    assert result.result == _polynomial(
        ("x", "y"),
        {(1, 0): 6, (0, 1): 6},
    )


def test_directional_derivative_uses_exact_rational_coordinates() -> None:
    source = _polynomial(("x", "y"), {(2, 0): 1, (0, 2): 1})
    result = compute_directional_derivative(
        DirectionalDerivativeRequest(
            polynomial=source,
            direction=(
                CanonicalRational(num="1", den="2"),
                CanonicalRational(num="1", den="1"),
            ),
        )
    )
    assert result.result == _polynomial(
        ("x", "y"),
        {(1, 0): 1, (0, 1): 2},
    )


def test_divergence_uses_one_authoritative_axis() -> None:
    result = compute_divergence(
        VectorFieldRequest(
            components=(
                _polynomial(("x", "y"), {(2, 0): 1}),
                _polynomial(("x", "y"), {(0, 2): 1}),
            )
        )
    )
    assert result.result == _polynomial(
        ("x", "y"),
        {(1, 0): 2, (0, 1): 2},
    )


def test_curl_is_rejected_at_the_request_boundary_outside_three_dimensions() -> None:
    with pytest.raises(ValidationError) as exc_info:
        CurlRequest(
            components=(
                _polynomial(("x", "y"), {(1, 0): 1}),
                _polynomial(("x", "y"), {(0, 1): 1}),
            )
        )
    assert (
        exc_info.value.errors()[0]["type"] == "polynomial_vector_calc.curl_dimensions"
    )


def test_curl_three_dimensional_orientation() -> None:
    variables = ("x", "y", "z")
    result = compute_curl(
        CurlRequest(
            components=(
                _polynomial(variables, {(0, 1, 0): 1}),
                _polynomial(variables, {}),
                _polynomial(variables, {}),
            )
        )
    )
    assert result.components == (
        _polynomial(variables, {}),
        _polynomial(variables, {}),
        _polynomial(variables, {(0, 0, 0): -1}),
    )


def test_vector_components_must_share_the_same_ring() -> None:
    with pytest.raises(ValidationError) as exc_info:
        VectorFieldRequest(
            components=(
                _polynomial(("x", "y"), {(1, 0): 1}),
                _polynomial(("y", "x"), {(1, 0): 1}),
            )
        )
    assert exc_info.value.errors()[0]["type"] == "polynomial_vector_calc.ordered_ring"


def test_vector_field_rejects_aggregate_result_term_growth() -> None:
    variables = ("x", "y")
    monomials: list[tuple[int, ...]] = [
        (left, right) for left in range(1, 64) for right in range(1, 64 - left)
    ]
    first = dict.fromkeys(monomials[:128], 1)
    second = dict.fromkeys(monomials[128:257], 1)
    with pytest.raises(ValidationError) as exc_info:
        VectorFieldRequest(
            components=(
                _polynomial(variables, first),
                _polynomial(variables, second),
            )
        )
    assert exc_info.value.errors()[0]["type"] == (
        "polynomial_vector_calc.derivative_term_budget"
    )


def test_direction_length_must_match_polynomial_axis() -> None:
    with pytest.raises(ValidationError) as exc_info:
        DirectionalDerivativeRequest(
            polynomial=_polynomial(("x", "y"), {(1, 0): 1}),
            direction=(CanonicalRational(num="1", den="1"),),
        )
    assert (
        exc_info.value.errors()[0]["type"] == "polynomial_vector_calc.direction_length"
    )
