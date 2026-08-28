"""Tests for canonical polynomial map operations."""

from __future__ import annotations

from fractions import Fraction

import pytest
from tests.math.polynomials._support import polynomial_validation_error

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.maps._models import (
    CompositionRequest,
    EvalRequest,
    VariablePoint,
)
from jacobian.math.polynomials.maps.operations import (
    compose_polynomials,
    evaluate_polynomial,
    jacobian_matrix,
)
from jacobian.math.polynomials.maps.values import RationalPolynomialMap
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _evaluate(request: EvalRequest):
    return evaluate_polynomial(request.polynomial, request.point)


def _compose(request: CompositionRequest):
    return compose_polynomials(
        request.outer,
        request.inner,
        outer_variable=request.outer_variable,
        inner_variable=request.inner_variable,
    )


def _polynomial(
    variables: tuple[str, ...],
    terms: dict[tuple[int, ...], int | Fraction],
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


def test_evaluation_returns_a_canonical_rational() -> None:
    request = EvalRequest(
        polynomial=_polynomial(("x", "y"), {(2, 0): 1, (0, 1): 2}),
        point=VariablePoint(
            variables=("x", "y"),
            values=(
                CanonicalRational(num="3", den="1"),
                CanonicalRational(num="1", den="1"),
            ),
        ),
    )
    assert _evaluate(request).value == CanonicalRational(num="11", den="1")


def test_evaluation_requires_the_complete_ordered_axis() -> None:
    request = EvalRequest(
        polynomial=_polynomial(("x", "y"), {(1, 0): 1}),
        point=VariablePoint(
            variables=("x",),
            values=(CanonicalRational(num="1", den="1"),),
        ),
    )
    with pytest.raises(OperationDomainValidationError):
        _evaluate(request)


def test_evaluation_rejects_a_point_whose_exact_value_exceeds_result_bound() -> None:
    request = EvalRequest(
        polynomial=_polynomial(("x",), {(64,): 1}),
        point=VariablePoint(
            variables=("x",),
            values=(CanonicalRational(num="1" + "0" * 600, den="1"),),
        ),
    )
    with pytest.raises(OperationDomainValidationError):
        _evaluate(request)


def test_jacobian_entries_are_directly_composable_polynomials() -> None:
    request = RationalPolynomialMap(
        input_variables=("x", "y"),
        output_polynomials=(
            _polynomial(("x", "y"), {(2, 0): 1}),
            _polynomial(("x", "y"), {(0, 2): 1}),
        ),
    )
    result = jacobian_matrix(request)
    assert result.n_inputs == 2
    assert result.n_outputs == 2
    assert result.entries == (
        _polynomial(("x", "y"), {(1, 0): 2}),
        _polynomial(("x", "y"), {}),
        _polynomial(("x", "y"), {}),
        _polynomial(("x", "y"), {(0, 1): 2}),
    )


def test_jacobian_rejects_a_mismatched_output_ring() -> None:
    with polynomial_validation_error():
        RationalPolynomialMap(
            input_variables=("x", "y"),
            output_polynomials=(_polynomial(("x",), {(2,): 1}),),
        )


def test_univariate_composition_returns_a_canonical_polynomial() -> None:
    result = _compose(
        CompositionRequest(
            outer=_polynomial(("u",), {(2,): 1}),
            inner=_polynomial(("x",), {(1,): 1, (0,): 1}),
            inner_variable="x",
            outer_variable="u",
        )
    )
    assert result.polynomial == _polynomial(
        ("x",),
        {(2,): 1, (1,): 2, (0,): 1},
    )


def test_composition_rejects_multivariate_operands() -> None:
    request = CompositionRequest(
        outer=_polynomial(("u", "v"), {(1, 0): 1}),
        inner=_polynomial(("x",), {(1,): 1}),
        inner_variable="x",
        outer_variable="u",
    )
    with pytest.raises(OperationDomainValidationError):
        _compose(request)
