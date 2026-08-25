"""Tests for canonical polynomial map operations."""

from __future__ import annotations

from fractions import Fraction

from tests.math.polynomials._support import polynomial_validation_error

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials.maps._models import (
    CompositionRequest,
    EvalRequest,
    RationalPolynomialMap,
    VariablePoint,
)
from jacobian.math.polynomials.maps._operations import (
    compose_polynomials,
    compute_jacobian,
    evaluate_polynomial,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
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
    assert evaluate_polynomial(request).value == CanonicalRational(num="11", den="1")


def test_evaluation_requires_the_complete_ordered_axis() -> None:
    with polynomial_validation_error():
        EvalRequest(
            polynomial=_polynomial(("x", "y"), {(1, 0): 1}),
            point=VariablePoint(
                variables=("x",),
                values=(CanonicalRational(num="1", den="1"),),
            ),
        )


def test_evaluation_rejects_a_point_whose_exact_value_exceeds_result_bound() -> None:
    with polynomial_validation_error():
        EvalRequest(
            polynomial=_polynomial(("x",), {(64,): 1}),
            point=VariablePoint(
                variables=("x",),
                values=(CanonicalRational(num="1" + "0" * 600, den="1"),),
            ),
        )


def test_jacobian_entries_are_directly_composable_polynomials() -> None:
    request = RationalPolynomialMap(
        input_variables=("x", "y"),
        output_polynomials=(
            _polynomial(("x", "y"), {(2, 0): 1}),
            _polynomial(("x", "y"), {(0, 2): 1}),
        ),
    )
    result = compute_jacobian(request)
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
    result = compose_polynomials(
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
    with polynomial_validation_error():
        CompositionRequest(
            outer=_polynomial(("u", "v"), {(1, 0): 1}),
            inner=_polynomial(("x",), {(1,): 1}),
            inner_variable="x",
            outer_variable="u",
        )
