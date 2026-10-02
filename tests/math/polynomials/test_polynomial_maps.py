"""Tests for canonical polynomial map operations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from fractions import Fraction
from itertools import islice, product
from math import prod

import pytest
from tests.math.polynomials._support import polynomial_validation_error

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials._elementary_kernel import rational_polynomial_evaluate
from jacobian.math.polynomials.maps._models import (
    CompositionRequest,
    CompositionResult,
    EvalRequest,
    EvalResult,
    VariablePoint,
)
from jacobian.math.polynomials.maps.operations import (
    compose_polynomials,
    evaluate_polynomial,
    jacobian_matrix,
    verify_jacobian,
)
from jacobian.math.polynomials.maps.values import RationalPolynomialMap
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _evaluate(request: EvalRequest) -> EvalResult:
    return evaluate_polynomial(request.polynomial, request.point)


def _compose(request: CompositionRequest) -> CompositionResult:
    return compose_polynomials(
        request.outer,
        request.inner,
        outer_variable=request.outer_variable,
        inner_variable=request.inner_variable,
    )


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


def test_evaluation_returns_a_canonical_rational() -> None:
    request = EvalRequest(
        polynomial=_polynomial(("x", "y"), {(2, 0): 1, (0, 1): 2}),
        point=VariablePoint(
            variables=("x", "y"),
            values=(
                CanonicalRational(num=3, den=1),
                CanonicalRational(num=1, den=1),
            ),
        ),
    )
    assert _evaluate(request).value == CanonicalRational(num=11, den=1)


def test_evaluation_requires_the_complete_ordered_axis() -> None:
    request = EvalRequest(
        polynomial=_polynomial(("x", "y"), {(1, 0): 1}),
        point=VariablePoint(
            variables=("x",),
            values=(CanonicalRational(num=1, den=1),),
        ),
    )
    with pytest.raises(OperationDomainValidationError):
        _evaluate(request)


def test_evaluation_rejects_a_point_whose_exact_value_exceeds_result_bound() -> None:
    request = EvalRequest(
        polynomial=_polynomial(("x",), {(64,): 1}),
        point=VariablePoint(
            variables=("x",),
            values=(CanonicalRational(num=10**600, den=1),),
        ),
    )
    with pytest.raises(OperationDomainValidationError):
        _evaluate(request)


@pytest.mark.parametrize("degree", (64, 65, 127))
@pytest.mark.parametrize("name", ("x", "t"))
@pytest.mark.parametrize("coordinate", (Fraction(0), Fraction(-2), Fraction(2, 3)))
def test_unified_evaluation_retains_univariate_envelope(
    degree: int, name: str, coordinate: Fraction
) -> None:
    source = _polynomial((name,), {(degree,): Fraction(-3, 7), (0,): Fraction(2, 5)})
    scalar = CanonicalRational.from_fraction(coordinate)
    point = VariablePoint(variables=(name,), values=(scalar,))
    result = evaluate_polynomial(source, point)
    assert result.value.as_fraction() == Fraction(
        -3, 7
    ) * coordinate**degree + Fraction(2, 5)
    assert result.value == rational_polynomial_evaluate(source, scalar).value
    decoded = EvalResult.model_validate_json(result.model_dump_json(), strict=True)
    assert decoded == result
    identity = _polynomial((name,), {(1,): 1})
    assert (
        evaluate_polynomial(
            identity, VariablePoint(variables=(name,), values=(decoded.value,))
        )
        == result
    )


def test_unified_evaluation_preserves_dense_and_high_coefficient_scalar_cases() -> None:
    source = _polynomial(("t",), {(degree,): 10**255 for degree in range(128)})
    point = VariablePoint(variables=("t",), values=(CanonicalRational(num=1, den=1),))
    assert evaluate_polynomial(source, point).value.num == 128 * 10**255


@pytest.mark.parametrize("variables", (("t",), ("y", "x"), tuple("abcdefgh")))
def test_unified_evaluation_preserves_zero_and_multivariate_fraction_oracle(
    variables: tuple[str, ...],
) -> None:
    values = tuple(Fraction(index - 2, index + 1) for index in range(len(variables)))
    point = VariablePoint(
        variables=variables,
        values=tuple(CanonicalRational.from_fraction(value) for value in values),
    )
    assert evaluate_polynomial(_polynomial(variables, {}), point).value.num == 0
    support = tuple(islice(product(range(3), repeat=len(variables)), 9))
    terms = {key: Fraction(index + 1, index + 2) for index, key in enumerate(support)}
    source = _polynomial(variables, terms)
    expected = sum(
        (
            coefficient
            * prod(value**exponent for value, exponent in zip(values, key, strict=True))
            for key, coefficient in terms.items()
        ),
        Fraction(),
    )
    assert evaluate_polynomial(source, point).value.as_fraction() == expected


def test_unified_evaluation_preserves_multivariate_source_term_boundary() -> None:
    terms = dict.fromkeys(islice(product(range(20), repeat=2), 256), 1)
    source = _polynomial(("x", "y"), terms)
    coordinates = (Fraction(1, 2), Fraction(-2, 3))
    point = VariablePoint(
        variables=source.variables,
        values=tuple(CanonicalRational.from_fraction(value) for value in coordinates),
    )
    expected = sum(
        (coordinates[0] ** left * coordinates[1] ** right for left, right in terms),
        Fraction(),
    )
    assert evaluate_polynomial(source, point).value.as_fraction() == expected


@pytest.mark.parametrize(
    ("variables", "terms"),
    [
        (("x",), {(128,): 1}),
        (("x",), {(1,): 10**256}),
        (("x", "y"), {(64, 1): 1}),
        (("x", "y"), {(1, 0): 10**128}),
        (("x", "y"), dict.fromkeys(islice(product(range(20), repeat=2), 257), 1)),
    ],
)
def test_unified_evaluation_preserves_source_regime_boundaries(
    variables: tuple[str, ...], terms: dict[tuple[int, ...], int]
) -> None:
    source = _polynomial(variables, terms)
    point = VariablePoint(
        variables=variables, values=(CanonicalRational(num=0, den=1),) * len(variables)
    )
    with pytest.raises(OperationDomainValidationError):
        evaluate_polynomial(source, point)


@pytest.mark.parametrize(
    ("variables", "point_variables"),
    [(("t",), ("x",)), (("x", "y"), ("y", "x"))],
)
def test_unified_evaluation_validates_full_axis_before_routing(
    variables: tuple[str, ...], point_variables: tuple[str, ...]
) -> None:
    source = _polynomial(variables, {})
    point = VariablePoint(
        variables=point_variables,
        values=(CanonicalRational(num=1, den=1),) * len(point_variables),
    )
    with pytest.raises(OperationDomainValidationError, match="complete ordered axis"):
        evaluate_polynomial(source, point)
    short_point = point.model_copy(update={"variables": variables, "values": ()})
    with pytest.raises(OperationDomainValidationError, match="complete ordered axis"):
        evaluate_polynomial(source, short_point)


def test_jacobian_entries_are_directly_composable_polynomials() -> None:
    request = RationalPolynomialMap(
        input_variables=("x", "y"),
        output_polynomials=(
            _polynomial(("x", "y"), {(2, 0): 1}),
            _polynomial(("x", "y"), {(0, 2): 1}),
        ),
    )
    result = jacobian_matrix(request)
    assert result.source == request
    assert result.matrix.input_variables == ("x", "y")
    assert result.matrix.entries == (
        (_polynomial(("x", "y"), {(1, 0): 2}), _polynomial(("x", "y"), {})),
        (_polynomial(("x", "y"), {}), _polynomial(("x", "y"), {(0, 1): 2})),
    )
    decoded = type(result).model_validate_json(result.model_dump_json())
    assert verify_jacobian(decoded)
    payload = result.model_dump(mode="json")
    payload["matrix"]["entries"][0][0]["polynomial"]["terms"][0]["coefficient"] = {
        "num": "99",
        "den": "1",
    }
    assert not verify_jacobian(type(result).model_validate_json(json.dumps(payload)))


def test_jacobian_preserves_axes_for_an_empty_output_map() -> None:
    source = RationalPolynomialMap(input_variables=("x",), output_polynomials=())
    result = jacobian_matrix(source)

    assert result.source == source
    assert result.matrix.input_variables == ("x",)
    assert result.matrix.entries == ()
    assert verify_jacobian(type(result).model_validate_json(result.model_dump_json()))


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


def _evaluate_direct(polynomial: RationalPolynomial, value: Fraction) -> Fraction:
    """Hand evaluation of a univariate sparse polynomial, no kernel reuse."""
    total = Fraction(0)
    for term in polynomial.polynomial.terms:
        total += term.coefficient.as_fraction() * value ** term.exponents[0]
    return total


def test_composition_matches_pointwise_evaluation_oracle() -> None:
    outer = _polynomial(("u",), {(3,): 1, (1,): 2})
    inner = _polynomial(("x",), {(2,): 1, (1,): 2})
    result = _compose(
        CompositionRequest(
            outer=outer,
            inner=inner,
            inner_variable="x",
            outer_variable="u",
        )
    )
    for point in (Fraction(0), Fraction(1), Fraction(-1, 2), Fraction(3, 2)):
        assert _evaluate_direct(result.polynomial, point) == _evaluate_direct(
            outer, _evaluate_direct(inner, point)
        )
    # Swapping inner and outer is a different polynomial.
    swapped = _compose(
        CompositionRequest(
            outer=_polynomial(("u",), {(2,): 1, (1,): 2}),
            inner=_polynomial(("x",), {(3,): 1, (1,): 2}),
            inner_variable="x",
            outer_variable="u",
        )
    )
    assert swapped.polynomial != result.polynomial
