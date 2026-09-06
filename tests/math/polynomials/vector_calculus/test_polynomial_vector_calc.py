"""Tests for canonical polynomial vector-calculus operations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)
from jacobian.math.polynomials.vector_calculus._models import (
    CurlRequest,
    DirectionalDerivativeRequest,
    ScalarFieldRequest,
    ScalarResult,
    VectorFieldRequest,
    VectorResult,
)
from jacobian.math.polynomials.vector_calculus._tools import TOOLS
from jacobian.math.polynomials.vector_calculus.operations import (
    curl,
    directional_derivative,
    divergence,
    gradient,
    laplacian,
    verify_curl,
    verify_directional_derivative,
    verify_divergence,
    verify_gradient,
    verify_laplacian,
)


def _run_gradient(request: ScalarFieldRequest) -> VectorResult:
    return gradient(request.polynomial)


def _run_laplacian(request: ScalarFieldRequest) -> ScalarResult:
    return laplacian(request.polynomial)


def _run_directional_derivative(request: DirectionalDerivativeRequest) -> ScalarResult:
    return directional_derivative(request.polynomial, request.direction)


def _run_divergence(request: VectorFieldRequest) -> ScalarResult:
    return divergence(request.components)


def _run_curl(request: CurlRequest) -> VectorResult:
    return curl(request.components)


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
    result = _run_gradient(ScalarFieldRequest(polynomial=source))
    assert result.components == (
        _polynomial(("x", "y"), {(1, 0): 2}),
        _polynomial(("x", "y"), {(0, 1): 2}),
    )


def test_serialized_vector_calculus_claims_verify_retained_sources() -> None:
    scalar = _polynomial(("x", "y"), {(2, 0): 1, (0, 2): 1})
    vector = (
        _polynomial(("x", "y", "z"), {(1, 0, 0): 1}),
        _polynomial(("x", "y", "z"), {(0, 1, 0): 1}),
        _polynomial(("x", "y", "z"), {(0, 0, 1): 1}),
    )
    direction = (
        CanonicalRational(num=1, den=2),
        CanonicalRational(num=1, den=1),
    )
    gradient_claim = gradient(scalar)
    laplacian_claim = laplacian(scalar)
    directional_claim = directional_derivative(scalar, direction)
    divergence_claim = divergence(vector)
    curl_claim = curl(vector)

    assert verify_gradient(
        VectorResult.model_validate_json(gradient_claim.model_dump_json())
    )
    assert verify_laplacian(
        ScalarResult.model_validate_json(laplacian_claim.model_dump_json())
    )
    assert verify_directional_derivative(
        ScalarResult.model_validate_json(directional_claim.model_dump_json())
    )
    assert verify_divergence(
        ScalarResult.model_validate_json(divergence_claim.model_dump_json())
    )
    assert verify_curl(VectorResult.model_validate_json(curl_claim.model_dump_json()))
    assert not verify_gradient(
        gradient_claim.model_copy(update={"source_polynomial": laplacian_claim.result})
    )
    assert not verify_directional_derivative(
        directional_claim.model_copy(
            update={
                "direction": (
                    direction[0],
                    CanonicalRational(num=2, den=1),
                )
            }
        )
    )

    missing_source = gradient_claim.model_dump(mode="json")
    missing_source.pop("source_polynomial")
    assert not verify_gradient(
        VectorResult.model_validate_json(json.dumps(missing_source))
    )

    wrong_source_kind = gradient_claim.model_dump(mode="json")
    wrong_source_kind.pop("source_polynomial")
    wrong_source_kind["source_components"] = [scalar.model_dump(mode="json")]
    assert not verify_gradient(
        VectorResult.model_validate_json(json.dumps(wrong_source_kind))
    )

    wrong_axis = gradient_claim.model_dump(mode="json")
    wrong_axis["source_polynomial"]["variables"] = ["u", "v"]
    assert not verify_gradient(VectorResult.model_validate_json(json.dumps(wrong_axis)))


def test_renaming_the_declared_axis_transports_the_gradient() -> None:
    original = _run_gradient(
        ScalarFieldRequest(polynomial=_polynomial(("x", "y"), {(2, 0): 1, (0, 1): 3}))
    )
    renamed = _run_gradient(
        ScalarFieldRequest(polynomial=_polynomial(("u", "v"), {(2, 0): 1, (0, 1): 3}))
    )

    assert tuple(component.polynomial for component in original.components) == tuple(
        component.polynomial for component in renamed.components
    )
    assert all(component.variables == ("x", "y") for component in original.components)
    assert all(component.variables == ("u", "v") for component in renamed.components)


def test_laplacian_returns_a_composable_polynomial() -> None:
    source = _polynomial(("x", "y"), {(3, 0): 1, (0, 3): 1})
    result = _run_laplacian(ScalarFieldRequest(polynomial=source))
    assert result.result == _polynomial(
        ("x", "y"),
        {(1, 0): 6, (0, 1): 6},
    )


def test_directional_derivative_uses_exact_rational_coordinates() -> None:
    source = _polynomial(("x", "y"), {(2, 0): 1, (0, 2): 1})
    result = _run_directional_derivative(
        DirectionalDerivativeRequest(
            polynomial=source,
            direction=(
                CanonicalRational(num=1, den=2),
                CanonicalRational(num=1, den=1),
            ),
        )
    )
    assert result.result == _polynomial(
        ("x", "y"),
        {(1, 0): 1, (0, 1): 2},
    )


def test_divergence_uses_one_authoritative_axis() -> None:
    result = _run_divergence(
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
    result = _run_curl(
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
    request = VectorFieldRequest(
        components=(
            _polynomial(variables, first),
            _polynomial(variables, second),
        )
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        _run_divergence(request)
    assert exc_info.value.errors()[0]["type"] == (
        "polynomial_vector_calc.derivative_term_budget"
    )


def test_direction_length_must_match_polynomial_axis() -> None:
    with pytest.raises(ValidationError) as exc_info:
        DirectionalDerivativeRequest(
            polynomial=_polynomial(("x", "y"), {(1, 0): 1}),
            direction=(CanonicalRational(num=1, den=1),),
        )
    assert (
        exc_info.value.errors()[0]["type"] == "polynomial_vector_calc.direction_length"
    )
