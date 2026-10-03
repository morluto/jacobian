"""Tests for canonical polynomial vector-calculus operations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from fractions import Fraction
from itertools import combinations_with_replacement, islice, product
from math import comb

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


def _termwise_gradient(
    polynomial: RationalPolynomial,
) -> tuple[RationalPolynomial, ...]:
    """Compute the defining sparse partials without invoking the kernel."""

    components: list[RationalPolynomial] = []
    for axis in range(len(polynomial.variables)):
        terms: dict[tuple[int, ...], Fraction] = {}
        for term in polynomial.polynomial.terms:
            exponent = term.exponents[axis]
            if exponent == 0:
                continue
            derived_exponents = list(term.exponents)
            derived_exponents[axis] -= 1
            key = tuple(derived_exponents)
            terms[key] = terms.get(key, Fraction()) + (
                term.coefficient.as_fraction() * exponent
            )
        components.append(_polynomial(polynomial.variables, terms))
    return tuple(components)


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "polynomial_field.scalar.gradient.compute",
        "polynomial_field.scalar.laplacian.compute",
        "polynomial_field.scalar.directional_derivative.compute",
        "polynomial_field.vector.divergence.compute",
        "polynomial_field.vector.curl.compute",
    }


def test_native_vector_operations_reject_malformed_fields_at_admission() -> None:
    with pytest.raises(OperationDomainValidationError) as exc_info:
        divergence(())
    assert (
        exc_info.value.errors()[0]["type"]
        == "polynomial_vector_calc.empty_vector_field"
    )

    x = _polynomial(("x", "y", "z"), {(1, 0, 0): 1})
    with pytest.raises(OperationDomainValidationError) as exc_info:
        divergence((x, x))
    assert (
        exc_info.value.errors()[0]["type"] == "polynomial_vector_calc.component_count"
    )

    planar = _polynomial(("x", "y"), {(1, 0): 1})
    with pytest.raises(OperationDomainValidationError) as exc_info:
        curl((planar, planar))
    assert (
        exc_info.value.errors()[0]["type"] == "polynomial_vector_calc.curl_dimensions"
    )

    y = _polynomial(("y", "x", "z"), {(0, 1, 0): 1})
    with pytest.raises(OperationDomainValidationError) as exc_info:
        divergence((x, y, x))
    assert exc_info.value.errors()[0]["type"] == "polynomial_vector_calc.ordered_ring"


def test_gradient_admits_sparse_inactive_axes_beyond_dense_proxy() -> None:
    """A vector result retains only the axes on which each monomial survives."""

    variables = ("x", "a", "b", "c", "d", "e", "f", "g")
    source = _polynomial(
        variables,
        {
            (degree, 0, 0, 0, 0, 0, 0, 0): 10**127 if degree == 34 else 1
            for degree in range(34, 1, -1)
        },
    )
    expected_components = _termwise_gradient(source)

    native = gradient(source)
    assert native.components == expected_components
    assert sum(len(component.polynomial.terms) for component in native.components) == 33
    assert all(not component.polynomial.terms for component in native.components[1:])

    public = TOOLS[0].run(ScalarFieldRequest(polynomial=source))
    assert (
        VectorResult.model_validate_json(json.dumps(public.model_dump(mode="json")))
        == native
    )
    assert verify_gradient(VectorResult.model_validate_json(native.model_dump_json()))


def test_gradient_exact_and_termwise_oracles_cover_simple_and_mixed_support() -> None:
    simple = _polynomial(("x", "y"), {(2, 0): 1, (0, 2): 1})
    assert _run_gradient(ScalarFieldRequest(polynomial=simple)).components == (
        _polynomial(("x", "y"), {(1, 0): 2}),
        _polynomial(("x", "y"), {(0, 1): 2}),
    )

    source = _polynomial(
        ("x", "y", "z"),
        {
            (2, 0, 1): 3,
            (1, 1, 0): Fraction(5, 7),
            (0, 4, 0): -2,
        },
    )

    assert gradient(source).components == _termwise_gradient(source)


def _termwise_scalar_derivative(
    polynomial: RationalPolynomial,
    *,
    order: int,
    weights: tuple[Fraction, ...],
) -> RationalPolynomial:
    """Independent exact coefficient-map definition, without SymPy."""

    terms: dict[tuple[int, ...], Fraction] = {}
    for term in polynomial.polynomial.terms:
        for axis, weight in enumerate(weights):
            exponent = term.exponents[axis]
            if exponent < order:
                continue
            key = tuple(
                value - order if index == axis else value
                for index, value in enumerate(term.exponents)
            )
            factor = exponent if order == 1 else exponent * (exponent - 1)
            terms[key] = terms.get(key, Fraction()) + (
                term.coefficient.as_fraction() * factor * weight
            )
    return _polynomial(polynomial.variables, terms)


def _assert_scalar_derivatives(
    source: RationalPolynomial, weights: tuple[Fraction, ...]
) -> tuple[ScalarResult, ScalarResult]:
    direction = tuple(CanonicalRational.from_fraction(weight) for weight in weights)
    laplace = laplacian(source)
    directional = directional_derivative(source, direction)
    assert laplace.result == _termwise_scalar_derivative(
        source, order=2, weights=(Fraction(1),) * len(source.variables)
    )
    assert directional.result == _termwise_scalar_derivative(
        source, order=1, weights=weights
    )
    for result in (laplace, directional):
        assert result.result.variables == source.variables
        assert ScalarResult.model_validate_json(result.model_dump_json()) == result
    assert verify_laplacian(laplace)
    assert verify_directional_derivative(directional)
    return laplace, directional


@pytest.mark.parametrize(
    "variables",
    [("x", "y", "z"), ("x", "y", "z", "w"), ("w", "z", "y", "x")],
)
def test_scalar_derivatives_admit_sparse_retained_axes(
    variables: tuple[str, ...],
) -> None:
    source = _polynomial(
        variables,
        {
            tuple(degree if name == "x" else 0 for name in variables): 1
            for degree in range(65)
        },
    )
    weights = tuple(Fraction(name == "x") for name in variables)
    laplace, directional = _assert_scalar_derivatives(source, weights)
    assert len(laplace.result.polynomial.terms) == 63
    assert len(directional.result.polynomial.terms) == 64


def test_harmonic_scalar_derivative_cancels_on_retained_axes() -> None:
    source = _polynomial(
        ("x", "y", "z", "w"),
        {(64 - k, k, 0, 0): (-1) ** (k // 2) * comb(64, k) for k in range(65)},
    )
    laplace, _ = _assert_scalar_derivatives(
        source, (Fraction(1, 2), Fraction(-2, 3), Fraction(), Fraction())
    )
    assert not laplace.result.polynomial.terms


def _fully_active_scalar(terms: int) -> RationalPolynomial:
    # Different derivative axes have distinct exponent residues modulo three.
    return _polynomial(
        ("x", "y", "z", "w"),
        {
            tuple(3 * value + 2 for value in exponents): 1
            for exponents in islice(product(range(5), repeat=4), terms)
        },
    )


def test_scalar_derivative_exact_output_term_boundary() -> None:
    source = _fully_active_scalar(64)
    results = _assert_scalar_derivatives(source, (Fraction(1),) * 4)
    assert all(len(result.result.polynomial.terms) == 256 for result in results)

    source = _fully_active_scalar(65)
    direction = (CanonicalRational(num=1, den=1),) * 4
    assert (
        len(
            _termwise_scalar_derivative(
                source, order=2, weights=(Fraction(1),) * 4
            ).polynomial.terms
        )
        == 260
    )
    assert (
        len(
            _termwise_scalar_derivative(
                source, order=1, weights=(Fraction(1),) * 4
            ).polynomial.terms
        )
        == 260
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        laplacian(source)
    assert (
        exc_info.value.errors()[0]["type"]
        == "polynomial_vector_calc.derivative_term_budget"
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        directional_derivative(source, direction)
    assert (
        exc_info.value.errors()[0]["type"]
        == "polynomial_vector_calc.derivative_term_budget"
    )


@pytest.mark.parametrize("weight", [Fraction(), Fraction(-2, 3)])
def test_direction_weights_control_surviving_support(weight: Fraction) -> None:
    source = _fully_active_scalar(65)
    weights = (weight, Fraction(), Fraction(), Fraction())
    direction = tuple(CanonicalRational.from_fraction(value) for value in weights)
    result = directional_derivative(source, direction)
    assert result.result == _termwise_scalar_derivative(
        source, order=1, weights=weights
    )
    assert len(result.result.polynomial.terms) == (65 if weight else 0)
    assert verify_directional_derivative(
        ScalarResult.model_validate_json(result.model_dump_json())
    )


def test_scalar_support_union_admits_collisions_at_source_limit() -> None:
    # 256 homogeneous quartics on eight axes have <=36 quadratic and <=120
    # cubic output monomials even when >256 partial contributions survive.
    variables = tuple("abcdefgh")
    quartics = (
        tuple(indices.count(axis) for axis in range(8))
        for indices in combinations_with_replacement(range(8), 4)
    )
    exponents = sorted(
        quartics, key=lambda key: sum(value > 1 for value in key), reverse=True
    )[:256]
    source = _polynomial(variables, dict.fromkeys(exponents, 1))
    assert sum(sum(value > 0 for value in key) for key in exponents) > 256
    assert sum(sum(value > 1 for value in key) for key in exponents) > 256
    results = _assert_scalar_derivatives(source, (Fraction(1),) * 8)
    assert len(results[0].result.polynomial.terms) <= 36
    assert len(results[1].result.polynomial.terms) <= 120


def test_scalar_derivative_coefficient_growth_remains_exact_and_decodable() -> None:
    variables = tuple("abcdefgh")
    denominators = tuple(10**127 + 2 * index + 1 for index in range(8))
    sources = tuple(
        _polynomial(
            variables,
            {
                tuple(order if axis == index else 0 for axis in range(8)): Fraction(
                    10**127, denominator
                )
                for index, denominator in enumerate(denominators)
            },
        )
        for order in (1, 2)
    )
    weights = tuple(Fraction(10**127, value + 32) for value in denominators)
    first = _assert_scalar_derivatives(sources[0], weights)[1]
    second = _assert_scalar_derivatives(sources[1], weights)[0]
    assert first.result.polynomial.terms[0].coefficient.den > 10**1900
    assert second.result.polynomial.terms[0].coefficient.den > 10**950


@pytest.mark.parametrize("direction_length", [0, 3, 5])
def test_native_direction_length_precedes_support_admission(
    direction_length: int,
) -> None:
    direction = (CanonicalRational(num=0, den=1),) * direction_length
    with pytest.raises(OperationDomainValidationError) as caught:
        directional_derivative(_fully_active_scalar(65), direction)
    assert caught.value.errors()[0]["type"] == "polynomial_vector_calc.direction_length"


def test_zero_direction_preserves_source_and_coordinate_validation() -> None:
    source = _fully_active_scalar(65)
    zero = (CanonicalRational(num=0, den=1),) * 4
    over_degree = _polynomial(source.variables, {(65, 0, 0, 0): 1})
    over_height = _polynomial(source.variables, {(1, 0, 0, 0): 10**128})
    over_total_degree = _polynomial(source.variables, {(33, 33, 0, 0): 1})
    for invalid_source in (
        over_degree,
        over_total_degree,
        over_height,
        _fully_active_scalar(257),
    ):
        with pytest.raises(OperationDomainValidationError):
            directional_derivative(invalid_source, zero)
        with pytest.raises(OperationDomainValidationError):
            laplacian(invalid_source)
    with pytest.raises(OperationDomainValidationError) as caught:
        directional_derivative(
            source, (CanonicalRational(num=10**128, den=1), *zero[1:])
        )
    assert caught.value.errors()[0]["type"] == (
        "polynomial_vector_calc.direction_coordinate_bound"
    )


def test_zero_and_multiaffine_sources_have_no_second_partial_support() -> None:
    variables = tuple("abcdefgh")
    for terms in ({}, dict.fromkeys(islice(product(range(2), repeat=8), 65), 1)):
        source = _polynomial(variables, terms)
        laplace, directional = _assert_scalar_derivatives(source, (Fraction(),) * 8)
        assert not laplace.result.polynomial.terms
        assert not directional.result.polynomial.terms


def test_scalar_derivative_verifiers_reject_changed_result_and_direction() -> None:
    source = _fully_active_scalar(65)
    direction = (CanonicalRational(num=1, den=1),) + (
        CanonicalRational(num=0, den=1),
    ) * 3
    result = directional_derivative(source, direction)
    assert not verify_directional_derivative(
        result.model_copy(update={"result": source})
    )
    assert not verify_directional_derivative(
        result.model_copy(update={"direction": direction[1:]})
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
