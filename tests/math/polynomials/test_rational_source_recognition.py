"""Shared source recognition retains the canonical exponent envelope."""

import pytest
import sympy

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials._conversions import rational_function_from_sympy
from jacobian.math.polynomials.rational_functions.composition import compose_maps
from jacobian.math.polynomials.rational_functions.gradient import (
    RationalFunctionGradient,
    gradient,
)
from jacobian.math.polynomials.rational_functions.maps import jacobian_matrix
from jacobian.math.polynomials.rational_functions.values import RationalFunctionMap
from jacobian.math.polynomials.values import RationalFunction


def _source(exponent: int, axes: tuple[str, ...]) -> RationalFunction:
    return rational_function_from_sympy(sympy.Symbol("x") ** exponent, axes)


def _map(source: RationalFunction) -> RationalFunctionMap:
    return RationalFunctionMap(
        source_variables=source.variables,
        target_coordinates=("u",),
        components=(source,),
    )


def _constant_outer() -> RationalFunctionMap:
    return RationalFunctionMap(
        source_variables=("u",),
        target_coordinates=("v",),
        components=(rational_function_from_sympy(1, ("u",)),),
    )


@pytest.mark.parametrize("axes", [("x",), ("unused", "x")])
def test_degree_65_gradient_and_jacobian_preserve_axes(axes: tuple[str, ...]) -> None:
    source = _source(65, axes)
    result = gradient(source)
    restored = RationalFunctionGradient.model_validate_json(result.model_dump_json())
    expected = tuple(
        rational_function_from_sympy(
            65 * sympy.Symbol("x") ** 64 if axis == "x" else 0, axes
        )
        for axis in axes
    )
    assert restored.source == source
    assert restored.variables == axes
    assert restored.partial_derivatives == expected
    assert jacobian_matrix(_map(source)).entries == (expected,)
    # Consume an actual newly admitted derivative unchanged after serialization.
    assert gradient(restored.partial_derivatives[axes.index("x")]).partial_derivatives[
        axes.index("x")
    ] == rational_function_from_sympy(4160 * sympy.Symbol("x") ** 63, axes)


@pytest.mark.parametrize("exponent", [65, 128])
@pytest.mark.parametrize("axes", [("x",), ("unused", "x")])
def test_constant_composition_recognizes_wide_source_without_expansion(
    exponent: int, axes: tuple[str, ...]
) -> None:
    inner = _map(_source(exponent, axes))
    result = compose_maps(_constant_outer(), inner)
    assert result.inner == inner
    assert result.composite.source_variables == axes
    assert result.composite.components == (rational_function_from_sympy(1, axes),)
    assert result.construction_locus_guard == ()


@pytest.mark.parametrize("exponent", [128, -65])
def test_wide_source_still_obeys_derivative_result_envelope(exponent: int) -> None:
    with pytest.raises(OperationResourceAdmissionError) as error:
        gradient(_source(exponent, ("x",)))
    assert (
        error.value.errors()[0]["type"] == "rational_function.gradient.result_exponent"
    )


@pytest.mark.parametrize("operation", ["gradient", "jacobian", "composition"])
def test_wide_source_recognition_still_rejects_common_factors(operation: str) -> None:
    source = _source(65, ("x",))
    unreduced = source.model_copy(update={"denominator": _source(64, ("x",)).numerator})
    with pytest.raises(OperationDomainValidationError) as error:
        if operation == "gradient":
            gradient(unreduced)
        elif operation == "jacobian":
            jacobian_matrix(_map(unreduced))
        else:
            compose_maps(_constant_outer(), _map(unreduced))
    assert error.value.errors()[0]["type"] == "polynomial.not_coprime"


def test_forged_source_shape_budget_is_a_structured_native_error() -> None:
    source = _source(128, ("x",))
    term = source.numerator.terms[0].model_copy(update={"exponents": (129,)})
    malformed = source.model_copy(
        update={"numerator": source.numerator.model_copy(update={"terms": (term,)})}
    )
    with pytest.raises(OperationDomainValidationError) as error:
        gradient(malformed)
    assert error.value.errors()[0]["type"] == "rational_function.source_admission"


@pytest.mark.parametrize("exponent", [-65, -128])
def test_constant_composition_admits_retained_denominator_guards(exponent: int) -> None:
    inner = _map(_source(exponent, ("x",)))
    with pytest.raises(OperationResourceAdmissionError) as error:
        compose_maps(_constant_outer(), inner)
    assert (
        error.value.errors()[0]["type"]
        == "rational_function_map.compose.guard_exponent"
    )
