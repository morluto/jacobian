"""Polynomial Hermite reduction uses bounded coefficientwise integration."""

import pytest
from sympy import Rational, Symbol, diff, expand

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_function_to_sympy,
)
from jacobian.math.polynomials.rational_functions._models import (
    HermiteReductionRequest,
    HermiteReductionResult,
)
from jacobian.math.polynomials.rational_functions._tools import (
    compute_hermite_reduction,
)
from jacobian.math.polynomials.rational_functions.operations import (
    verify_hermite_reduction,
)


def _compute(expression: object) -> HermiteReductionResult:
    return compute_hermite_reduction(
        HermiteReductionRequest(
            function=rational_function_from_sympy(expression, ("t",)),
        )
    )


@pytest.mark.parametrize("degree", [7, 31, 63])
def test_polynomial_primitive_reaches_carrier_degree_boundary(degree: int) -> None:
    t = Symbol("t")
    source = sum(Rational(i + 1, i + 2) * t**i for i in range(degree + 1))
    result = _compute(source)
    primitive = rational_function_to_sympy(result.rational_part)
    assert expand(diff(primitive, t) - source) == 0
    assert primitive.subs(t, 0) == 0
    assert not result.remainder.numerator.terms
    decoded = type(result).model_validate_json(result.model_dump_json())
    assert decoded == result
    assert verify_hermite_reduction(decoded)


def test_polynomial_large_numerator_does_not_reserve_denominator_digits() -> None:
    t = Symbol("t")
    result = _compute((10**128 - 1) * t**63)
    assert (
        rational_function_to_sympy(result.rational_part)
        == Rational(10**128 - 1, 64) * t**64
    )


def test_primitive_beyond_carrier_degree_is_resource_refused() -> None:
    with pytest.raises(OperationResourceAdmissionError):
        _compute(Symbol("t") ** 64)


def test_polynomial_denominator_growth_is_resource_refused() -> None:
    with pytest.raises(OperationResourceAdmissionError):
        _compute(Symbol("t") ** 63 / (10**128 - 1))
