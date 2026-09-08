"""Exact identities for rational metric pullbacks."""

import pytest
from sympy import symbols

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.differential.metrics import RationalCoordinateMetric
from jacobian.math.geometry.differential.pullback import pullback_metric
from jacobian.math.geometry.differential.values import RationalCoordinateTensor
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_function_to_sympy,
)
from jacobian.math.polynomials.rational_functions.values import RationalFunctionMap


def rf(value, axis):
    return rational_function_from_sympy(value, tuple(str(item) for item in axis))


def metric(values, axis):
    return RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=tuple(axis),
            variance=("COVARIANT", "COVARIANT"),
            components=tuple(rf(value, axis) for value in values),
        )
    )


def map_value(values, source, target):
    return RationalFunctionMap(
        source_variables=tuple(source),
        target_coordinates=tuple(target),
        components=tuple(rf(value, source) for value in values),
    )


def test_identity_map_recovers_metric() -> None:
    y = symbols("y")
    source = metric((1 + y**2,), ("y",))
    x = symbols("x")
    result = pullback_metric(source, map_value((x,), ("x",), ("y",)))
    assert [
        rational_function_to_sympy(value) for value in result.pullback.components
    ] == [1 + symbols("x") ** 2]


def test_rank_deficient_map_returns_degenerate_tensor() -> None:
    x, _y = symbols("x y")
    result = pullback_metric(
        metric((1, 0, 0, 1), ("u", "v")),
        map_value((x, x), ("x", "y"), ("u", "v")),
    )
    assert [
        rational_function_to_sympy(value) for value in result.pullback.components
    ] == [2, 0, 0, 0]


def test_rational_substitution_retains_map_and_metric_guards() -> None:
    x, y = symbols("x y")
    source_tensor = RationalCoordinateTensor(
        coordinate_axis=("y",),
        variance=("COVARIANT", "COVARIANT"),
        components=(rf(1 / y, (y,)),),
        retained_nonzero_denominators=(rf(y, (y,)).numerator,),
    )
    source = RationalCoordinateMetric(tensor=source_tensor)
    result = pullback_metric(
        source,
        map_value((x / (x - 1),), ("x",), ("y",)),
    )
    assert len(result.pullback_locus_guard) == 3
    assert all(guard.terms for guard in result.pullback_locus_guard)


def test_metric_determinant_guard_is_substituted_and_singular_pullback_rejected() -> (
    None
):
    u, _v, x, y = symbols("u v x y")
    source = metric((1, 0, 0, u**2), ("u", "v"))
    result = pullback_metric(source, map_value((x, y), ("x", "y"), ("u", "v")))
    assert any(
        guard.terms[-1].coefficient.num == 1 and guard.terms[-1].exponents == (2, 0)
        for guard in result.pullback_locus_guard
    )
    with pytest.raises(OperationDomainValidationError, match="vanishes identically"):
        pullback_metric(
            metric((1, 0, 0, 0), ("u", "v")),
            map_value((x, y), ("x", "y"), ("u", "v")),
        )
