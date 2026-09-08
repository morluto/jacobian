"""Exact identities for the rational Laplace--Beltrami operation."""

from fractions import Fraction
from time import monotonic

import pytest
from sympy import cancel, symbols

from jacobian._execution import OperationExecutionTimeoutError, request_execution
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.differential.laplace_beltrami import (
    RationalLaplaceBeltramiResult,
    laplace_beltrami,
)
from jacobian.math.geometry.differential.metrics import RationalCoordinateMetric
from jacobian.math.geometry.differential.values import RationalCoordinateTensor
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_function_to_sympy,
)
from jacobian.math.polynomials.values import RationalFunction


def _metric(
    values: tuple[object, ...], axis: tuple[str, ...]
) -> RationalCoordinateMetric:
    return RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=axis,
            variance=("COVARIANT", "COVARIANT"),
            components=tuple(
                rational_function_from_sympy(value, axis) for value in values
            ),
        )
    )


def _scalar(value: object, axis: tuple[str, ...]) -> RationalFunction:
    return rational_function_from_sympy(value, axis)


def test_polar_metric_matches_exact_coordinate_values() -> None:
    r, theta = symbols("r theta")
    metric = _metric((1, 0, 0, r**2), ("r", "theta"))

    assert (
        rational_function_to_sympy(
            laplace_beltrami(metric, _scalar(r**2, ("r", "theta"))).value
        )
        == 4
    )
    assert (
        rational_function_to_sympy(
            laplace_beltrami(metric, _scalar(theta, ("r", "theta"))).value
        )
        == 0
    )


def test_euclidean_and_constant_metrics_match_independent_formula() -> None:
    x, y = symbols("x y")
    axis = ("x", "y")
    scalar = x**3 + x * y**2 + 1 / (x + 2)
    result = laplace_beltrami(_metric((1, 0, 0, 1), axis), _scalar(scalar, axis))
    expected = scalar.diff(x, 2) + scalar.diff(y, 2)
    assert cancel(rational_function_to_sympy(result.value) - expected) == 0

    weighted = laplace_beltrami(_metric((2, 0, 0, 4), axis), _scalar(x**2 + y**2, axis))
    assert rational_function_to_sympy(weighted.value) == Fraction(3, 2)


def test_conformal_metric_agrees_with_divergence_formula() -> None:
    x, y = symbols("x y")
    axis = ("x", "y")
    conformal = 1 + x**2 + y**2
    metric = _metric((conformal, 0, 0, conformal), axis)
    scalar = x**2 * y + 1 / (x + y + 3)
    result = laplace_beltrami(metric, _scalar(scalar, axis))
    # In two dimensions sqrt(det(g)) = conformal and
    # sqrt(det(g)) * g^ij = delta^ij.
    expected = sum(scalar.diff(coordinate, 2) for coordinate in (x, y)) / conformal
    assert cancel(rational_function_to_sympy(result.value) - expected) == 0


def test_zero_constant_serialization_and_axis_validation() -> None:
    x, _y = symbols("x y")
    axis = ("x", "y")
    metric = _metric((1, 0, 0, 1), axis)
    result = laplace_beltrami(metric, _scalar(7, axis))
    assert rational_function_to_sympy(result.value) == 0
    assert (
        RationalCoordinateTensor.model_validate_json(metric.tensor.model_dump_json())
        == metric.tensor
    )
    assert type(result).model_validate_json(result.model_dump_json()) == result
    with pytest.raises(OperationDomainValidationError, match="same coordinate axis"):
        laplace_beltrami(metric, _scalar(x, ("y", "x")))


def test_shared_deadline_is_honored() -> None:
    x = symbols("x")
    metric = _metric((1,), ("x",))
    with (
        request_execution(monotonic() - 200),
        pytest.raises(OperationExecutionTimeoutError),
    ):
        laplace_beltrami(metric, _scalar(x**2, ("x",)))


def test_identically_singular_metric_rejects_before_cancellation() -> None:
    x = symbols("x")
    metric = _metric((1, x, x, x**2), ("x", "y"))
    with pytest.raises(OperationDomainValidationError, match="identically zero"):
        laplace_beltrami(metric, _scalar(x, ("x", "y")))


def test_result_locus_guard_budget_is_bounded() -> None:
    x = symbols("x")
    axis = ("x",)
    metric = _metric((1,), axis)
    one = _scalar(1, axis)
    guards = tuple(_scalar(x + offset, axis).numerator for offset in range(1, 770))

    with pytest.raises(ValueError, match="at most 768"):
        RationalLaplaceBeltramiResult(
            metric=metric,
            scalar=one,
            value=one,
            retained_nonzero_denominators=guards,
        )
