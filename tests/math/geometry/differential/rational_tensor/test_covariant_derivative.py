"""Exact coordinate identities for rational covariant derivatives."""

from importlib import import_module
from itertools import product
from typing import Any

import pytest
from sympy import cancel, symbols

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.differential.metrics import RationalCoordinateMetric
from jacobian.math.geometry.differential.rational_tensor.covariant_derivative import (
    RationalCovariantDerivativeProfile,
    covariant_derivative,
)
from jacobian.math.geometry.differential.values import (
    RationalCoordinateTensor,
    TensorVariance,
)
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_function_to_sympy,
)

r, theta = symbols("r theta")


def tensor(
    values: list[Any],
    variance: tuple[TensorVariance, ...],
    axis: tuple[str, ...] = ("r", "theta"),
) -> RationalCoordinateTensor:
    return RationalCoordinateTensor(
        coordinate_axis=axis,
        variance=variance,
        components=tuple(rational_function_from_sympy(value, axis) for value in values),
    )


def polar_metric() -> RationalCoordinateMetric:
    return RationalCoordinateMetric(
        tensor=tensor([1, 0, 0, r**2], ("COVARIANT", "COVARIANT"))
    )


def expressions(value: RationalCoordinateTensor) -> tuple[Any, ...]:
    return tuple(
        rational_function_to_sympy(component) for component in value.components
    )


def test_scalar_derivative_agrees_with_rational_gradient_and_retains_det_guard() -> (
    None
):
    result = covariant_derivative(polar_metric(), tensor([r**2], ()))

    assert expressions(result.covariant_derivative) == (2 * r, 0)
    assert len(result.covariant_derivative.retained_nonzero_denominators) == 1
    assert result.covariant_derivative.retained_nonzero_denominators[0].terms[
        0
    ].exponents == (
        2,
        0,
    )


def test_polar_metric_is_covariantly_constant() -> None:
    result = covariant_derivative(polar_metric(), polar_metric().tensor)

    assert expressions(result.covariant_derivative) == (0,) * 8


def test_vector_and_covector_connection_signs() -> None:
    metric = polar_metric()
    vector = covariant_derivative(metric, tensor([1, 0], ("CONTRAVARIANT",)))
    covector = covariant_derivative(metric, tensor([1, 0], ("COVARIANT",)))

    assert expressions(vector.covariant_derivative) == (0, 0, 0, 1 / r)
    assert expressions(covector.covariant_derivative) == (0, 0, 0, r)


def test_mixed_tensor_formula_replays_exactly() -> None:
    metric = polar_metric()
    source = tensor([r, 0, 0, 1], ("CONTRAVARIANT", "COVARIANT"))
    result = covariant_derivative(metric, source)
    actual = expressions(result.covariant_derivative)
    gamma = (
        ((0, 0), (0, -r)),
        ((0, 1 / r), (1 / r, 0)),
    )
    source_values = expressions(source)
    expected = []
    for derivative_axis, upper, lower in product(range(2), repeat=3):
        value = source_values[upper * 2 + lower]
        expression = value.diff((r, theta)[derivative_axis])
        for replacement in range(2):
            expression += (
                gamma[upper][derivative_axis][replacement]
                * source_values[replacement * 2 + lower]
            )
            expression -= (
                gamma[replacement][derivative_axis][lower]
                * source_values[upper * 2 + replacement]
            )
        expected.append(expression)
    assert all(
        cancel(left - right) == 0 for left, right in zip(actual, expected, strict=True)
    )


def test_profile_serialization_and_axis_mismatch() -> None:
    result = covariant_derivative(polar_metric(), tensor([r**2], ()))
    assert (
        RationalCovariantDerivativeProfile.model_validate_json(result.model_dump_json())
        == result
    )
    with pytest.raises(OperationDomainValidationError, match="same coordinate axis"):
        covariant_derivative(polar_metric(), tensor([r**2], (), axis=("theta", "r")))


def test_rank_four_output_is_rejected_before_backend_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric = RationalCoordinateMetric(
        tensor=tensor(
            [1 if i == j else 0 for i in range(4) for j in range(4)],
            ("COVARIANT", "COVARIANT"),
            axis=("x", "y", "z", "w"),
        )
    )
    source = tensor(
        [0] * 256,
        ("COVARIANT",) * 4,
        axis=("x", "y", "z", "w"),
    )
    operations = import_module(
        "jacobian.math.geometry.differential.rational_tensor.covariant_derivative.operations"
    )
    monkeypatch.setattr(
        operations,
        "evaluate_admitted_covariant_derivative",
        lambda *args, **kwargs: pytest.fail("backend execution must follow admission"),
    )
    with pytest.raises(OperationResourceAdmissionError, match="component"):
        covariant_derivative(metric, source)


def test_rank_eight_source_is_rejected_before_rank_nine_result() -> None:
    metric = RationalCoordinateMetric(
        tensor=tensor([1], ("COVARIANT", "COVARIANT"), axis=("x",))
    )
    source = tensor([1], ("COVARIANT",) * 8, axis=("x",))
    with pytest.raises(OperationResourceAdmissionError, match="rank-8"):
        covariant_derivative(metric, source)


def test_determinant_guards_are_capped_before_backend_expansion() -> None:
    x, y = symbols("x y")
    axis = ("x", "y")
    metric = RationalCoordinateMetric(
        tensor=tensor(
            [x**64, 1, 1, x**64],
            ("COVARIANT", "COVARIANT"),
            axis=axis,
        )
    )
    with pytest.raises(OperationResourceAdmissionError, match="determinant locus"):
        covariant_derivative(metric, tensor([1], (), axis=axis))
