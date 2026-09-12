"""Exact identities for rational metric pullbacks."""

import time
from typing import Any

import pytest
from sympy import Matrix, simplify, symbols

from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_execution,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.differential.metrics import RationalCoordinateMetric
from jacobian.math.geometry.differential.pullback import pullback_metric
from jacobian.math.geometry.differential.values import RationalCoordinateTensor
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_function_to_sympy,
)
from jacobian.math.polynomials.rational_functions.values import RationalFunctionMap
from jacobian.math.polynomials.values import RationalFunction


def rf(value: Any, axis: tuple[Any, ...]) -> RationalFunction:
    return rational_function_from_sympy(value, tuple(str(item) for item in axis))


def metric(values: tuple[Any, ...], axis: tuple[Any, ...]) -> RationalCoordinateMetric:
    return RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=tuple(axis),
            variance=("COVARIANT", "COVARIANT"),
            components=tuple(rf(value, axis) for value in values),
        )
    )


def map_value(
    values: tuple[Any, ...], source: tuple[Any, ...], target: tuple[Any, ...]
) -> RationalFunctionMap:
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


def test_mixed_pullback_entries_match_independent_jacobian_oracle_and_axis_permutation() -> (
    None
):
    x, z, u, v = symbols("x z u v")
    target_axis = ("v", "u")
    source_axis = ("x", "z")
    source_metric = metric((u + v, u, u, 2 * v), target_axis)
    mapping = map_value((x + z, x - z), source_axis, target_axis)
    result = pullback_metric(source_metric, mapping)

    g = Matrix([[u + v, u], [u, 2 * v]])
    jacobian = Matrix([[1, 1], [1, -1]])
    expected = jacobian.T * g.subs({v: x + z, u: x - z}) * jacobian
    actual = Matrix(
        2,
        2,
        [rational_function_to_sympy(value) for value in result.pullback.components],
    )
    assert actual == expected
    assert result.pullback.coordinate_axis == source_axis


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


def test_substituted_metric_denominator_zero_is_typed_domain_error() -> None:
    u = symbols("u")
    component = rf(1 / u, (u,))
    source = RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=("u",),
            variance=("COVARIANT", "COVARIANT"),
            components=(component,),
            retained_nonzero_denominators=(component.denominator,),
        )
    )
    with pytest.raises(OperationDomainValidationError, match="denominator"):
        pullback_metric(source, map_value((0,), ("x",), ("u",)))


def test_expired_shared_deadline_stops_admission() -> None:
    x = symbols("x")
    with request_execution(time.monotonic()):
        bind_request_deadline(time.monotonic() - 1)
        with pytest.raises(OperationExecutionTimeoutError):
            pullback_metric(metric((1,), ("u",)), map_value((x,), ("x",), ("u",)))


def test_noncanonical_authored_map_is_rejected_after_raw_admission() -> None:
    x = symbols("x")
    authored = map_value((x,), ("x",), ("u",))
    component = authored.components[0]
    forged_component = component.model_copy(update={"denominator": component.numerator})
    forged_map = authored.model_copy(update={"components": (forged_component,)})
    with pytest.raises(OperationDomainValidationError, match="canonical"):
        pullback_metric(metric((1,), ("u",)), forged_map)


def test_nonmonomial_rational_map_uses_canonical_worker_path() -> None:
    x = symbols("x")
    authored = map_value((x / (x + 1),), ("x",), ("u",))
    # Make a deliberately unreduced, nonmonomial presentation with a genuine
    # polynomial denominator so the worker-backed GCD recognition is exercised.
    numerator = rf((x + 1) * (x + 2), (x,)).numerator
    denominator = rf(x + 1, (x,)).numerator
    forged = authored.components[0].model_copy(
        update={"numerator": numerator, "denominator": denominator}
    )
    value = authored.model_copy(update={"components": (forged,)})
    with pytest.raises(OperationDomainValidationError, match="canonical"):
        pullback_metric(metric((1,), ("u",)), value)
    result = pullback_metric(metric((1,), ("u",)), authored)
    assert (
        simplify(
            rational_function_to_sympy(result.pullback.components[0]) - 1 / (x + 1) ** 4
        )
        == 0
    )


def test_four_dimensional_translated_map_round_trips_into_tensor_consumer() -> None:
    from jacobian.math.geometry.differential.operations import lie_derivative

    axis = ("x0", "x1", "x2", "x3")
    target = ("u0", "u1", "u2", "u3")
    x = symbols("x0:4")
    hadamard = Matrix([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]])
    mapping = map_value(
        tuple(sum(hadamard[i, j] * x[j] for j in range(4)) + 10**40 for i in range(4)),
        axis,
        target,
    )
    source = metric(tuple(int(i == j) for i in range(4) for j in range(4)), target)
    result = pullback_metric(source, mapping)
    restored = type(result).model_validate_json(result.model_dump_json())
    assert tuple(
        rational_function_to_sympy(value) for value in restored.pullback.components
    ) == tuple(4 * int(i == j) for i in range(4) for j in range(4))
    vector = RationalCoordinateTensor(
        coordinate_axis=axis,
        variance=("CONTRAVARIANT",),
        components=tuple(rf(1, axis) for _ in axis),
    )
    derivative = lie_derivative(vector, restored.pullback)
    assert all(
        not value.numerator.terms for value in derivative.lie_derivative.components
    )


def test_inherited_guard_that_cancels_after_substitution_is_rejected() -> None:
    u, v, w, x, y = symbols("u v w x y")
    axis = ("u", "v", "w")
    ones = (
        rf(1, axis),
        rf(0, axis),
        rf(0, axis),
        rf(0, axis),
        rf(1, axis),
        rf(0, axis),
        rf(0, axis),
        rf(0, axis),
        rf(1, axis),
    )
    guard = rf(u - v - w, axis).numerator
    source = RationalCoordinateMetric(
        tensor=RationalCoordinateTensor(
            coordinate_axis=axis,
            variance=("COVARIANT", "COVARIANT"),
            components=ones,
            retained_nonzero_denominators=(guard,),
        )
    )
    with pytest.raises(OperationDomainValidationError, match="vanishes identically"):
        pullback_metric(source, map_value((x + y, x, y), ("x", "y"), axis))


def test_pullback_dag_is_bound_to_pullback_admission_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.geometry.differential.metrics._dag import Dag

    seen: list[object] = []
    original = Dag.__init__

    def capturing_init(self, dimension, *, reject=None, label=None):
        seen.append((reject, label))
        original(self, dimension, reject=reject, label=label)

    monkeypatch.setattr(Dag, "__init__", capturing_init)
    x = symbols("x")
    pullback_metric(metric((1,), ("u",)), map_value((x,), ("x",), ("u",)))
    assert seen
    reject, label = seen[0]
    assert reject is not None
    assert label == "rational metric pullback"
    with pytest.raises(OperationResourceAdmissionError) as error:
        reject("work", "complete rational metric pullback DAG exceeds 50,000,000 work units")
    assert "pullback" in error.value.errors()[0]["type"]
    assert "curvature" not in error.value.errors()[0]["type"]

