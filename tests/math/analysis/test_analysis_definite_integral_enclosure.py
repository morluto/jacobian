from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from fractions import Fraction
from itertools import pairwise
from threading import Event
from time import monotonic
from typing import Any

import pytest
from pydantic import ValidationError
from tests.fixtures.accounting import assert_charged_work_parity

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_execution,
)
from jacobian._flint import flint_workprec
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math import analysis
from jacobian.math.analysis._box_enclosure import (
    _evaluate_box_expression,
    _preflight_box_expression,
)
from jacobian.math.analysis._definite_integral_enclosure import (
    MAX_DEFINITE_INTEGRAL_LEAVES,
    MAX_DEFINITE_INTEGRAL_PRECISION_WORK,
    MAX_DEFINITE_INTEGRAL_WALL_SECONDS,
    DefiniteIntegralBudgetExhausted,
    DefiniteIntegralDomainUnproven,
    DefiniteIntegralDomainUnprovenLeaf,
    DefiniteIntegralEnclosedLeaf,
    DefiniteIntegralEnclosureRequest,
    DefiniteIntegralEnclosureResult,
    DefiniteIntegralTargetMet,
    DefiniteIntegralZeroMeasureLeaf,
    _admit_definite_integral,
    _compute_definite_integral_enclosure,
    _enclosure_width,
    _interval_at_path,
)
from jacobian.math.analysis._models import (
    DyadicClosedInterval,
    ExactDyadic,
    IntervalExpressionNode,
    RationalIntervalBox,
    _bounded_expression_nodes,
)
from jacobian.math.analysis.intervals import ClosedRationalInterval
from jacobian.process import bounded_process_cancellation


def _q(value: Fraction | int, denominator: int = 1) -> dict[str, str]:
    fraction = value if isinstance(value, Fraction) else Fraction(value, denominator)
    return {"num": str(fraction.numerator), "den": str(fraction.denominator)}


def _validate_json(model: Any, payload: Any) -> Any:
    return model.model_validate_json(json.dumps(payload))


def _var(name: str = "t") -> dict[str, Any]:
    return {"op": "var", "variable": name}


def _const(value: Fraction | int) -> dict[str, Any]:
    return {"op": "const", "value": _q(value)}


def _request(
    expression: dict[str, Any],
    *,
    lower: Fraction = Fraction(0),
    upper: Fraction = Fraction(1),
    target_mantissa: int = 1,
    target_exponent: int = -2,
    precision_bits: int = 128,
    max_leaves: int = 8,
    wall_seconds: int = 30,
    variable: str = "t",
) -> DefiniteIntegralEnclosureRequest:
    return DefiniteIntegralEnclosureRequest.model_validate_json(
        json.dumps(
            {
                "expression": expression,
                "box": {
                    "variables": [variable],
                    "intervals": [
                        {"lower": _q(lower), "upper": _q(upper)},
                    ],
                },
                "precision_bits": precision_bits,
                "target_width": {
                    "mantissa": str(target_mantissa),
                    "exponent": target_exponent,
                },
                "max_leaves": max_leaves,
                "wall_seconds": wall_seconds,
            }
        )
    )


def _run(expression: dict[str, Any], **kwargs: Any) -> DefiniteIntegralEnclosureResult:
    return _compute_definite_integral_enclosure(_request(expression, **kwargs))


def _contains(enclosure: DyadicClosedInterval, value: Fraction) -> bool:
    return enclosure.lower.as_fraction() <= value <= enclosure.upper.as_fraction()


def _linear_integral(lower: Fraction, upper: Fraction) -> Fraction:
    return (upper * upper - lower * lower) / 2


def _quadratic_integral(lower: Fraction, upper: Fraction) -> Fraction:
    def antiderivative(value: Fraction) -> Fraction:
        return value * value / 2 - value * value * value / 3

    return antiderivative(upper) - antiderivative(lower)


def _balanced_sum_expression(depth: int) -> dict[str, Any]:
    if depth == 0:
        return _var()
    child = _balanced_sum_expression(depth - 1)
    return {"op": "add", "children": [child, deepcopy(child)]}


def test_constant_one_has_the_exact_integral_singleton() -> None:
    result = _run(_const(1), target_mantissa=0, target_exponent=0, max_leaves=1)

    assert isinstance(result.outcome, DefiniteIntegralTargetMet)
    assert result.outcome.enclosure.lower.as_fraction() == 1
    assert result.outcome.enclosure.upper.as_fraction() == 1
    assert len(result.outcome.leaves) == 1
    leaf = result.outcome.leaves[0]
    assert isinstance(leaf, DefiniteIntegralEnclosedLeaf)
    assert leaf.contribution == result.outcome.enclosure


def test_degenerate_interval_uses_the_explicit_zero_integral_convention() -> None:
    expression = {
        "op": "div",
        "children": [_const(1), _const(0)],
    }
    result = _run(
        expression,
        lower=Fraction(7, 3),
        upper=Fraction(7, 3),
        target_mantissa=0,
        target_exponent=0,
        max_leaves=1,
    )

    assert isinstance(result.outcome, DefiniteIntegralTargetMet)
    assert result.outcome.enclosure.lower.as_fraction() == 0
    assert result.outcome.enclosure.upper.as_fraction() == 0
    assert len(result.outcome.leaves) == 1
    assert isinstance(result.outcome.leaves[0], DefiniteIntegralZeroMeasureLeaf)


def test_linear_integral_contains_one_half_and_reconstructs_each_leaf() -> None:
    result = _run(_var(), target_mantissa=1, target_exponent=-2, max_leaves=8)

    assert isinstance(result.outcome, DefiniteIntegralTargetMet)
    assert _contains(result.outcome.enclosure, Fraction(1, 2))
    assert tuple(leaf.path for leaf in result.outcome.leaves) == (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    )
    source = result.box.intervals[0]
    intervals = tuple(
        _interval_at_path(source, leaf.path) for leaf in result.outcome.leaves
    )
    assert intervals[0].lower == source.lower
    assert intervals[-1].upper == source.upper
    assert all(left.upper == right.lower for left, right in pairwise(intervals))
    assert sum((_interval_width(interval) for interval in intervals), Fraction()) == 1
    for leaf, interval in zip(result.outcome.leaves, intervals, strict=True):
        assert isinstance(leaf, DefiniteIntegralEnclosedLeaf)
        assert _contains(
            leaf.contribution,
            _linear_integral(
                interval.lower.as_fraction(), interval.upper.as_fraction()
            ),
        )
    summed_lower = sum(
        (leaf.contribution.lower.as_fraction() for leaf in result.outcome.leaves),
        Fraction(),
    )
    summed_upper = sum(
        (leaf.contribution.upper.as_fraction() for leaf in result.outcome.leaves),
        Fraction(),
    )
    assert result.outcome.enclosure.lower.as_fraction() <= summed_lower
    assert result.outcome.enclosure.upper.as_fraction() >= summed_upper


def _interval_width(interval: ClosedRationalInterval) -> Fraction:
    return interval.upper.as_fraction() - interval.lower.as_fraction()


def test_quadratic_refinement_contains_one_six_and_tightens_the_sum() -> None:
    expression = {
        "op": "mul",
        "children": [
            _var(),
            {
                "op": "sub",
                "children": [_const(1), _var()],
            },
        ],
    }
    coarse = _run(expression, target_mantissa=0, target_exponent=0, max_leaves=1)
    refined = _run(expression, target_mantissa=1, target_exponent=-2, max_leaves=8)

    assert isinstance(coarse.outcome, DefiniteIntegralBudgetExhausted)
    assert isinstance(refined.outcome, DefiniteIntegralTargetMet)
    assert _contains(refined.outcome.enclosure, Fraction(1, 6))
    assert _enclosure_width(refined.outcome.enclosure) < _enclosure_width(
        coarse.outcome.enclosure
    )
    source = refined.box.intervals[0]
    for leaf in refined.outcome.leaves:
        assert isinstance(leaf, DefiniteIntegralEnclosedLeaf)
        interval = _interval_at_path(source, leaf.path)
        assert _contains(
            leaf.contribution,
            _quadratic_integral(
                interval.lower.as_fraction(), interval.upper.as_fraction()
            ),
        )


def test_sendov_shaped_polynomial_contains_its_exact_integral() -> None:
    expression = {
        "op": "mul",
        "children": [
            {"op": "pow", "exponent": 3, "children": [_var()]},
            {
                "op": "add",
                "children": [
                    {"op": "sub", "children": [_const(1), _var()]},
                    {
                        "op": "mul",
                        "children": [
                            _const(Fraction(9, 16)),
                            {
                                "op": "pow",
                                "exponent": 2,
                                "children": [_var()],
                            },
                        ],
                    },
                ],
            },
        ],
    }
    result = _run(expression, target_mantissa=1, target_exponent=-5, max_leaves=32)

    assert isinstance(result.outcome, DefiniteIntegralTargetMet)
    assert _contains(result.outcome.enclosure, Fraction(23, 160))
    assert 1 < len(result.outcome.leaves) <= 32


def test_sine_integral_matches_an_independent_endpoint_antiderivative() -> None:
    from flint import arb

    expression = {"op": "sin", "children": [_var()]}
    coarse = _run(expression, target_mantissa=0, target_exponent=0, max_leaves=1)
    refined = _run(expression, target_mantissa=1, target_exponent=-4, max_leaves=16)

    assert isinstance(coarse.outcome, DefiniteIntegralBudgetExhausted)
    assert isinstance(refined.outcome, DefiniteIntegralTargetMet)
    with flint_workprec(512):
        antiderivative = arb(1) - arb(1).cos()
        exact_lower = antiderivative.lower().man_exp()
        exact_upper = antiderivative.upper().man_exp()
    lower = Fraction(int(exact_lower[0])) * Fraction(2) ** int(exact_lower[1])
    upper = Fraction(int(exact_upper[0])) * Fraction(2) ** int(exact_upper[1])
    assert refined.outcome.enclosure.lower.as_fraction() <= lower
    assert refined.outcome.enclosure.upper.as_fraction() >= upper
    assert _enclosure_width(refined.outcome.enclosure) < _enclosure_width(
        coarse.outcome.enclosure
    )


def test_negative_linear_integral_preserves_orientation_and_sign() -> None:
    result = _run(
        _var(),
        lower=Fraction(-1),
        upper=Fraction(0),
        target_mantissa=1,
        target_exponent=-2,
        max_leaves=8,
    )

    assert not isinstance(result.outcome, DefiniteIntegralDomainUnproven)
    assert _contains(result.outcome.enclosure, Fraction(-1, 2))


def test_equal_contribution_widths_use_lexicographic_path_ties() -> None:
    result = _run(_var(), target_mantissa=0, target_exponent=0, max_leaves=3)

    assert isinstance(result.outcome, DefiniteIntegralBudgetExhausted)
    assert tuple(leaf.path for leaf in result.outcome.leaves) == (
        (0, 0),
        (0, 1),
        (1,),
    )


def test_budget_exhaustion_retains_a_sound_complete_enclosure() -> None:
    result = _run(_var(), target_mantissa=1, target_exponent=-4, max_leaves=2)

    assert isinstance(result.outcome, DefiniteIntegralBudgetExhausted)
    assert _contains(result.outcome.enclosure, Fraction(1, 2))
    assert len(result.outcome.leaves) == result.max_leaves
    assert all(
        isinstance(leaf, DefiniteIntegralEnclosedLeaf) for leaf in result.outcome.leaves
    )


@pytest.mark.parametrize(
    "expression",
    [
        {"op": "div", "children": [_const(1), _var()]},
        {"op": "log", "children": [_var()]},
        {"op": "sqrt", "children": [_var()]},
    ],
)
def test_unproved_real_domain_never_returns_an_integral_conclusion(
    expression: dict[str, Any],
) -> None:
    lower = Fraction(-1) if expression["op"] != "log" else Fraction(0)
    result = _run(
        expression,
        lower=lower,
        upper=Fraction(1),
        target_mantissa=1,
        target_exponent=8,
        max_leaves=8,
    )

    assert isinstance(result.outcome, DefiniteIntegralDomainUnproven)
    assert len(result.outcome.leaves) == result.max_leaves
    assert any(
        isinstance(leaf, DefiniteIntegralDomainUnprovenLeaf)
        for leaf in result.outcome.leaves
    )


def test_arb_domain_uncertainty_is_a_typed_leaf_nonconclusion() -> None:
    expression = {
        "op": "log",
        "children": [
            {
                "op": "add",
                "children": [_var(), _const(Fraction(1, 10**127))],
            }
        ],
    }

    result = _run(
        expression,
        precision_bits=32,
        target_mantissa=1,
        target_exponent=8,
        max_leaves=1,
    )

    assert isinstance(result.outcome, DefiniteIntegralDomainUnproven)
    assert len(result.outcome.leaves) == 1
    leaf = result.outcome.leaves[0]
    assert isinstance(leaf, DefiniteIntegralDomainUnprovenLeaf)
    assert leaf.path == ()
    assert leaf.domain_failure.operation == "log"
    assert leaf.domain_failure.reason == "LOG_ARGUMENT_NOT_STRICTLY_POSITIVE"


def test_arb_domain_uncertainty_can_resolve_under_midpoint_refinement() -> None:
    expression = {
        "op": "log",
        "children": [
            {
                "op": "add",
                "children": [_var(), _const(Fraction(1, 2**40))],
            }
        ],
    }

    result = _run(
        expression,
        precision_bits=32,
        target_mantissa=1,
        target_exponent=8,
        max_leaves=16,
    )

    assert isinstance(result.outcome, DefiniteIntegralTargetMet)
    assert len(result.outcome.leaves) == 9
    assert all(
        isinstance(leaf, DefiniteIntegralEnclosedLeaf) for leaf in result.outcome.leaves
    )


def test_admitted_domain_proof_is_inherited_without_late_readmission() -> None:
    pole = Fraction(1) + Fraction(1, 2**200)
    expression = {
        "op": "add",
        "children": [
            {
                "op": "div",
                "children": [
                    _const(1),
                    {"op": "sub", "children": [_const(pole), _var()]},
                ],
            },
            {"op": "pow", "exponent": 64, "children": [_var()]},
        ],
    }

    result = _run(
        expression,
        precision_bits=256,
        target_mantissa=0,
        target_exponent=0,
        max_leaves=200,
        wall_seconds=120,
    )

    assert isinstance(result.outcome, DefiniteIntegralBudgetExhausted)
    assert len(result.outcome.leaves) == 200
    assert all(
        isinstance(leaf, DefiniteIntegralEnclosedLeaf) for leaf in result.outcome.leaves
    )


def test_dependency_domain_obstruction_can_be_resolved_by_subdivision() -> None:
    expression = {
        "op": "log",
        "children": [
            {
                "op": "add",
                "children": [
                    _const(1),
                    {"op": "sub", "children": [_var(), _var()]},
                ],
            }
        ],
    }
    result = _run(
        expression,
        target_mantissa=1,
        target_exponent=4,
        max_leaves=2,
    )

    assert isinstance(result.outcome, DefiniteIntegralTargetMet)
    assert _contains(result.outcome.enclosure, Fraction(0))
    assert tuple(leaf.path for leaf in result.outcome.leaves) == ((0,), (1,))
    assert all(
        isinstance(leaf, DefiniteIntegralEnclosedLeaf) for leaf in result.outcome.leaves
    )


def test_target_boundary_is_inclusive() -> None:
    met = _run(_var(), target_mantissa=1, target_exponent=-2, max_leaves=4)
    missed = _run(_var(), target_mantissa=1, target_exponent=-3, max_leaves=4)

    assert isinstance(met.outcome, DefiniteIntegralTargetMet)
    assert isinstance(missed.outcome, DefiniteIntegralBudgetExhausted)
    assert _enclosure_width(met.outcome.enclosure) == Fraction(1, 4)


def test_zero_target_is_admitted_but_negative_target_is_not() -> None:
    zero = _request(_const(1), target_mantissa=0, target_exponent=0, max_leaves=1)
    assert zero.target_width == ExactDyadic(mantissa=0, exponent=0)

    with pytest.raises(ValidationError, match="must be nonnegative"):
        _request(_const(1), target_mantissa=-1, target_exponent=0)


@pytest.mark.parametrize(
    ("box_variables", "expression", "message"),
    [
        ([], _const(1), "exactly one"),
        (["t", "u"], _var(), "exactly one"),
        (["t"], {"op": "var"}, "must be named"),
        (["t"], _var("u"), "must match"),
    ],
)
def test_request_requires_one_explicit_integration_axis(
    box_variables: list[str], expression: dict[str, Any], message: str
) -> None:
    intervals = [{"lower": _q(0), "upper": _q(1)} for _ in box_variables]
    with pytest.raises(ValidationError, match=message):
        _validate_json(
            DefiniteIntegralEnclosureRequest,
            {
                "expression": expression,
                "box": {"variables": box_variables, "intervals": intervals},
                "target_width": {"mantissa": "1", "exponent": -2},
                "max_leaves": 4,
                "wall_seconds": 30,
            },
        )


def test_constant_expression_retains_an_otherwise_unused_integration_axis() -> None:
    request = _request(_const(3), variable="x")
    assert request.box.variables == ("x",)


def test_request_precision_overrides_the_ambient_arb_context() -> None:
    request = _request(_var(), max_leaves=4)
    with flint_workprec(64):
        low_ambient = _compute_definite_integral_enclosure(request)
    with flint_workprec(512):
        high_ambient = _compute_definite_integral_enclosure(request)
    assert low_ambient == high_ambient


def test_concurrent_precision_requests_do_not_overlap_the_arb_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flint import ctx

    import jacobian.math.analysis._definite_integral_enclosure as integral

    ambient_precision = ctx.prec
    expression = {"op": "exp", "children": [_var()]}
    low_request = _request(
        expression,
        precision_bits=32,
        target_mantissa=0,
        target_exponent=0,
        max_leaves=1,
    )
    high_request = _request(
        expression,
        precision_bits=512,
        target_mantissa=0,
        target_exponent=0,
        max_leaves=1,
    )
    expected_low = _compute_definite_integral_enclosure(low_request)
    expected_high = _compute_definite_integral_enclosure(high_request)
    assert expected_low != expected_high

    low_evaluating = Event()
    high_attempting = Event()
    high_evaluating = Event()
    low_checked_context = Event()
    high_finished = Event()
    observations: dict[str, int | bool] = {}
    original_evaluate = _evaluate_box_expression

    def observe_real_evaluation(*args: Any, **kwargs: Any) -> Any:
        if not low_evaluating.is_set():
            observations["low_entered_precision"] = ctx.prec
            low_evaluating.set()
            assert high_attempting.wait(timeout=1)
            observations["contexts_overlapped"] = high_evaluating.wait(timeout=0.25)
            observations["low_precision_during_high_attempt"] = ctx.prec
            low_checked_context.set()
            if observations["contexts_overlapped"]:
                assert high_finished.wait(timeout=1)
        else:
            observations["high_entered_precision"] = ctx.prec
            high_evaluating.set()
            assert low_checked_context.wait(timeout=1)
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(integral, "_evaluate_box_expression", observe_real_evaluation)

    def run_high_precision() -> DefiniteIntegralEnclosureResult:
        assert low_evaluating.wait(timeout=1)
        high_attempting.set()
        try:
            return _compute_definite_integral_enclosure(high_request)
        finally:
            high_finished.set()

    with ThreadPoolExecutor(max_workers=2) as workers:
        low_future = workers.submit(
            _compute_definite_integral_enclosure,
            low_request,
        )
        high_future = workers.submit(run_high_precision)
        low_result = low_future.result(timeout=3)
        high_result = high_future.result(timeout=3)

    assert low_result == expected_low
    assert high_result == expected_high
    assert ctx.prec == ambient_precision
    assert observations == {
        "low_entered_precision": 32,
        "contexts_overlapped": False,
        "low_precision_during_high_attempt": 32,
        "high_entered_precision": 512,
    }


def test_result_round_trips_through_public_json() -> None:
    result = _run(_var(), max_leaves=4)
    assert (
        DefiniteIntegralEnclosureResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_structural_validation_rejects_an_invalid_partition_path() -> None:
    result = _run(_var(), max_leaves=4)
    payload = deepcopy(result.model_dump(mode="json"))
    payload["outcome"]["leaves"][0]["path"] = []

    with pytest.raises(ValidationError):
        _validate_json(DefiniteIntegralEnclosureResult, payload)


def test_result_deserialization_does_not_replay_computed_math(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.analysis._definite_integral_enclosure as integral

    result = _run(_var(), max_leaves=4)

    def replayed(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("computed mathematics was replayed during deserialization")

    for name in (
        "_interval_at_path",
        "_leaf_contribution",
        "_summed_enclosure",
        "_target_met",
    ):
        monkeypatch.setattr(integral, name, replayed)

    parsed = _validate_json(
        DefiniteIntegralEnclosureResult, result.model_dump(mode="json")
    )
    assert parsed == result


def test_result_rejects_a_leaf_deeper_than_the_requested_budget() -> None:
    result = _run(_var(), target_mantissa=0, target_exponent=0, max_leaves=3)
    payload = deepcopy(result.model_dump(mode="json"))
    payload["outcome"]["leaves"][0]["path"] = [0, 0, 0]

    with pytest.raises(ValidationError, match="requested partition-depth"):
        _validate_json(DefiniteIntegralEnclosureResult, payload)


def test_domain_unproven_result_cannot_smuggle_a_global_enclosure() -> None:
    result = _run(
        {"op": "div", "children": [_const(1), _var()]},
        lower=Fraction(-1),
        max_leaves=2,
    )
    payload = deepcopy(result.model_dump(mode="json"))
    payload["outcome"]["enclosure"] = {
        "lower": {"mantissa": "0", "exponent": 0},
        "upper": {"mantissa": "0", "exponent": 0},
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _validate_json(DefiniteIntegralEnclosureResult, payload)


def test_domain_failure_evidence_is_bound_to_the_expression_node() -> None:
    result = _run(
        {"op": "div", "children": [_const(1), _var()]},
        lower=Fraction(-1),
        max_leaves=1,
    )
    payload = deepcopy(result.model_dump(mode="json"))
    leaf = payload["outcome"]["leaves"][0]
    leaf["domain_failure"] = {
        "node_path": [],
        "operation": "log",
        "reason": "LOG_ARGUMENT_NOT_STRICTLY_POSITIVE",
    }

    with pytest.raises(ValidationError, match="does not match"):
        _validate_json(DefiniteIntegralEnclosureResult, payload)


def test_result_schema_exposes_both_discriminated_branch_families() -> None:
    schema = DefiniteIntegralEnclosureResult.model_json_schema()
    outcome = schema["$defs"]["DefiniteIntegralOutcome"]
    assert outcome["discriminator"]["propertyName"] == "status"
    assert set(outcome["discriminator"]["mapping"]) == {
        "TARGET_MET",
        "BUDGET_EXHAUSTED",
        "DOMAIN_UNPROVEN",
    }
    domain_properties = schema["$defs"]["DefiniteIntegralDomainUnproven"]["properties"]
    assert "enclosure" not in domain_properties
    budget_properties = schema["$defs"]["DefiniteIntegralBudgetExhausted"]["properties"]
    assert "reason" not in budget_properties
    concluded_leaf_mapping = schema["$defs"]["DefiniteIntegralConcludedLeaf"][
        "discriminator"
    ]["mapping"]
    assert set(concluded_leaf_mapping) == {"ENCLOSED", "ZERO_MEASURE"}


def test_wire_only_integral_contract_is_not_published_as_a_native_api() -> None:
    assert "definite_integral_enclosure" not in analysis.__all__
    assert not hasattr(analysis, "definite_integral_enclosure")


def test_dispatch_start_time_is_part_of_the_owner_deadline() -> None:
    request = _request(_var(), wall_seconds=1)
    with (
        request_execution(monotonic() - 2),
        pytest.raises(OperationExecutionTimeoutError, match="semantic preflight"),
    ):
        _compute_definite_integral_enclosure(request)


def test_owner_checkpoints_observe_request_cancellation() -> None:
    cancellation = Event()
    cancellation.set()
    request = _request(_var())

    with (
        bounded_process_cancellation(cancellation),
        pytest.raises(OperationExecutionCancelledError, match="cancelled"),
    ):
        _compute_definite_integral_enclosure(request)


def test_precision_work_boundary_is_derived_from_actual_subproblems() -> None:
    expression = _balanced_sum_expression(5)  # 63 nodes
    node_count = 63
    subproblems = 2 * MAX_DEFINITE_INTEGRAL_LEAVES - 1
    accepted_precision = MAX_DEFINITE_INTEGRAL_PRECISION_WORK // (
        node_count * subproblems
    )
    accepted = _request(
        expression,
        precision_bits=accepted_precision,
        max_leaves=MAX_DEFINITE_INTEGRAL_LEAVES,
    )
    _admit_definite_integral(accepted, started_at=monotonic())
    assert (
        node_count * subproblems * accepted_precision
        <= MAX_DEFINITE_INTEGRAL_PRECISION_WORK
    )

    rejected = _request(
        expression,
        precision_bits=accepted_precision + 1,
        max_leaves=MAX_DEFINITE_INTEGRAL_LEAVES,
    )
    with pytest.raises(
        OperationDomainValidationError,
        match="precision work",
    ):
        _admit_definite_integral(rejected, started_at=monotonic())


def test_widened_box_endpoints_retain_definite_integral_admission() -> None:
    lower = Fraction(10**128)
    request = _request(
        _var(),
        lower=lower,
        upper=lower + 1,
        max_leaves=MAX_DEFINITE_INTEGRAL_LEAVES,
    )

    _admit_definite_integral(request, started_at=monotonic())


def test_leaf_and_wall_field_bounds_reject_one_beyond() -> None:
    payload = _request(_var()).model_dump(mode="json")
    payload["max_leaves"] = MAX_DEFINITE_INTEGRAL_LEAVES + 1
    with pytest.raises(ValidationError):
        _validate_json(DefiniteIntegralEnclosureRequest, payload)

    payload = _request(_var()).model_dump(mode="json")
    payload["wall_seconds"] = MAX_DEFINITE_INTEGRAL_WALL_SECONDS + 1
    with pytest.raises(ValidationError):
        _validate_json(DefiniteIntegralEnclosureRequest, payload)


def test_admission_charges_every_executed_partition_primitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.analysis._definite_integral_enclosure as integral

    executed = {
        "arb_node_bit": 0,
        "expression_node": 0,
        "midpoint_split": 0,
        "output_leaf": 0,
        "selection_candidate": 0,
        "subproblem": 0,
        "summation_endpoint": 0,
    }
    admissions = 0
    expression = {
        "op": "mul",
        "children": [
            _var(),
            {"op": "sub", "children": [_const(1), _var()]},
        ],
    }
    request = _request(expression, target_mantissa=0, target_exponent=0, max_leaves=8)
    node_count = len(_bounded_expression_nodes(request.expression))

    original_admit = integral._admit_definite_integral
    original_evaluate = integral._evaluate_integral_leaf
    original_preflight = _preflight_box_expression
    original_arb = _evaluate_box_expression
    original_select = integral._select_leaf
    original_split = integral._split_interval
    original_sum = integral._summed_enclosure

    def admitting(*args: Any, **kwargs: Any) -> Any:
        nonlocal admissions
        admissions += 1
        return original_admit(*args, **kwargs)

    def evaluating(*args: Any, **kwargs: Any) -> Any:
        executed["subproblem"] += 1
        return original_evaluate(*args, **kwargs)

    def preflighting(*args: Any, **kwargs: Any) -> Any:
        executed["expression_node"] += node_count
        return original_preflight(*args, **kwargs)

    def evaluating_arb(*args: Any, **kwargs: Any) -> Any:
        executed["expression_node"] += node_count
        executed["arb_node_bit"] += node_count * request.precision_bits
        return original_arb(*args, **kwargs)

    def selecting(leaves: tuple[Any, ...]) -> Any:
        executed["selection_candidate"] += len(leaves)
        return original_select(leaves)

    def splitting(*args: Any, **kwargs: Any) -> Any:
        executed["midpoint_split"] += 1
        return original_split(*args, **kwargs)

    def summing(contributions: tuple[Any, ...], precision_bits: int) -> Any:
        executed["summation_endpoint"] += 2 * len(contributions)
        return original_sum(contributions, precision_bits)

    monkeypatch.setattr(integral, "_admit_definite_integral", admitting)
    monkeypatch.setattr(integral, "_evaluate_integral_leaf", evaluating)
    monkeypatch.setattr(integral, "_preflight_box_expression", preflighting)
    monkeypatch.setattr(integral, "_evaluate_box_expression", evaluating_arb)
    monkeypatch.setattr(integral, "_select_leaf", selecting)
    monkeypatch.setattr(integral, "_split_interval", splitting)
    monkeypatch.setattr(integral, "_summed_enclosure", summing)

    result = _compute_definite_integral_enclosure(request)
    assert admissions == 1
    executed["output_leaf"] = len(result.outcome.leaves)
    maximum_subproblems = 2 * request.max_leaves - 1
    assert_charged_work_parity(
        charged={
            "arb_node_bit": (node_count * maximum_subproblems * request.precision_bits),
            "expression_node": 2 * node_count * maximum_subproblems,
            "midpoint_split": request.max_leaves - 1,
            "output_leaf": request.max_leaves,
            "selection_candidate": (request.max_leaves * (request.max_leaves - 1) // 2),
            "subproblem": maximum_subproblems,
            "summation_endpoint": (request.max_leaves * (request.max_leaves + 1)),
        },
        executed=executed,
    )
    assert executed == {
        "arb_node_bit": 15 * node_count * request.precision_bits,
        "expression_node": 16 * node_count,
        "midpoint_split": 7,
        "output_leaf": 8,
        "selection_candidate": 28,
        "subproblem": 15,
        "summation_endpoint": 72,
    }


def test_public_source_envelope_is_retained_without_backend_objects() -> None:
    request = _request(_var(), max_leaves=4)
    result = _compute_definite_integral_enclosure(request)
    assert result.expression == request.expression
    assert result.box == request.box
    assert result.precision_bits == request.precision_bits
    assert result.target_width == request.target_width
    assert result.max_leaves == request.max_leaves
    assert result.wall_seconds == request.wall_seconds
    assert isinstance(result.expression, IntervalExpressionNode)
    assert isinstance(result.box, RationalIntervalBox)
