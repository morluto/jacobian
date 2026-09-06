from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from fractions import Fraction
from math import sin
from threading import Event
from time import monotonic
from typing import Any

import pytest
from flint import ctx
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from tests.fixtures.accounting import assert_charged_work_parity
from tests.math.analysis._analysis_support import analysis_validation_error

from jacobian._execution import OperationExecutionTimeoutError, request_execution
from jacobian._flint import flint_workprec
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.analysis._adaptive_range_enclosure import (
    MAX_ADAPTIVE_RANGE_DYADIC_EXPONENT,
    AdaptiveRangeBudgetExhausted,
    AdaptiveRangeDomainUnproven,
    AdaptiveRangeDomainUnprovenLeaf,
    AdaptiveRangeEnclosureRequest,
    AdaptiveRangeEnclosureResult,
    AdaptiveRangeLeaf,
    AdaptiveRangeTargetMet,
    _admit_adaptive_range,
    _compute_adaptive_range_enclosure,
    _enclosure_width,
    _problem_from_request,
    adaptive_range_enclosure,
)
from jacobian.math.analysis._box_enclosure import (
    IntervalExpressionBoxEnclosureRequest,
    _box_expression_enclosure,
)
from jacobian.math.analysis._models import (
    MAX_RATIONAL_BOX_ENDPOINT_DIGITS,
    DyadicClosedInterval,
)


def _q(numerator: int, denominator: int = 1) -> dict[str, str]:
    value = Fraction(numerator, denominator)
    return {"num": str(value.numerator), "den": str(value.denominator)}


def _var(name: str) -> dict[str, Any]:
    return {"op": "var", "variable": name}


def _const(value: int, denominator: int = 1) -> dict[str, Any]:
    return {"op": "const", "value": _q(value, denominator)}


def _request(
    expression: dict[str, Any],
    coordinates: tuple[tuple[str, Fraction, Fraction], ...],
    *,
    target_width: Fraction,
    precision_bits: int = 128,
    maximum_precision_bits: int = 128,
    max_leaves: int = 8,
    max_depth: int = 3,
    max_evaluations: int = 32,
    wall_seconds: int = 30,
) -> AdaptiveRangeEnclosureRequest:
    return AdaptiveRangeEnclosureRequest.model_validate_json(
        json.dumps(
            {
                "expression": expression,
                "box": {
                    "variables": [name for name, _, _ in coordinates],
                    "intervals": [
                        {
                            "lower": _q(lower.numerator, lower.denominator),
                            "upper": _q(upper.numerator, upper.denominator),
                        }
                        for _, lower, upper in coordinates
                    ],
                },
                "target_width": _q(target_width.numerator, target_width.denominator),
                "precision_bits": precision_bits,
                "maximum_precision_bits": maximum_precision_bits,
                "max_leaves": max_leaves,
                "max_depth": max_depth,
                "max_evaluations": max_evaluations,
                "wall_seconds": wall_seconds,
            }
        )
    )


def _run(
    expression: dict[str, Any],
    coordinates: tuple[tuple[str, Fraction, Fraction], ...],
    **kwargs: Any,
) -> AdaptiveRangeEnclosureResult:
    return _compute_adaptive_range_enclosure(
        _request(expression, coordinates, **kwargs)
    )


def _quadratic() -> dict[str, Any]:
    return {
        "op": "mul",
        "children": [
            _var("x"),
            {"op": "sub", "children": [_const(1), _var("x")]},
        ],
    }


def _balanced_sum(variables: tuple[str, ...]) -> dict[str, Any]:
    level = [_var(variable) for variable in variables]
    while len(level) > 1:
        next_level = [
            {"op": "add", "children": level[index : index + 2]}
            for index in range(0, len(level) - 1, 2)
        ]
        if len(level) % 2:
            next_level.append(level[-1])
        level = next_level
    return level[0]


def _result_enclosure(
    result: AdaptiveRangeEnclosureResult,
) -> DyadicClosedInterval:
    assert result.enclosure is not None
    return result.enclosure


def _enclosed_leaves(
    result: AdaptiveRangeEnclosureResult,
) -> tuple[AdaptiveRangeLeaf, ...]:
    assert all(isinstance(leaf, AdaptiveRangeLeaf) for leaf in result.leaves)
    return tuple(leaf for leaf in result.leaves if isinstance(leaf, AdaptiveRangeLeaf))


def test_quadratic_target_met_reconstructs_the_complete_four_leaf_cover() -> None:
    result = _run(
        _quadratic(),
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(7, 16),
        max_leaves=4,
        max_depth=2,
        max_evaluations=7,
    )

    assert isinstance(result.disposition, AdaptiveRangeTargetMet)
    assert tuple(leaf.path for leaf in result.leaves) == (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    )
    assert tuple(
        (
            leaf.box.intervals[0].lower.as_fraction(),
            leaf.box.intervals[0].upper.as_fraction(),
        )
        for leaf in result.leaves
    ) == (
        (Fraction(0), Fraction(1, 4)),
        (Fraction(1, 4), Fraction(1, 2)),
        (Fraction(1, 2), Fraction(3, 4)),
        (Fraction(3, 4), Fraction(1)),
    )
    leaves = _enclosed_leaves(result)
    enclosure = _result_enclosure(result)
    expected_lower = min(
        (leaf.enclosure.lower for leaf in leaves),
        key=lambda endpoint: endpoint.as_fraction(),
    )
    expected_upper = max(
        (leaf.enclosure.upper for leaf in leaves),
        key=lambda endpoint: endpoint.as_fraction(),
    )
    assert enclosure.lower == expected_lower
    assert enclosure.upper == expected_upper
    assert enclosure.lower.as_fraction() <= 0
    assert enclosure.upper.as_fraction() >= Fraction(1, 4)
    assert _enclosure_width(enclosure) <= Fraction(7, 16)
    assert result.evaluations_used == 7


def test_each_returned_leaf_box_composes_with_the_existing_box_operation() -> None:
    result = _run(
        _quadratic(),
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(7, 16),
        max_leaves=4,
        max_depth=2,
        max_evaluations=7,
    )

    for leaf in _enclosed_leaves(result):
        replay = _box_expression_enclosure(
            IntervalExpressionBoxEnclosureRequest(
                expression=result.expression,
                box=leaf.box,
                precision_bits=result.maximum_precision_bits_used,
            )
        )
        assert replay.status == "ENCLOSED"
        assert replay.lower == leaf.enclosure.lower
        assert replay.upper == leaf.enclosure.upper


def test_sine_enclosure_contains_both_interior_extrema_on_zero_to_seven() -> None:
    result = _run(
        {"op": "sin", "children": [_var("x")]},
        (("x", Fraction(0), Fraction(7)),),
        target_width=Fraction(3),
        max_leaves=1,
        max_depth=0,
        max_evaluations=1,
    )

    assert isinstance(result.disposition, AdaptiveRangeTargetMet)
    enclosure = _result_enclosure(result)
    assert enclosure.lower.as_fraction() <= -1
    assert enclosure.upper.as_fraction() >= 1


def test_two_variable_coupling_contains_a_secondary_grid_oracle() -> None:
    expression = {
        "op": "add",
        "children": [
            {
                "op": "sin",
                "children": [{"op": "add", "children": [_var("x"), _var("y")]}],
            },
            {"op": "mul", "children": [_var("x"), _var("y")]},
        ],
    }
    result = _run(
        expression,
        (
            ("x", Fraction(0), Fraction(1, 2)),
            ("y", Fraction(0), Fraction(1, 2)),
        ),
        target_width=Fraction(2),
        max_leaves=4,
        max_depth=2,
        max_evaluations=7,
    )

    enclosure = _result_enclosure(result)
    lower = float(enclosure.lower.as_fraction())
    upper = float(enclosure.upper.as_fraction())
    for x_index in range(17):
        for y_index in range(17):
            x = x_index / 32
            y = y_index / 32
            assert lower <= sin(x + y) + x * y <= upper


def test_dependency_expression_never_loses_zero() -> None:
    result = _run(
        {"op": "sub", "children": [_var("x"), _var("x")]},
        (("x", Fraction(-1), Fraction(1)),),
        target_width=Fraction(1, 2),
        max_leaves=8,
        max_depth=3,
        max_evaluations=15,
    )

    enclosure = _result_enclosure(result)
    assert enclosure.lower.as_fraction() <= 0
    assert enclosure.upper.as_fraction() >= 0


def test_leaf_and_coordinate_ties_follow_path_then_source_axis_order() -> None:
    sum_xy = {"op": "add", "children": [_var("x"), _var("y")]}
    result = _run(
        {"op": "sub", "children": [sum_xy, deepcopy(sum_xy)]},
        (
            ("x", Fraction(0), Fraction(1)),
            ("y", Fraction(0), Fraction(1)),
        ),
        target_width=Fraction(1, 100),
        max_leaves=3,
        max_depth=2,
        max_evaluations=5,
    )

    assert tuple(leaf.path for leaf in result.leaves) == ((0, 0), (0, 1), (1,))
    first = result.leaves[0].box.intervals
    assert (first[0].lower.as_fraction(), first[0].upper.as_fraction()) == (
        Fraction(0),
        Fraction(1, 2),
    )
    assert (first[1].lower.as_fraction(), first[1].upper.as_fraction()) == (
        Fraction(0),
        Fraction(1, 2),
    )


def test_reduced_matrix_norm_fixture_needs_refinement_for_one_third_bound() -> None:
    # For the real symmetric matrix [[a, b], [b, -a]], the squared operator
    # norm is a^2+b^2.  This reduced HRT-shaped fixture uses a=sin(x)-x and
    # b=cos(x)-1, so the source's strict norm < 1/3 target is upper^2 < 1/9.
    sine_remainder = {
        "op": "sub",
        "children": [{"op": "sin", "children": [_var("x")]}, _var("x")],
    }
    cosine_remainder = {
        "op": "sub",
        "children": [{"op": "cos", "children": [_var("x")]}, _const(1)],
    }
    expression = {
        "op": "add",
        "children": [
            {
                "op": "pow",
                "exponent": 2,
                "children": [sine_remainder],
            },
            {
                "op": "pow",
                "exponent": 2,
                "children": [cosine_remainder],
            },
        ],
    }
    source_box = ("x", Fraction(0), Fraction(1, 2))
    one_box = _box_expression_enclosure(
        IntervalExpressionBoxEnclosureRequest.model_validate_json(
            json.dumps(
                {
                    "expression": expression,
                    "box": {
                        "variables": ["x"],
                        "intervals": [{"lower": _q(0), "upper": _q(1, 2)}],
                    },
                    "precision_bits": 128,
                }
            )
        )
    )
    result = _run(
        expression,
        (source_box,),
        target_width=Fraction(1, 8),
        max_leaves=4,
        max_depth=2,
        max_evaluations=7,
    )

    assert one_box.status == "ENCLOSED"
    assert one_box.upper is not None
    assert one_box.upper.as_fraction() >= Fraction(1, 9)
    assert isinstance(result.disposition, AdaptiveRangeTargetMet)
    assert len(result.leaves) == 4
    assert _result_enclosure(result).upper.as_fraction() < Fraction(1, 9)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"max_leaves": 1, "max_depth": 3, "max_evaluations": 3}, "MAX_LEAVES"),
        (
            {"max_leaves": 4, "max_depth": 3, "max_evaluations": 1},
            "MAX_EVALUATIONS",
        ),
        ({"max_leaves": 4, "max_depth": 0, "max_evaluations": 3}, "MAX_DEPTH"),
    ],
)
def test_budget_exhaustion_reasons_are_deterministic(
    kwargs: dict[str, int], reason: str
) -> None:
    result = _run(
        _var("x"),
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(1, 8),
        **kwargs,
    )

    assert isinstance(result.disposition, AdaptiveRangeBudgetExhausted)
    assert result.disposition.reason == reason
    enclosure = _result_enclosure(result)
    assert enclosure.lower.as_fraction() <= 0
    assert enclosure.upper.as_fraction() >= 1


def test_zero_dimensional_precision_schedule_has_a_typed_exhaustion() -> None:
    result = _run(
        {"op": "exp", "children": [_const(1)]},
        (),
        target_width=Fraction(1, 10**60),
        precision_bits=32,
        maximum_precision_bits=100,
        max_leaves=4,
        max_depth=4,
        max_evaluations=5,
    )

    assert isinstance(result.disposition, AdaptiveRangeBudgetExhausted)
    assert result.disposition.reason == "MAX_PRECISION"
    assert result.evaluations_used == 3
    assert result.maximum_precision_bits_used == 100
    assert result.leaves[0].path == ()
    assert result.leaves[0].box.variables == ()


@pytest.mark.parametrize(
    ("max_leaves", "max_evaluations"),
    [(1, 5), (2, 3)],
)
def test_no_split_precision_cause_precedes_irrelevant_budget_caps(
    max_leaves: int, max_evaluations: int
) -> None:
    result = _run(
        {"op": "exp", "children": [_const(1)]},
        (),
        target_width=Fraction(1, 10**60),
        precision_bits=32,
        maximum_precision_bits=100,
        max_leaves=max_leaves,
        max_depth=4,
        max_evaluations=max_evaluations,
    )

    assert isinstance(result.disposition, AdaptiveRangeBudgetExhausted)
    assert result.disposition.reason == "MAX_PRECISION"


@pytest.mark.parametrize(
    ("max_leaves", "max_evaluations"),
    [(1, 1), (4, 1)],
)
def test_no_split_depth_cause_precedes_irrelevant_budget_caps(
    max_leaves: int, max_evaluations: int
) -> None:
    result = _run(
        _var("x"),
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(1, 8),
        max_leaves=max_leaves,
        max_depth=0,
        max_evaluations=max_evaluations,
    )

    assert isinstance(result.disposition, AdaptiveRangeBudgetExhausted)
    assert result.disposition.reason == "MAX_DEPTH"


@pytest.mark.parametrize(
    "expression",
    [
        {"op": "log", "children": [_var("x")]},
        {"op": "div", "children": [_const(1), _var("x")]},
        {"op": "sqrt", "children": [_var("x")]},
    ],
)
def test_source_domain_failure_is_request_rejection(
    expression: dict[str, Any],
) -> None:
    request = _request(
        expression,
        (("x", Fraction(-1), Fraction(1)),),
        target_width=Fraction(1),
    )

    with pytest.raises(OperationDomainValidationError) as caught:
        _compute_adaptive_range_enclosure(request)
    assert caught.value.errors()[0]["type"] == "analysis.adaptive_range.domain_unproven"


def test_arb_domain_uncertainty_is_a_typed_leaf_nonconclusion() -> None:
    expression = {
        "op": "log",
        "children": [
            {
                "op": "add",
                "children": [_var("x"), _const(1, 10**127)],
            }
        ],
    }

    result = _run(
        expression,
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(100),
        precision_bits=32,
        maximum_precision_bits=32,
        max_leaves=1,
        max_depth=8,
        max_evaluations=1,
    )

    assert isinstance(result.disposition, AdaptiveRangeDomainUnproven)
    assert result.disposition.reason == "MAX_LEAVES"
    assert result.enclosure is None
    assert len(result.leaves) == 1
    leaf = result.leaves[0]
    assert isinstance(leaf, AdaptiveRangeDomainUnprovenLeaf)
    assert leaf.path == ()
    assert leaf.domain_failure.operation == "log"
    assert leaf.domain_failure.reason == "LOG_ARGUMENT_NOT_STRICTLY_POSITIVE"


def test_arb_domain_uncertainty_can_resolve_under_midpoint_refinement() -> None:
    expression = {
        "op": "log",
        "children": [
            {
                "op": "add",
                "children": [_var("x"), _const(1, 2**40)],
            }
        ],
    }

    result = _run(
        expression,
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(100),
        precision_bits=32,
        maximum_precision_bits=32,
        max_leaves=16,
        max_depth=15,
        max_evaluations=31,
    )

    assert isinstance(result.disposition, AdaptiveRangeTargetMet)
    assert result.enclosure is not None
    assert all(isinstance(leaf, AdaptiveRangeLeaf) for leaf in result.leaves)
    assert any(len(leaf.path) > 0 for leaf in result.leaves)


def test_arb_domain_uncertainty_resolves_before_partition_at_higher_precision() -> None:
    expression = {
        "op": "log",
        "children": [
            {
                "op": "add",
                "children": [_var("x"), _const(1, 2**40)],
            }
        ],
    }

    result = _run(
        expression,
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(100),
        precision_bits=32,
        maximum_precision_bits=128,
        max_leaves=1,
        max_depth=8,
        max_evaluations=3,
    )

    assert isinstance(result.disposition, AdaptiveRangeTargetMet)
    assert result.evaluations_used == 2
    assert result.maximum_precision_bits_used == 64
    assert tuple(leaf.path for leaf in result.leaves) == ((),)
    assert all(isinstance(leaf, AdaptiveRangeLeaf) for leaf in result.leaves)


def test_unresolved_leaf_depth_precedes_unrelated_refinement_budgets() -> None:
    expression = {
        "op": "log",
        "children": [
            {
                "op": "add",
                "children": [_var("x"), _const(1, 2**40)],
            }
        ],
    }

    result = _run(
        expression,
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(100),
        precision_bits=32,
        maximum_precision_bits=32,
        max_leaves=16,
        max_depth=4,
        max_evaluations=31,
    )

    assert isinstance(result.disposition, AdaptiveRangeDomainUnproven)
    assert result.disposition.reason == "MAX_DEPTH"
    assert len(result.leaves) == 5
    assert any(
        isinstance(leaf, AdaptiveRangeDomainUnprovenLeaf)
        and len(leaf.path) == result.max_depth
        for leaf in result.leaves
    )


def test_raw_target_and_cross_precision_bounds_reject_before_execution() -> None:
    payload = {
        "expression": _var("x"),
        "box": {
            "variables": ["x"],
            "intervals": [{"lower": _q(0), "upper": _q(1)}],
        },
        "target_width": {"num": "9" * 129, "den": "1"},
        "precision_bits": 128,
        "maximum_precision_bits": 64,
    }
    with analysis_validation_error():
        AdaptiveRangeEnclosureRequest.model_validate_json(json.dumps(payload))

    payload["target_width"] = _q(1)
    with analysis_validation_error():
        AdaptiveRangeEnclosureRequest.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("target", [Fraction(0), Fraction(-1)])
def test_nonpositive_target_is_semantic_request_rejection(target: Fraction) -> None:
    request = _request(
        _var("x"),
        (("x", Fraction(0), Fraction(1)),),
        target_width=target,
    )

    with pytest.raises(OperationDomainValidationError) as caught:
        _compute_adaptive_range_enclosure(request)
    assert caught.value.errors()[0]["type"] == "analysis.adaptive_range.target_width"


def test_depth_zero_eight_variable_request_reserves_one_result_leaf() -> None:
    variables = tuple(f"x{index}" for index in range(8))
    request = _request(
        _balanced_sum(variables),
        tuple((variable, Fraction(0), Fraction(10**127)) for variable in variables),
        target_width=Fraction(1),
        precision_bits=4096,
        maximum_precision_bits=4096,
        max_leaves=1024,
        max_depth=0,
        max_evaluations=4096,
    )
    problem = _problem_from_request(request)

    admission = _admit_adaptive_range(problem, started_at=monotonic())
    result = _compute_adaptive_range_enclosure(request)

    assert admission.planned_evaluations == 1
    assert admission.plan.planned_leaf_count == 1
    assert admission.planned_node_evaluations == 15
    assert isinstance(result.disposition, AdaptiveRangeBudgetExhausted)
    assert result.disposition.reason == "MAX_DEPTH"
    assert result.evaluations_used == 1
    assert len(result.leaves) == 1


@pytest.mark.scale
def test_quadratic_full_leaf_boundary_retains_exact_partition_enclosures() -> None:
    request = _request(
        {"op": "mul", "children": [_var("x"), _var("x")]},
        (("x", Fraction(-1), Fraction(1)),),
        target_width=Fraction(1, 10**9),
        precision_bits=64,
        maximum_precision_bits=64,
        max_leaves=1024,
        max_depth=32,
        max_evaluations=2047,
    )

    result = _compute_adaptive_range_enclosure(request)

    assert isinstance(result.disposition, AdaptiveRangeBudgetExhausted)
    assert result.disposition.reason == "MAX_LEAVES"
    assert result.evaluations_used == 2047
    assert len(result.leaves) == 1024
    assert result.enclosure.lower.as_fraction() <= 0
    assert result.enclosure.upper.as_fraction() >= 1
    for leaf in result.leaves:
        assert isinstance(leaf, AdaptiveRangeLeaf)
        interval = leaf.box.intervals[0]
        lower = interval.lower.as_fraction()
        upper = interval.upper.as_fraction()
        exact_lower = 0 if lower <= 0 <= upper else min(lower * lower, upper * upper)
        exact_upper = max(lower * lower, upper * upper)
        assert leaf.enclosure.lower.as_fraction() <= exact_lower
        assert leaf.enclosure.upper.as_fraction() >= exact_upper


def test_precision_weighted_work_is_rejected_before_arb() -> None:
    level = [_var("x") for _ in range(32)]
    while len(level) > 1:
        level = [
            {"op": "add", "children": level[index : index + 2]}
            for index in range(0, len(level), 2)
        ]
    expression = level[0]
    request = _request(
        expression,
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(1),
        precision_bits=4096,
        maximum_precision_bits=4096,
        max_leaves=1024,
        max_depth=32,
        max_evaluations=4096,
    )

    with pytest.raises(OperationDomainValidationError) as caught:
        _compute_adaptive_range_enclosure(request)
    assert caught.value.errors()[0]["type"] == "analysis.adaptive_range.precision_work"


def test_depth_zero_33_node_request_charges_only_the_reachable_root() -> None:
    expression = _balanced_sum(("x",) * 17)
    request = _request(
        expression,
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(20),
        precision_bits=4096,
        maximum_precision_bits=4096,
        max_leaves=1024,
        max_depth=0,
        max_evaluations=4096,
    )

    admission = _admit_adaptive_range(
        _problem_from_request(request), started_at=monotonic()
    )
    result = _compute_adaptive_range_enclosure(request)

    assert admission.planned_evaluations == 1
    assert admission.plan.planned_leaf_count == 1
    assert admission.planned_node_evaluations == 33
    assert isinstance(result.disposition, AdaptiveRangeTargetMet)
    assert result.evaluations_used == 1
    assert len(result.leaves) == 1


@pytest.mark.parametrize(
    ("interval", "maximum_precision", "max_evaluations", "expected"),
    [
        ((Fraction(0), Fraction(1)), 512, 3, (3, 0, 1, 0, 224)),
        ((Fraction(0), Fraction(1)), 32, 5, (5, 2, 3, 2, 160)),
        ((Fraction(0), Fraction(0)), 32, 4096, (1, 0, 1, 0, 32)),
    ],
)
def test_admission_plan_intersects_schedule_evaluation_and_split_caps(
    interval: tuple[Fraction, Fraction],
    maximum_precision: int,
    max_evaluations: int,
    expected: tuple[int, int, int, int, int],
) -> None:
    request = _request(
        _var("x"),
        (("x", interval[0], interval[1]),),
        target_width=Fraction(2),
        precision_bits=32,
        maximum_precision_bits=maximum_precision,
        max_leaves=1024,
        max_depth=32,
        max_evaluations=max_evaluations,
    )

    admission = _admit_adaptive_range(
        _problem_from_request(request), started_at=monotonic()
    )
    plan = admission.plan

    assert (
        plan.planned_evaluations,
        plan.planned_splits,
        plan.planned_leaf_count,
        plan.planned_maximum_leaf_depth,
        plan.precision_bits_per_expression_node,
    ) == expected
    assert admission.planned_node_evaluations == plan.planned_evaluations


@pytest.mark.parametrize(
    ("field", "at_limit", "above_limit"),
    [
        ("max_leaves", 1024, 1025),
        ("max_depth", 32, 33),
        ("max_evaluations", 4096, 4097),
        ("maximum_precision_bits", 4096, 4097),
    ],
)
def test_adaptive_budget_field_boundaries_are_schema_visible(
    field: str, at_limit: int, above_limit: int
) -> None:
    payload: dict[str, Any] = {
        "expression": _var("x"),
        "box": {
            "variables": ["x"],
            "intervals": [{"lower": _q(0), "upper": _q(1)}],
        },
        "target_width": _q(2),
        "precision_bits": 32,
        "maximum_precision_bits": 32,
        "max_leaves": 1,
        "max_depth": 0,
        "max_evaluations": 1,
        "wall_seconds": 30,
    }
    payload[field] = at_limit
    AdaptiveRangeEnclosureRequest.model_validate_json(json.dumps(payload))

    payload[field] = above_limit
    with analysis_validation_error():
        AdaptiveRangeEnclosureRequest.model_validate_json(json.dumps(payload))


def test_past_request_context_expires_before_semantic_preflight() -> None:
    request = _request(
        _var("x"),
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(2),
        wall_seconds=1,
    )

    with (
        request_execution(monotonic() - 2),
        pytest.raises(OperationExecutionTimeoutError, match="before semantic"),
    ):
        _compute_adaptive_range_enclosure(request)


def test_native_and_request_paths_return_the_same_exact_result() -> None:
    request = _request(
        _quadratic(),
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(7, 16),
        max_leaves=4,
        max_depth=2,
        max_evaluations=7,
    )
    dispatched = _compute_adaptive_range_enclosure(request)
    native = adaptive_range_enclosure(
        request.expression,
        request.box,
        request.target_width,
        precision_bits=request.precision_bits,
        maximum_precision_bits=request.maximum_precision_bits,
        max_leaves=request.max_leaves,
        max_depth=request.max_depth,
        max_evaluations=request.max_evaluations,
        wall_seconds=request.wall_seconds,
    )
    assert native == dispatched


def test_derived_midpoint_box_round_trips_into_fixed_box_consumer() -> None:
    q = 6 * 10**127 + 1
    result = _run(
        _var("x"),
        (("x", Fraction(0), Fraction(1, q)),),
        target_width=Fraction(1, 9 * 10**127),
        precision_bits=128,
        maximum_precision_bits=128,
        max_leaves=2,
        max_depth=1,
        max_evaluations=3,
    )

    assert len(result.leaves) == 2
    first_leaf = result.leaves[0]
    assert isinstance(first_leaf, AdaptiveRangeLeaf)
    midpoint = first_leaf.box.intervals[0].upper
    assert midpoint.as_fraction() == Fraction(1, 2 * q)
    assert len(str(midpoint.den)) == 129

    parsed = AdaptiveRangeEnclosureResult.model_validate_json(
        result.model_dump_json(), strict=True
    )
    parsed_first_leaf = parsed.leaves[0]
    assert isinstance(parsed_first_leaf, AdaptiveRangeLeaf)
    consumer_request = IntervalExpressionBoxEnclosureRequest(
        expression=parsed.expression,
        box=parsed_first_leaf.box,
        precision_bits=128,
    )
    consumed = _box_expression_enclosure(consumer_request)

    assert consumed.status == "ENCLOSED"
    assert consumed.lower is not None and consumed.upper is not None
    assert consumed.lower.as_fraction() <= 0
    assert consumed.upper.as_fraction() >= Fraction(1, 2 * q)


def test_concurrent_adaptive_operations_isolate_32_and_512_bit_precision() -> None:
    expression = {"op": "exp", "children": [_const(1)]}
    low_request = _request(
        expression,
        (),
        target_width=Fraction(1),
        precision_bits=32,
        maximum_precision_bits=32,
        max_leaves=1,
        max_depth=0,
        max_evaluations=1,
    )
    high_request = low_request.model_copy(
        update={"precision_bits": 512, "maximum_precision_bits": 512}
    )
    expected_low = _compute_adaptive_range_enclosure(low_request)
    expected_high = _compute_adaptive_range_enclosure(high_request)
    original_precision = ctx.prec
    holder_entered = Event()
    high_attempting = Event()
    high_finished = Event()

    def low_worker() -> AdaptiveRangeEnclosureResult:
        with flint_workprec(32):
            holder_entered.set()
            assert high_attempting.wait(timeout=1)
            assert not high_finished.wait(timeout=0.25)
            assert ctx.prec == 32
            return _compute_adaptive_range_enclosure(low_request)

    def high_worker() -> AdaptiveRangeEnclosureResult:
        assert holder_entered.wait(timeout=1)
        high_attempting.set()
        try:
            return _compute_adaptive_range_enclosure(high_request)
        finally:
            high_finished.set()

    with ThreadPoolExecutor(max_workers=2) as workers:
        low_future = workers.submit(low_worker)
        high_future = workers.submit(high_worker)
        low = low_future.result(timeout=2)
        high = high_future.result(timeout=2)

    assert low == expected_low
    assert high == expected_high
    assert _result_enclosure(low) != _result_enclosure(high)
    assert ctx.prec == original_precision


def test_admission_charge_covers_every_real_leaf_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.analysis._adaptive_range_enclosure as adaptive

    request = _request(
        _quadratic(),
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(7, 16),
        max_leaves=4,
        max_depth=2,
        max_evaluations=7,
    )
    admission = _admit_adaptive_range(
        _problem_from_request(request), started_at=monotonic()
    )
    original = adaptive._evaluate_leaf
    calls = 0

    def counting(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(adaptive, "_evaluate_leaf", counting)
    result = _compute_adaptive_range_enclosure(request)

    assert_charged_work_parity(
        charged={"arb_expression_evaluations": admission.planned_evaluations},
        executed={"arb_expression_evaluations": calls},
    )
    assert calls == result.evaluations_used == 7


def test_round_trip_rejects_forged_partition_path_structure() -> None:
    result = _run(
        _quadratic(),
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(7, 16),
        max_leaves=4,
        max_depth=2,
        max_evaluations=7,
    )
    assert (
        AdaptiveRangeEnclosureResult.model_validate_json(
            result.model_dump_json(), strict=True
        )
        == result
    )
    payload = deepcopy(result.model_dump(mode="json"))
    payload["leaves"][0]["path"] = [0, 1]

    with pytest.raises(ValidationError):
        AdaptiveRangeEnclosureResult.model_validate_json(json.dumps(payload))


def test_result_deserialization_does_not_reconstruct_partition_boxes() -> None:
    result = _run(
        _quadratic(),
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(7, 16),
        max_leaves=4,
        max_depth=2,
        max_evaluations=7,
    )
    payload = deepcopy(result.model_dump(mode="json"))
    payload["leaves"][0]["box"]["intervals"][0]["upper"] = _q(1, 3)

    parsed = AdaptiveRangeEnclosureResult.model_validate_json(json.dumps(payload))

    assert parsed.leaves[0].box.intervals[0].upper.as_fraction() == Fraction(1, 3)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("evaluations_used", 8, "requested evaluation budget"),
        ("maximum_precision_bits_used", 256, "requested precision range"),
    ],
)
def test_result_structural_counters_remain_within_the_request(
    field: str, value: int, message: str
) -> None:
    result = _run(
        _quadratic(),
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(7, 16),
        max_leaves=4,
        max_depth=2,
        max_evaluations=7,
    )
    payload = result.model_dump(mode="json")
    payload[field] = value

    with pytest.raises(ValidationError, match=message):
        AdaptiveRangeEnclosureResult.model_validate_json(json.dumps(payload))


def test_result_preflights_authored_leaf_endpoints_before_fraction_work() -> None:
    result = _run(
        _quadratic(),
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(7, 16),
        max_leaves=4,
        max_depth=2,
        max_evaluations=7,
    )
    payload = result.model_dump(mode="json")
    payload["leaves"][0]["box"]["intervals"][0]["upper"] = {
        "num": "9" * (MAX_RATIONAL_BOX_ENDPOINT_DIGITS + 1),
        "den": "1",
    }

    with pytest.raises(
        ValidationError, match=rf"{MAX_RATIONAL_BOX_ENDPOINT_DIGITS}-digit bound"
    ):
        AdaptiveRangeEnclosureResult.model_validate_json(json.dumps(payload))


def test_result_structurally_bounds_authored_dyadic_exponents() -> None:
    result = _run(
        _quadratic(),
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(7, 16),
        max_leaves=4,
        max_depth=2,
        max_evaluations=7,
    )
    payload = result.model_dump(mode="json")
    payload["leaves"][0]["enclosure"]["lower"]["exponent"] = (
        MAX_ADAPTIVE_RANGE_DYADIC_EXPONENT + 1
    )

    with pytest.raises(ValidationError, match="source-and-precision bound"):
        AdaptiveRangeEnclosureResult.model_validate_json(json.dumps(payload))


def test_result_deserialization_does_not_replay_computed_math(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.analysis._adaptive_range_enclosure as adaptive

    result = _run(
        _quadratic(),
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(7, 16),
        max_leaves=4,
        max_depth=2,
        max_evaluations=7,
    )

    def replayed(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("computed mathematics was replayed during deserialization")

    for name in (
        "_enclosure_width",
        "_evaluated_hull",
        "_precision_schedule",
    ):
        monkeypatch.setattr(adaptive, name, replayed)

    parsed = AdaptiveRangeEnclosureResult.model_validate_json(result.model_dump_json())
    assert parsed == result


def test_domain_unproven_result_cannot_carry_a_global_enclosure() -> None:
    expression = {
        "op": "log",
        "children": [
            {
                "op": "add",
                "children": [_var("x"), _const(1, 10**127)],
            }
        ],
    }
    result = _run(
        expression,
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(100),
        precision_bits=32,
        maximum_precision_bits=32,
        max_leaves=1,
        max_depth=8,
        max_evaluations=1,
    )
    payload = result.model_dump(mode="json")
    payload["enclosure"] = {
        "lower": {"mantissa": "0", "exponent": 0},
        "upper": {"mantissa": "1", "exponent": 0},
    }

    with pytest.raises(ValidationError, match="cannot carry a global enclosure"):
        AdaptiveRangeEnclosureResult.model_validate_json(json.dumps(payload))


def test_domain_failure_evidence_is_bound_to_the_source_expression_node() -> None:
    expression = {
        "op": "log",
        "children": [
            {
                "op": "add",
                "children": [_var("x"), _const(1, 10**127)],
            }
        ],
    }
    result = _run(
        expression,
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(100),
        precision_bits=32,
        maximum_precision_bits=32,
        max_leaves=1,
        max_depth=8,
        max_evaluations=1,
    )
    payload = result.model_dump(mode="json")
    payload["leaves"][0]["domain_failure"]["node_path"] = [0]

    with pytest.raises(ValidationError, match="does not match the source expression"):
        AdaptiveRangeEnclosureResult.model_validate_json(json.dumps(payload))


def test_result_schema_has_status_discriminated_conclusion_branches() -> None:
    schema = AdaptiveRangeEnclosureResult.model_json_schema()
    branch_references = {
        branch["$ref"].removeprefix("#/$defs/") for branch in schema["oneOf"]
    }
    assert branch_references == {
        "AdaptiveRangeConcludedResult",
        "AdaptiveRangeDomainUnprovenResult",
    }
    concluded = schema["$defs"]["AdaptiveRangeConcludedResult"]
    domain_unproven = schema["$defs"]["AdaptiveRangeDomainUnprovenResult"]
    reference = concluded["properties"]["disposition"]["$ref"]
    disposition = schema["$defs"][reference.removeprefix("#/$defs/")]
    leaf_reference = domain_unproven["properties"]["leaves"]["items"]["$ref"]
    leaf = schema["$defs"][leaf_reference.removeprefix("#/$defs/")]

    assert disposition["discriminator"]["propertyName"] == "status"
    assert set(disposition["discriminator"]["mapping"]) == {
        "BUDGET_EXHAUSTED",
        "TARGET_MET",
    }
    assert leaf["discriminator"]["propertyName"] == "status"
    assert set(leaf["discriminator"]["mapping"]) == {
        "DOMAIN_UNPROVEN",
        "ENCLOSED",
    }
    assert concluded["properties"]["enclosure"] == {
        "$ref": "#/$defs/DyadicClosedInterval"
    }
    assert domain_unproven["properties"]["enclosure"]["type"] == "null"
    assert domain_unproven["properties"]["leaves"]["minContains"] == 1


def test_derived_box_endpoint_envelope_is_schema_visible_to_producer_and_consumer() -> (
    None
):
    adaptive_schema = AdaptiveRangeEnclosureRequest.model_json_schema()
    consumer_schema = IntervalExpressionBoxEnclosureRequest.model_json_schema()
    endpoint_bound = str(MAX_RATIONAL_BOX_ENDPOINT_DIGITS)

    assert endpoint_bound in adaptive_schema["properties"]["box"]["description"]
    for schema in (adaptive_schema, consumer_schema):
        intervals = schema["$defs"]["RationalIntervalBox"]["properties"]["intervals"]
        assert endpoint_bound in intervals["description"]


def test_result_schema_and_parser_reject_contradictory_outcome_shapes() -> None:
    concluded = _run(
        _quadratic(),
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(7, 16),
        max_leaves=4,
        max_depth=2,
        max_evaluations=7,
    )
    uncertain = _run(
        {
            "op": "log",
            "children": [
                {
                    "op": "add",
                    "children": [_var("x"), _const(1, 10**127)],
                }
            ],
        },
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(100),
        precision_bits=32,
        maximum_precision_bits=32,
        max_leaves=1,
        max_depth=8,
        max_evaluations=1,
    )
    concluded_payload = json.loads(concluded.model_dump_json())
    uncertain_payload = json.loads(uncertain.model_dump_json())
    schema = AdaptiveRangeEnclosureResult.model_json_schema()
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    assert not list(validator.iter_errors(concluded_payload))
    assert not list(validator.iter_errors(uncertain_payload))

    concluded_without_enclosure = deepcopy(concluded_payload)
    concluded_without_enclosure["enclosure"] = None
    domain_missing_enclosure = deepcopy(uncertain_payload)
    domain_missing_enclosure.pop("enclosure")
    domain_with_enclosure = deepcopy(uncertain_payload)
    domain_with_enclosure["enclosure"] = concluded_payload["enclosure"]
    domain_without_uncertain_leaf = deepcopy(concluded_payload)
    domain_without_uncertain_leaf["enclosure"] = None
    domain_without_uncertain_leaf["disposition"] = {
        "status": "DOMAIN_UNPROVEN",
        "reason": "MAX_LEAVES",
    }
    concluded_with_uncertain_leaf = deepcopy(uncertain_payload)
    concluded_with_uncertain_leaf["enclosure"] = concluded_payload["enclosure"]
    concluded_with_uncertain_leaf["disposition"] = {"status": "TARGET_MET"}

    for forged in (
        concluded_without_enclosure,
        domain_missing_enclosure,
        domain_with_enclosure,
        domain_without_uncertain_leaf,
        concluded_with_uncertain_leaf,
    ):
        assert list(validator.iter_errors(forged))
        with pytest.raises(ValidationError):
            AdaptiveRangeEnclosureResult.model_validate_json(json.dumps(forged))

    target_contradiction = deepcopy(concluded_payload)
    target_contradiction["target_width"] = {"num": "1", "den": "1000000"}
    with pytest.raises(ValidationError, match="TARGET_MET"):
        AdaptiveRangeEnclosureResult.model_validate_json(
            json.dumps(target_contradiction)
        )

    budget_contradiction = deepcopy(concluded_payload)
    budget_contradiction["disposition"] = {
        "status": "BUDGET_EXHAUSTED",
        "reason": "MAX_LEAVES",
    }
    with pytest.raises(ValidationError, match="BUDGET_EXHAUSTED"):
        AdaptiveRangeEnclosureResult.model_validate_json(
            json.dumps(budget_contradiction)
        )
