from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from math import sin
from time import monotonic
from typing import Any

import pytest
from pydantic import ValidationError
from tests.fixtures.accounting import assert_charged_work_parity
from tests.math.analysis._analysis_support import analysis_validation_error

from jacobian._execution import OperationExecutionTimeoutError, request_execution
from jacobian.canonical import canonicalize_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.analysis._adaptive_range_enclosure import (
    MAX_ADAPTIVE_RANGE_RESULT_BYTES,
    AdaptiveRangeBudgetExhausted,
    AdaptiveRangeEnclosureRequest,
    AdaptiveRangeEnclosureResult,
    AdaptiveRangeTargetMet,
    _admit_adaptive_range,
    _compute_adaptive_range_enclosure,
    _enclosure_width,
    _estimated_result_bytes,
    _interval_hull,
    adaptive_range_enclosure,
)
from jacobian.math.analysis._box_enclosure import (
    IntervalExpressionBoxEnclosureRequest,
    _box_expression_enclosure,
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
    return AdaptiveRangeEnclosureRequest.model_validate(
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
    assert result.enclosure == _interval_hull(result.leaves)
    assert result.enclosure.lower.as_fraction() <= 0
    assert result.enclosure.upper.as_fraction() >= Fraction(1, 4)
    assert _enclosure_width(result.enclosure) <= Fraction(7, 16)
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

    for leaf in result.leaves:
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
    assert result.enclosure.lower.as_fraction() <= -1
    assert result.enclosure.upper.as_fraction() >= 1


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

    lower = float(result.enclosure.lower.as_fraction())
    upper = float(result.enclosure.upper.as_fraction())
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

    assert result.enclosure.lower.as_fraction() <= 0
    assert result.enclosure.upper.as_fraction() >= 0


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
        IntervalExpressionBoxEnclosureRequest.model_validate(
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
    result = _run(
        expression,
        (source_box,),
        target_width=Fraction(1, 8),
        max_leaves=4,
        max_depth=2,
        max_evaluations=7,
    )

    assert one_box.status == "ENCLOSED"
    assert one_box.upper.as_fraction() >= Fraction(1, 9)
    assert isinstance(result.disposition, AdaptiveRangeTargetMet)
    assert len(result.leaves) == 4
    assert result.enclosure.upper.as_fraction() < Fraction(1, 9)


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
    assert result.enclosure.lower.as_fraction() <= 0
    assert result.enclosure.upper.as_fraction() >= 1


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
    ("coordinates", "request_overrides", "forged_reason"),
    [
        (
            (),
            {
                "expression": {"op": "exp", "children": [_const(1)]},
                "precision_bits": 32,
                "maximum_precision_bits": 100,
                "max_leaves": 1,
                "max_depth": 4,
                "max_evaluations": 5,
            },
            "MAX_LEAVES",
        ),
        (
            (("x", Fraction(0), Fraction(1)),),
            {
                "expression": _var("x"),
                "max_leaves": 4,
                "max_depth": 0,
                "max_evaluations": 1,
            },
            "MAX_EVALUATIONS",
        ),
    ],
)
def test_round_trip_rejects_an_irrelevant_no_split_budget_reason(
    coordinates: tuple[tuple[str, Fraction, Fraction], ...],
    request_overrides: dict[str, Any],
    forged_reason: str,
) -> None:
    expression = request_overrides.pop("expression")
    result = _run(
        expression,
        coordinates,
        target_width=Fraction(1, 10**60),
        **request_overrides,
    )
    payload = result.model_dump(mode="json")
    payload["disposition"] = {
        "status": "BUDGET_EXHAUSTED",
        "reason": forged_reason,
    }

    with pytest.raises(ValidationError):
        AdaptiveRangeEnclosureResult.model_validate(payload)


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
        AdaptiveRangeEnclosureRequest.model_validate(payload)

    payload["target_width"] = _q(1)
    with analysis_validation_error():
        AdaptiveRangeEnclosureRequest.model_validate(payload)


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


def test_result_sensitive_output_bound_rejects_only_the_large_envelope() -> None:
    variables = tuple(f"x{index}" for index in range(8))
    expression: dict[str, Any] = _var(variables[0])
    for variable in variables[1:]:
        expression = {"op": "add", "children": [expression, _var(variable)]}
    large = _request(
        expression,
        tuple((variable, Fraction(0), Fraction(10**127)) for variable in variables),
        target_width=Fraction(1),
        max_leaves=1024,
        max_depth=32,
        max_evaluations=1,
    )
    assert _estimated_result_bytes(large) > MAX_ADAPTIVE_RANGE_RESULT_BYTES
    with pytest.raises(OperationDomainValidationError) as caught:
        _compute_adaptive_range_enclosure(large)
    assert caught.value.errors()[0]["type"] == "analysis.adaptive_range.result_bytes"

    small = _request(
        _var("x"),
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(2),
        max_leaves=1024,
        max_depth=32,
        max_evaluations=1,
    )
    assert _estimated_result_bytes(small) < MAX_ADAPTIVE_RANGE_RESULT_BYTES
    assert isinstance(
        _compute_adaptive_range_enclosure(small).disposition,
        AdaptiveRangeTargetMet,
    )


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
    AdaptiveRangeEnclosureRequest.model_validate(payload)

    payload[field] = above_limit
    with analysis_validation_error():
        AdaptiveRangeEnclosureRequest.model_validate(payload)


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
    admission = _admit_adaptive_range(request, started_at=monotonic())
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


@pytest.mark.parametrize("mutation", ["path", "box", "hull", "disposition"])
def test_round_trip_rejects_forged_partition_or_target_claim(mutation: str) -> None:
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
    if mutation == "path":
        payload["leaves"][0]["path"] = [0, 1]
    elif mutation == "box":
        payload["leaves"][0]["box"]["intervals"][0]["upper"] = _q(1, 3)
    elif mutation == "hull":
        payload["enclosure"]["lower"] = {"mantissa": "0", "exponent": 0}
    else:
        payload["disposition"] = {
            "status": "BUDGET_EXHAUSTED",
            "reason": "MAX_LEAVES",
        }

    with pytest.raises(ValidationError):
        AdaptiveRangeEnclosureResult.model_validate(payload)


def test_result_reservation_dominates_actual_canonical_output() -> None:
    request = _request(
        _quadratic(),
        (("x", Fraction(0), Fraction(1)),),
        target_width=Fraction(7, 16),
        max_leaves=4,
        max_depth=2,
        max_evaluations=7,
    )
    result = _compute_adaptive_range_enclosure(request)

    assert len(canonicalize_json(result.model_dump(mode="json"))) <= (
        _estimated_result_bytes(request)
    )


def test_disposition_schema_is_a_status_discriminated_union() -> None:
    schema = AdaptiveRangeEnclosureResult.model_json_schema()
    reference = schema["properties"]["disposition"]["$ref"]
    disposition = schema["$defs"][reference.removeprefix("#/$defs/")]

    assert disposition["discriminator"]["propertyName"] == "status"
    assert set(disposition["discriminator"]["mapping"]) == {
        "BUDGET_EXHAUSTED",
        "TARGET_MET",
    }
