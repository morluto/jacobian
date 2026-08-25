from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from itertools import product
from typing import Any

import pytest
from pydantic import ValidationError
from tests.math._analysis_support import analysis_validation_error

from jacobian.math.analysis._models import (
    IntervalExpressionBoxEnclosureRequest,
    IntervalExpressionBoxEnclosureResult,
    IntervalExpressionEnclosureRequest,
)
from jacobian.math.analysis._operations import (
    _box_expression_enclosure,
    _expression_enclosure,
)


def _q(numerator: int, denominator: int = 1) -> dict[str, str]:
    value = Fraction(numerator, denominator)
    return {"num": str(value.numerator), "den": str(value.denominator)}


def _request(
    expression: dict[str, Any],
    coordinates: tuple[tuple[str, Fraction, Fraction], ...],
    *,
    precision_bits: int = 128,
) -> IntervalExpressionBoxEnclosureRequest:
    return IntervalExpressionBoxEnclosureRequest.model_validate(
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
            "precision_bits": precision_bits,
        }
    )


def _run(
    expression: dict[str, Any],
    coordinates: tuple[tuple[str, Fraction, Fraction], ...],
    *,
    precision_bits: int = 128,
) -> IntervalExpressionBoxEnclosureResult:
    return _box_expression_enclosure(
        _request(expression, coordinates, precision_bits=precision_bits)
    )


def _var(name: str) -> dict[str, Any]:
    return {"op": "var", "variable": name}


def _const(value: int) -> dict[str, Any]:
    return {"op": "const", "value": _q(value)}


@pytest.mark.parametrize("op", ["exp", "log"])
def test_monotone_box_enclosures_contain_both_endpoint_enclosures(op: str) -> None:
    lower = Fraction(1) if op == "log" else Fraction(0)
    upper = Fraction(2) if op == "log" else Fraction(1)
    expression = {"op": op, "children": [_var("x")]}
    box_result = _run(expression, (("x", lower, upper),))

    point_results = []
    for argument in (lower, upper):
        point_results.append(
            _expression_enclosure(
                IntervalExpressionEnclosureRequest.model_validate(
                    {
                        "expression": {
                            "op": op,
                            "children": [{"op": "var"}],
                        },
                        "argument": _q(argument.numerator, argument.denominator),
                        "precision_bits": 128,
                    }
                )
            )
        )

    assert box_result.status == "ENCLOSED"
    assert box_result.lower is not None and box_result.upper is not None
    lower_point = point_results[0].lower
    upper_point = point_results[1].upper
    assert lower_point is not None and upper_point is not None
    assert box_result.lower.compare(lower_point) <= 0
    assert box_result.upper.compare(upper_point) >= 0


def test_degenerate_point_box_agrees_with_point_expression_operation() -> None:
    box_result = _run(
        {"op": "exp", "children": [_var("x")]},
        (("x", Fraction(1), Fraction(1)),),
    )
    point_result = _expression_enclosure(
        IntervalExpressionEnclosureRequest.model_validate(
            {
                "expression": {"op": "exp", "children": [{"op": "var"}]},
                "argument": _q(1),
                "precision_bits": 128,
            }
        )
    )

    assert box_result.status == point_result.status == "ENCLOSED"
    assert box_result.lower == point_result.lower
    assert box_result.upper == point_result.upper


def test_request_precision_overrides_the_ambient_arb_context() -> None:
    from flint import ctx

    request = _request(
        {"op": "exp", "children": [_var("x")]},
        (("x", Fraction(0), Fraction(1)),),
        precision_bits=128,
    )
    with ctx.workprec(64):
        low_ambient_precision = _box_expression_enclosure(request)
    with ctx.workprec(512):
        high_ambient_precision = _box_expression_enclosure(request)

    assert low_ambient_precision == high_ambient_precision


def test_sine_box_contains_an_interior_extremum() -> None:
    result = _run(
        {"op": "sin", "children": [_var("x")]},
        (("x", Fraction(0), Fraction(2)),),
    )

    assert result.status == "ENCLOSED"
    assert result.lower is not None and result.upper is not None
    assert result.lower.as_fraction() <= 1 <= result.upper.as_fraction()


def test_dependency_enclosure_is_conservative_but_sound() -> None:
    result = _run(
        {"op": "sub", "children": [_var("x"), _var("x")]},
        (("x", Fraction(0), Fraction(1)),),
    )

    assert result.status == "ENCLOSED"
    assert result.lower is not None and result.upper is not None
    assert result.lower.as_fraction() <= 0 <= result.upper.as_fraction()


def test_coupled_multivariate_box_contains_every_exact_corner_value() -> None:
    expression = {
        "op": "add",
        "children": [
            {"op": "mul", "children": [_var("q"), _var("s")]},
            {"op": "div", "children": [_var("t"), _var("z")]},
        ],
    }
    coordinates = (
        ("q", Fraction(1), Fraction(2)),
        ("s", Fraction(2), Fraction(3)),
        ("t", Fraction(1), Fraction(2)),
        ("z", Fraction(2), Fraction(4)),
    )
    result = _run(expression, coordinates)

    assert result.status == "ENCLOSED"
    assert result.lower is not None and result.upper is not None
    for q, s, t, z in product(*(pair[1:] for pair in coordinates)):
        value = q * s + t / z
        assert result.lower.as_fraction() <= value <= result.upper.as_fraction()


def test_equivalent_variable_axis_permutations_preserve_the_enclosure() -> None:
    expression = {
        "op": "sub",
        "children": [
            {"op": "mul", "children": [_var("x"), _var("y")]},
            _var("x"),
        ],
    }
    xy = _run(
        expression,
        (("x", Fraction(1), Fraction(2)), ("y", Fraction(3), Fraction(4))),
    )
    yx = _run(
        expression,
        (("y", Fraction(3), Fraction(4)), ("x", Fraction(1), Fraction(2))),
    )

    assert xy.status == yx.status == "ENCLOSED"
    assert xy.lower == yx.lower
    assert xy.upper == yx.upper


def test_narrow_positive_box_does_not_lose_its_proved_log_domain() -> None:
    tiny = Fraction(1, 10**50)
    result = _run(
        {"op": "log", "children": [_var("x")]},
        (("x", tiny, Fraction(1)),),
    )

    assert result.status == "ENCLOSED"


def test_variable_enclosure_contains_both_exact_rational_box_endpoints() -> None:
    result = _run(
        _var("x"),
        (("x", Fraction(1, 3), Fraction(2, 3)),),
    )

    assert result.status == "ENCLOSED"
    assert result.lower is not None and result.upper is not None
    assert result.lower.as_fraction() <= Fraction(1, 3)
    assert result.upper.as_fraction() >= Fraction(2, 3)


def test_narrow_negative_box_does_not_gain_a_spurious_reciprocal_pole() -> None:
    tiny = Fraction(1, 10**50)
    result = _run(
        {"op": "div", "children": [_const(1), _var("x")]},
        (("x", Fraction(-1), -tiny),),
    )

    assert result.status == "ENCLOSED"
    assert result.lower is not None and result.upper is not None
    assert result.lower.as_fraction() <= -(10**50) <= result.upper.as_fraction()
    assert result.lower.as_fraction() <= -1 <= result.upper.as_fraction()


@pytest.mark.parametrize(
    ("expression", "operation", "reason"),
    [
        (
            {"op": "div", "children": [_const(1), _var("x")]},
            "div",
            "DENOMINATOR_CONTAINS_ZERO",
        ),
        (
            {"op": "pow", "exponent": -2, "children": [_var("x")]},
            "pow",
            "NEGATIVE_POWER_BASE_CONTAINS_ZERO",
        ),
        (
            {"op": "log", "children": [_var("x")]},
            "log",
            "LOG_ARGUMENT_NOT_STRICTLY_POSITIVE",
        ),
        (
            {"op": "sqrt", "children": [_var("x")]},
            "sqrt",
            "SQRT_ARGUMENT_NOT_NONNEGATIVE",
        ),
    ],
)
def test_domain_rejections_identify_the_source_node(
    expression: dict[str, Any], operation: str, reason: str
) -> None:
    result = _run(expression, (("x", Fraction(-1), Fraction(1)),))

    assert result.status == "DOMAIN_UNPROVEN"
    assert result.lower is result.upper is None
    assert result.domain_failure is not None
    assert result.domain_failure.node_path == ()
    assert result.domain_failure.operation == operation
    assert result.domain_failure.reason == reason


def test_first_domain_rejection_uses_deterministic_left_to_right_tree_order() -> None:
    result = _run(
        {
            "op": "add",
            "children": [
                {"op": "log", "children": [_var("x")]},
                {"op": "sqrt", "children": [_var("y")]},
            ],
        },
        (
            ("x", Fraction(-1), Fraction(1)),
            ("y", Fraction(-1), Fraction(1)),
        ),
    )

    assert result.domain_failure is not None
    assert result.domain_failure.node_path == (0,)
    assert result.domain_failure.operation == "log"


def test_first_domain_rejection_short_circuits_later_preflight_growth() -> None:
    result = _run(
        {
            "op": "add",
            "children": [
                {"op": "log", "children": [_var("x")]},
                {"op": "exp", "children": [_const(4096)]},
            ],
        },
        (("x", Fraction(-1), Fraction(1)),),
    )

    assert result.status == "DOMAIN_UNPROVEN"
    assert result.domain_failure is not None
    assert result.domain_failure.node_path == (0,)


def test_producer_result_round_trips_through_source_replay() -> None:
    result = _run(
        {"op": "exp", "children": [_var("x")]},
        (("x", Fraction(0), Fraction(1)),),
    )

    assert (
        IntervalExpressionBoxEnclosureResult.model_validate(result.model_dump())
        == result
    )


@pytest.mark.parametrize("mutation", ["expression", "box", "endpoint"])
def test_source_or_endpoint_mutation_is_rejected_by_replay(mutation: str) -> None:
    result = _run(
        {"op": "exp", "children": [_var("x")]},
        (("x", Fraction(0), Fraction(1)),),
    )
    payload = deepcopy(result.model_dump())
    if mutation == "expression":
        payload["expression"]["op"] = "sin"
    elif mutation == "box":
        payload["box"]["intervals"][0]["upper"] = _q(2)
    else:
        payload["lower"] = {"mantissa": "0", "exponent": 0}

    with analysis_validation_error():
        IntervalExpressionBoxEnclosureResult.model_validate(payload)


def test_domain_rejection_is_also_bound_to_its_source() -> None:
    result = _run(
        {"op": "div", "children": [_const(1), _var("x")]},
        (("x", Fraction(-1), Fraction(1)),),
    )
    assert (
        IntervalExpressionBoxEnclosureResult.model_validate(result.model_dump())
        == result
    )
    payload = deepcopy(result.model_dump())
    payload["box"]["intervals"] = (
        {
            "lower": _q(1),
            "upper": _q(2),
        },
    )

    with analysis_validation_error():
        IntervalExpressionBoxEnclosureResult.model_validate(payload)


def test_producer_does_not_pay_a_second_backend_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flint import ctx

    import jacobian.math.analysis._operations as operations

    calls = 0
    work_precisions: list[int] = []
    original = operations._evaluate_box_expression

    def counting(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        if len(args) == 2 and "path" not in kwargs:
            calls += 1
            work_precisions.append(ctx.prec)
        return original(*args, **kwargs)

    monkeypatch.setattr(operations, "_evaluate_box_expression", counting)
    _run(
        {"op": "exp", "children": [_var("x")]},
        (("x", Fraction(0), Fraction(1)),),
    )
    assert calls == 1
    assert work_precisions == [128]


def test_backend_value_error_has_a_typed_nonconclusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.analysis._operations as operations

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("synthetic backend rejection")

    monkeypatch.setattr(operations, "_evaluate_box_expression", fail)
    result = _run(
        {"op": "exp", "children": [_var("x")]},
        (("x", Fraction(0), Fraction(1)),),
    )
    assert result.status == "BACKEND_ERROR"
    assert result.lower is None and result.upper is None
    assert result.domain_failure is None


def test_backend_error_result_round_trips_when_the_failure_does_not_recur(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.analysis._operations as operations

    original = operations._evaluate_box_expression
    calls = 0

    def transient_then_original(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("transient backend rejection")
        return original(*args, **kwargs)

    monkeypatch.setattr(operations, "_evaluate_box_expression", transient_then_original)
    result = _run(
        {"op": "exp", "children": [_var("x")]},
        (("x", Fraction(0), Fraction(1)),),
    )
    assert result.status == "BACKEND_ERROR"

    assert (
        IntervalExpressionBoxEnclosureResult.model_validate(result.model_dump())
        == result
    )
    assert IntervalExpressionBoxEnclosureResult.model_validate_json(
        result.model_dump_json()
    )
    assert calls == 1


@pytest.mark.parametrize(
    ("field", "payload_patch"),
    [
        (
            "endpoints",
            {
                "lower": {"mantissa": "0", "exponent": 0},
                "upper": {"mantissa": "1", "exponent": 0},
            },
        ),
        (
            "domain_failure",
            {
                "domain_failure": {
                    "node_path": [],
                    "operation": "log",
                    "reason": "LOG_ARGUMENT_NOT_STRICTLY_POSITIVE",
                },
            },
        ),
    ],
)
def test_backend_error_payload_cannot_smuggle_conclusion_evidence(
    field: str, payload_patch: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import jacobian.math.analysis._operations as operations

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("synthetic backend rejection")

    monkeypatch.setattr(operations, "_evaluate_box_expression", fail)
    result = _run(
        {"op": "exp", "children": [_var("x")]},
        (("x", Fraction(1), Fraction(2)),),
    )
    assert result.status == "BACKEND_ERROR"
    payload = {**deepcopy(result.model_dump()), **payload_patch}

    with pytest.raises(ValidationError):
        IntervalExpressionBoxEnclosureResult.model_validate(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "expression": {"op": "var"},
                "box": {
                    "variables": ["x"],
                    "intervals": [{"lower": _q(0), "upper": _q(1)}],
                },
            },
            "must be named",
        ),
        (
            {
                "expression": _var("x"),
                "box": {"variables": [], "intervals": []},
            },
            "missing from the box",
        ),
        (
            {
                "expression": _var("x"),
                "box": {
                    "variables": ["x", "y"],
                    "intervals": [
                        {"lower": _q(0), "upper": _q(1)},
                        {"lower": _q(0), "upper": _q(1)},
                    ],
                },
            },
            "unused by the expression",
        ),
        (
            {
                "expression": _var("x"),
                "box": {
                    "variables": ["x", "x"],
                    "intervals": [
                        {"lower": _q(0), "upper": _q(1)},
                        {"lower": _q(0), "upper": _q(1)},
                    ],
                },
            },
            "must be unique",
        ),
    ],
)
def test_expression_and_box_must_share_one_complete_named_axis(
    payload: dict[str, Any], message: str
) -> None:
    with analysis_validation_error():
        IntervalExpressionBoxEnclosureRequest.model_validate(payload)


def test_named_variables_are_not_accepted_by_the_point_expression_contract() -> None:
    with analysis_validation_error():
        IntervalExpressionEnclosureRequest.model_validate(
            {
                "expression": _var("x"),
                "argument": _q(0),
            }
        )


def test_box_interval_endpoints_are_ordered_and_bounded() -> None:
    with analysis_validation_error():
        _request(
            _var("x"),
            (("x", Fraction(1), Fraction(0)),),
        )

    too_many_digits = "1" * 129
    with analysis_validation_error():
        IntervalExpressionBoxEnclosureRequest.model_validate(
            {
                "expression": _var("x"),
                "box": {
                    "variables": ["x"],
                    "intervals": [
                        {
                            "lower": {"num": "0", "den": "1"},
                            "upper": {"num": too_many_digits, "den": "1"},
                        }
                    ],
                },
            }
        )


def test_variable_names_are_non_evaluating_identifiers() -> None:
    with analysis_validation_error():
        _request(
            _var("__import__('os').system('false')"),
            (("x", Fraction(0), Fraction(1)),),
        )

    with analysis_validation_error():
        IntervalExpressionBoxEnclosureRequest.model_validate(
            {
                "expression": {"op": "var", "variable": "x", "source": "x+1"},
                "box": {
                    "variables": ["x"],
                    "intervals": [{"lower": _q(0), "upper": _q(1)}],
                },
            }
        )


def test_point_and_box_requests_compose_through_strict_json_transport() -> None:
    point = IntervalExpressionEnclosureRequest.model_validate(
        {
            "expression": {"op": "exp", "children": [{"op": "var"}]},
            "argument": _q(1),
            "precision_bits": 128,
        }
    )
    box = _request(
        {"op": "exp", "children": [_var("x")]},
        (("x", Fraction(0), Fraction(1)),),
    )

    assert (
        IntervalExpressionEnclosureRequest.model_validate_json(
            point.model_dump_json(), strict=True
        )
        == point
    )
    assert (
        IntervalExpressionBoxEnclosureRequest.model_validate_json(
            box.model_dump_json(), strict=True
        )
        == box
    )


def test_variable_count_is_rejected_before_evaluation() -> None:
    variables = tuple(f"x{index}" for index in range(9))
    with analysis_validation_error():
        IntervalExpressionBoxEnclosureRequest.model_validate(
            {
                "expression": _var("x0"),
                "box": {
                    "variables": variables,
                    "intervals": [{"lower": _q(0), "upper": _q(1)} for _ in variables],
                },
            }
        )


def test_raw_rationals_and_domain_paths_are_bounded_before_nested_parsing() -> None:
    oversized = {"num": "x" * 129, "den": "1"}
    with analysis_validation_error():
        IntervalExpressionEnclosureRequest.model_validate(
            {
                "expression": {"op": "var"},
                "argument": oversized,
            }
        )
    with analysis_validation_error():
        IntervalExpressionBoxEnclosureRequest.model_validate(
            {
                "expression": {"op": "const", "value": oversized},
                "box": {"variables": [], "intervals": []},
            }
        )
    with analysis_validation_error():
        IntervalExpressionBoxEnclosureRequest.model_validate(
            {
                "expression": _var("x"),
                "box": {
                    "variables": ["x"],
                    "intervals": [{"lower": _q(0), "upper": oversized}],
                },
            }
        )

    domain_result = _run(
        {"op": "log", "children": [_var("x")]},
        (("x", Fraction(-1), Fraction(1)),),
    )
    payload = domain_result.model_dump(mode="json")
    payload["domain_failure"]["node_path"] = [0] * 16
    with analysis_validation_error():
        IntervalExpressionBoxEnclosureResult.model_validate(payload)


def test_raw_expression_size_is_bounded_before_recursive_model_parsing() -> None:
    leaves: list[dict[str, Any]] = [
        {"op": "var", "variable": "not valid!"} for _ in range(64)
    ]
    while len(leaves) > 1:
        leaves = [
            {"op": "add", "children": leaves[index : index + 2]}
            for index in range(0, len(leaves), 2)
        ]

    with analysis_validation_error():
        IntervalExpressionBoxEnclosureRequest.model_validate(
            {
                "expression": leaves[0],
                "box": {"variables": [], "intervals": []},
            }
        )


def test_eight_variable_boundary_is_admitted() -> None:
    variables = tuple(f"x{index}" for index in range(8))
    expression = _var(variables[0])
    for variable in variables[1:]:
        expression = {"op": "add", "children": [expression, _var(variable)]}

    result = _run(
        expression,
        tuple((variable, Fraction(0), Fraction(1)) for variable in variables),
    )
    assert result.status == "ENCLOSED"


def test_intermediate_growth_is_rejected_during_request_validation() -> None:
    _request(
        {"op": "exp", "children": [_const(4095)]},
        (),
    )
    with analysis_validation_error():
        _request(
            {"op": "exp", "children": [_const(4096)]},
            (),
        )

    comparison_boundary = Fraction(10**127 + 51, 10**127 + 87)
    powered = {
        "op": "pow",
        "exponent": 19,
        "children": [
            {
                "op": "const",
                "value": _q(
                    comparison_boundary.numerator,
                    comparison_boundary.denominator,
                ),
            }
        ],
    }
    with analysis_validation_error():
        _request(
            {"op": "mul", "children": [powered, deepcopy(powered)]},
            (),
        )

    wide_power_base = 1 << 128
    with analysis_validation_error():
        _request(
            {
                "op": "pow",
                "exponent": 64,
                "children": [_const(wide_power_base)],
            },
            (),
        )


def test_constant_expression_uses_the_zero_dimensional_box() -> None:
    result = _run(
        {"op": "pow", "exponent": 3, "children": [_const(2)]},
        (),
    )

    assert result.status == "ENCLOSED"
    assert result.lower is not None and result.upper is not None
    assert result.lower.as_fraction() == result.upper.as_fraction() == 8
