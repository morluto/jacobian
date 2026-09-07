from __future__ import annotations

import json
from fractions import Fraction
from typing import Literal

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from tests.math.analysis._analysis_support import analysis_validation_error

from jacobian.math.analysis._arb import dyadic_endpoints
from jacobian.math.analysis._expression_enclosure import (
    IntervalExpressionEnclosureRequest,
    IntervalExpressionEnclosureResult,
)
from jacobian.math.analysis._models import (
    MAX_DYADIC_EXPONENT,
    MAX_DYADIC_MANTISSA_DIGITS,
    ExactDyadic,
)
from jacobian.math.analysis.operations import expression_enclosure


@pytest.mark.parametrize("mode", ["validation", "serialization"])
@pytest.mark.parametrize("sign", [1, -1])
def test_dyadic_signed_mantissa_boundary(
    mode: Literal["validation", "serialization"], sign: int
) -> None:
    digits = MAX_DYADIC_MANTISSA_DIGITS - int(sign < 0)
    accepted = sign * (10**digits - 1)
    rejected = sign * (10**digits + 1)
    native = ExactDyadic(mantissa=accepted, exponent=0)
    assert ExactDyadic.model_validate_json(native.model_dump_json()) == native
    validator = Draft202012Validator(ExactDyadic.model_json_schema(mode=mode))
    assert validator.is_valid(native.model_dump(mode="json"))
    wire = {"mantissa": str(rejected), "exponent": 0}
    assert not validator.is_valid(wire)
    with pytest.raises(ValidationError):
        ExactDyadic(mantissa=rejected, exponent=0)
    with pytest.raises(ValidationError):
        ExactDyadic.model_validate_json(json.dumps(wire))


def _run(
    expression: dict[str, object], argument: str = "0"
) -> IntervalExpressionEnclosureResult:
    request = IntervalExpressionEnclosureRequest.model_validate_json(
        json.dumps(
            {
                "expression": expression,
                "argument": {"num": argument, "den": "1"},
                "precision_bits": 128,
            }
        )
    )
    return expression_enclosure(
        request.expression, request.argument, request.precision_bits
    )


@pytest.mark.parametrize(
    ("op", "num", "den"),
    [("log", "137", "80"), ("sqrt", "2", "1"), ("exp", "1", "1"), ("sin", "1", "1")],
)
def test_transcendental_known_answers_are_rigorously_enclosed(
    op: str, num: str, den: str
) -> None:
    result = _run(
        {"op": op, "children": [{"op": "const", "value": {"num": num, "den": den}}]}
    )

    assert result.status == "ENCLOSED"
    assert result.lower is not None and result.upper is not None
    assert result.lower.as_fraction() <= result.upper.as_fraction()
    assert result.exact is False


def test_exact_polynomial_preserves_its_defining_value() -> None:
    result = _run(
        {
            "op": "pow",
            "exponent": 2,
            "children": [
                {
                    "op": "add",
                    "children": [
                        {"op": "var"},
                        {"op": "const", "value": {"num": "1", "den": "1"}},
                    ],
                }
            ],
        },
        "3",
    )

    assert result.status == "ENCLOSED"
    assert result.lower is not None and result.upper is not None
    assert result.lower.as_fraction() == Fraction(16)
    assert result.upper.as_fraction() == Fraction(16)
    assert result.exact is True


@pytest.mark.parametrize(
    "expression",
    [
        {
            "op": "div",
            "children": [
                {"op": "const", "value": {"num": "1", "den": "1"}},
                {"op": "const", "value": {"num": "0", "den": "1"}},
            ],
        },
        {
            "op": "log",
            "children": [{"op": "const", "value": {"num": "-1", "den": "1"}}],
        },
        {
            "op": "sqrt",
            "children": [{"op": "const", "value": {"num": "-1", "den": "1"}}],
        },
    ],
)
def test_real_domain_failures_return_typed_results(
    expression: dict[str, object],
) -> None:
    assert _run(expression).status == "DOMAIN_ERROR"


def test_nested_domain_failure_propagates_as_a_typed_result() -> None:
    result = _run(
        {
            "op": "add",
            "children": [
                {
                    "op": "log",
                    "children": [{"op": "const", "value": {"num": "-1", "den": "1"}}],
                },
                {"op": "const", "value": {"num": "1", "den": "1"}},
            ],
        }
    )
    assert result.status == "DOMAIN_ERROR"


def test_uncertain_denominator_reports_precision_instead_of_domain_error() -> None:
    exp_one = {
        "op": "exp",
        "children": [{"op": "const", "value": {"num": "1", "den": "1"}}],
    }
    request = IntervalExpressionEnclosureRequest.model_validate_json(
        json.dumps(
            {
                "expression": {
                    "op": "div",
                    "children": [
                        {"op": "const", "value": {"num": "1", "den": "1"}},
                        {
                            "op": "add",
                            "children": [
                                {
                                    "op": "const",
                                    "value": {"num": "1", "den": str(2**100)},
                                },
                                {"op": "sub", "children": [exp_one, exp_one]},
                            ],
                        },
                    ],
                },
                "argument": {"num": "0", "den": "1"},
                "precision_bits": 32,
            }
        )
    )
    result = expression_enclosure(
        request.expression, request.argument, request.precision_bits
    )
    assert result.status == "PRECISION_INSUFFICIENT"


def test_nonfinite_intermediate_is_not_consumed_by_parent_arithmetic() -> None:
    result = _run(
        {
            "op": "div",
            "children": [
                {"op": "const", "value": {"num": "1", "den": "1"}},
                {
                    "op": "exp",
                    "children": [
                        {
                            "op": "exp",
                            "children": [
                                {
                                    "op": "const",
                                    "value": {
                                        "num": "100000000000000000000",
                                        "den": "1",
                                    },
                                }
                            ],
                        }
                    ],
                },
            ],
        }
    )
    assert result.status == "NONFINITE"


def test_expression_depth_is_rejected_before_arb() -> None:
    expression: dict[str, object] = {"op": "var"}
    for _ in range(16):
        expression = {"op": "neg", "children": [expression]}
    with analysis_validation_error():
        IntervalExpressionEnclosureRequest.model_validate(
            {"expression": expression, "argument": {"num": "0", "den": "1"}}
        )


def test_operation_payloads_are_structurally_typed() -> None:
    with analysis_validation_error():
        IntervalExpressionEnclosureRequest.model_validate(
            {
                "expression": {"op": "var", "value": {"num": "1", "den": "1"}},
                "argument": {"num": "0", "den": "1"},
            }
        )


def test_dyadic_enclosure_order_avoids_expanding_huge_binary_exponents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_expanded(self: ExactDyadic) -> Fraction:
        raise AssertionError("endpoint comparison must not materialize a power of two")

    monkeypatch.setattr(ExactDyadic, "as_fraction", fail_if_expanded)
    result = IntervalExpressionEnclosureResult(
        status="ENCLOSED",
        precision_bits=128,
        lower=ExactDyadic(mantissa=1, exponent=MAX_DYADIC_EXPONENT),
        upper=ExactDyadic(mantissa=3, exponent=MAX_DYADIC_EXPONENT - 1),
        relative_accuracy_bits=100,
        detail="synthetic compact dyadic enclosure",
    )
    assert result.lower is not None and result.upper is not None


def test_non_interoperable_dyadic_exponents_are_not_materialized() -> None:
    assert dyadic_endpoints(1, MAX_DYADIC_EXPONENT + 1, 3, 0) is None


def test_unrepresentable_expression_enclosure_is_an_execution_failure() -> None:
    with pytest.raises(RuntimeError, match="outside the interoperable dyadic"):
        _run({"op": "exp", "children": [{"op": "var"}]}, str(10**17))
