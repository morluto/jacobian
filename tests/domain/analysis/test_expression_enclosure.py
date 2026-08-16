from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.validated_analysis import (
    IntervalExpressionEnclosureRequest,
    IntervalExpressionNode,
)
from jacobian.domains.analysis.expression_enclosure import compute_expression_enclosure


def test_log_of_rational():
    """log(137/80) should be enclosed and not exact."""
    request = IntervalExpressionEnclosureRequest.model_validate({
        "expression": {
            "op": "log",
            "children": [{"op": "const", "value": {"num": "137", "den": "80"}}],
        },
        "argument": {"num": "1", "den": "1"},
        "precision_bits": 128,
    })
    result = compute_expression_enclosure(request)
    assert result.status == "ENCLOSED"
    assert result.lower is not None
    assert result.upper is not None
    assert not result.exact
    assert result.lower.as_fraction() < result.upper.as_fraction()


def test_sqrt_of_two():
    """sqrt(2) should be enclosed and not exact."""
    request = IntervalExpressionEnclosureRequest.model_validate({
        "expression": {
            "op": "sqrt",
            "children": [{"op": "const", "value": {"num": "2", "den": "1"}}],
        },
        "argument": {"num": "1", "den": "1"},
        "precision_bits": 256,
    })
    result = compute_expression_enclosure(request)
    assert result.status == "ENCLOSED"
    assert not result.exact
    lower = result.lower.as_fraction()
    upper = result.upper.as_fraction()
    assert lower * lower <= 2 <= upper * upper


def test_exp_of_one():
    """exp(1) should be enclosed and not exact."""
    request = IntervalExpressionEnclosureRequest.model_validate({
        "expression": {
            "op": "exp",
            "children": [{"op": "const", "value": {"num": "1", "den": "1"}}],
        },
        "argument": {"num": "1", "den": "1"},
        "precision_bits": 128,
    })
    result = compute_expression_enclosure(request)
    assert result.status == "ENCLOSED"
    assert not result.exact


def test_sin_of_one():
    """sin(1) should be enclosed and not exact."""
    request = IntervalExpressionEnclosureRequest.model_validate({
        "expression": {
            "op": "sin",
            "children": [{"op": "const", "value": {"num": "1", "den": "1"}}],
        },
        "argument": {"num": "1", "den": "1"},
        "precision_bits": 128,
    })
    result = compute_expression_enclosure(request)
    assert result.status == "ENCLOSED"
    assert not result.exact


def test_polynomial_inequality():
    """x^2 + 2x + 1 = (x+1)^2 at x=3 should be exactly 16."""
    request = IntervalExpressionEnclosureRequest.model_validate({
        "expression": {
            "op": "add",
            "children": [
                {
                    "op": "add",
                    "children": [
                        {"op": "pow", "exponent": 2, "children": [{"op": "var"}]},
                        {
                            "op": "mul",
                            "children": [
                                {"op": "const", "value": {"num": "2", "den": "1"}},
                                {"op": "var"},
                            ],
                        },
                    ],
                },
                {"op": "const", "value": {"num": "1", "den": "1"}},
            ],
        },
        "argument": {"num": "3", "den": "1"},
        "precision_bits": 128,
    })
    result = compute_expression_enclosure(request)
    assert result.status == "ENCLOSED"
    assert result.exact
    assert result.lower.as_fraction() == result.upper.as_fraction()
    assert result.lower.as_fraction() == 16


def test_division_by_zero_fails_closed():
    """Division by zero should return INVALID status."""
    request = IntervalExpressionEnclosureRequest.model_validate({
        "expression": {
            "op": "div",
            "children": [
                {"op": "const", "value": {"num": "1", "den": "1"}},
                {"op": "const", "value": {"num": "0", "den": "1"}},
            ],
        },
        "argument": {"num": "1", "den": "1"},
        "precision_bits": 128,
    })
    result = compute_expression_enclosure(request)
    assert result.status == "INVALID"


def test_const_node_requires_value():
    """A const node without a value should fail validation."""
    with pytest.raises(ValidationError, match="const node requires a value"):
        IntervalExpressionNode.model_validate({"op": "const"})


def test_pow_node_requires_exponent():
    """A pow node without an exponent should fail validation."""
    with pytest.raises(ValidationError, match="pow node requires"):
        IntervalExpressionNode.model_validate(
            {"op": "pow", "children": [{"op": "var"}]}
        )


def test_binary_op_requires_two_children():
    """An add node with one child should fail validation."""
    with pytest.raises(ValidationError, match="add node requires exactly two children"):
        IntervalExpressionNode.model_validate(
            {"op": "add", "children": [{"op": "var"}]}
        )


def test_operation_discoverable():
    """The operation should be discoverable via the analysis factory."""
    from jacobian.domains.analysis import real_analysis_operations

    ops = real_analysis_operations()
    assert any(op.operation_id == "interval.compute.enclosure" for op in ops)
