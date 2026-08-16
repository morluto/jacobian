"""Expression-tree enclosure backed by Arb ball arithmetic."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.contracts.validated_analysis import (
    ExactDyadic,
    IntervalExpressionEnclosureRequest,
    IntervalExpressionEnclosureResult,
    IntervalExpressionNode,
)


def _eval_node(node: IntervalExpressionNode, var_value: object) -> object:
    """Evaluate one expression node using Arb ball arithmetic."""
    from flint import arb, fmpq

    op = node.op
    if op == "const":
        num, den = node.value.as_integer_ratio()  # type: ignore[union-attr]
        return arb(fmpq(num, den))
    if op == "var":
        return var_value
    if op == "neg":
        return -_eval_node(node.children[0], var_value)
    if op == "add":
        return _eval_node(node.children[0], var_value) + _eval_node(
            node.children[1], var_value
        )
    if op == "sub":
        return _eval_node(node.children[0], var_value) - _eval_node(
            node.children[1], var_value
        )
    if op == "mul":
        return _eval_node(node.children[0], var_value) * _eval_node(
            node.children[1], var_value
        )
    if op == "div":
        denominator = _eval_node(node.children[1], var_value)
        if denominator.contains(0):
            raise ZeroDivisionError("division by zero in expression enclosure")
        return _eval_node(node.children[0], var_value) / denominator
    if op == "pow":
        base = _eval_node(node.children[0], var_value)
        exp = node.exponent
        assert exp is not None
        if base.contains(0) and exp < 0:
            raise ZeroDivisionError("zero raised to negative power")
        return base ** int(exp)
    child_value = _eval_node(node.children[0], var_value)
    if op == "exp":
        return child_value.exp()
    if op == "log":
        if child_value.contains(0):
            raise ValueError("log of non-positive value")
        return child_value.log()
    if op == "sqrt":
        return child_value.sqrt()
    if op == "sin":
        return child_value.sin()
    if op == "cos":
        return child_value.cos()
    raise ValueError(f"unsupported node op: {op}")


def compute_expression_enclosure(
    request: IntervalExpressionEnclosureRequest,
) -> IntervalExpressionEnclosureResult:
    """Compute a rigorous Arb ball enclosure for one expression tree."""
    from flint import arb, ctx, fmpq

    numerator, denominator = request.argument.as_integer_ratio()
    with ctx.workprec(request.precision_bits):
        var_value = arb(fmpq(numerator, denominator))
        try:
            result = _eval_node(request.expression, var_value)
        except ZeroDivisionError:
            return IntervalExpressionEnclosureResult(
                status="INVALID",
                precision_bits=request.precision_bits,
                detail="Expression evaluation encountered a zero denominator.",
            )
        except ValueError as exc:
            return IntervalExpressionEnclosureResult(
                status="INVALID",
                precision_bits=request.precision_bits,
                detail=f"Expression evaluation failed: {exc}",
            )
        if not result.is_finite():
            return IntervalExpressionEnclosureResult(
                status="NONFINITE",
                precision_bits=request.precision_bits,
                detail="Arb returned a non-finite ball; no enclosure conclusion is available.",
            )
        lower_mantissa, lower_exponent = result.lower().man_exp()
        upper_mantissa, upper_exponent = result.upper().man_exp()
        exact = bool(result.is_exact())
    return IntervalExpressionEnclosureResult(
        status="ENCLOSED",
        precision_bits=request.precision_bits,
        lower=ExactDyadic(
            mantissa=format_canonical_integer(int(lower_mantissa)),
            exponent=int(lower_exponent),
        ),
        upper=ExactDyadic(
            mantissa=format_canonical_integer(int(upper_mantissa)),
            exponent=int(upper_exponent),
        ),
        relative_accuracy_bits=None if exact else int(result.rel_accuracy_bits()),
        exact=exact,
        detail="Arb ball arithmetic returned an outward-rounded enclosure with exact dyadic endpoints.",
    )
