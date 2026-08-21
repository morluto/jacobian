"""Validated real-function operations backed by Arb."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from jacobian.canonical import format_canonical_integer
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.analysis._models import (
    MAX_DYADIC_EXPONENT,
    ArbPointEnclosureRequest,
    ArbPointEnclosureResult,
    ExactDyadic,
    IntervalExpressionEnclosureRequest,
    IntervalExpressionEnclosureResult,
    IntervalExpressionNode,
)


class _EvaluationFailure(StrEnum):
    DOMAIN_ERROR = "DOMAIN_ERROR"
    NONFINITE = "NONFINITE"
    PRECISION_INSUFFICIENT = "PRECISION_INSUFFICIENT"


def _dyadic_endpoints(
    lower_mantissa: Any,
    lower_exponent: Any,
    upper_mantissa: Any,
    upper_exponent: Any,
) -> tuple[ExactDyadic, ExactDyadic] | None:
    """Serialize Arb endpoints only when their exponents fit the wire contract."""

    if (
        abs(lower_exponent) > MAX_DYADIC_EXPONENT
        or abs(upper_exponent) > MAX_DYADIC_EXPONENT
    ):
        return None
    return (
        ExactDyadic(
            mantissa=format_canonical_integer(int(lower_mantissa)),
            exponent=int(lower_exponent),
        ),
        ExactDyadic(
            mantissa=format_canonical_integer(int(upper_mantissa)),
            exponent=int(upper_exponent),
        ),
    )


def _apply_binary(node: IntervalExpressionNode, left: Any, right: Any) -> Any:
    if node.op == "add":
        return left + right
    if node.op == "sub":
        return left - right
    if node.op == "mul":
        return left * right
    if right.contains(0):
        return (
            _EvaluationFailure.DOMAIN_ERROR
            if right.is_exact()
            else _EvaluationFailure.PRECISION_INSUFFICIENT
        )
    return left / right


def _apply_unary(node: IntervalExpressionNode, value: Any) -> Any:
    if node.op == "neg":
        return -value
    if node.op == "pow":
        assert node.exponent is not None
        if node.exponent < 0 and value.contains(0):
            return (
                _EvaluationFailure.DOMAIN_ERROR
                if value.is_exact()
                else _EvaluationFailure.PRECISION_INSUFFICIENT
            )
        return value**node.exponent
    if node.op == "log" and not value > 0:
        return (
            _EvaluationFailure.DOMAIN_ERROR
            if value <= 0
            else _EvaluationFailure.PRECISION_INSUFFICIENT
        )
    if node.op == "sqrt" and not value >= 0:
        return (
            _EvaluationFailure.DOMAIN_ERROR
            if value < 0
            else _EvaluationFailure.PRECISION_INSUFFICIENT
        )
    return getattr(value, node.op)()


def _evaluate_expression(node: IntervalExpressionNode, variable: Any) -> Any:
    from flint import arb, fmpq

    if node.op == "const":
        assert node.value is not None
        numerator, denominator = node.value.as_integer_ratio()
        return arb(fmpq(numerator, denominator))
    if node.op == "var":
        return variable
    values = tuple(_evaluate_expression(child, variable) for child in node.children)
    failure = next(
        (value for value in values if isinstance(value, _EvaluationFailure)), None
    )
    if failure is not None:
        return failure
    if any(not value.is_finite() for value in values):
        return _EvaluationFailure.NONFINITE
    if len(values) == 2:
        return _apply_binary(node, values[0], values[1])
    return _apply_unary(node, values[0])


def _expression_enclosure(
    request: IntervalExpressionEnclosureRequest,
) -> IntervalExpressionEnclosureResult:
    from flint import arb, ctx, fmpq

    numerator, denominator = request.argument.as_integer_ratio()
    with ctx.workprec(request.precision_bits):
        result = _evaluate_expression(
            request.expression, arb(fmpq(numerator, denominator))
        )
        if isinstance(result, _EvaluationFailure):
            return IntervalExpressionEnclosureResult(
                status=result.value,
                precision_bits=request.precision_bits,
                detail=(
                    "The expression is outside its real domain at the supplied argument."
                    if result is _EvaluationFailure.DOMAIN_ERROR
                    else (
                        "An intermediate Arb value was non-finite."
                        if result is _EvaluationFailure.NONFINITE
                        else "The requested precision cannot determine a denominator or domain boundary."
                    )
                ),
            )
        if not result.is_finite():
            return IntervalExpressionEnclosureResult(
                status="NONFINITE",
                precision_bits=request.precision_bits,
                detail="Arb returned a non-finite ball.",
            )
        lower_mantissa, lower_exponent = result.lower().man_exp()
        upper_mantissa, upper_exponent = result.upper().man_exp()
        exact = bool(result.is_exact())
        endpoints = _dyadic_endpoints(
            lower_mantissa, lower_exponent, upper_mantissa, upper_exponent
        )
    if endpoints is None:
        return IntervalExpressionEnclosureResult(
            status="OUTPUT_MAGNITUDE_EXCEEDED",
            precision_bits=request.precision_bits,
            detail="Arb produced finite endpoints outside the interoperable dyadic exponent range.",
        )
    return IntervalExpressionEnclosureResult(
        status="ENCLOSED",
        precision_bits=request.precision_bits,
        lower=endpoints[0],
        upper=endpoints[1],
        relative_accuracy_bits=None if exact else int(result.rel_accuracy_bits()),
        exact=exact,
        detail="Arb returned an outward-rounded enclosure with exact dyadic endpoints.",
    )


def _point_enclosure(
    request: ArbPointEnclosureRequest,
) -> ArbPointEnclosureResult:
    from flint import arb, ctx, fmpq

    numerator, denominator = request.argument.as_integer_ratio()
    with ctx.workprec(request.precision_bits):
        value = arb(fmpq(numerator, denominator))
        result = getattr(value, request.function.value.lower())()
        if not result.is_finite():
            return ArbPointEnclosureResult(
                status="NONFINITE",
                function=request.function,
                argument=request.argument,
                precision_bits=request.precision_bits,
                detail="Arb returned a non-finite ball; no enclosure conclusion is available.",
            )
        lower_mantissa, lower_exponent = result.lower().man_exp()
        upper_mantissa, upper_exponent = result.upper().man_exp()
        exact = bool(result.is_exact())
        endpoints = _dyadic_endpoints(
            lower_mantissa, lower_exponent, upper_mantissa, upper_exponent
        )
    if endpoints is None:
        return ArbPointEnclosureResult(
            status="OUTPUT_MAGNITUDE_EXCEEDED",
            function=request.function,
            argument=request.argument,
            precision_bits=request.precision_bits,
            detail="Arb produced finite endpoints outside the interoperable dyadic exponent range.",
        )
    return ArbPointEnclosureResult(
        status="ENCLOSED",
        function=request.function,
        argument=request.argument,
        precision_bits=request.precision_bits,
        lower=endpoints[0],
        upper=endpoints[1],
        relative_accuracy_bits=None if exact else int(result.rel_accuracy_bits()),
        exact=exact,
        detail="Pinned Arb ball arithmetic returned an outward-rounded enclosure with exact dyadic endpoints.",
    )


POINT_ENCLOSURE_OPERATIONS = (
    MathTool(
        operation_id="analysis.real_function.point_enclosure.compute",
        version="1",
        title="Enclose a real function at a rational point",
        description=(
            "Use pinned Arb ball arithmetic to enclose one supported real "
            "function (square root, logarithm, exponential, sine, or cosine) "
            "at one exact rational point."
        ),
        request_type=ArbPointEnclosureRequest,
        result_type=ArbPointEnclosureResult,
        run=_point_enclosure,
        tags=(
            "analysis",
            "validated",
            "arb",
            "enclosure",
            "bounded",
            "square-root",
            "sqrt",
            "logarithm",
            "log",
            "exponential",
            "exp",
            "sine",
            "sin",
            "cosine",
            "cos",
        ),
        examples=(
            example(
                "sqrt_zero",
                "Enclose sqrt(0) at 32-bit precision.",
                {
                    "function": "SQRT",
                    "argument": {"num": "0", "den": "1"},
                    "precision_bits": 32,
                },
            ),
        ),
    ),
)

EXPRESSION_ENCLOSURE_OPERATIONS = (
    MathTool(
        operation_id="interval.compute.enclosure",
        version="1",
        title="Enclose a univariate expression at a rational point",
        description="Use Arb ball arithmetic to enclose a bounded expression tree over one variable at one exact rational point.",
        request_type=IntervalExpressionEnclosureRequest,
        result_type=IntervalExpressionEnclosureResult,
        run=_expression_enclosure,
        tags=("analysis", "interval", "expression", "arb", "exact", "bounded"),
        examples=(
            example(
                "log_137_80",
                "Enclose log(137/80); the expression must use the bounded typed tree grammar.",
                {
                    "expression": {
                        "op": "log",
                        "children": [
                            {"op": "const", "value": {"num": "137", "den": "80"}}
                        ],
                    },
                    "argument": {"num": "0", "den": "1"},
                    "precision_bits": 128,
                },
            ),
        ),
    ),
)

__all__ = ["EXPRESSION_ENCLOSURE_OPERATIONS", "POINT_ENCLOSURE_OPERATIONS"]
