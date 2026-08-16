"""Validated real-function operations backed by Arb."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.contracts.validated_analysis import (
    ArbPointEnclosureRequest,
    ArbPointEnclosureResult,
    ExactDyadic,
    IntervalExpressionEnclosureRequest,
    IntervalExpressionEnclosureResult,
)
from jacobian.domains._examples import example
from jacobian.domains.analysis.expression_enclosure import compute_expression_enclosure
from jacobian.math_tools import MathTool


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
    return ArbPointEnclosureResult(
        status="ENCLOSED",
        function=request.function,
        argument=request.argument,
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
        detail="Pinned Arb ball arithmetic returned an outward-rounded enclosure with exact dyadic endpoints.",
    )


ANALYSIS_OPERATIONS: tuple[MathTool, ...] = (
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
    MathTool(
        operation_id="interval.compute.enclosure",
        version="1",
        title="Enclose a user-defined expression at a rational point",
        description=(
            "Use Arb ball arithmetic to enclose a user-supplied univariate "
            "expression tree at one exact rational point with declared "
            "precision.  Supports +, -, *, /, ^n, exp, log, sqrt, sin, cos."
        ),
        request_type=IntervalExpressionEnclosureRequest,
        result_type=IntervalExpressionEnclosureResult,
        run=compute_expression_enclosure,
        tags=(
            "analysis",
            "validated",
            "arb",
            "enclosure",
            "bounded",
            "interval",
            "expression",
            "transcendental",
        ),
        examples=(
            example(
                "log_137_80",
                "Enclose log(137/80) at 128-bit precision.",
                {
                    "expression": {
                        "op": "log",
                        "children": [
                            {"op": "const", "value": {"num": "137", "den": "80"}}
                        ],
                    },
                    "argument": {"num": "1", "den": "1"},
                    "precision_bits": 128,
                },
            ),
        ),
    ),
)

# Backward-compatible alias
POINT_ENCLOSURE_OPERATIONS = ANALYSIS_OPERATIONS

__all__ = ["POINT_ENCLOSURE_OPERATIONS", "ANALYSIS_OPERATIONS"]
