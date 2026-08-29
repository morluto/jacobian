"""Validated real-analysis operation declarations."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.analysis._box_enclosure import BOX_EXPRESSION_ENCLOSURE_OPERATIONS
from jacobian.math.analysis._expression_enclosure import (
    IntervalExpressionEnclosureRequest,
    IntervalExpressionEnclosureResult,
)
from jacobian.math.analysis._point_enclosure import (
    ArbPointEnclosureRequest,
    ArbPointEnclosureResult,
    PointEnclosureCheckRequest,
    PointEnclosureCheckResult,
    _check_point_enclosure,
    _point_enclosure,
)
from jacobian.math.analysis._second_jet import (
    IntervalExpressionSecondJetEnclosureRequest,
    IntervalExpressionSecondJetEnclosureResult,
)
from jacobian.math.analysis.operations import expression_enclosure, second_jet_enclosure


def _expression_request(
    request: IntervalExpressionEnclosureRequest,
) -> IntervalExpressionEnclosureResult:
    return expression_enclosure(
        request.expression, request.argument, request.precision_bits
    )


def _second_jet_request(
    request: IntervalExpressionSecondJetEnclosureRequest,
) -> IntervalExpressionSecondJetEnclosureResult:
    return second_jet_enclosure(request.expression, request.box, request.precision_bits)


POINT_ENCLOSURE_OPERATIONS = (
    MathTool(
        operation_id="analysis.real_function.point_enclosure.compute",
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
        operation_id="analysis.real_function.point_enclosure.check",
        title="Check a claimed real-function point enclosure",
        description=(
            "Independently check one claimed exact-dyadic enclosure of LOG or "
            "SQRT at an exact rational point. The result is ACCEPTED, REJECTED, "
            "or NON_RESULT and retains the complete claim. LOG verification is "
            "capped at 128 exact series terms."
        ),
        request_type=PointEnclosureCheckRequest,
        result_type=PointEnclosureCheckResult,
        run=_check_point_enclosure,
        tags=(
            "analysis",
            "check",
            "enclosure",
            "exact",
            "bounded",
            "square-root",
            "sqrt",
            "logarithm",
            "log",
        ),
        examples=(
            example(
                "sqrt_zero",
                "Independently check the exact claimed enclosure sqrt(0) = 0.",
                {
                    "enclosure": {
                        "function": "SQRT",
                        "argument": {"num": "0", "den": "1"},
                        "precision_bits": 128,
                        "lower": {"mantissa": "0", "exponent": 0},
                        "upper": {"mantissa": "0", "exponent": 0},
                    },
                },
            ),
        ),
    ),
)

EXPRESSION_ENCLOSURE_OPERATIONS = (
    MathTool(
        operation_id="interval.compute.enclosure",
        title="Enclose a univariate expression at a rational point",
        description="Use Arb ball arithmetic to enclose a bounded expression tree over one variable at one exact rational point.",
        request_type=IntervalExpressionEnclosureRequest,
        result_type=IntervalExpressionEnclosureResult,
        run=_expression_request,
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

SECOND_JET_ENCLOSURE_OPERATIONS = (
    MathTool(
        operation_id="interval.expression.second_jet_enclosure.compute",
        title="Enclose an expression, gradient, and Hessian over a rational box",
        description=(
            "Use pinned Arb forward automatic differentiation to enclose one bounded "
            "named-variable elementary expression, every first partial, and its "
            "symmetric Hessian over a complete ordered rational box. The fixed "
            "envelope admits at most 8 variables, 64 nodes, depth 16, 128-digit "
            "rationals, absolute power exponents up to 64, 4,096-bit Arb precision, "
            "and 16,384 forward-jet scalar arithmetic units charged by dimension."
        ),
        request_type=IntervalExpressionSecondJetEnclosureRequest,
        result_type=IntervalExpressionSecondJetEnclosureResult,
        run=_second_jet_request,
        tags=(
            "analysis",
            "interval",
            "expression",
            "box",
            "gradient",
            "hessian",
            "automatic-differentiation",
            "arb",
            "validated",
            "bounded",
        ),
        examples=(
            example(
                "quadratic_unit_box",
                "Enclose x^2 + y^2 and all derivatives over the unit square.",
                {
                    "expression": {
                        "op": "add",
                        "children": [
                            {
                                "op": "pow",
                                "exponent": 2,
                                "children": [{"op": "var", "variable": "x"}],
                            },
                            {
                                "op": "pow",
                                "exponent": 2,
                                "children": [{"op": "var", "variable": "y"}],
                            },
                        ],
                    },
                    "box": {
                        "variables": ["x", "y"],
                        "intervals": [
                            {
                                "lower": {"num": "0", "den": "1"},
                                "upper": {"num": "1", "den": "1"},
                            },
                            {
                                "lower": {"num": "0", "den": "1"},
                                "upper": {"num": "1", "den": "1"},
                            },
                        ],
                    },
                    "precision_bits": 128,
                },
            ),
        ),
    ),
)

TOOLS: MathTools = (
    *POINT_ENCLOSURE_OPERATIONS,
    *EXPRESSION_ENCLOSURE_OPERATIONS,
    *BOX_EXPRESSION_ENCLOSURE_OPERATIONS,
    *SECOND_JET_ENCLOSURE_OPERATIONS,
)

__all__ = ["TOOLS"]
