"""Validated real-function enclosure operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.analysis._arb import arb_source_interval, dyadic_endpoints
from jacobian.math.analysis._expression_enclosure import (
    IntervalExpressionEnclosureRequest,
    IntervalExpressionEnclosureResult,
)
from jacobian.math.analysis._models import (
    DyadicClosedInterval,
    IntervalExpressionDomainFailure,
    IntervalExpressionNode,
    _rational_box_bounds,
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
    FirstPartialEnclosure,
    HessianEntryEnclosure,
    IntervalExpressionSecondJetEnclosureRequest,
    IntervalExpressionSecondJetEnclosureResult,
    _preflight_second_jet_expression,
)


class _EvaluationFailure(StrEnum):
    DOMAIN_ERROR = "DOMAIN_ERROR"
    NONFINITE = "NONFINITE"
    PRECISION_INSUFFICIENT = "PRECISION_INSUFFICIENT"


class _SecondJetEvaluationFailure(StrEnum):
    BACKEND_ERROR = "BACKEND_ERROR"


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
        endpoints = dyadic_endpoints(
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


@dataclass(frozen=True, slots=True)
class _SecondJet:
    """One private Arb value with all forward derivatives through order two."""

    value: Any
    gradient: tuple[Any, ...]
    hessian: tuple[tuple[Any, ...], ...]


def _second_jet_is_finite(jet: _SecondJet) -> bool:
    return jet.value.is_finite() and all(
        value.is_finite()
        for value in (*jet.gradient, *(entry for row in jet.hessian for entry in row))
    )


def _constant_second_jet(value: Any, dimension: int) -> _SecondJet:
    from flint import arb

    zero = arb(0)
    return _SecondJet(
        value=value,
        gradient=(zero,) * dimension,
        hessian=tuple((zero,) * dimension for _ in range(dimension)),
    )


def _compose_second_jet_unary(
    value: Any, first: Any, second: Any, child: _SecondJet
) -> _SecondJet:
    return _SecondJet(
        value=value,
        gradient=tuple(first * partial for partial in child.gradient),
        hessian=tuple(
            tuple(
                first * child.hessian[row][column]
                + second * child.gradient[row] * child.gradient[column]
                for column in range(len(child.gradient))
            )
            for row in range(len(child.gradient))
        ),
    )


def _add_second_jets(left: _SecondJet, right: _SecondJet) -> _SecondJet:
    return _SecondJet(
        value=left.value + right.value,
        gradient=tuple(
            left_partial + right_partial
            for left_partial, right_partial in zip(
                left.gradient, right.gradient, strict=True
            )
        ),
        hessian=tuple(
            tuple(
                left_entry + right_entry
                for left_entry, right_entry in zip(left_row, right_row, strict=True)
            )
            for left_row, right_row in zip(left.hessian, right.hessian, strict=True)
        ),
    )


def _negate_second_jet(child: _SecondJet) -> _SecondJet:
    return _SecondJet(
        value=-child.value,
        gradient=tuple(-partial for partial in child.gradient),
        hessian=tuple(tuple(-entry for entry in row) for row in child.hessian),
    )


def _multiply_second_jets(left: _SecondJet, right: _SecondJet) -> _SecondJet:
    return _SecondJet(
        value=left.value * right.value,
        gradient=tuple(
            left_partial * right.value + left.value * right_partial
            for left_partial, right_partial in zip(
                left.gradient, right.gradient, strict=True
            )
        ),
        hessian=tuple(
            tuple(
                left.hessian[row][column] * right.value
                + left.gradient[row] * right.gradient[column]
                + left.gradient[column] * right.gradient[row]
                + left.value * right.hessian[row][column]
                for column in range(len(left.gradient))
            )
            for row in range(len(left.gradient))
        ),
    )


def _power_second_jet(child: _SecondJet, exponent: int) -> _SecondJet:
    """Apply the exact integer-power chain rules to one second-order jet."""

    from flint import arb

    value = child.value**exponent
    if exponent == 1:
        return _compose_second_jet_unary(value, arb(1), arb(0), child)
    first = exponent * child.value ** (exponent - 1)
    if exponent == -1:
        second = 2 * child.value**-3
    else:
        second = exponent * (exponent - 1) * child.value ** (exponent - 2)
    return _compose_second_jet_unary(value, first, second, child)


def _unary_second_jet(
    node: IntervalExpressionNode, child: _SecondJet
) -> _SecondJet | _SecondJetEvaluationFailure:
    if node.op == "neg":
        return _negate_second_jet(child)
    if node.op == "pow":
        assert node.exponent is not None
        if node.exponent < 0 and child.value.contains(0):
            return _SecondJetEvaluationFailure.BACKEND_ERROR
        return _power_second_jet(child, node.exponent)
    if node.op == "exp":
        value = child.value.exp()
        return _compose_second_jet_unary(value, value, value, child)
    if node.op == "log":
        if not child.value > 0:
            return _SecondJetEvaluationFailure.BACKEND_ERROR
        value = child.value.log()
        return _compose_second_jet_unary(
            value, 1 / child.value, -1 / child.value**2, child
        )
    if node.op == "sqrt":
        if not child.value > 0:
            return _SecondJetEvaluationFailure.BACKEND_ERROR
        value = child.value.sqrt()
        return _compose_second_jet_unary(
            value,
            1 / (2 * value),
            -1 / (4 * child.value * value),
            child,
        )
    if node.op == "sin":
        value = child.value.sin()
        return _compose_second_jet_unary(value, child.value.cos(), -value, child)
    if node.op == "cos":
        value = child.value.cos()
        return _compose_second_jet_unary(value, -child.value.sin(), -value, child)
    raise AssertionError(f"unsupported unary expression operation: {node.op}")


def _evaluate_second_jet(
    node: IntervalExpressionNode, variables: dict[str, Any], dimension: int
) -> _SecondJet | _SecondJetEvaluationFailure:
    from flint import arb, fmpq

    if node.op == "const":
        assert node.value is not None
        return _constant_second_jet(
            arb(fmpq(*node.value.as_integer_ratio())), dimension
        )
    if node.op == "var":
        assert node.variable is not None
        zero = arb(0)
        one = arb(1)
        index = tuple(variables).index(node.variable)
        return _SecondJet(
            value=variables[node.variable],
            gradient=tuple(
                one if coordinate == index else zero for coordinate in range(dimension)
            ),
            hessian=tuple((zero,) * dimension for _ in range(dimension)),
        )

    children: list[_SecondJet] = []
    for child_node in node.children:
        child = _evaluate_second_jet(child_node, variables, dimension)
        if isinstance(child, _SecondJetEvaluationFailure) or not _second_jet_is_finite(
            child
        ):
            return _SecondJetEvaluationFailure.BACKEND_ERROR
        children.append(child)
    left = children[0]
    if len(children) == 1:
        result = _unary_second_jet(node, left)
    else:
        right = children[1]
        assert isinstance(right, _SecondJet)
        if node.op == "add":
            result = _add_second_jets(left, right)
        elif node.op == "sub":
            result = _add_second_jets(left, _negate_second_jet(right))
        elif node.op == "mul":
            result = _multiply_second_jets(left, right)
        elif node.op == "div":
            if right.value.contains(0):
                return _SecondJetEvaluationFailure.BACKEND_ERROR
            result = _multiply_second_jets(left, _power_second_jet(right, -1))
        else:
            raise AssertionError(f"unsupported binary expression operation: {node.op}")
    if isinstance(result, _SecondJetEvaluationFailure) or not _second_jet_is_finite(
        result
    ):
        return _SecondJetEvaluationFailure.BACKEND_ERROR
    return result


def _dyadic_closed_interval(value: Any) -> DyadicClosedInterval | None:
    lower_mantissa, lower_exponent = value.lower().man_exp()
    upper_mantissa, upper_exponent = value.upper().man_exp()
    endpoints = dyadic_endpoints(
        lower_mantissa, lower_exponent, upper_mantissa, upper_exponent
    )
    if endpoints is None:
        return None
    return DyadicClosedInterval(lower=endpoints[0], upper=endpoints[1])


def _second_jet_enclosure(
    request: IntervalExpressionSecondJetEnclosureRequest,
) -> IntervalExpressionSecondJetEnclosureResult:
    preflight = _preflight_second_jet_expression(
        request.expression, _rational_box_bounds(request.box)
    )
    if isinstance(preflight, IntervalExpressionDomainFailure):
        return IntervalExpressionSecondJetEnclosureResult._from_kernel(
            request,
            status="DOMAIN_UNPROVEN",
            domain_failure=preflight,
            detail=(
                "The exact admission interval extension could not establish twice "
                "differentiability at the reported source node."
            ),
        )

    try:
        from flint import ctx

        with ctx.workprec(request.precision_bits):
            variables = {
                variable: arb_source_interval(interval)
                for variable, interval in zip(
                    request.box.variables, request.box.intervals, strict=True
                )
            }
            jet = _evaluate_second_jet(
                request.expression, variables, len(request.box.variables)
            )
            if isinstance(jet, _SecondJetEvaluationFailure):
                return IntervalExpressionSecondJetEnclosureResult._from_kernel(
                    request,
                    status="BACKEND_ERROR",
                    detail=(
                        "Pinned Arb returned no finite second-order enclosure within "
                        "the admitted fixed-precision envelope."
                    ),
                )
            value = _dyadic_closed_interval(jet.value)
            gradient_intervals = tuple(
                _dyadic_closed_interval(entry) for entry in jet.gradient
            )
            hessian_intervals = tuple(
                tuple(_dyadic_closed_interval(entry) for entry in row)
                for row in jet.hessian
            )
    except (OverflowError, ValueError, ZeroDivisionError):
        return IntervalExpressionSecondJetEnclosureResult._from_kernel(
            request,
            status="BACKEND_ERROR",
            detail=(
                "Pinned Arb rejected an admitted bounded second-order computation; "
                "no enclosure conclusion is available."
            ),
        )

    if (
        value is None
        or any(entry is None for entry in gradient_intervals)
        or any(entry is None for row in hessian_intervals for entry in row)
    ):
        return IntervalExpressionSecondJetEnclosureResult._from_kernel(
            request,
            status="BACKEND_ERROR",
            detail=(
                "Pinned Arb produced endpoints outside the admitted dyadic wire "
                "envelope; no enclosure conclusion is available."
            ),
        )

    gradient = tuple(
        FirstPartialEnclosure(variable=variable, enclosure=entry)
        for variable, entry in zip(
            request.box.variables, gradient_intervals, strict=True
        )
        if entry is not None
    )
    hessian_entries: list[HessianEntryEnclosure] = []
    for first_index, first in enumerate(request.box.variables):
        for second_index, second in enumerate(
            request.box.variables[first_index:], first_index
        ):
            enclosure = hessian_intervals[first_index][second_index]
            assert enclosure is not None
            hessian_entries.append(
                HessianEntryEnclosure(
                    first_variable=first,
                    second_variable=second,
                    enclosure=enclosure,
                )
            )
    return IntervalExpressionSecondJetEnclosureResult._from_kernel(
        request,
        status="ENCLOSED",
        value=value,
        gradient=gradient,
        hessian=tuple(hessian_entries),
        detail=(
            "Pinned Arb forward automatic differentiation returned outward-rounded "
            "value, gradient, and symmetric Hessian enclosures with exact dyadic endpoints."
        ),
    )


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
        run=_second_jet_enclosure,
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

__all__ = [
    "EXPRESSION_ENCLOSURE_OPERATIONS",
    "POINT_ENCLOSURE_OPERATIONS",
    "SECOND_JET_ENCLOSURE_OPERATIONS",
]
