"""Validated real-function operations backed by Arb."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from jacobian.canonical import format_canonical_integer
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.analysis._models import (
    MAX_BOX_PREFLIGHT_TEMPORARY_BITS,
    MAX_DYADIC_EXPONENT,
    ArbPointEnclosureRequest,
    ArbPointEnclosureResult,
    ExactDyadic,
    IntervalExpressionBoxEnclosureRequest,
    IntervalExpressionBoxEnclosureResult,
    IntervalExpressionBoxEnclosureStatus,
    IntervalExpressionDomainFailure,
    IntervalExpressionEnclosureRequest,
    IntervalExpressionEnclosureResult,
    IntervalExpressionNode,
    RationalClosedInterval,
    _preflight_box_expression,
    _rational_box_bounds,
)


class _EvaluationFailure(StrEnum):
    DOMAIN_ERROR = "DOMAIN_ERROR"
    NONFINITE = "NONFINITE"
    PRECISION_INSUFFICIENT = "PRECISION_INSUFFICIENT"


class _BoxEvaluationFailure(StrEnum):
    BACKEND_ERROR = "BACKEND_ERROR"


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


def _normalize_dyadic_pair(mantissa: int, exponent: int) -> tuple[int, int]:
    if mantissa == 0:
        return 0, 0
    while mantissa % 2 == 0:
        mantissa //= 2
        exponent += 1
    return mantissa, exponent


def _add_dyadic_pairs(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    common_exponent = min(left[1], right[1])
    return _normalize_dyadic_pair(
        (left[0] << (left[1] - common_exponent))
        + (right[0] << (right[1] - common_exponent)),
        common_exponent,
    )


def _negate_dyadic_pair(value: tuple[int, int]) -> tuple[int, int]:
    return -value[0], value[1]


def _halve_dyadic_pair(value: tuple[int, int]) -> tuple[int, int]:
    return _normalize_dyadic_pair(value[0], value[1] - 1)


def _exact_arb_dyadic_pair(value: Any) -> tuple[int, int]:
    mantissa, exponent = value.man_exp()
    return _normalize_dyadic_pair(int(mantissa), int(exponent))


def _arb_source_interval(interval: RationalClosedInterval) -> Any:
    """Build one Arb ball that contains the exact rational source interval.

    Arb radii have a fixed implementation precision.  Anchoring a one-sided
    interval at its endpoint nearest zero preserves a proved sign even when
    the radius is rounded upward by python-flint's public constructor.
    """

    from flint import arb, fmpq

    lower_ratio = interval.lower.as_integer_ratio()
    upper_ratio = interval.upper.as_integer_ratio()
    if lower_ratio == upper_ratio:
        return arb(fmpq(*lower_ratio))

    lower = _exact_arb_dyadic_pair(arb(fmpq(*lower_ratio)).lower())
    upper = _exact_arb_dyadic_pair(arb(fmpq(*upper_ratio)).upper())
    half_width = _halve_dyadic_pair(
        _add_dyadic_pairs(upper, _negate_dyadic_pair(lower))
    )
    actual_radius = _exact_arb_dyadic_pair(arb((0, 0), half_width).upper())
    if interval.lower.as_fraction() >= 0:
        midpoint = _add_dyadic_pairs(lower, actual_radius)
    elif interval.upper.as_fraction() <= 0:
        midpoint = _add_dyadic_pairs(upper, _negate_dyadic_pair(actual_radius))
    else:
        midpoint = _halve_dyadic_pair(_add_dyadic_pairs(lower, upper))
    return arb(midpoint, half_width)


def _box_domain_failure(
    node: IntervalExpressionNode, path: tuple[int, ...]
) -> IntervalExpressionDomainFailure:
    if node.op == "div":
        return IntervalExpressionDomainFailure(
            node_path=path,
            operation="div",
            reason="DENOMINATOR_CONTAINS_ZERO",
        )
    if node.op == "pow":
        return IntervalExpressionDomainFailure(
            node_path=path,
            operation="pow",
            reason="NEGATIVE_POWER_BASE_CONTAINS_ZERO",
        )
    if node.op == "log":
        return IntervalExpressionDomainFailure(
            node_path=path,
            operation="log",
            reason="LOG_ARGUMENT_NOT_STRICTLY_POSITIVE",
        )
    if node.op == "sqrt":
        return IntervalExpressionDomainFailure(
            node_path=path,
            operation="sqrt",
            reason="SQRT_ARGUMENT_NOT_NONNEGATIVE",
        )
    raise AssertionError("only real-domain operations can produce a domain failure")


def _evaluate_box_unary(
    node: IntervalExpressionNode, value: Any, path: tuple[int, ...]
) -> Any:
    if node.op == "neg":
        return -value
    if node.op == "pow":
        assert node.exponent is not None
        if node.exponent < 0 and value.contains(0):
            return _box_domain_failure(node, path)
        return value**node.exponent
    if node.op == "log":
        if not value > 0:
            return _box_domain_failure(node, path)
        return value.log()
    if node.op == "sqrt":
        if not value >= 0:
            return _box_domain_failure(node, path)
        return value.sqrt()
    if node.op in ("exp", "sin", "cos"):
        return getattr(value, node.op)()
    raise AssertionError(f"unsupported unary expression operation: {node.op}")


def _evaluate_box_binary(
    node: IntervalExpressionNode,
    left: Any,
    right: Any,
    path: tuple[int, ...],
) -> Any:
    if node.op == "add":
        return left + right
    if node.op == "sub":
        return left - right
    if node.op == "mul":
        return left * right
    if node.op == "div":
        if right.contains(0):
            return _box_domain_failure(node, path)
        return left / right
    raise AssertionError(f"unsupported binary expression operation: {node.op}")


def _evaluate_box_expression(
    node: IntervalExpressionNode,
    variables: dict[str, Any],
    path: tuple[int, ...] = (),
) -> Any:
    from flint import arb, fmpq

    if node.op == "const":
        assert node.value is not None
        return arb(fmpq(*node.value.as_integer_ratio()))
    if node.op == "var":
        assert node.variable is not None
        return variables[node.variable]

    values: list[Any] = []
    for index, child in enumerate(node.children):
        value = _evaluate_box_expression(child, variables, (*path, index))
        if isinstance(value, (IntervalExpressionDomainFailure, _BoxEvaluationFailure)):
            return value
        if not value.is_finite():
            return _BoxEvaluationFailure.BACKEND_ERROR
        values.append(value)

    left = values[0]
    if len(values) == 1:
        return _evaluate_box_unary(node, left, path)

    right = values[1]
    return _evaluate_box_binary(node, left, right, path)


def _constructed_box_result(
    request: IntervalExpressionBoxEnclosureRequest,
    *,
    status: IntervalExpressionBoxEnclosureStatus,
    detail: str,
    lower: ExactDyadic | None = None,
    upper: ExactDyadic | None = None,
    domain_failure: IntervalExpressionDomainFailure | None = None,
) -> IntervalExpressionBoxEnclosureResult:
    """Build the producer result without paying a second backend replay."""

    return IntervalExpressionBoxEnclosureResult.model_construct(
        expression=request.expression,
        box=request.box,
        precision_bits=request.precision_bits,
        status=status,
        lower=lower,
        upper=upper,
        domain_failure=domain_failure,
        method="ARB_NATURAL_INTERVAL_EXTENSION",
        detail=detail,
    )


def _box_expression_enclosure(
    request: IntervalExpressionBoxEnclosureRequest,
) -> IntervalExpressionBoxEnclosureResult:
    preflight = _preflight_box_expression(
        request.expression, _rational_box_bounds(request.box)
    )
    if isinstance(preflight, IntervalExpressionDomainFailure):
        return _constructed_box_result(
            request,
            status="DOMAIN_UNPROVEN",
            domain_failure=preflight,
            detail=(
                "The exact admission interval extension could not establish the "
                "real domain at the reported source node."
            ),
        )

    from flint import ctx

    try:
        with ctx.workprec(request.precision_bits):
            variables = {
                variable: _arb_source_interval(interval)
                for variable, interval in zip(
                    request.box.variables, request.box.intervals, strict=True
                )
            }
            result = _evaluate_box_expression(request.expression, variables)
            if isinstance(result, IntervalExpressionDomainFailure):
                return _constructed_box_result(
                    request,
                    status="DOMAIN_UNPROVEN",
                    domain_failure=result,
                    detail=(
                        "Arb's natural interval extension could not establish the "
                        "real domain at the reported source node."
                    ),
                )
            if isinstance(result, _BoxEvaluationFailure) or not result.is_finite():
                return _constructed_box_result(
                    request,
                    status="BACKEND_ERROR",
                    detail=(
                        "Pinned Arb returned no finite enclosure within the admitted "
                        "fixed-precision envelope."
                    ),
                )
            lower_mantissa, lower_exponent = result.lower().man_exp()
            upper_mantissa, upper_exponent = result.upper().man_exp()
            endpoints = _dyadic_endpoints(
                lower_mantissa, lower_exponent, upper_mantissa, upper_exponent
            )
    except (OverflowError, ValueError):
        return _constructed_box_result(
            request,
            status="BACKEND_ERROR",
            detail=(
                "Pinned Arb rejected an admitted bounded computation; no enclosure "
                "conclusion is available."
            ),
        )

    if endpoints is None:
        return _constructed_box_result(
            request,
            status="BACKEND_ERROR",
            detail=(
                "Pinned Arb produced endpoints outside the admitted dyadic wire "
                "envelope; no enclosure conclusion is available."
            ),
        )
    return _constructed_box_result(
        request,
        status="ENCLOSED",
        lower=endpoints[0],
        upper=endpoints[1],
        detail=(
            "Pinned Arb natural interval arithmetic returned an outward-rounded "
            "enclosure with exact dyadic endpoints."
        ),
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

BOX_EXPRESSION_ENCLOSURE_OPERATIONS = (
    MathTool(
        operation_id="interval.expression.box_enclosure.compute",
        version="1",
        title="Enclose an elementary expression over a rational box",
        description=(
            "Use pinned Arb natural interval arithmetic to enclose one bounded "
            "named-variable elementary expression over a complete ordered rational "
            "box, or report the first source node whose real domain is unproved. "
            "The fixed envelope admits at most 8 variables, 64 nodes, depth 16, "
            "128-digit rationals, absolute power exponents up to 64, 4,096-bit "
            "Arb precision, 8,192-bit retained exact admission bounds, and "
            f"{MAX_BOX_PREFLIGHT_TEMPORARY_BITS:,}-bit Fraction temporaries."
        ),
        request_type=IntervalExpressionBoxEnclosureRequest,
        result_type=IntervalExpressionBoxEnclosureResult,
        run=_box_expression_enclosure,
        tags=(
            "analysis",
            "interval",
            "expression",
            "box",
            "multivariate",
            "arb",
            "validated",
            "bounded",
        ),
        examples=(
            example(
                "exp_unit_interval",
                "Enclose exp(x) over the exact rational interval 0 <= x <= 1.",
                {
                    "expression": {
                        "op": "exp",
                        "children": [{"op": "var", "variable": "x"}],
                    },
                    "box": {
                        "variables": ["x"],
                        "intervals": [
                            {
                                "lower": {"num": "0", "den": "1"},
                                "upper": {"num": "1", "den": "1"},
                            }
                        ],
                    },
                    "precision_bits": 128,
                },
            ),
        ),
    ),
)

__all__ = [
    "BOX_EXPRESSION_ENCLOSURE_OPERATIONS",
    "EXPRESSION_ENCLOSURE_OPERATIONS",
    "POINT_ENCLOSURE_OPERATIONS",
]
