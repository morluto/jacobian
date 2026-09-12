"""Rigorous endpoint-logarithmic improper-integral enclosures."""

from __future__ import annotations

from fractions import Fraction
from math import factorial
from time import monotonic
from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    BackendFailureReason,
    OperationBackendError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
    OperationResourceAdmissionError,
)
from jacobian.math.analysis._box_enclosure import _preflight_box_expression
from jacobian.math.analysis._definite_integral_enclosure import (
    DefiniteIntegralDomainUnproven,
    DefiniteIntegralEnclosureRequest,
    DefiniteIntegralEnclosureResult,
    _compare_nonnegative_fraction_to_dyadic,
    _compute_definite_integral_enclosure,
)
from jacobian.math.analysis._models import (
    MAX_DYADIC_EXPONENT,
    ExactDyadic,
    IntervalExpressionDomainFailure,
    IntervalExpressionNode,
    IntervalExpressionOp,
    RationalIntervalBox,
    _bounded_expression_nodes,
    _rational_box_bounds,
    _RationalBounds,
)
from jacobian.math.analysis.intervals import ClosedRationalInterval

MAX_ENDPOINT_LOG_POWER = 12
MAX_IMPROPER_TRUNCATION = 64
MAX_IMPROPER_SMOOTH_NODES = 16


class EndpointLogImproperIntegralRequest(StrictModel):
    """Integrate a closed-box-safe smooth factor times normalized endpoint logs."""

    smooth_expression: IntervalExpressionNode
    interval: ClosedRationalInterval
    variable: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
    left_log_power: StrictInt = Field(default=0, ge=0, le=MAX_ENDPOINT_LOG_POWER)
    right_log_power: StrictInt = Field(default=0, ge=0, le=MAX_ENDPOINT_LOG_POWER)
    target_width: ExactDyadic
    precision_bits: StrictInt = Field(default=256, ge=32, le=4_096)
    max_leaves: StrictInt = Field(default=128, ge=1, le=1_024)
    wall_seconds: StrictInt = Field(default=30, ge=1, le=120)

    @model_validator(mode="after")
    def require_source(self) -> Self:
        if self.interval.lower.as_fraction() >= self.interval.upper.as_fraction():
            raise ValueError("improper integration requires a nonempty interval")
        nodes = _bounded_expression_nodes(self.smooth_expression)
        variables = {node.variable for node in nodes if node.op == "var"}
        if variables - {self.variable} or None in variables:
            raise ValueError(
                "smooth expression variables must use the integration axis"
            )
        if len(nodes) > MAX_IMPROPER_SMOOTH_NODES:
            raise ValueError(
                "improper smooth factors admit at most 16 expression nodes"
            )
        if self.target_width.mantissa <= 0:
            raise ValueError("improper integration requires a positive target width")
        if self.target_width.exponent < -MAX_DYADIC_EXPONENT + 2:
            raise ValueError("target width is too small to reserve quadrature error")
        return self


class EndpointTailEnclosure(StrictModel):
    truncation: StrictInt = Field(ge=1, le=MAX_IMPROPER_TRUNCATION)
    enclosure: ClosedRationalInterval


class EndpointLogImproperIntegralResult(StrictModel):
    source: EndpointLogImproperIntegralRequest
    left_quadrature: DefiniteIntegralEnclosureResult
    right_quadrature: DefiniteIntegralEnclosureResult
    left_tail: EndpointTailEnclosure
    right_tail: EndpointTailEnclosure
    enclosure: ClosedRationalInterval

    @model_validator(mode="after")
    def bind_structural_decomposition(self) -> Self:
        truncation = self.left_tail.truncation
        if (
            self.right_tail.truncation != truncation
            or not _symmetric_tail(self.left_tail)
            or not _symmetric_tail(self.right_tail)
            or not _quadrature_shape_matches(
                self.left_quadrature, self.source, truncation
            )
            or not _quadrature_shape_matches(
                self.right_quadrature, self.source, truncation
            )
        ):
            raise ValueError(
                "improper-integral quadratures and tails must match the source decomposition"
            )
        for quadrature in (self.left_quadrature, self.right_quadrature):
            if isinstance(quadrature.outcome, DefiniteIntegralDomainUnproven):
                raise ValueError(
                    "an improper-integral result requires concluded quadratures"
                )
        return self


def _const(value: Fraction) -> IntervalExpressionNode:
    return IntervalExpressionNode(
        op="const", value=CanonicalRational.from_fraction(value)
    )


def _unary(
    op: IntervalExpressionOp,
    child: IntervalExpressionNode,
    *,
    exponent: int | None = None,
) -> IntervalExpressionNode:
    return IntervalExpressionNode(op=op, exponent=exponent, children=(child,))


def _binary(
    op: IntervalExpressionOp,
    left: IntervalExpressionNode,
    right: IntervalExpressionNode,
) -> IntervalExpressionNode:
    return IntervalExpressionNode(op=op, children=(left, right))


def _multiply(factors: list[IntervalExpressionNode]) -> IntervalExpressionNode:
    result = factors[0]
    for factor in factors[1:]:
        result = _binary("mul", result, factor)
    return result


def _substitute(
    node: IntervalExpressionNode,
    variable: str,
    replacement: IntervalExpressionNode,
) -> IntervalExpressionNode:
    if node.op == "var" and node.variable == variable:
        return replacement
    return IntervalExpressionNode(
        op=node.op,
        value=node.value,
        variable=node.variable,
        exponent=node.exponent,
        children=tuple(
            _substitute(child, variable, replacement) for child in node.children
        ),
    )


def _transformed_expression(
    request: EndpointLogImproperIntegralRequest, *, left: bool
) -> IntervalExpressionNode:
    a = request.interval.lower.as_fraction()
    b = request.interval.upper.as_fraction()
    half_width = (b - a) / 2
    t = IntervalExpressionNode(op="var", variable="t")
    exponential = _unary("exp", _unary("neg", t))
    displacement = _binary("mul", _const(half_width), exponential)
    x = (
        _binary("add", _const(a), displacement)
        if left
        else _binary("sub", _const(b), displacement)
    )
    near = _binary("mul", _const(Fraction(1, 2)), exponential)
    far = _binary("sub", _const(Fraction(1)), near)
    normalized_left, normalized_right = (near, far) if left else (far, near)
    left_log = _unary("neg", _unary("log", normalized_left))
    right_log = _unary("neg", _unary("log", normalized_right))
    factors = [
        _substitute(request.smooth_expression, request.variable, x),
        displacement,
    ]
    if request.left_log_power:
        factors.append(_unary("pow", left_log, exponent=request.left_log_power))
    if request.right_log_power:
        factors.append(_unary("pow", right_log, exponent=request.right_log_power))
    return _multiply(factors)


def _tail_bound(
    magnitude: Fraction,
    half_width: Fraction,
    polynomial_power: int,
    decay: int,
    truncation: int,
) -> Fraction:
    if not magnitude:
        return Fraction()
    polynomial = sum(
        (
            Fraction(factorial(polynomial_power), factorial(polynomial_power - j))
            * Fraction((truncation + 1) ** (polynomial_power - j), decay ** (j + 1))
            for j in range(polynomial_power + 1)
        ),
        Fraction(),
    )
    return magnitude * half_width * polynomial * Fraction(1, 2 ** (decay * truncation))


def _tail_interval(bound: Fraction) -> ClosedRationalInterval:
    return ClosedRationalInterval(
        lower=CanonicalRational.from_fraction(-bound),
        upper=CanonicalRational.from_fraction(bound),
    )


def _smooth_magnitude(request: EndpointLogImproperIntegralRequest) -> Fraction:
    source_box = RationalIntervalBox(
        variables=(request.variable,), intervals=(request.interval,)
    )
    try:
        preflight = _preflight_box_expression(
            request.smooth_expression, _rational_box_bounds(source_box)
        )
    except ValueError as error:
        raise OperationDomainValidationError(
            location=("smooth_expression",),
            code="analysis.improper_integral.smooth_factor_intermediate_bound",
            message="smooth factor exceeds the admitted exact preflight bound",
        ) from error
    if isinstance(preflight, IntervalExpressionDomainFailure):
        raise OperationDomainValidationError(
            location=("smooth_expression",),
            code="analysis.improper_integral.smooth_factor_not_closed_box_safe",
            message=(
                "smooth factor must have a proven real domain on the closed "
                "source interval"
            ),
        )
    assert isinstance(preflight, _RationalBounds)
    return max(abs(preflight.lower), abs(preflight.upper))


def _tail_plan(
    request: EndpointLogImproperIntegralRequest,
) -> tuple[int, Fraction, Fraction]:
    magnitude = _smooth_magnitude(request)
    half_width = (
        request.interval.upper.as_fraction() - request.interval.lower.as_fraction()
    ) / 2
    for truncation in range(1, MAX_IMPROPER_TRUNCATION + 1):
        left_tail = _tail_bound(
            magnitude,
            half_width,
            request.left_log_power,
            request.right_log_power + 1,
            truncation,
        )
        right_tail = _tail_bound(
            magnitude,
            half_width,
            request.right_log_power,
            request.left_log_power + 1,
            truncation,
        )
        if (
            _compare_nonnegative_fraction_to_dyadic(
                8 * (left_tail + right_tail), request.target_width
            )
            <= 0
        ):
            return truncation, left_tail, right_tail
    raise OperationResourceAdmissionError(
        location=("target_width",),
        code="analysis.improper_integral.tail_truncation_bound",
        message=(
            "the analytic endpoint-tail bound does not meet the requested "
            "width within truncation 64"
        ),
    )


def _quarter_target(target: ExactDyadic) -> ExactDyadic:
    return ExactDyadic(mantissa=target.mantissa, exponent=target.exponent - 2)


def _quadrature_request(
    source: EndpointLogImproperIntegralRequest,
    expression: IntervalExpressionNode,
    truncation: int,
) -> DefiniteIntegralEnclosureRequest:
    return DefiniteIntegralEnclosureRequest(
        expression=expression,
        box=RationalIntervalBox(
            variables=("t",),
            intervals=(
                ClosedRationalInterval(
                    lower=CanonicalRational(num=0, den=1),
                    upper=CanonicalRational(num=truncation, den=1),
                ),
            ),
        ),
        precision_bits=source.precision_bits,
        target_width=_quarter_target(source.target_width),
        max_leaves=source.max_leaves,
        wall_seconds=source.wall_seconds,
    )


def _symmetric_tail(tail: EndpointTailEnclosure) -> bool:
    return tail.enclosure.lower.as_fraction() == -tail.enclosure.upper.as_fraction()


def _quadrature_shape_matches(
    result: DefiniteIntegralEnclosureResult,
    source: EndpointLogImproperIntegralRequest,
    truncation: int,
) -> bool:
    return (
        result.box.variables == ("t",)
        and result.box.intervals[0].lower.as_fraction() == 0
        and result.box.intervals[0].upper.as_fraction() == truncation
        and result.precision_bits == source.precision_bits
        and result.target_width == _quarter_target(source.target_width)
        and result.max_leaves == source.max_leaves
        and result.wall_seconds == source.wall_seconds
    )


def _enclose_endpoint_log_improper_integral(
    request: EndpointLogImproperIntegralRequest,
) -> EndpointLogImproperIntegralResult:
    execution = current_request_execution()
    started_at = execution.started_at if execution is not None else monotonic()
    deadline = started_at + request.wall_seconds
    if execution is not None and execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    request_checkpoint("before improper-integral admission")
    try:
        left_expression = _transformed_expression(request, left=True)
        right_expression = _transformed_expression(request, left=False)
        _bounded_expression_nodes(left_expression)
        _bounded_expression_nodes(right_expression)
    except ValueError as error:
        raise OperationResourceAdmissionError(
            location=("smooth_expression",),
            code="analysis.improper_integral.transformed_expression_bound",
            message="endpoint substitution exceeds the admitted expression size",
        ) from error
    request_checkpoint("after improper-integral transformed-expression admission")
    truncation, left_tail, right_tail = _tail_plan(request)
    request_checkpoint("after improper-integral tail admission")
    left = _compute_definite_integral_enclosure(
        _quadrature_request(request, left_expression, truncation)
    )
    right = _compute_definite_integral_enclosure(
        _quadrature_request(request, right_expression, truncation)
    )
    if isinstance(left.outcome, DefiniteIntegralDomainUnproven) or isinstance(
        right.outcome, DefiniteIntegralDomainUnproven
    ):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    # The tails are symmetric, so adding their exact lower/upper bounds keeps
    # the returned interval an exact replay of the proof decomposition.
    lower = (
        left.outcome.enclosure.lower.as_fraction()
        + right.outcome.enclosure.lower.as_fraction()
        - left_tail
        - right_tail
    )
    upper = (
        left.outcome.enclosure.upper.as_fraction()
        + right.outcome.enclosure.upper.as_fraction()
        + left_tail
        + right_tail
    )
    result = EndpointLogImproperIntegralResult(
        source=request,
        left_quadrature=left,
        right_quadrature=right,
        left_tail=EndpointTailEnclosure(
            truncation=truncation, enclosure=_tail_interval(left_tail)
        ),
        right_tail=EndpointTailEnclosure(
            truncation=truncation, enclosure=_tail_interval(right_tail)
        ),
        enclosure=ClosedRationalInterval(
            lower=CanonicalRational.from_fraction(lower),
            upper=CanonicalRational.from_fraction(upper),
        ),
    )
    request_checkpoint("after combined improper-integral result construction")
    return result


def enclose_endpoint_log_improper_integral(
    request: EndpointLogImproperIntegralRequest,
) -> EndpointLogImproperIntegralResult:
    """Enclose one endpoint-logarithmic integral under one request deadline."""

    if not isinstance(request, EndpointLogImproperIntegralRequest):
        raise OperationDomainValidationError(
            location=("request",),
            code="analysis.improper_integral.request_type",
            message="improper integration requires a validated endpoint-log request",
        )
    if current_request_execution() is None:
        with request_execution(monotonic()):
            return _enclose_endpoint_log_improper_integral(request)
    return _enclose_endpoint_log_improper_integral(request)


IMPROPER_INTEGRAL_OPERATIONS = (
    MathTool(
        operation_id=(
            "interval.expression.endpoint_log_improper_integral_enclosure.compute"
        ),
        title="Enclose an endpoint-logarithmic improper integral",
        description=(
            "Enclose a closed-box-safe smooth expression multiplied by bounded "
            "powers of normalized left and right endpoint logarithms. Each half "
            "uses an endpoint-specific exponential substitution and the proper "
            "Arb integrator; separate analytic exponential-polynomial tail "
            "enclosures are retained."
        ),
        request_type=EndpointLogImproperIntegralRequest,
        result_type=EndpointLogImproperIntegralResult,
        run=enclose_endpoint_log_improper_integral,
        tags=(
            "analysis",
            "integral",
            "improper",
            "endpoint",
            "logarithm",
            "arb",
            "validated",
        ),
        examples=(
            OperationExample(
                name="left_log_unit_interval",
                description="Enclose the integral of -log(x) over the unit interval.",
                input={
                    "smooth_expression": {
                        "op": "const",
                        "value": {"num": "1", "den": "1"},
                    },
                    "interval": {
                        "lower": {"num": "0", "den": "1"},
                        "upper": {"num": "1", "den": "1"},
                    },
                    "variable": "x",
                    "left_log_power": 1,
                    "target_width": {"mantissa": "1", "exponent": -2},
                    "max_leaves": 128,
                },
            ),
        ),
    ),
)
