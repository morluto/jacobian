"""Contracts, admission, replay, and execution for expression box enclosures."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.analysis._arb import arb_source_interval, dyadic_endpoints
from jacobian.math.analysis._models import (
    MAX_BOX_PREFLIGHT_TEMPORARY_BITS,
    ExactDyadic,
    IntervalExpressionDomainFailure,
    IntervalExpressionNode,
    _bounded_rational_bounds,
    _BoxPreflight,
    _IntervalExpressionBoxRequest,
    _preflight_box_binary,
    _preflight_box_unary,
    _rational_box_bounds,
    _RationalBounds,
    _validation_error,
)

type IntervalExpressionBoxEnclosureStatus = Literal[
    "ENCLOSED", "DOMAIN_UNPROVEN", "BACKEND_ERROR"
]


class IntervalExpressionBoxEnclosureRequest(_IntervalExpressionBoxRequest):
    """Enclose one named-variable expression on a complete rational box."""

    @model_validator(mode="after")
    def require_bounded_box_admission(self) -> Self:
        _preflight_box_expression(self.expression, _rational_box_bounds(self.box))
        return self


def _preflight_box_expression(
    node: IntervalExpressionNode,
    variables: dict[str, _RationalBounds],
    path: tuple[int, ...] = (),
) -> _BoxPreflight:
    """Bound exact admission work and locate the first domain obstruction."""

    if node.op == "const":
        assert node.value is not None
        value = node.value.as_fraction()
        return _bounded_rational_bounds(value, value)
    if node.op == "var":
        assert node.variable is not None
        return variables[node.variable]

    children: list[_BoxPreflight] = []
    for index, child_node in enumerate(node.children):
        child = _preflight_box_expression(child_node, variables, (*path, index))
        if isinstance(child, IntervalExpressionDomainFailure):
            return child
        children.append(child)

    left = children[0]
    assert isinstance(left, _RationalBounds)
    if len(children) == 1:
        return _preflight_box_unary(node, left, path)

    right = children[1]
    assert isinstance(right, _RationalBounds)
    return _preflight_box_binary(node, left, right, path)


class _BoxEvaluationFailure(StrEnum):
    BACKEND_ERROR = "BACKEND_ERROR"


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


class IntervalExpressionBoxEnclosureResult(IntervalExpressionBoxEnclosureRequest):
    """A replayable enclosure bound to its expression, axis, and source box.

    For ``ENCLOSED``, every defined real source-box value lies between the two
    exact dyadic endpoints. Full validation recomputes that canonical claim.
    ``DOMAIN_UNPROVEN`` replays its deterministic first-obstruction evidence.
    ``BACKEND_ERROR`` asserts no enclosure conclusion at all, so it is
    validated structurally: rerunning Arb would reject the operation's own
    serialized result whenever a transient backend condition does not recur.
    """

    status: IntervalExpressionBoxEnclosureStatus
    lower: ExactDyadic | None = None
    upper: ExactDyadic | None = None
    domain_failure: IntervalExpressionDomainFailure | None = None
    method: Literal["ARB_NATURAL_INTERVAL_EXTENSION"] = "ARB_NATURAL_INTERVAL_EXTENSION"
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_enclosure_to_source(self) -> Self:
        enclosed = self.status == "ENCLOSED"
        if enclosed != (self.lower is not None and self.upper is not None):
            raise _validation_error(
                "only an enclosed result may carry dyadic endpoints"
            )
        if enclosed:
            assert self.lower is not None and self.upper is not None
            if self.lower.compare(self.upper) > 0:
                raise _validation_error(
                    "enclosure lower endpoint exceeds upper endpoint"
                )
            if self.domain_failure is not None:
                raise _validation_error(
                    "an enclosed result cannot carry a domain failure"
                )
        else:
            domain_unproven = self.status == "DOMAIN_UNPROVEN"
            if domain_unproven != (self.domain_failure is not None):
                raise _validation_error(
                    "domain-failure evidence must agree with DOMAIN_UNPROVEN status"
                )
            if self.status == "BACKEND_ERROR":
                return self

        request = IntervalExpressionBoxEnclosureRequest.model_construct(
            expression=self.expression,
            box=self.box,
            precision_bits=self.precision_bits,
        )
        if self != _box_expression_enclosure(request):
            raise _validation_error(
                "box enclosure does not replay from its expression, axis, and source box"
            )
        return self


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
                variable: arb_source_interval(interval)
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
            endpoints = dyadic_endpoints(
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


BOX_EXPRESSION_ENCLOSURE_OPERATIONS = (
    MathTool(
        operation_id="interval.expression.box_enclosure.compute",
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
