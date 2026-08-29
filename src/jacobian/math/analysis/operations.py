"""Validated real-function enclosure operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.analysis._arb import arb_source_interval, dyadic_endpoints
from jacobian.math.analysis._expression_enclosure import (
    IntervalExpressionEnclosureResult,
)
from jacobian.math.analysis._models import (
    MAX_RATIONAL_DIGITS,
    DyadicClosedInterval,
    IntervalExpressionDomainFailure,
    IntervalExpressionNode,
    RationalIntervalBox,
    _bounded_expression_nodes,
    _rational_box_bounds,
)
from jacobian.math.analysis._second_jet import (
    MAX_SECOND_JET_RESULT_INTERVALS,
    MAX_SECOND_JET_WORK_UNITS,
    FirstPartialEnclosure,
    HessianEntryEnclosure,
    IntervalExpressionSecondJetEnclosureResult,
    _preflight_second_jet_expression,
    _second_jet_node_arithmetic_units,
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


def expression_enclosure(
    expression: IntervalExpressionNode,
    argument: CanonicalRational,
    precision_bits: int,
) -> IntervalExpressionEnclosureResult:
    try:
        require_bounded_rational(
            argument,
            max_digits=MAX_RATIONAL_DIGITS,
            label="interval-enclosure argument",
        )
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("argument",),
            code="analysis.expression.argument_bound",
            message=str(exc),
        ) from exc
    if any(
        node.op == "var" and node.variable is not None
        for node in _bounded_expression_nodes(expression)
    ):
        raise OperationDomainValidationError(
            location=("expression",),
            code="analysis.expression.named_variable",
            message="point-enclosure variable nodes must remain anonymous",
        )
    from flint import arb, ctx, fmpq

    numerator, denominator = argument.as_integer_ratio()
    with ctx.workprec(precision_bits):
        result = _evaluate_expression(expression, arb(fmpq(numerator, denominator)))
        if isinstance(result, _EvaluationFailure):
            return IntervalExpressionEnclosureResult(
                status=result.value,
                precision_bits=precision_bits,
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
                precision_bits=precision_bits,
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
            precision_bits=precision_bits,
            detail="Arb produced finite endpoints outside the interoperable dyadic exponent range.",
        )
    return IntervalExpressionEnclosureResult(
        status="ENCLOSED",
        precision_bits=precision_bits,
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


def _second_jet_result(
    expression: IntervalExpressionNode,
    box: RationalIntervalBox,
    precision_bits: int,
    *,
    status: str,
    detail: str,
    value: DyadicClosedInterval | None = None,
    gradient: tuple[FirstPartialEnclosure, ...] = (),
    hessian: tuple[HessianEntryEnclosure, ...] = (),
    domain_failure: IntervalExpressionDomainFailure | None = None,
) -> IntervalExpressionSecondJetEnclosureResult:
    return IntervalExpressionSecondJetEnclosureResult.model_construct(
        expression=expression,
        box=box,
        precision_bits=precision_bits,
        status=status,
        value=value,
        gradient=gradient,
        hessian=hessian,
        domain_failure=domain_failure,
        detail=detail,
    )


def second_jet_enclosure(
    expression: IntervalExpressionNode,
    box: RationalIntervalBox,
    precision_bits: int,
) -> IntervalExpressionSecondJetEnclosureResult:
    dimension = len(box.variables)
    result_intervals = 1 + dimension + dimension * (dimension + 1) // 2
    if result_intervals > MAX_SECOND_JET_RESULT_INTERVALS:
        raise AssertionError("second-jet result interval accounting is inconsistent")
    work_units = len(_bounded_expression_nodes(expression)) * (
        _second_jet_node_arithmetic_units(dimension)
    )
    if work_units > MAX_SECOND_JET_WORK_UNITS:
        raise OperationDomainValidationError(
            location=("expression",),
            code="analysis.second_jet.work_bound",
            message=(
                f"second-jet forward arithmetic work of {work_units} scalar units "
                f"exceeds its {MAX_SECOND_JET_WORK_UNITS}-unit bound at this dimension"
            ),
        )
    try:
        preflight = _preflight_second_jet_expression(
            expression, _rational_box_bounds(box)
        )
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("expression",),
            code="analysis.second_jet.intermediate_bound",
            message=str(exc),
        ) from exc
    if isinstance(preflight, IntervalExpressionDomainFailure):
        return _second_jet_result(
            expression,
            box,
            precision_bits,
            status="DOMAIN_UNPROVEN",
            domain_failure=preflight,
            detail=(
                "The exact admission interval extension could not establish twice "
                "differentiability at the reported source node."
            ),
        )

    try:
        from flint import ctx

        with ctx.workprec(precision_bits):
            variables = {
                variable: arb_source_interval(interval)
                for variable, interval in zip(box.variables, box.intervals, strict=True)
            }
            jet = _evaluate_second_jet(expression, variables, len(box.variables))
            if isinstance(jet, _SecondJetEvaluationFailure):
                return _second_jet_result(
                    expression,
                    box,
                    precision_bits,
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
        return _second_jet_result(
            expression,
            box,
            precision_bits,
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
        return _second_jet_result(
            expression,
            box,
            precision_bits,
            status="BACKEND_ERROR",
            detail=(
                "Pinned Arb produced endpoints outside the admitted dyadic wire "
                "envelope; no enclosure conclusion is available."
            ),
        )

    gradient = tuple(
        FirstPartialEnclosure(variable=variable, enclosure=entry)
        for variable, entry in zip(box.variables, gradient_intervals, strict=True)
        if entry is not None
    )
    hessian_entries: list[HessianEntryEnclosure] = []
    for first_index, first in enumerate(box.variables):
        for second_index, second in enumerate(box.variables[first_index:], first_index):
            enclosure = hessian_intervals[first_index][second_index]
            assert enclosure is not None
            hessian_entries.append(
                HessianEntryEnclosure(
                    first_variable=first,
                    second_variable=second,
                    enclosure=enclosure,
                )
            )
    return _second_jet_result(
        expression,
        box,
        precision_bits,
        status="ENCLOSED",
        value=value,
        gradient=gradient,
        hessian=tuple(hessian_entries),
        detail=(
            "Pinned Arb forward automatic differentiation returned outward-rounded "
            "value, gradient, and symmetric Hessian enclosures with exact dyadic endpoints."
        ),
    )


__all__ = ["expression_enclosure", "second_jet_enclosure"]
