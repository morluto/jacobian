"""Contracts for bounded second-order interval-expression enclosures."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from jacobian._models import StrictModel
from jacobian.math.analysis._models import (
    MAX_BOX_VARIABLES,
    DyadicClosedInterval,
    IntervalExpressionDomainFailure,
    IntervalExpressionNode,
    IntervalVariable,
    _bounded_expression_nodes,
    _bounded_rational_bounds,
    _BoxPreflight,
    _IntervalExpressionBoxRequest,
    _preflight_box_binary,
    _preflight_box_unary,
    _rational_box_bounds,
    _RationalBounds,
    _validation_error,
)

MAX_SECOND_JET_VARIABLES = MAX_BOX_VARIABLES
MAX_SECOND_JET_RESULT_INTERVALS = (
    1
    + MAX_SECOND_JET_VARIABLES
    + MAX_SECOND_JET_VARIABLES * (MAX_SECOND_JET_VARIABLES + 1) // 2
)


def _second_jet_node_arithmetic_units(dimension: int) -> int:
    """Bound the scalar Arb steps of one source node's dense second-order jet.

    Every jet carries a value, gradient, and dense Hessian, so one node's
    arithmetic grows quadratically in the jet dimension. A division is the
    most expensive node: one reciprocal power jet followed by one product jet
    costs about ``10 * dimension**2 + 4 * dimension`` scalar steps, and the
    padded constant covers transcendental chain-rule values. At three
    variables this reproduces the former flat 128-unit bound.
    """
    return 10 * dimension * dimension + 4 * dimension + 16


# Twice the former 64-node x 128-unit envelope. Every previously accepted
# three-variable request stays admitted while the dimension-derived charge
# admits small full-axis jets such as an eight-variable affine form.
MAX_SECOND_JET_WORK_UNITS = 16_384

type IntervalExpressionSecondJetEnclosureStatus = Literal[
    "ENCLOSED", "DOMAIN_UNPROVEN", "BACKEND_ERROR"
]


def _preflight_second_jet_expression(
    node: IntervalExpressionNode,
    variables: dict[str, _RationalBounds],
    path: tuple[int, ...] = (),
) -> _BoxPreflight:
    """Prove the source tree is twice differentiable on the complete box."""

    if node.op == "const":
        assert node.value is not None
        value = node.value.as_fraction()
        return _bounded_rational_bounds(value, value)
    if node.op == "var":
        assert node.variable is not None
        return variables[node.variable]

    children: list[_BoxPreflight] = []
    for index, child_node in enumerate(node.children):
        child = _preflight_second_jet_expression(child_node, variables, (*path, index))
        if isinstance(child, IntervalExpressionDomainFailure):
            return child
        children.append(child)

    left = children[0]
    assert isinstance(left, _RationalBounds)
    if len(children) == 1:
        if node.op == "sqrt" and left.lower <= 0:
            return IntervalExpressionDomainFailure(
                node_path=path,
                operation="sqrt",
                reason="SQRT_ARGUMENT_NOT_STRICTLY_POSITIVE_FOR_SECOND_JET",
            )
        return _preflight_box_unary(node, left, path)

    right = children[1]
    assert isinstance(right, _RationalBounds)
    return _preflight_box_binary(node, left, right, path)


class IntervalExpressionSecondJetEnclosureRequest(_IntervalExpressionBoxRequest):
    """Enclose an elementary expression and all derivatives through order two."""

    @model_validator(mode="after")
    def require_small_twice_differentiable_source(self) -> Self:
        dimension = len(self.box.variables)
        result_intervals = 1 + dimension + dimension * (dimension + 1) // 2
        if result_intervals > MAX_SECOND_JET_RESULT_INTERVALS:
            raise AssertionError(
                "second-jet result interval accounting is inconsistent"
            )
        work_units = len(
            _bounded_expression_nodes(self.expression)
        ) * _second_jet_node_arithmetic_units(dimension)
        if work_units > MAX_SECOND_JET_WORK_UNITS:
            raise _validation_error(
                f"second-jet forward arithmetic work of {work_units} scalar "
                f"units exceeds its {MAX_SECOND_JET_WORK_UNITS}-unit bound at "
                "this dimension"
            )
        _preflight_second_jet_expression(
            self.expression, _rational_box_bounds(self.box)
        )
        return self


class FirstPartialEnclosure(StrictModel):
    """One first-partial enclosure bound to the authoritative source axis."""

    variable: IntervalVariable
    enclosure: DyadicClosedInterval


class HessianEntryEnclosure(StrictModel):
    """One canonical upper-triangular Hessian entry over the source box."""

    first_variable: IntervalVariable
    second_variable: IntervalVariable
    enclosure: DyadicClosedInterval


class IntervalExpressionSecondJetEnclosureResult(
    IntervalExpressionSecondJetEnclosureRequest
):
    """A source-bound rigorous value, gradient, and symmetric Hessian enclosure.

    For ``ENCLOSED``, full validation recomputes the canonical jet claim from
    its expression, axis, and source box.  ``DOMAIN_UNPROVEN`` replays its
    deterministic first-obstruction evidence.  ``BACKEND_ERROR`` asserts no
    enclosure conclusion at all, so it is validated structurally: rerunning
    Arb would reject the operation's own serialized result whenever a
    transient backend condition does not recur.
    """

    status: IntervalExpressionSecondJetEnclosureStatus
    value: DyadicClosedInterval | None = None
    gradient: tuple[FirstPartialEnclosure, ...] = Field(
        default=(), max_length=MAX_SECOND_JET_VARIABLES
    )
    hessian: tuple[HessianEntryEnclosure, ...] = Field(
        default=(),
        max_length=MAX_SECOND_JET_VARIABLES * (MAX_SECOND_JET_VARIABLES + 1) // 2,
    )
    domain_failure: IntervalExpressionDomainFailure | None = None
    method: Literal["ARB_FORWARD_SECOND_ORDER_JET"] = "ARB_FORWARD_SECOND_ORDER_JET"
    detail: str = Field(min_length=1, max_length=1024)

    @field_validator("gradient", mode="before")
    @classmethod
    def preserve_gradient_json_array_composition(cls, value: object) -> object:
        if isinstance(value, (list, tuple)) and len(value) > MAX_SECOND_JET_VARIABLES:
            raise _validation_error(
                "second-jet result exceeds its declared interval bound"
            )
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("hessian", mode="before")
    @classmethod
    def preserve_hessian_json_array_composition(cls, value: object) -> object:
        maximum = MAX_SECOND_JET_VARIABLES * (MAX_SECOND_JET_VARIABLES + 1) // 2
        if isinstance(value, (list, tuple)) and len(value) > maximum:
            raise _validation_error(
                "second-jet result exceeds its declared interval bound"
            )
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def bind_second_jet_to_source(self) -> Self:
        enclosed = self.status == "ENCLOSED"
        expected_pairs = tuple(
            (first, second)
            for first_index, first in enumerate(self.box.variables)
            for second in self.box.variables[first_index:]
        )
        if enclosed != (self.value is not None):
            raise _validation_error(
                "only an enclosed second jet may carry a value enclosure"
            )
        if enclosed:
            if tuple(entry.variable for entry in self.gradient) != self.box.variables:
                raise _validation_error(
                    "gradient entries must match the ordered source axis"
                )
            if (
                tuple(
                    (entry.first_variable, entry.second_variable)
                    for entry in self.hessian
                )
                != expected_pairs
            ):
                raise _validation_error(
                    "Hessian entries must be the canonical upper triangle of the source axis"
                )
            if self.domain_failure is not None:
                raise _validation_error(
                    "an enclosed second jet cannot carry a domain failure"
                )
        else:
            if self.value is not None or self.gradient or self.hessian:
                raise _validation_error(
                    "a non-enclosed second jet cannot carry partial enclosures"
                )
            domain_unproven = self.status == "DOMAIN_UNPROVEN"
            if domain_unproven != (self.domain_failure is not None):
                raise _validation_error(
                    "domain-failure evidence must agree with DOMAIN_UNPROVEN status"
                )
            if self.status == "BACKEND_ERROR":
                return self

        from jacobian.math.analysis._operations import _second_jet_enclosure

        request = IntervalExpressionSecondJetEnclosureRequest.model_construct(
            expression=self.expression,
            box=self.box,
            precision_bits=self.precision_bits,
        )
        if self != _second_jet_enclosure(request):
            raise _validation_error(
                "second-jet enclosure does not replay from its expression, axis, and source box"
            )
        return self


__all__ = [
    "MAX_SECOND_JET_RESULT_INTERVALS",
    "MAX_SECOND_JET_VARIABLES",
    "MAX_SECOND_JET_WORK_UNITS",
    "FirstPartialEnclosure",
    "HessianEntryEnclosure",
    "IntervalExpressionSecondJetEnclosureRequest",
    "IntervalExpressionSecondJetEnclosureResult",
    "IntervalExpressionSecondJetEnclosureStatus",
]
