"""Contracts for bounded pointwise interval-expression enclosures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.analysis._models import (
    MAX_RATIONAL_DIGITS,
    ExactDyadic,
    IntervalExpressionNode,
    _bound_raw_expression,
    _bound_raw_rational,
    _bounded_expression_nodes,
    _validation_error,
)


class IntervalExpressionEnclosureRequest(StrictModel):
    """Evaluate a bounded expression at one exact rational argument using Arb."""

    expression: IntervalExpressionNode
    argument: CanonicalRational
    precision_bits: StrictInt = Field(default=128, ge=32, le=4096)

    @model_validator(mode="before")
    @classmethod
    def bound_raw_tree(cls, value: object) -> object:
        value = canonicalize_json_containers(value)
        if isinstance(value, Mapping):
            _bound_raw_expression(value.get("expression"))
            _bound_raw_rational(value.get("argument"), "interval-enclosure argument")
        return value

    @model_validator(mode="after")
    def require_bounded_tree(self) -> Self:
        require_bounded_rational(
            self.argument,
            max_digits=MAX_RATIONAL_DIGITS,
            label="interval-enclosure argument",
        )
        nodes = _bounded_expression_nodes(self.expression)
        if any(node.op == "var" and node.variable is not None for node in nodes):
            raise _validation_error(
                "point-enclosure variable nodes must remain anonymous"
            )
        return self


class IntervalExpressionEnclosureResult(StrictModel):
    status: Literal[
        "ENCLOSED",
        "DOMAIN_ERROR",
        "PRECISION_INSUFFICIENT",
        "NONFINITE",
        "OUTPUT_MAGNITUDE_EXCEEDED",
    ]
    precision_bits: StrictInt = Field(ge=32, le=4096)
    lower: ExactDyadic | None = None
    upper: ExactDyadic | None = None
    relative_accuracy_bits: StrictInt | None = None
    exact: bool = False
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_enclosure_to_status(self) -> Self:
        enclosed = self.status == "ENCLOSED"
        if enclosed != (self.lower is not None and self.upper is not None):
            raise _validation_error(
                "only an enclosed result may carry dyadic endpoints"
            )
        if not enclosed and (self.relative_accuracy_bits is not None or self.exact):
            raise _validation_error(
                "a non-enclosure cannot claim accuracy or exactness"
            )
        if enclosed:
            assert self.lower is not None and self.upper is not None
            if self.lower.compare(self.upper) > 0:
                raise _validation_error(
                    "enclosure lower endpoint exceeds upper endpoint"
                )
            if self.exact != (self.relative_accuracy_bits is None):
                raise _validation_error(
                    "exact enclosures omit relative accuracy; inexact ones report it"
                )
        return self


__all__ = [
    "IntervalExpressionEnclosureRequest",
    "IntervalExpressionEnclosureResult",
]
