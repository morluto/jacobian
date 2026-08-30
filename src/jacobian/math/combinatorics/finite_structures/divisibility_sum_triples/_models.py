"""Typed contracts for divisibility-sum triple hypergraph construction."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_LABEL_LENGTH,
    FiniteHypergraph,
)

MAX_INTERVAL_SIZE = 200

IntervalBound = Annotated[
    str,
    StringConstraints(
        pattern=r"^(?:0|-?[1-9][0-9]*)$",
        max_length=MAX_LABEL_LENGTH,
        strict=True,
    ),
]


def _validate_interval_shape(lower_bound: int, upper_bound: int) -> None:
    if not isinstance(lower_bound, int) or isinstance(lower_bound, bool):
        raise PydanticCustomError(
            "divisibility_sum.bound_type", "interval bounds must be integers"
        )
    if not isinstance(upper_bound, int) or isinstance(upper_bound, bool):
        raise PydanticCustomError(
            "divisibility_sum.bound_type", "interval bounds must be integers"
        )
    if lower_bound > upper_bound:
        raise PydanticCustomError(
            "divisibility_sum.invalid_bounds",
            "lower_bound must not exceed upper_bound",
        )


class DivisibilitySumTriplesRequest(StrictModel):
    """Request to construct the divisibility-sum triple hypergraph."""

    lower_bound: IntervalBound
    upper_bound: IntervalBound

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        _validate_interval_shape(
            parse_canonical_integer(self.lower_bound),
            parse_canonical_integer(self.upper_bound),
        )
        return self


class DivisibilitySumTriplesResult(StrictModel):
    """The divisibility-sum triple hypergraph."""

    lower_bound: CanonicalInteger
    upper_bound: CanonicalInteger
    hypergraph: FiniteHypergraph


__all__ = [
    "MAX_INTERVAL_SIZE",
    "DivisibilitySumTriplesRequest",
    "DivisibilitySumTriplesResult",
    "_validate_interval_shape",
]
