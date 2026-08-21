"""Typed bounded contracts for finite universal-algebra operations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.universal_algebra.values import (
    MAX_CARRIER_SIZE,
    FiniteAlgebra,
    FlatTerm,
    require_term_for_algebra,
)

MAX_ENUMERATION_WORK = 1_000_000
CarrierBlock = Annotated[
    tuple[int, ...],
    Field(min_length=1, max_length=MAX_CARRIER_SIZE),
]


def _require_partition(
    algebra: FiniteAlgebra,
    partition: tuple[CarrierBlock, ...],
) -> None:
    expected = set(range(len(algebra.carrier)))
    seen: set[int] = set()
    for block in partition:
        for element in block:
            if element not in expected:
                raise ValueError("partition element out of carrier range")
            if element in seen:
                raise ValueError("partition blocks must be disjoint")
            seen.add(element)
    if seen != expected:
        raise ValueError("partition blocks must exactly cover the carrier")


def _congruence_work(algebra: FiniteAlgebra) -> int:
    size = len(algebra.carrier)
    return sum(
        size**symbol.arity * max(1, symbol.arity) * size
        for symbol in algebra.operations
    )


class EvaluateRequest(StrictModel):
    algebra: FiniteAlgebra
    term: FlatTerm
    assignment: tuple[int, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def require_total_assignment(self) -> Self:
        require_term_for_algebra(self.term, self.algebra)
        if len(self.assignment) != self.term.variable_count:
            raise ValueError("assignment must cover exactly the referenced variables")
        size = len(self.algebra.carrier)
        if any(not 0 <= value < size for value in self.assignment):
            raise ValueError("assignment value out of carrier range")
        return self


class EvaluateResult(StrictModel):
    value: int = Field(ge=0)


class EquationProfileRequest(StrictModel):
    algebra: FiniteAlgebra
    left: FlatTerm
    right: FlatTerm
    variable_count: int = Field(ge=1, le=8, strict=True)

    @model_validator(mode="after")
    def require_bounded_complete_profile(self) -> Self:
        require_term_for_algebra(self.left, self.algebra)
        require_term_for_algebra(self.right, self.algebra)
        if (
            max(self.left.variable_count, self.right.variable_count)
            > self.variable_count
        ):
            raise ValueError("variable_count must cover every referenced variable")
        work = len(self.algebra.carrier) ** self.variable_count
        if work > MAX_ENUMERATION_WORK:
            raise ValueError("equation profile exceeds the assignment work budget")
        return self


class EquationCounterexample(StrictModel):
    assignment: tuple[int, ...] = Field(max_length=8)
    left_value: int = Field(ge=0, le=MAX_CARRIER_SIZE - 1)
    right_value: int = Field(ge=0, le=MAX_CARRIER_SIZE - 1)


class EquationProfileResult(StrictModel):
    status: Literal["HOLDS", "FAILS"]
    satisfying_count: int = Field(ge=0, le=MAX_ENUMERATION_WORK)
    first_counterassignment: EquationCounterexample | None = None

    @model_validator(mode="after")
    def bind_status(self) -> Self:
        if (self.status == "FAILS") != (self.first_counterassignment is not None):
            raise ValueError("FAILS must carry exactly one first counterassignment")
        return self


class SubalgebraRequest(StrictModel):
    algebra: FiniteAlgebra
    generators: tuple[int, ...] = Field(
        default=(),
        max_length=MAX_CARRIER_SIZE,
    )

    @model_validator(mode="after")
    def require_valid_bounded_generators(self) -> Self:
        size = len(self.algebra.carrier)
        if any(not 0 <= generator < size for generator in self.generators):
            raise ValueError("generator out of carrier range")
        work = sum(size**symbol.arity for symbol in self.algebra.operations) * size
        if work > MAX_ENUMERATION_WORK:
            raise ValueError("subalgebra closure exceeds the operation work budget")
        return self


class SubalgebraResult(StrictModel):
    generated_carrier: tuple[int, ...]
    rounds: int = Field(ge=1)
    is_closed: bool


class _PartitionRequest(StrictModel):
    algebra: FiniteAlgebra
    partition: tuple[CarrierBlock, ...] = Field(
        min_length=1,
        max_length=MAX_CARRIER_SIZE,
    )

    @model_validator(mode="after")
    def require_complete_bounded_partition(self) -> Self:
        _require_partition(self.algebra, self.partition)
        if _congruence_work(self.algebra) > MAX_ENUMERATION_WORK:
            raise ValueError("congruence check exceeds the operation work budget")
        return self


class CongruenceRequest(_PartitionRequest):
    """Check one complete carrier partition for operation compatibility."""


class CongruenceResult(StrictModel):
    is_congruence: bool
    obstruction: str | None = None


class QuotientRequest(_PartitionRequest):
    """Construct ``A/theta`` for an admitted congruence partition."""


class QuotientResult(StrictModel):
    """A directly composable quotient algebra and its carrier map."""

    algebra: FiniteAlgebra
    quotient_map: tuple[int, ...] = Field(
        min_length=1,
        max_length=MAX_CARRIER_SIZE,
    )

    @model_validator(mode="after")
    def require_map_into_quotient(self) -> Self:
        if any(
            not 0 <= value < len(self.algebra.carrier) for value in self.quotient_map
        ):
            raise ValueError("quotient map value is outside the quotient carrier")
        return self


__all__ = [
    "CongruenceRequest",
    "CongruenceResult",
    "EquationCounterexample",
    "EquationProfileRequest",
    "EquationProfileResult",
    "EvaluateRequest",
    "EvaluateResult",
    "QuotientRequest",
    "QuotientResult",
    "SubalgebraRequest",
    "SubalgebraResult",
]
