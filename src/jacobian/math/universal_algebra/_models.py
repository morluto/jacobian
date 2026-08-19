"""Typed wire contracts for universal-algebra operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.universal_algebra.values import FiniteAlgebra, FlatTerm


class EvaluateRequest(StrictModel):
    """Evaluate a source-bound term under a complete assignment."""

    algebra: FiniteAlgebra
    term: FlatTerm
    assignment: tuple[int, ...] = Field(default=())

    @model_validator(mode="after")
    def require_valid_assignment(self) -> Self:
        n = len(self.algebra.carrier)
        if any(not 0 <= v < n for v in self.assignment):
            raise ValueError("assignment value out of carrier range")
        return self


class EvaluateResult(StrictModel):
    """The exact carrier value t^A(alpha)."""

    value: int = Field(ge=0)


class EquationProfileRequest(StrictModel):
    """Evaluate s = t over all assignments."""

    algebra: FiniteAlgebra
    left: FlatTerm
    right: FlatTerm
    variable_count: int = Field(ge=1)


class EquationProfileResult(StrictModel):
    """HOLDS with satisfying count, or FAILS with first counterassignment."""

    status: str
    satisfying_count: int = Field(ge=0)
    first_counterassignment: dict[str, object] | None = None

    @model_validator(mode="after")
    def bind_status(self) -> Self:
        if self.status not in ("HOLDS", "FAILS"):
            raise ValueError("status must be HOLDS or FAILS")
        return self


class SubalgebraRequest(StrictModel):
    """Compute the least subalgebra containing the generating set."""

    algebra: FiniteAlgebra
    generators: tuple[int, ...] = Field(default=())

    @model_validator(mode="after")
    def require_valid_generators(self) -> Self:
        n = len(self.algebra.carrier)
        if any(not 0 <= g < n for g in self.generators):
            raise ValueError("generator out of carrier range")
        return self


class SubalgebraResult(StrictModel):
    """The generated carrier, closure rounds, and closed-ness."""

    generated_carrier: tuple[int, ...]
    rounds: int = Field(ge=1)
    is_closed: bool


class CongruenceRequest(StrictModel):
    """Check whether a carrier partition is a congruence."""

    algebra: FiniteAlgebra
    partition: tuple[tuple[int, ...], ...]

    @model_validator(mode="after")
    def require_partition_covers_carrier(self) -> Self:
        n = len(self.algebra.carrier)
        seen = set()
        for block in self.partition:
            for elem in block:
                if not 0 <= elem < n:
                    raise ValueError("partition element out of carrier range")
                if elem in seen:
                    raise ValueError("partition blocks must be disjoint")
                seen.add(elem)
        return self


class CongruenceResult(StrictModel):
    """Whether the partition is a congruence, with obstruction if not."""

    is_congruence: bool
    obstruction: str | None = None


class QuotientRequest(StrictModel):
    """Compute the quotient algebra A/theta."""

    algebra: FiniteAlgebra
    partition: tuple[tuple[int, ...], ...]

    @model_validator(mode="after")
    def require_partition_covers_carrier(self) -> Self:
        n = len(self.algebra.carrier)
        seen: set[int] = set()
        for block in self.partition:
            for elem in block:
                if not 0 <= elem < n:
                    raise ValueError("partition element out of carrier range")
                if elem in seen:
                    raise ValueError("partition blocks must be disjoint")
                seen.add(elem)
        return self


class QuotientResult(StrictModel):
    """The quotient algebra carrier, operations, and tables."""

    carrier: tuple[str, ...]
    operations: tuple[tuple[str, int], ...]
    tables: tuple[tuple[int, ...], ...]


__all__ = [
    "CongruenceRequest",
    "CongruenceResult",
    "EquationProfileRequest",
    "EquationProfileResult",
    "EvaluateRequest",
    "EvaluateResult",
    "QuotientRequest",
    "QuotientResult",
    "SubalgebraRequest",
    "SubalgebraResult",
]
