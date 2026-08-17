"""Private wire models for bounded rational linear optimization."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel

MAX_RATIONAL_DIGITS = 128
type RationalLinearProgramStatus = Literal[
    "OPTIMAL",
    "PRIMAL_FEASIBLE",
    "INFEASIBLE",
    "UNBOUNDED",
]


class StandardFormRationalLinearProgram(StrictModel):
    """Minimize ``cᵀx`` subject to ``Ax=b`` and ``x>=0``."""

    variables: tuple[str, ...] = Field(min_length=1, max_length=32)
    objective: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=32)
    coefficients: tuple[tuple[CanonicalRational, ...], ...] = Field(
        min_length=1,
        max_length=64,
    )
    rhs: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_canonical_dimensions(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("linear-program variable names must be unique")
        if any(
            not name
            or len(name) > 64
            or not (name[0].isalpha() or name[0] == "_")
            or any(not (char.isalnum() or char == "_") for char in name)
            for name in self.variables
        ):
            raise ValueError("linear-program variable names must be identifiers")
        width = len(self.variables)
        if len(self.objective) != width:
            raise ValueError("objective length must equal the variable count")
        if len(self.coefficients) != len(self.rhs):
            raise ValueError("coefficient row count must equal the rhs length")
        if any(len(row) != width for row in self.coefficients):
            raise ValueError("every coefficient row must match the variable count")
        for value in (
            *self.objective,
            *self.rhs,
            *(item for row in self.coefficients for item in row),
        ):
            require_bounded_rational(
                value,
                max_digits=MAX_RATIONAL_DIGITS,
                label="validated-analysis rational",
            )
        return self


class RationalLinearProgramRequest(StrictModel):
    program: StandardFormRationalLinearProgram


class RationalLinearProgramResult(StrictModel):
    """The direct mathematical outcome of one rational linear program."""

    status: RationalLinearProgramStatus
    primal_candidate: tuple[CanonicalRational, ...] | None = None
    dual_candidate: tuple[CanonicalRational, ...] | None = None
    primal_objective: CanonicalRational | None = None
    dual_objective: CanonicalRational | None = None
    primal_residuals: tuple[CanonicalRational, ...] | None = None
    dual_slacks: tuple[CanonicalRational, ...] | None = None

    @model_validator(mode="after")
    def bind_result_fields(self) -> Self:
        optimal = self.status == "OPTIMAL"
        primal_fields = (
            self.primal_candidate,
            self.primal_objective,
            self.primal_residuals,
        )
        dual_fields = (
            self.dual_candidate,
            self.dual_objective,
            self.dual_slacks,
        )
        has_primal = self.status in {"OPTIMAL", "PRIMAL_FEASIBLE"}
        if has_primal and not all(value is not None for value in primal_fields):
            raise ValueError(
                "a primal result requires a candidate, objective, and residuals"
            )
        if not has_primal and any(value is not None for value in primal_fields):
            raise ValueError("an infeasible or unbounded result cannot carry a point")
        if optimal and not all(value is not None for value in dual_fields):
            raise ValueError(
                "an optimal result requires a dual candidate, objective, and slacks"
            )
        if not optimal and any(value is not None for value in dual_fields):
            raise ValueError("only an optimal result can carry dual data")
        return self


__all__ = [
    "RationalLinearProgramRequest",
    "RationalLinearProgramResult",
    "RationalLinearProgramStatus",
    "StandardFormRationalLinearProgram",
]
