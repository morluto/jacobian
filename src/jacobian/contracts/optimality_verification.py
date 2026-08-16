"""Typed contracts for exact optimality verification."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.validated_analysis import (
    MAX_RATIONAL_DIGITS,
    StandardFormRationalLinearProgram,
    require_bounded_rational,
)


class RationalOptimalityVerifyRequest(ContractModel):
    """Request to verify an exact LP optimum from primal and dual evidence."""

    program: StandardFormRationalLinearProgram
    claimed_objective: CanonicalRational
    primal_candidate: tuple[CanonicalRational, ...]
    dual_candidate: tuple[CanonicalRational, ...]

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        n_vars = len(self.program.variables)
        n_constraints = len(self.program.coefficients)
        if len(self.primal_candidate) != n_vars:
            raise ValueError(
                f"primal candidate has {len(self.primal_candidate)} values but "
                f"the program has {n_vars} variables"
            )
        if len(self.dual_candidate) != n_constraints:
            raise ValueError(
                f"dual candidate has {len(self.dual_candidate)} values but "
                f"the program has {n_constraints} constraints"
            )
        require_bounded_rational(
            self.claimed_objective,
            max_digits=MAX_RATIONAL_DIGITS,
            label="claimed objective",
        )
        return self


class RationalOptimalityVerifyResult(ContractModel):
    """Result of an exact LP optimality verification."""

    status: Literal["VERIFIED", "REJECTED", "INVALID"]
    primal_objective: CanonicalRational | None = None
    dual_objective: CanonicalRational | None = None
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_status_to_evidence(self) -> Self:
        if self.status == "VERIFIED":
            if self.primal_objective is None or self.dual_objective is None:
                raise ValueError("a verified result requires both objectives")
        if self.status != "VERIFIED" and self.primal_objective is not None:
            if self.dual_objective is not None:
                pass  # REJECTED can carry evidence showing the mismatch
        return self
