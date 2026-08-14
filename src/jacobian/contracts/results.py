"""Orthogonal operational and mathematical result fields."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian.contracts.base import ContractModel as ContractModel


class ExecutionStatus(StrEnum):
    """Operational completion state, independent of mathematical truth."""

    COMPLETED = "COMPLETED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


class InputStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class Conclusion(StrEnum):
    """Mathematical conclusion; UNKNOWN is never interpreted as false."""

    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Arithmetic(StrEnum):
    EXACT_INTEGER = "EXACT_INTEGER"
    EXACT_RATIONAL = "EXACT_RATIONAL"
    EXACT_ALGEBRAIC = "EXACT_ALGEBRAIC"
    VERIFIED_INTERVAL = "VERIFIED_INTERVAL"
    SYMBOLIC = "SYMBOLIC"
    FLOATING_HEURISTIC = "FLOATING_HEURISTIC"


class Method(StrEnum):
    DIRECT_WITNESS = "DIRECT_WITNESS"
    EXHAUSTIVE_FINITE = "EXHAUSTIVE_FINITE"
    CHECKED_CERTIFICATE = "CHECKED_CERTIFICATE"
    BOUNDED_SEARCH = "BOUNDED_SEARCH"


class Coverage(StrEnum):
    EXHAUSTIVE = "EXHAUSTIVE"
    BOUNDED = "BOUNDED"
    RESTRICTED = "RESTRICTED"
    SAMPLED = "SAMPLED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Execution(ContractModel):
    status: ExecutionStatus
    runtime_ms: StrictInt | None = Field(default=None, ge=0)
    detail: str | None = None


class InputValidation(ContractModel):
    status: InputStatus
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def accepted_input_has_no_errors(self) -> Self:
        if self.status == InputStatus.ACCEPTED and self.errors:
            raise ValueError("accepted input cannot carry validation errors")
        if self.status == InputStatus.REJECTED and not self.errors:
            raise ValueError("rejected input requires at least one error")
        return self
