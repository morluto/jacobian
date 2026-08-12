"""Orthogonal operational and mathematical result fields."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian.contracts.base import ContractModel as ContractModel
from jacobian.contracts.common import ArtifactUri, Sha256Digest


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


class VerificationResult(ContractModel):
    """Internal result of one separately authorized checker execution."""

    schema_version: Literal["1"] = "1"
    execution: Execution
    input: InputValidation
    conclusion: Conclusion
    scope_uri: ArtifactUri | None = None
    claim_digest: Sha256Digest | None = None
    semantics_digest: Sha256Digest | None = None
    candidate_digest: Sha256Digest | None = None
    evidence_uris: tuple[ArtifactUri, ...] = ()
    verification_record_uri: ArtifactUri | None = None

    @model_validator(mode="after")
    def enforce_fail_closed_state(self) -> Self:
        execution_failed = self.execution.status != ExecutionStatus.COMPLETED
        input_rejected = self.input.status == InputStatus.REJECTED

        if (execution_failed or input_rejected) and self.conclusion not in {
            Conclusion.UNKNOWN,
            Conclusion.NOT_APPLICABLE,
        }:
            raise ValueError(
                "non-completed or rejected results cannot carry a "
                "mathematical conclusion"
            )
        self._require_verification_evidence()
        return self

    def _require_verification_evidence(self) -> None:
        if self.verification_record_uri is not None:
            if self.conclusion not in {Conclusion.TRUE, Conclusion.FALSE}:
                raise ValueError(
                    "verified results require a decisive mathematical conclusion"
                )
            if self.claim_digest is None or self.semantics_digest is None:
                raise ValueError(
                    "verified results require claim and semantics bindings"
                )
            if not self.evidence_uris:
                raise ValueError("verified results require bound evidence")
            if self.candidate_digest is None:
                raise ValueError("verified results require a candidate binding")
