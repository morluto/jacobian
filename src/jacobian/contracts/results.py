"""Orthogonal operational, mathematical, and assurance result fields."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from jacobian.contracts.common import ArtifactUri, CheckerUri, Sha256Digest


class ContractModel(BaseModel):
    """Closed, immutable base for public wire models."""

    model_config = ConfigDict(extra="forbid", frozen=True)


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
    SAMPLING = "SAMPLING"
    HEURISTIC = "HEURISTIC"


class Coverage(StrEnum):
    EXHAUSTIVE = "EXHAUSTIVE"
    BOUNDED = "BOUNDED"
    RESTRICTED = "RESTRICTED"
    SAMPLED = "SAMPLED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Verification(StrEnum):
    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"


class Execution(ContractModel):
    status: ExecutionStatus
    runtime_ms: StrictInt | None = Field(default=None, ge=0)
    detail: str | None = None


class InputValidation(ContractModel):
    status: InputStatus
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def accepted_input_has_no_errors(self) -> Self:
        if self.status == InputStatus.ACCEPTED and self.errors:
            raise ValueError("accepted input cannot carry validation errors")
        if self.status == InputStatus.REJECTED and not self.errors:
            raise ValueError("rejected input requires at least one error")
        return self


class Assurance(ContractModel):
    """Arithmetic, method, coverage, and checker identity for a conclusion."""

    arithmetic: Arithmetic
    method: Method
    coverage: Coverage
    verification: Verification
    checker_id: CheckerUri | None = None
    checker_digest: Sha256Digest | None = None
    scope_uri: ArtifactUri | None = None

    @model_validator(mode="after")
    def require_checker_identity_for_verified_assurance(self) -> Self:
        if self.verification == Verification.VERIFIED:
            if self.checker_id is None or self.checker_digest is None:
                raise ValueError(
                    "verified assurance requires checker_id and checker_digest"
                )
            if self.method not in {
                Method.DIRECT_WITNESS,
                Method.EXHAUSTIVE_FINITE,
                Method.CHECKED_CERTIFICATE,
            }:
                raise ValueError(
                    "verified assurance requires a replayable verification method"
                )
            if self.arithmetic == Arithmetic.FLOATING_HEURISTIC:
                raise ValueError(
                    "floating-point heuristic arithmetic cannot be verified"
                )
            if self.coverage in {Coverage.RESTRICTED, Coverage.SAMPLED}:
                raise ValueError("restricted or sampled coverage cannot be verified")
            if (
                self.method == Method.DIRECT_WITNESS
                and self.coverage != Coverage.NOT_APPLICABLE
            ):
                raise ValueError("direct witness verification has no coverage claim")
            if (
                self.method == Method.EXHAUSTIVE_FINITE
                and self.coverage != Coverage.EXHAUSTIVE
            ):
                raise ValueError(
                    "exhaustive finite verification requires exhaustive coverage"
                )
            if self.coverage == Coverage.BOUNDED and self.scope_uri is None:
                raise ValueError(
                    "bounded verified assurance requires an explicit scope"
                )
        return self


class ResultEnvelope(ContractModel):
    """Separate execution, input, conclusion, assurance, and evidence."""

    schema_version: Literal["1"] = "1"
    execution: Execution
    input: InputValidation
    conclusion: Conclusion
    assurance: Assurance
    claim_digest: Sha256Digest | None = None
    semantics_digest: Sha256Digest | None = None
    candidate_digest: Sha256Digest | None = None
    evidence_uris: tuple[ArtifactUri, ...] = ()
    trace_uri: ArtifactUri | None = None
    verification_record_uri: ArtifactUri | None = None

    @model_validator(mode="after")
    def enforce_fail_closed_state(self) -> Self:
        execution_failed = self.execution.status != ExecutionStatus.COMPLETED
        input_rejected = self.input.status == InputStatus.REJECTED

        if execution_failed or input_rejected:
            if self.conclusion not in {
                Conclusion.UNKNOWN,
                Conclusion.NOT_APPLICABLE,
            }:
                raise ValueError(
                    "non-completed or rejected results cannot carry a "
                    "mathematical conclusion"
                )
            if self.assurance.verification != Verification.UNVERIFIED:
                raise ValueError("non-completed or rejected results cannot be verified")

        self._require_verification_evidence()
        return self

    def _require_verification_evidence(self) -> None:
        if self.assurance.verification == Verification.VERIFIED:
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
            if self.verification_record_uri is None:
                raise ValueError(
                    "verified results require an immutable verification record"
                )
            if self.candidate_digest is None:
                raise ValueError("verified results require a candidate binding")
        elif self.verification_record_uri is not None:
            raise ValueError("an unverified result cannot carry a verification record")


def validate_result_envelope(
    value: ResultEnvelope | dict[str, object],
) -> ResultEnvelope:
    """Revalidate a result at a trust or serialization boundary.

    Pydantic's ``model_construct`` deliberately bypasses validation. Public
    adapters and persistence code must call this function instead of trusting
    the runtime type of an incoming model instance.
    """

    payload = (
        value.model_dump(mode="json", warnings=False)
        if isinstance(value, ResultEnvelope)
        else value
    )
    return ResultEnvelope.model_validate(payload)
