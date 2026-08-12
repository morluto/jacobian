"""Contracts for checker-backed validation of an exact Lean proof edit."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictBool, model_validator

from jacobian.contracts.common import ArtifactUri
from jacobian.contracts.lean import LeanDiagnostic, LeanEnvironment
from jacobian.contracts.results import ContractModel, ExecutionStatus


class LeanProofEditRequest(ContractModel):
    environment: LeanEnvironment = LeanEnvironment.CORE
    statement: str = Field(min_length=1, max_length=2_000)
    original_proof: str = Field(min_length=1, max_length=20_000)
    edited_proof: str = Field(min_length=1, max_length=20_000)

    @model_validator(mode="after")
    def require_exact_edit(self) -> Self:
        if "\n" in self.statement or "\r" in self.statement or ":=" in self.statement:
            raise ValueError("statement must be one Lean expression")
        if self.original_proof == self.edited_proof:
            raise ValueError("edited_proof must differ from original_proof")
        return self


class LeanProofEditArtifact(ContractModel):
    proof_edit_schema_version: Literal["3"] = "3"
    environment: LeanEnvironment
    statement: str
    original_proof: str
    edited_proof: str
    unified_diff: str = Field(min_length=1, max_length=50_000)
    baseline_checker_execution_status: ExecutionStatus
    baseline_accepted: StrictBool
    baseline_diagnostics: tuple[LeanDiagnostic, ...] = ()
    baseline_candidate_uri: ArtifactUri
    baseline_certificate_uri: ArtifactUri
    baseline_verification_record_uri: ArtifactUri | None = None
    checker_execution_status: ExecutionStatus
    accepted: StrictBool
    diagnostics: tuple[LeanDiagnostic, ...] = ()
    claim_uri: ArtifactUri
    candidate_uri: ArtifactUri
    certificate_uri: ArtifactUri
    verification_record_uri: ArtifactUri | None = None

    @model_validator(mode="after")
    def bind_acceptance_to_verification(self) -> Self:
        if self.baseline_accepted and self.baseline_verification_record_uri is None:
            raise ValueError("accepted proof baselines require a verification record")
        if (
            self.baseline_accepted
            and self.baseline_checker_execution_status is not ExecutionStatus.COMPLETED
        ):
            raise ValueError("baseline acceptance requires completed checker execution")
        if self.accepted and not self.baseline_accepted:
            raise ValueError("an accepted proof edit requires an accepted baseline")
        if self.accepted and self.verification_record_uri is None:
            raise ValueError("accepted proof edits require a verification record")
        if (
            self.accepted
            and self.checker_execution_status is not ExecutionStatus.COMPLETED
        ):
            raise ValueError(
                "proof-edit acceptance requires completed checker execution"
            )
        return self


class LeanProofEditOutput(LeanProofEditArtifact):
    proof_edit_uri: ArtifactUri
