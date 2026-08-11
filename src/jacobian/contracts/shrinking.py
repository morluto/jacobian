"""Contracts for preservation-checked candidate and witness shrinking."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import Field, StrictBool, StrictInt, model_validator

from jacobian.canonical import canonicalize_json
from jacobian.contracts.common import ArtifactUri
from jacobian.contracts.results import (
    ContractModel,
    Execution,
    ExecutionStatus,
    InputStatus,
    InputValidation,
    ResultEnvelope,
)


class ShrinkTargetKind(StrEnum):
    CANDIDATE = "candidate"
    WITNESS = "witness"


class Minimality(StrEnum):
    NONE = "NONE"
    LOCAL = "LOCAL"
    ONE_STEP = "ONE_STEP"
    BOUNDED_GLOBAL = "BOUNDED_GLOBAL"
    PROVED_GLOBAL = "PROVED_GLOBAL"


class ReductionProposal(ContractModel):
    reducer: str = Field(min_length=1, max_length=128)
    payload: Any
    objectives: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_canonical_payload(self) -> Self:
        canonicalize_json(self.payload)
        canonicalize_json(self.objectives)
        return self


class PluginReductionResponse(ContractModel):
    response_version: Literal["1"] = "1"
    current_objectives: dict[str, Any] = Field(default_factory=dict)
    reductions: tuple[ReductionProposal, ...] = ()
    detail: str = ""

    @model_validator(mode="after")
    def require_canonical_current_objectives(self) -> Self:
        canonicalize_json(self.current_objectives)
        return self


class ShrinkStep(ContractModel):
    index: StrictInt = Field(ge=0)
    reducer: str
    from_uri: ArtifactUri
    proposed_uri: ArtifactUri | None = None
    accepted: StrictBool
    execution_status: ExecutionStatus | None = None
    input_status: InputStatus | None = None
    verification_record_uri: ArtifactUri | None = None
    objectives: dict[str, Any] = Field(default_factory=dict)
    detail: str = ""

    @model_validator(mode="after")
    def bind_acceptance_to_verified_proposal(self) -> Self:
        canonicalize_json(self.objectives)
        if self.accepted:
            if (
                self.proposed_uri is None
                or self.execution_status is not ExecutionStatus.COMPLETED
                or self.input_status is not InputStatus.ACCEPTED
                or self.verification_record_uri is None
            ):
                raise ValueError(
                    "an accepted shrink step requires a completed accepted proposal record"
                )
        elif self.verification_record_uri is not None:
            raise ValueError(
                "a rejected shrink step cannot carry a verification record"
            )
        return self


class ShrinkResult(ContractModel):
    schema_version: Literal["1"] = "1"
    execution: Execution
    input: InputValidation
    result: ResultEnvelope
    target_kind: ShrinkTargetKind
    initial_target_uri: ArtifactUri
    final_target_uri: ArtifactUri
    minimality: Minimality
    evaluations: StrictInt = Field(ge=0)
    steps: tuple[ShrinkStep, ...] = ()
    objectives: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def v01_reports_only_implemented_minimality_levels(self) -> Self:
        if self.minimality not in {Minimality.NONE, Minimality.LOCAL}:
            raise ValueError(
                "the current contract supports only NONE and LOCAL minimality; stronger claims "
                "require independently checked neighborhood or global evidence"
            )
        return self
