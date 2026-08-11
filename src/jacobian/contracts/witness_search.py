"""Contracts for untrusted adversarial witness search."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from jacobian.canonical import canonicalize_json
from jacobian.contracts.common import ArtifactUri
from jacobian.contracts.evidence import (
    FormatIdentifier,
    WitnessRole,
)
from jacobian.contracts.results import (
    Arithmetic,
    ContractModel,
    Coverage,
    ResultEnvelope,
    Verification,
)


class WitnessSearchStatus(StrEnum):
    FOUND = "FOUND"
    NONE_CERTIFIED = "NONE_CERTIFIED"
    SEARCH_EXHAUSTED = "SEARCH_EXHAUSTED"
    NOT_FOUND_WITHIN_SCOPE = "NOT_FOUND_WITHIN_SCOPE"
    UNKNOWN = "UNKNOWN"


class PluginWitnessResponse(ContractModel):
    response_version: Literal["1"] = "1"
    status: WitnessSearchStatus
    witness: dict[str, Any] | None = None
    witness_format: FormatIdentifier | None = None
    format_version: str | None = Field(default=None, min_length=1, max_length=64)
    role: WitnessRole | None = None
    certificate_uri: ArtifactUri | None = None
    arithmetic: Arithmetic
    coverage: Coverage
    detail: str = ""

    @model_validator(mode="after")
    def found_response_has_a_self_describing_witness(self) -> Self:
        if self.status == WitnessSearchStatus.FOUND:
            if (
                self.witness is None
                or self.witness_format is None
                or self.format_version is None
                or self.role is None
            ):
                raise ValueError("FOUND requires a witness, format, version, and role")
            canonicalize_json(self.witness)
            if self.certificate_uri is not None:
                raise ValueError("FOUND cannot also carry a certificate")
        elif self.status == WitnessSearchStatus.NONE_CERTIFIED:
            if self.certificate_uri is None:
                raise ValueError("NONE_CERTIFIED proposal requires a certificate URI")
            if any(
                value is not None
                for value in (
                    self.witness,
                    self.witness_format,
                    self.format_version,
                    self.role,
                )
            ):
                raise ValueError("NONE_CERTIFIED cannot carry direct witness fields")
        elif any(
            value is not None
            for value in (
                self.witness,
                self.witness_format,
                self.format_version,
                self.role,
                self.certificate_uri,
            )
        ):
            raise ValueError("non-FOUND search results cannot carry a witness")
        return self


class WitnessFindResult(ContractModel):
    schema_version: Literal["1"] = "1"
    status: WitnessSearchStatus
    result: ResultEnvelope
    claim_uri: ArtifactUri
    candidate_uri: ArtifactUri
    plugin_id: ArtifactUri
    witness_uri: ArtifactUri | None = None
    certificate_uri: ArtifactUri | None = None
    detail: str = ""

    @model_validator(mode="after")
    def evidence_matches_status(self) -> Self:
        if self.status == WitnessSearchStatus.FOUND and self.witness_uri is None:
            raise ValueError("FOUND requires a stored witness")
        if (
            self.status == WitnessSearchStatus.FOUND
            and self.certificate_uri is not None
        ):
            raise ValueError("FOUND cannot carry a no-witness certificate")
        if (
            self.status == WitnessSearchStatus.NONE_CERTIFIED
            and self.certificate_uri is None
        ):
            raise ValueError("NONE_CERTIFIED requires a verified certificate")
        if self.status == WitnessSearchStatus.NONE_CERTIFIED:
            if self.result.assurance.verification != Verification.VERIFIED:
                raise ValueError("NONE_CERTIFIED requires verified assurance")
            if self.certificate_uri not in self.result.evidence_uris:
                raise ValueError(
                    "NONE_CERTIFIED certificate must be bound as verified evidence"
                )
            if self.witness_uri is not None:
                raise ValueError("NONE_CERTIFIED cannot carry a direct witness")
        if self.status not in {
            WitnessSearchStatus.FOUND,
            WitnessSearchStatus.NONE_CERTIFIED,
        } and (self.witness_uri is not None or self.certificate_uri is not None):
            raise ValueError("non-evidentiary search status cannot carry evidence")
        return self
