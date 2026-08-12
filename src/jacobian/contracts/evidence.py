"""Self-describing witness and certificate envelopes."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from jacobian.canonical import canonicalize_json
from jacobian.contracts.common import Sha256Digest
from jacobian.contracts.results import ContractModel

FormatIdentifier = Annotated[
    str,
    StringConstraints(
        pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$",
        min_length=3,
        max_length=128,
        strict=True,
    ),
]


class WitnessRole(StrEnum):
    DEFEATS_CANDIDATE = "DEFEATS_CANDIDATE"
    RESCUES_CANDIDATE = "RESCUES_CANDIDATE"
    SUPPORTS_CLAIM = "SUPPORTS_CLAIM"
    REFUTES_CLAIM = "REFUTES_CLAIM"


class EvidenceBindings(ContractModel):
    claim_digest: Sha256Digest
    semantics_digest: Sha256Digest
    candidate_digest: Sha256Digest | None = None
    scope_digest: Sha256Digest | None = None
    encoding_digest: Sha256Digest | None = None


class WitnessEnvelope(ContractModel):
    evidence_schema_version: Literal["1"] = "1"
    witness_format: FormatIdentifier
    format_version: str = Field(min_length=1, max_length=64)
    role: WitnessRole
    bindings: EvidenceBindings
    payload: Any

    @model_validator(mode="after")
    def witness_requires_candidate_binding(self) -> Self:
        if self.bindings.candidate_digest is None:
            raise ValueError("a direct witness must bind a candidate")
        canonicalize_json(self.payload)
        return self


class CertificateEnvelope(ContractModel):
    evidence_schema_version: Literal["1"] = "1"
    certificate_type: FormatIdentifier
    format_version: str = Field(min_length=1, max_length=64)
    bindings: EvidenceBindings
    payload_digest: Sha256Digest
    payload: Any

    @model_validator(mode="after")
    def payload_matches_declared_digest(self) -> Self:
        canonical_payload = canonicalize_json(self.payload)
        computed = "sha256:" + hashlib.sha256(canonical_payload).hexdigest()
        if computed != self.payload_digest:
            raise ValueError("certificate payload does not match payload_digest")
        return self
