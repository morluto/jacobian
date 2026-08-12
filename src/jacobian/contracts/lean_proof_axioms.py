"""Domain-owned facts about a Lean proof's reported trust base."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.common import ArtifactUri, Sha256Digest
from jacobian.contracts.lean import LeanEnvironment
from jacobian.contracts.lean_statement import LeanElaborationDiagnostic
from jacobian.contracts.results import ContractModel


class LeanProofAxiomsInspectRequest(ContractModel):
    """One bounded statement/proof source to inspect in a pinned environment."""

    environment: LeanEnvironment = LeanEnvironment.CORE
    statement: str = Field(min_length=1, max_length=2_000)
    proof: str = Field(min_length=1, max_length=20_000)

    @model_validator(mode="after")
    def require_source_boundary(self) -> Self:
        if "\n" in self.statement or "\r" in self.statement:
            raise ValueError("statement must be one Lean expression")
        if ":=" in self.statement:
            raise ValueError("statement must not contain ':='")
        if any(
            marker in self.statement or marker in self.proof
            for marker in ("--", "/-", "-/")
        ):
            raise ValueError("comments are outside the source boundary")
        if "\x00" in self.proof:
            raise ValueError("proof must not contain null bytes")
        return self


class LeanProofAxiomsArtifact(ContractModel):
    """Computed Lean facts; no field in this artifact is verification authority."""

    proof_axioms_schema_version: Literal["1"] = "1"
    environment: LeanEnvironment
    environment_digest: Sha256Digest
    provider_runtime_digest: Sha256Digest
    lean_version: str = Field(min_length=1, max_length=128)
    lean_commit: str = Field(min_length=1, max_length=128)
    mathlib_commit: str | None = Field(default=None, max_length=128)
    imports: tuple[str, ...] = Field(max_length=16)
    package_manifest_digest: Sha256Digest | None = None
    statement: str
    proof: str
    elaborated: bool
    inspection_complete: bool
    axioms_reported: bool
    axioms: tuple[str, ...] = Field(max_length=128)
    sorry_count: int = Field(ge=0, le=64)
    admit_count: int = Field(ge=0, le=64)
    diagnostics: tuple[LeanElaborationDiagnostic, ...] = Field(max_length=128)
    semantic_scope: Literal["AXIOM_DEPENDENCY_ONLY"] = "AXIOM_DEPENDENCY_ONLY"

    @model_validator(mode="after")
    def validate_inspection_shape(self) -> Self:
        if tuple(sorted(self.axioms)) != self.axioms:
            raise ValueError("reported axioms must be sorted")
        if len(set(self.axioms)) != len(self.axioms):
            raise ValueError("reported axioms must be unique")
        if self.axioms_reported and not self.elaborated:
            raise ValueError("axioms can be reported only for an elaborated proof")
        if self.inspection_complete and not self.axioms_reported:
            raise ValueError("complete inspection requires a reported axiom closure")
        if self.inspection_complete and not self.elaborated:
            raise ValueError("complete inspection requires elaboration")
        return self


class LeanProofAxiomsInspectOutput(LeanProofAxiomsArtifact):
    proof_axioms_uri: ArtifactUri


__all__ = [
    "LeanProofAxiomsArtifact",
    "LeanProofAxiomsInspectOutput",
    "LeanProofAxiomsInspectRequest",
]
