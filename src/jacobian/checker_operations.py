"""Typed declarations for operator-authorized checker implementations."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.contracts.capabilities import CapabilityProviderRuntime
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.results import ContractModel


@dataclass(frozen=True, slots=True)
class ExactReplayCheckerDeclaration:
    """Domain-owned declaration of an independently replayable exact result."""

    capability_id: str
    request_model: type[ContractModel]
    function: str
    format_id: str
    entrypoint_module: str = "jacobian_checkers.exact_domain_operations"
    replay_method: str = "Python-FLINT exact replay"
    reason: str = (
        "operator-authorized Python-FLINT exact replay independent of the "
        "SymPy producer"
    )
    verification_capability_id: str | None = None
    verification_title: str | None = None
    verification_description: str | None = None
    verification_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field, value in {
            "capability_id": self.capability_id,
            "function": self.function,
            "format_id": self.format_id,
            "entrypoint_module": self.entrypoint_module,
            "replay_method": self.replay_method,
            "reason": self.reason,
        }.items():
            if not value.strip():
                raise ValueError(
                    f"exact replay checker declaration {field} must not be empty"
                )
        verification_fields = (
            self.verification_capability_id,
            self.verification_title,
            self.verification_description,
        )
        if any(value is not None for value in verification_fields) and not all(
            isinstance(value, str) and value.strip() for value in verification_fields
        ):
            raise ValueError(
                "verification capability ID, title, and description must be "
                "declared together"
            )
        if self.verification_tags and self.verification_capability_id is None:
            raise ValueError(
                "verification tags require verification capability metadata"
            )


@dataclass(frozen=True, slots=True)
class CheckerOperation:
    """One independently executable checker and its compatibility scope."""

    name: str
    entrypoint: str
    evidence_kind: EvidenceKind
    format_id: str
    format_version: str
    claim_schema_uris: tuple[str, ...]
    semantics_uris: tuple[str, ...]
    candidate_schema_uris: tuple[str, ...]
    reason: str
    target_schema_uris: tuple[str, ...] = ()
    target_semantics_uris: tuple[str, ...] = ()
    provider_runtime: CapabilityProviderRuntime | None = None

    def __post_init__(self) -> None:
        required_text = {
            "name": self.name,
            "entrypoint": self.entrypoint,
            "format_id": self.format_id,
            "format_version": self.format_version,
            "reason": self.reason,
        }
        for field, value in required_text.items():
            if not value.strip():
                raise ValueError(f"checker operation {field} must not be empty")
        if not self.claim_schema_uris:
            raise ValueError("checker operation must declare a claim schema")
        if not self.semantics_uris:
            raise ValueError("checker operation must declare semantics")


@dataclass(frozen=True, slots=True)
class InstalledChecker:
    """Authorization result for one checker operation."""

    operation: CheckerOperation
    checker_id: str | None

    @property
    def authorized(self) -> bool:
        return self.checker_id is not None

    def require_checker_id(self) -> str:
        if self.checker_id is None:
            raise ValueError("checker operation is not authorized")
        return self.checker_id
