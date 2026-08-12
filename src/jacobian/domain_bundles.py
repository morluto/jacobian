"""Installed declarations that bind semantic operations into a domain bundle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.capabilities import CapabilityProviderRuntime
from jacobian.operation_bindings import InstalledOperation
from jacobian.operations import DomainDiagnostics, DomainSemantics


@dataclass(frozen=True, slots=True)
class DomainBundle:
    """Explicit installed declaration owned by one mathematical domain."""

    domain_id: str
    schema_namespace: str
    semantics: DomainSemantics
    provider_runtime: CapabilityProviderRuntime
    backend_version: str
    capabilities: tuple[InstalledOperation[Any, Any], ...]
    diagnostics: DomainDiagnostics
    checker_declarations: tuple[ExactReplayCheckerDeclaration, ...] = ()

    @property
    def capability_ids(self) -> tuple[str, ...]:
        """Return the capability IDs declared by exactly one installation mode."""

        return tuple(operation.spec.operation_id for operation in self.capabilities)


__all__ = ["DomainBundle"]
