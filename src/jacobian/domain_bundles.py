"""Installed declarations that bind semantic operations into a domain bundle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.operations import ProviderObservation
from jacobian.operation_bindings import InstalledOperation
from jacobian.operation_declarations import OperationDeclaration
from jacobian.operations import DomainDiagnostics, DomainSemantics


@dataclass(frozen=True, slots=True)
class DomainBundle:
    """Explicit installed declaration owned by one mathematical domain."""

    domain_id: str
    schema_namespace: str
    semantics: DomainSemantics
    provider_runtime: ProviderObservation
    backend_version: str
    operations: tuple[
        OperationDeclaration[Any, Any] | InstalledOperation[Any, Any],
        ...,
    ]
    diagnostics: DomainDiagnostics
    checker_declarations: tuple[ExactReplayCheckerDeclaration, ...] = ()

    @property
    def operation_ids(self) -> tuple[str, ...]:
        """Return the operation IDs declared by this domain."""

        return tuple(
            (
                operation.operation_id
                if isinstance(operation, OperationDeclaration)
                else operation.spec.operation_id
            )
            for operation in self.operations
        )


__all__ = ["DomainBundle"]
