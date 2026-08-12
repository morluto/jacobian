"""Small shared installation facts for mathematical operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jacobian.artifacts import ArtifactService
from jacobian.contracts.capabilities import CapabilityProviderRuntime
from jacobian.contracts.results import ContractModel
from jacobian.domain_bundles import DomainBundle
from jacobian.operation_bindings import InstalledOperation
from jacobian.value_references import ValueReferenceStore

type DomainOperation = InstalledOperation[Any, Any]


@dataclass(frozen=True, slots=True)
class OperationResources:
    artifacts: ArtifactService
    values: ValueReferenceStore
    semantics_uri: str
    input_schema_uris: dict[type[ContractModel], str]
    result_schema_uris: dict[str, str]


def operation_runtime(
    operation: DomainOperation,
    bundle: DomainBundle,
) -> CapabilityProviderRuntime:
    """Resolve provider identity at operation granularity."""

    return operation.provider_binding.runtime or bundle.provider_runtime


__all__ = [
    "DomainOperation",
    "OperationResources",
    "operation_runtime",
]
