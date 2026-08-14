"""Small shared installation facts for mathematical operations."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.artifacts import ArtifactService
from jacobian.contracts.results import ContractModel
from jacobian.operation_declarations import OperationSpec
from jacobian.value_references import ValueReferenceStore

type DomainOperation = OperationSpec


@dataclass(frozen=True, slots=True)
class OperationResources:
    artifacts: ArtifactService
    values: ValueReferenceStore
    semantics_uri: str
    input_schema_uris: dict[type[ContractModel], str]
    result_schema_uris: dict[str, str]


__all__ = [
    "DomainOperation",
    "OperationResources",
]
