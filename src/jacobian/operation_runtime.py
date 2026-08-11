"""Small shared installation facts for mathematical operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.contracts.capabilities import (
    CapabilityDiagnostic,
    CapabilityProviderRuntime,
)
from jacobian.contracts.results import ContractModel
from jacobian.operation_bindings import InstalledOperation
from jacobian.operations import DomainBundle

type DomainOperation = InstalledOperation[Any, Any]


@dataclass(frozen=True, slots=True)
class OperationResources:
    artifacts: ArtifactService
    semantics_uri: str
    input_schema_uris: dict[type[ContractModel], str]
    result_schema_uris: dict[str, str]


def operation_runtime(
    operation: DomainOperation,
    bundle: DomainBundle,
) -> CapabilityProviderRuntime:
    """Resolve provider identity at operation granularity."""

    return operation.provider_binding.runtime or bundle.provider_runtime


def enriched_invalid_request(
    base: CapabilityDiagnostic,
    exc: ValidationError,
) -> CapabilityDiagnostic:
    """Add the first Pydantic error location to a bundle diagnostic."""

    errors = exc.errors()
    if not errors:
        return base
    first = errors[0]
    loc = first.get("loc", ())
    path = "/".join(str(part) for part in loc) if loc else None
    message = str(first.get("msg", ""))
    if message.startswith("Value error, "):
        message = message[len("Value error, ") :]
    return base.model_copy(
        update={
            "path": path,
            "hint": message if message else base.hint,
        }
    )


__all__ = [
    "DomainOperation",
    "OperationResources",
    "enriched_invalid_request",
    "operation_runtime",
]
