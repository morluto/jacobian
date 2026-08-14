"""Authoritative typed request and response contracts for ``math.find``."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, RootModel, StrictInt

from jacobian.contracts.base import ContractModel
from jacobian.contracts.common import ArtifactUri
from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationDiscoveryMatch,
    OperationId,
    OperationInputKind,
    OperationValuePort,
    ProviderAvailability,
)


class OperationSearchRequest(ContractModel):
    op: Literal["search"]
    query: Annotated[str, Field(min_length=1, max_length=512)]
    domain: Annotated[
        str | None,
        Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,127}$"),
    ] = None
    input_kind: OperationInputKind | None = None
    artifact_type: Annotated[
        str | None,
        Field(pattern=r"^artifact://sha256/[0-9a-f]{64}$"),
    ] = None
    limit: Annotated[StrictInt, Field(ge=1, le=20)] = 5
    cursor: Annotated[
        str | None,
        Field(
            max_length=128,
            pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$",
        ),
    ] = None


class OperationInspectRequest(ContractModel):
    op: Literal["inspect"]
    operation_id: OperationId


OperationFindRequest = Annotated[
    OperationSearchRequest | OperationInspectRequest,
    Field(discriminator="op"),
]


class OperationFindCallArguments(ContractModel):
    request: OperationSearchRequest


class OperationSearchRecoveryPath(ContractModel):
    action: Literal["search"]
    tool: Literal["math.find"] = "math.find"
    arguments: OperationFindCallArguments


class OperationCatalogPointer(ContractModel):
    action: Literal["inspect_catalog"]
    resource_uri: Literal["operation://catalog"] = "operation://catalog"


OperationErrorRecoveryPath = OperationSearchRecoveryPath | OperationCatalogPointer


class OperationDiscoveryErrorDetail(ContractModel):
    code: Literal["INVALID_CURSOR", "UNKNOWN_OPERATION"]
    stage: Literal["operation_discovery", "operation_resolution"]
    message: str
    hint: str
    nearby_operation_ids: tuple[OperationId, ...] = ()
    available_recovery_paths: tuple[OperationErrorRecoveryPath, ...] = ()


class OperationDiscoveryCard(OperationDiscoveryMatch):
    accepted_input_kinds: tuple[OperationInputKind, ...]
    accepted_artifact_types: tuple[ArtifactUri, ...]
    produced_artifact_types: tuple[ArtifactUri, ...]
    input_ports: tuple[OperationValuePort, ...]
    output_ports: tuple[OperationValuePort, ...]
    provider_availability: ProviderAvailability | Literal["UNKNOWN"]


class OperationSearchResult(ContractModel):
    kind: Literal["discovery"]
    discovery_version: Literal["1"]
    query: str
    domain: str | None = None
    input_kind: OperationInputKind | None = None
    artifact_type: str | None = None
    matches: tuple[OperationDiscoveryCard, ...]
    total_matches: StrictInt
    truncated: bool
    next_cursor: str | None = None
    catalog_resource: Literal["operation://catalog"] = "operation://catalog"
    response_byte_limit: StrictInt
    truncation_reason: str | None = None
    match_metadata_truncated: bool = False


class OperationInspectionResult(ContractModel):
    kind: Literal["operation"]
    operation: OperationDescriptor


class OperationDiscoveryError(ContractModel):
    kind: Literal["error"]
    error: OperationDiscoveryErrorDetail


class OperationFindResponse(
    RootModel[
        Annotated[
            OperationSearchResult | OperationInspectionResult | OperationDiscoveryError,
            Field(discriminator="kind"),
        ]
    ]
):
    """Closed discriminated output returned directly through MCP SDK 2.0."""

    model_config = ConfigDict(json_schema_extra={"type": "object"})


__all__ = [
    "OperationCatalogPointer",
    "OperationDiscoveryCard",
    "OperationDiscoveryError",
    "OperationDiscoveryErrorDetail",
    "OperationFindRequest",
    "OperationFindResponse",
    "OperationInspectRequest",
    "OperationInspectionResult",
    "OperationSearchRecoveryPath",
    "OperationSearchRequest",
    "OperationSearchResult",
]
