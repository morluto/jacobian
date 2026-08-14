"""Authoritative typed request and response contracts for ``math.find``."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, RootModel, StrictInt

from jacobian.contracts.base import ContractModel
from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationDiscoveryMatch,
    OperationId,
)


class OperationSearchRequest(ContractModel):
    op: Literal["search"]
    query: Annotated[str, Field(min_length=1, max_length=512)]
    domain: Annotated[
        str | None,
        Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,127}$"),
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


class OperationDiscoveryErrorDetail(ContractModel):
    code: Literal["INVALID_CURSOR", "UNKNOWN_OPERATION"]
    stage: Literal["operation_discovery", "operation_resolution"]
    message: str
    hint: str


class OperationSearchResult(ContractModel):
    kind: Literal["discovery"]
    discovery_version: Literal["1"]
    query: str
    domain: str | None = None
    matches: tuple[OperationDiscoveryMatch, ...]
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
    "OperationDiscoveryError",
    "OperationDiscoveryErrorDetail",
    "OperationFindRequest",
    "OperationFindResponse",
    "OperationInspectRequest",
    "OperationInspectionResult",
    "OperationSearchRequest",
    "OperationSearchResult",
]
