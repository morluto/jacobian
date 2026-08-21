"""Authoritative typed request and response contracts for ``math.find``."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, RootModel, StrictInt

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationBrowseCard,
    OperationDescriptor,
    OperationDiscoveryMatch,
    OperationId,
)


class OperationSearchRequest(StrictModel):
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


class OperationBrowseRequest(StrictModel):
    op: Literal["browse"]
    domain: Annotated[
        str | None,
        Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,127}$"),
    ] = None
    limit: Annotated[StrictInt, Field(ge=1, le=20)] = 20
    cursor: Annotated[
        str | None,
        Field(
            max_length=128,
            pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$",
        ),
    ] = None


class OperationInspectRequest(StrictModel):
    op: Literal["inspect"]
    operation_id: OperationId


OperationFindRequest = Annotated[
    OperationSearchRequest | OperationBrowseRequest | OperationInspectRequest,
    Field(discriminator="op"),
]


class OperationDiscoveryErrorDetail(StrictModel):
    code: Literal["INVALID_CURSOR", "UNKNOWN_OPERATION"]
    stage: Literal["operation_discovery", "operation_resolution"]
    message: str
    hint: str


class OperationSearchResult(StrictModel):
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


class OperationBrowseResult(StrictModel):
    kind: Literal["browse"]
    discovery_version: Literal["1"]
    domain: str | None = None
    operations: tuple[OperationBrowseCard, ...]
    total_operations: StrictInt
    truncated: bool
    next_cursor: str | None = None
    catalog_resource: Literal["operation://catalog"] = "operation://catalog"
    response_byte_limit: StrictInt
    truncation_reason: str | None = None
    operation_metadata_truncated: bool = False


class OperationInspectionResult(StrictModel):
    kind: Literal["operation"]
    operation: OperationDescriptor


class OperationValidationIssue(StrictModel):
    """One field-level recovery item for a selected operation payload."""

    location: tuple[Annotated[str, Field(max_length=128)] | StrictInt, ...] = Field(
        max_length=32
    )
    code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=1_024)
    input: Any | None = None


class OperationInvalidRequestData(StrictModel):
    """Structured MCP error data for operation-owned request validation."""

    code: Literal["INVALID_REQUEST"] = "INVALID_REQUEST"
    stage: Literal["operation_validation"] = "operation_validation"
    operation_id: OperationId
    errors: tuple[OperationValidationIssue, ...] = Field(
        min_length=1,
        max_length=64,
    )
    hint: str = (
        "Inspect the operation with math.find and correct the fields at the "
        "reported locations before retrying."
    )


class OperationDiscoveryError(StrictModel):
    kind: Literal["error"]
    error: OperationDiscoveryErrorDetail


class OperationFindResponse(
    RootModel[
        Annotated[
            OperationSearchResult
            | OperationBrowseResult
            | OperationInspectionResult
            | OperationDiscoveryError,
            Field(discriminator="kind"),
        ]
    ]
):
    """Closed discriminated output returned directly through MCP SDK 2.0."""

    model_config = ConfigDict(json_schema_extra={"type": "object"})


__all__ = [
    "OperationBrowseRequest",
    "OperationBrowseResult",
    "OperationDiscoveryError",
    "OperationDiscoveryErrorDetail",
    "OperationFindRequest",
    "OperationFindResponse",
    "OperationInspectRequest",
    "OperationInspectionResult",
    "OperationInvalidRequestData",
    "OperationSearchRequest",
    "OperationSearchResult",
    "OperationValidationIssue",
]
