"""Authoritative typed request and response contracts for ``math.find``."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AfterValidator, ConfigDict, Field, RootModel, StrictInt

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDescriptor,
    OperationDiscoveryMatch,
    OperationId,
)


def _require_nonblank_need(value: str) -> str:
    if not value.strip():
        raise ValueError("need must contain a non-whitespace character")
    return value


OperationNeed = Annotated[
    str,
    Field(
        min_length=1,
        max_length=4_096,
        description=(
            "A concise description of the local mathematical result needed. Preserve "
            "the supplied objects and constraints, the computation or decision, and "
            "whether a value, witness, certificate, profile, or exhaustive result is "
            "required. Ordinary mathematical language is preferred to catalog tags."
        ),
    ),
    AfterValidator(_require_nonblank_need),
]
OperationNamespace = Annotated[
    str | None,
    Field(
        pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,127}$",
        description=(
            "Optional exact primary operation-ID namespace. Omit it unless already "
            "known with high confidence."
        ),
    ),
]
OperationMatchLimit = Annotated[
    StrictInt,
    Field(ge=1, le=10, description="Maximum compact matches to return."),
]
OperationCursor = Annotated[
    str | None,
    Field(
        max_length=128,
        pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$",
        description="Continuation cursor from a prior call with the same need.",
    ),
]


class OperationMatchRequest(StrictModel):
    op: Literal["match"]
    need: OperationNeed
    namespace: OperationNamespace = None
    limit: OperationMatchLimit = 5
    cursor: OperationCursor = None


class OperationInspectRequest(StrictModel):
    op: Literal["inspect"]
    operation_id: OperationId


OperationFindRequest = Annotated[
    OperationMatchRequest | OperationInspectRequest,
    Field(discriminator="op"),
]


class OperationDiscoveryErrorDetail(StrictModel):
    code: Literal["INVALID_CURSOR", "UNKNOWN_OPERATION"]
    stage: Literal["operation_discovery", "operation_resolution"]
    message: str
    hint: str


class OperationFindResult(StrictModel):
    kind: Literal["matches"]
    need: str
    namespace: str | None = None
    matches: tuple[OperationDiscoveryMatch, ...]
    total_matches: StrictInt
    next_cursor: str | None = None
    catalog_resource: Literal["operation://catalog"] = "operation://catalog"


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
        "Inspect the operation with math.find and correct the fields at the reported "
        "locations before retrying."
    )


class OperationDiscoveryError(StrictModel):
    kind: Literal["error"]
    error: OperationDiscoveryErrorDetail


class OperationFindResponse(
    RootModel[
        Annotated[
            OperationFindResult | OperationInspectionResult | OperationDiscoveryError,
            Field(discriminator="kind"),
        ]
    ]
):
    """Closed discriminated output returned directly through MCP SDK 2.0."""

    model_config = ConfigDict(json_schema_extra={"type": "object"})


__all__ = [
    "OperationCursor",
    "OperationDiscoveryError",
    "OperationDiscoveryErrorDetail",
    "OperationFindRequest",
    "OperationFindResponse",
    "OperationFindResult",
    "OperationInspectRequest",
    "OperationInspectionResult",
    "OperationInvalidRequestData",
    "OperationMatchLimit",
    "OperationMatchRequest",
    "OperationNamespace",
    "OperationNeed",
    "OperationValidationIssue",
]
