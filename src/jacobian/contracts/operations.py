"""Model-facing contracts for extensible mathematical operations."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from jacobian.canonical import canonicalize_json
from jacobian.contracts.base import ContractModel

OperationId = Annotated[
    str,
    StringConstraints(
        pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$",
        min_length=3,
        max_length=128,
        strict=True,
    ),
]


class OperationExample(ContractModel):
    """One operator-authored, schema-valid example."""

    name: str = Field(
        pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$",
        min_length=1,
        max_length=64,
    )
    description: str = Field(min_length=1, max_length=256)
    input: dict[str, Any]

    @model_validator(mode="after")
    def require_canonical_input(self) -> Self:
        canonicalize_json(self.input)
        return self


class OperationDiscoveryRequest(ContractModel):
    """Compact installed-portfolio search, independent of any transport."""

    query: str = Field(min_length=1, max_length=512)
    domain: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,127}$",
    )
    limit: int = Field(default=5, ge=1, le=20, strict=True)
    cursor: OperationId | None = None

    @model_validator(mode="after")
    def reject_blank_filters(self) -> Self:
        if not self.query.strip():
            raise ValueError("query must contain a non-whitespace character")
        if self.domain is not None and not self.domain.strip():
            raise ValueError("domain must contain a non-whitespace character")
        return self


class OperationDiscoveryMatch(ContractModel):
    """One compact installed outcome returned by operation discovery."""

    operation_id: OperationId
    title: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=512)
    tags: tuple[str, ...] = ()
    relevance_score: int = Field(default=0, ge=0, strict=True)
    applicability: Literal[
        "INCOMPATIBLE",
        "NEEDS_MORE_TYPED_REQUIREMENTS",
    ]
    applicability_code: Literal["FULL_REQUEST_REQUIRED",]


class OperationDiscoveryResult(ContractModel):
    """Deterministically ranked compact installed outcomes."""

    discovery_version: Literal["1"] = "1"
    query: str
    domain: str | None = None
    matches: tuple[OperationDiscoveryMatch, ...]
    total_matches: int = Field(ge=0, strict=True)
    truncated: bool
    next_cursor: OperationId | None = None

    @model_validator(mode="after")
    def bind_page_metadata(self) -> Self:
        operation_ids = tuple(match.operation_id for match in self.matches)
        if len(set(operation_ids)) != len(operation_ids):
            raise ValueError("discovery matches must have unique operation IDs")
        if self.total_matches < len(self.matches):
            raise ValueError("total_matches cannot be smaller than the returned page")
        if self.truncated != (self.next_cursor is not None):
            raise ValueError("truncated must agree with next_cursor")
        if self.next_cursor is not None and (
            not operation_ids or self.next_cursor != operation_ids[-1]
        ):
            raise ValueError("next_cursor must identify the final returned match")
        return self


class OperationDescriptor(ContractModel):
    """One installed operation advertised by an operator-installed adapter."""

    descriptor_version: Literal["1"] = "1"
    operation_id: OperationId
    version: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=512)
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    read_only: bool = False
    tags: tuple[str, ...] = ()
    examples: tuple[OperationExample, ...] = ()

    @model_validator(mode="after")
    def require_canonical_schemas(self) -> Self:
        if len({example.name for example in self.examples}) != len(self.examples):
            raise ValueError("operation invocation example names must be unique")
        canonicalize_json(self.input_schema)
        canonicalize_json(self.output_schema)
        return self


class OperationResult(ContractModel):
    """The final transport envelope around one direct mathematical result."""

    operation_id: OperationId
    operation_version: str = Field(min_length=1, max_length=64)
    runtime_ms: int = Field(ge=0, strict=True)
    output: dict[str, Any]

    @model_validator(mode="after")
    def enforce_lane_and_canonical_output(self) -> Self:
        canonicalize_json(self.output)
        return self


class OperationCatalogSnapshot(ContractModel):
    catalog_version: Literal["1"] = "1"
    operations: tuple[OperationDescriptor, ...]

    @model_validator(mode="after")
    def require_unique_sorted_operations(self) -> Self:
        operation_ids = tuple(descriptor.operation_id for descriptor in self.operations)
        if operation_ids != tuple(sorted(set(operation_ids))):
            raise ValueError("catalog operation IDs must be unique and sorted")
        return self
