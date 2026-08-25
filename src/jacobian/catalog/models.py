"""Typed declarations and discovery values for mathematical operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import canonicalize_json

OperationId = Annotated[
    str,
    StringConstraints(
        pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$",
        min_length=3,
        max_length=128,
        strict=True,
    ),
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"catalog.{reason}", message)


class OperationExample(StrictModel):
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


class OperationDiscoveryRequest(StrictModel):
    """Compact installed-portfolio search, independent of any transport."""

    query: str = Field(min_length=1)
    domain: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,127}$",
    )
    limit: int = Field(default=5, ge=1, le=20, strict=True)
    cursor: OperationId | None = None

    @model_validator(mode="after")
    def reject_blank_filters(self) -> Self:
        if not self.query.strip():
            raise _validation_error(
                "blank_query", "query must contain a non-whitespace character"
            )
        if self.domain is not None and not self.domain.strip():
            raise _validation_error(
                "blank_domain", "domain must contain a non-whitespace character"
            )
        return self


class OperationDiscoveryMatch(StrictModel):
    """One compact installed outcome returned by operation discovery."""

    operation_id: OperationId
    title: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1)
    tags: tuple[str, ...] = ()
    relevance_score: int = Field(default=0, ge=0, strict=True)
    applicability: Literal[
        "INCOMPATIBLE",
        "NEEDS_MORE_TYPED_REQUIREMENTS",
    ]
    applicability_code: Literal["FULL_REQUEST_REQUIRED",]


class OperationDiscoveryResult(StrictModel):
    """Deterministically ranked compact installed outcomes."""

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
            raise _validation_error(
                "duplicate_match_id", "discovery matches must have unique operation IDs"
            )
        if self.total_matches < len(self.matches):
            raise _validation_error(
                "match_count", "total_matches cannot be smaller than the returned page"
            )
        if self.truncated != (self.next_cursor is not None):
            raise _validation_error(
                "cursor_state", "truncated must agree with next_cursor"
            )
        if self.next_cursor is not None and (
            not operation_ids or self.next_cursor != operation_ids[-1]
        ):
            raise _validation_error(
                "cursor_position", "next_cursor must identify the final returned match"
            )
        return self


class OperationBrowseCard(StrictModel):
    """One compact operation card in deterministic catalog order."""

    operation_id: OperationId
    title: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1)
    tags: tuple[str, ...] = ()


class OperationBrowseResult(StrictModel):
    """One cursor-paged, unranked view of the immutable operation library."""

    domain: str | None = None
    operations: tuple[OperationBrowseCard, ...]
    total_operations: int = Field(ge=0, strict=True)
    truncated: bool
    next_cursor: OperationId | None = None

    @model_validator(mode="after")
    def bind_page_metadata(self) -> Self:
        operation_ids = tuple(operation.operation_id for operation in self.operations)
        if operation_ids != tuple(sorted(set(operation_ids))):
            raise _validation_error(
                "browse_order",
                "browse operations must have unique sorted operation IDs",
            )
        if self.total_operations < len(self.operations):
            raise _validation_error(
                "browse_count",
                "total_operations cannot be smaller than the returned page",
            )
        if self.truncated != (self.next_cursor is not None):
            raise _validation_error(
                "cursor_state", "truncated must agree with next_cursor"
            )
        if self.next_cursor is not None and (
            not operation_ids or self.next_cursor != operation_ids[-1]
        ):
            raise _validation_error(
                "cursor_position",
                "next_cursor must identify the final returned operation",
            )
        return self


class OperationDescriptor(StrictModel):
    """One installed operation advertised by the immutable catalog."""

    operation_id: OperationId
    title: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1)
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    read_only: bool = False
    tags: tuple[str, ...] = ()
    examples: tuple[OperationExample, ...] = ()

    @model_validator(mode="after")
    def require_canonical_schemas(self) -> Self:
        if len({example.name for example in self.examples}) != len(self.examples):
            raise _validation_error(
                "duplicate_example_name",
                "operation invocation example names must be unique",
            )
        canonicalize_json(self.input_schema)
        canonicalize_json(self.output_schema)
        return self


class OperationResult(StrictModel):
    """The final transport envelope around one direct mathematical result."""

    operation_id: OperationId
    runtime_ms: int = Field(ge=0, strict=True)
    output: dict[str, Any]

    @model_validator(mode="after")
    def require_canonical_output(self) -> Self:
        canonicalize_json(self.output)
        return self


class OperationCatalogSnapshot(StrictModel):
    operations: tuple[OperationDescriptor, ...]

    @model_validator(mode="after")
    def require_unique_sorted_operations(self) -> Self:
        operation_ids = tuple(descriptor.operation_id for descriptor in self.operations)
        if operation_ids != tuple(sorted(set(operation_ids))):
            raise _validation_error(
                "operation_order", "catalog operation IDs must be unique and sorted"
            )
        return self


@dataclass(frozen=True, slots=True)
class MathTool[RequestT: StrictModel, ResultT: StrictModel]:
    """One discoverable mathematical function and its public typed contract."""

    operation_id: str
    title: str
    description: str
    request_type: type[RequestT]
    result_type: type[ResultT]
    run: Callable[[RequestT], ResultT]
    tags: tuple[str, ...] = ()
    examples: tuple[OperationExample, ...] = ()

    def __post_init__(self) -> None:
        if not self.operation_id.strip():
            raise ValueError("math tools require an ID")
        if not self.title.strip() or not self.description.strip():
            raise ValueError("math tools require title and description")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("math tool tags must be unique")
        if any(not tag.strip() for tag in self.tags):
            raise ValueError("math tool tags must not be empty")
        if len({example.name for example in self.examples}) != len(self.examples):
            raise ValueError("math tool example names must be unique")


type MathTools = tuple[MathTool[Any, Any], ...]

__all__ = [
    "MathTool",
    "MathTools",
    "OperationBrowseCard",
    "OperationBrowseResult",
    "OperationCatalogSnapshot",
    "OperationDescriptor",
    "OperationDiscoveryMatch",
    "OperationDiscoveryRequest",
    "OperationDiscoveryResult",
    "OperationExample",
    "OperationId",
    "OperationResult",
]
