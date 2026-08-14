"""Pure declarations for explicitly built-in mathematical operations.

Importing this module or a domain declaration module must not inspect providers, open
state, register schemas, or construct runtime services.  A declaration names
what execution requires; operator lifecycle code measures and binds those
requirements later.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from jacobian.contracts.base import ContractModel
from jacobian.contracts.operations import (
    OperationDiagnostic,
    OperationExample,
)
from jacobian.operation_ports import InputPort, OutputPort, validate_ports
from jacobian.operations import (
    SUPPORTED,
    Effect,
    OperationAbortError,
    OperationFailure,
    OperationRefusalError,
    PreflightResult,
    PreflightStatus,
)


@dataclass(frozen=True, slots=True)
class InlinePublication:
    """Publish one bounded mathematical value inline."""

    maximum_bytes: int = 10 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.maximum_bytes < 1:
            raise ValueError("inline publication byte limit must be positive")


@dataclass(frozen=True, slots=True)
class DurablePublication[ResultT: ContractModel, PreviewT: ContractModel]:
    """Publish a durable result with an optional typed bounded preview."""

    resource_reason: str
    preview_type: type[PreviewT] | None = None
    preview: Callable[[ResultT], PreviewT] | None = None
    preview_complete: bool = False

    def __post_init__(self) -> None:
        if not self.resource_reason.strip():
            raise ValueError("durable publication requires an explicit reason")
        if self.preview_complete and self.preview is None:
            raise ValueError("a complete durable preview requires a preview")


type PublicationPolicy[ResultT: ContractModel] = (
    InlinePublication | DurablePublication[ResultT, ContractModel]
)


@dataclass(frozen=True, slots=True)
class OperationDeclaration[RequestT: ContractModel, ResultT: ContractModel]:
    """Immutable declaration for one built-in mathematical operation."""

    operation_id: str
    version: str
    title: str
    description: str
    request_type: type[RequestT]
    result_type: type[ResultT]
    execute: Callable[[RequestT], ResultT]
    tags: tuple[str, ...] = ()
    publication: PublicationPolicy[ResultT] = InlinePublication()
    input_ports: tuple[InputPort[Any], ...] = ()
    output_ports: tuple[OutputPort[Any], ...] = ()
    examples: tuple[OperationExample, ...] = ()
    invalid_request: OperationDiagnostic | None = None
    enrich_invalid_request: bool = False
    preflight: Callable[[RequestT], PreflightResult] | None = None
    postcondition: Callable[[RequestT, ResultT], None] | None = None
    effect: Effect = Effect.READ_ONLY

    def __post_init__(self) -> None:
        if not self.operation_id.strip() or not self.version.strip():
            raise ValueError("operation declarations require an ID and version")
        if not self.title.strip() or not self.description.strip():
            raise ValueError("operation declarations require title and description")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("operation tags must be unique")
        if any(not tag.strip() for tag in self.tags):
            raise ValueError("operation tags must not be empty")
        if len({example.name for example in self.examples}) != len(self.examples):
            raise ValueError("operation example names must be unique")
        if isinstance(self.publication, DurablePublication) and self.output_ports:
            raise ValueError(
                "durable operations cannot publish request-local output references"
            )
        validate_ports(
            self.request_type,
            self.result_type,
            self.input_ports,
            self.output_ports,
        )


type OperationDeclarations = tuple[OperationDeclaration[Any, Any], ...]


def with_invalid_request(
    operations: OperationDeclarations,
    diagnostic: OperationDiagnostic,
) -> OperationDeclarations:
    """Attach one domain-owned boundary diagnostic without wrapping execution."""

    return tuple(
        operation
        if operation.invalid_request is not None
        else replace(
            operation,
            invalid_request=diagnostic,
            enrich_invalid_request=True,
        )
        for operation in operations
    )


@dataclass(frozen=True, slots=True)
class InlineOperationFactory:
    """Declare inline operations with one provider and domain error policy."""

    failure: OperationFailure

    def __call__[
        RequestT: ContractModel,
        ResultT: ContractModel,
    ](
        self,
        operation_id: str,
        title: str,
        description: str,
        request_type: type[RequestT],
        result_type: type[ResultT],
        operation: Callable[[RequestT], ResultT],
        *tags: str,
        examples: tuple[OperationExample, ...] = (),
        version: str = "2",
    ) -> OperationDeclaration[RequestT, ResultT]:
        def execute(request: RequestT) -> ResultT:
            try:
                return operation(request)
            except self.failure.exceptions as exc:
                raise OperationRefusalError(self.failure.diagnostic(exc)) from exc

        return OperationDeclaration(
            operation_id=operation_id,
            version=version,
            title=title,
            description=description,
            tags=tags,
            request_type=request_type,
            result_type=result_type,
            execute=execute,
            publication=InlinePublication(),
            examples=examples,
        )


@dataclass(frozen=True, slots=True)
class DurableOperationFactory:
    """Declare durable operations with one provider and domain error policy."""

    failure: OperationFailure

    def __call__[
        RequestT: ContractModel,
        ResultT: ContractModel,
    ](
        self,
        operation_id: str,
        title: str,
        description: str,
        request_type: type[RequestT],
        result_type: type[ResultT],
        operation: Callable[[RequestT], ResultT],
        *tags: str,
        resource_reason: str,
        examples: tuple[OperationExample, ...] = (),
        preview: Callable[[ResultT], ResultT] | None = None,
        preview_complete: bool = False,
        version: str = "2",
    ) -> OperationDeclaration[RequestT, ResultT]:
        def execute(request: RequestT) -> ResultT:
            try:
                return operation(request)
            except self.failure.exceptions as exc:
                raise OperationRefusalError(self.failure.diagnostic(exc)) from exc

        return OperationDeclaration(
            operation_id=operation_id,
            version=version,
            title=title,
            description=description,
            tags=tags,
            request_type=request_type,
            result_type=result_type,
            execute=execute,
            publication=DurablePublication(
                resource_reason=resource_reason,
                preview_type=result_type,
                preview=preview,
                preview_complete=preview_complete,
            ),
            examples=examples,
        )


__all__ = [
    "SUPPORTED",
    "DurableOperationFactory",
    "DurablePublication",
    "Effect",
    "InlineOperationFactory",
    "InlinePublication",
    "OperationAbortError",
    "OperationDeclaration",
    "OperationDeclarations",
    "OperationExample",
    "OperationFailure",
    "OperationRefusalError",
    "PreflightResult",
    "PreflightStatus",
    "PublicationPolicy",
    "with_invalid_request",
]
