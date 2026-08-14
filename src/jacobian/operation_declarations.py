"""Pure declarations for explicitly built-in mathematical operations.

Importing this module or a domain declaration module must not inspect providers, open
state, register schemas, or construct runtime services.  A declaration names
what execution requires; operator lifecycle code measures and binds those
requirements later.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, overload

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


@dataclass(frozen=True, slots=True)
class InlineOperation[RequestT: ContractModel, ResultT: ContractModel]:
    """Immutable declaration for one ordinary inline mathematical operation."""

    operation_id: str
    version: str
    title: str
    description: str
    request_type: type[RequestT]
    result_type: type[ResultT]
    run: Callable[[RequestT], ResultT]
    tags: tuple[str, ...] = ()
    examples: tuple[OperationExample, ...] = ()
    invalid_request: OperationDiagnostic | None = None
    enrich_invalid_request: bool = False

    def __post_init__(self) -> None:
        if not self.operation_id.strip() or not self.version.strip():
            raise ValueError("inline operations require an ID and version")
        if not self.title.strip() or not self.description.strip():
            raise ValueError("inline operations require title and description")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("operation tags must be unique")
        if any(not tag.strip() for tag in self.tags):
            raise ValueError("operation tags must not be empty")
        if len({example.name for example in self.examples}) != len(self.examples):
            raise ValueError("operation example names must be unique")

    @property
    def execute(self) -> Callable[[RequestT], ResultT]:
        return self.run


type OperationSpec = OperationDeclaration[Any, Any] | InlineOperation[Any, Any]
type OperationDeclarations = tuple[OperationSpec, ...]


def _declaration_for_inline(
    spec: OperationSpec,
) -> OperationDeclaration[Any, Any]:
    if isinstance(spec, OperationDeclaration):
        return spec
    return OperationDeclaration(
        operation_id=spec.operation_id,
        version=spec.version,
        title=spec.title,
        description=spec.description,
        request_type=spec.request_type,
        result_type=spec.result_type,
        execute=spec.run,
        tags=spec.tags,
        examples=spec.examples,
        invalid_request=spec.invalid_request,
        enrich_invalid_request=spec.enrich_invalid_request,
    )


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
    ) -> InlineOperation[RequestT, ResultT]:
        def run(request: RequestT) -> ResultT:
            try:
                return operation(request)
            except self.failure.exceptions as exc:
                raise OperationRefusalError(self.failure.diagnostic(exc)) from exc

        return InlineOperation(
            operation_id=operation_id,
            version=version,
            title=title,
            description=description,
            tags=tags,
            request_type=request_type,
            result_type=result_type,
            run=run,
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


@overload
def inline_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
](
    spec: OperationDeclaration[RequestT, ResultT],
) -> InlineOperation[RequestT, ResultT]: ...


@overload
def inline_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
](
    spec: OperationDeclaration[RequestT, ResultT],
    *,
    input_ports: tuple[InputPort[Any], ...],
    output_ports: tuple[OutputPort[Any], ...] = (),
) -> OperationDeclaration[RequestT, ResultT]: ...


@overload
def inline_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
](
    spec: OperationDeclaration[RequestT, ResultT],
    *,
    output_ports: tuple[OutputPort[Any], ...],
    input_ports: tuple[InputPort[Any], ...] = (),
) -> OperationDeclaration[RequestT, ResultT]: ...


def inline_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
](
    spec: OperationSpec,
    *,
    input_ports: tuple[InputPort[Any], ...] = (),
    output_ports: tuple[OutputPort[Any], ...] = (),
) -> InlineOperation[Any, Any] | OperationDeclaration[Any, Any]:
    """Bind a semantic operation to inline publication."""

    if input_ports or output_ports:
        declaration = _declaration_for_inline(spec)
        return replace(
            declaration,
            publication=InlinePublication(),
            input_ports=input_ports,
            output_ports=output_ports,
        )
    if isinstance(spec, OperationDeclaration) and spec.effect is not Effect.READ_ONLY:
        return replace(spec, publication=InlinePublication())
    if isinstance(spec, InlineOperation):
        return spec
    return InlineOperation(
        operation_id=spec.operation_id,
        version=spec.version,
        title=spec.title,
        description=spec.description,
        request_type=spec.request_type,
        result_type=spec.result_type,
        run=spec.execute,
        tags=spec.tags,
        examples=spec.examples,
        invalid_request=spec.invalid_request,
        enrich_invalid_request=spec.enrich_invalid_request,
    )


def durable_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
    PreviewT: ContractModel,
](
    spec: OperationDeclaration[RequestT, ResultT],
    *,
    resource_reason: str,
    preview_type: type[PreviewT] | None = None,
    preview: Callable[[ResultT], PreviewT] | None = None,
    preview_complete: bool = False,
    input_ports: tuple[InputPort[Any], ...] = (),
    output_ports: tuple[OutputPort[Any], ...] = (),
) -> OperationDeclaration[RequestT, ResultT]:
    """Bind a semantic operation to durable artifact publication."""

    return replace(
        spec,
        publication=DurablePublication(
            resource_reason=resource_reason,
            preview_type=preview_type,
            preview=preview,
            preview_complete=preview_complete,
        ),
        input_ports=input_ports,
        output_ports=output_ports,
    )


__all__ = [
    "SUPPORTED",
    "DurableOperationFactory",
    "DurablePublication",
    "Effect",
    "InlineOperation",
    "InlineOperationFactory",
    "InlinePublication",
    "OperationAbortError",
    "OperationDeclaration",
    "OperationDeclarations",
    "OperationExample",
    "OperationFailure",
    "OperationRefusalError",
    "OperationSpec",
    "PreflightResult",
    "PreflightStatus",
    "PublicationPolicy",
    "durable_operation",
    "inline_operation",
    "with_invalid_request",
]
