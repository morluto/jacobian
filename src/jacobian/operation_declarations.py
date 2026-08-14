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
    preflight: Callable[[RequestT], PreflightResult] | None = None

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
        preflight=spec.preflight,
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


def inline_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
](
    spec: OperationSpec,
) -> InlineOperation[Any, Any]:
    """Bind a semantic operation to inline publication."""

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
        preflight=spec.preflight if isinstance(spec, OperationDeclaration) else None,
    )


__all__ = [
    "SUPPORTED",
    "Effect",
    "InlineOperation",
    "InlineOperationFactory",
    "OperationAbortError",
    "OperationDeclaration",
    "OperationDeclarations",
    "OperationExample",
    "OperationFailure",
    "OperationRefusalError",
    "OperationSpec",
    "PreflightResult",
    "PreflightStatus",
    "inline_operation",
    "with_invalid_request",
]
