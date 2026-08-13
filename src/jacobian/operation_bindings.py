"""Runtime bindings for semantic mathematical operation specifications.

The semantic declaration lives in :mod:`jacobian.operations`.  This module
pairs it with provider selection and transport-only publication policy without
making either concern part of the mathematical function contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from jacobian.contracts.operations import OperationExample
from jacobian.contracts.results import ContractModel
from jacobian.operation_declarations import (
    DurablePublication,
    InlinePublication,
    OperationDeclaration,
    PublicationPolicy,
)
from jacobian.operation_ports import InputPort, OutputPort
from jacobian.operations import (
    OperationFailure,
    OperationRefusalError,
)


def inline_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
](
    spec: OperationDeclaration[RequestT, ResultT],
    *,
    input_ports: tuple[InputPort[Any], ...] = (),
    output_ports: tuple[OutputPort[Any], ...] = (),
) -> OperationDeclaration[RequestT, ResultT]:
    """Bind a semantic operation to inline publication."""

    return replace(
        spec,
        publication=InlinePublication(),
        input_ports=input_ports,
        output_ports=output_ports,
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


@dataclass(frozen=True, slots=True)
class InlineOperationFactory:
    """Build inline installed operations with one domain error policy."""

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

        return inline_operation(
            OperationDeclaration(
                operation_id=operation_id,
                version=version,
                request_type=request_type,
                result_type=result_type,
                execute=execute,
                title=title,
                description=description,
                tags=tags,
                examples=examples,
            ),
        )


@dataclass(frozen=True, slots=True)
class DurableOperationFactory:
    """Build durable installed operations with one domain error policy."""

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
        resource_reason: str,
        preview: Callable[[ResultT], ResultT] | None = None,
        preview_complete: bool = False,
        version: str = "2",
    ) -> OperationDeclaration[RequestT, ResultT]:
        def execute(request: RequestT) -> ResultT:
            try:
                return operation(request)
            except self.failure.exceptions as exc:
                raise OperationRefusalError(self.failure.diagnostic(exc)) from exc

        return durable_operation(
            OperationDeclaration(
                operation_id=operation_id,
                version=version,
                request_type=request_type,
                result_type=result_type,
                execute=execute,
                title=title,
                description=description,
                tags=tags,
                examples=examples,
            ),
            resource_reason=resource_reason,
            preview_type=result_type,
            preview=preview,
            preview_complete=preview_complete,
        )


__all__ = [
    "DurableOperationFactory",
    "DurablePublication",
    "InlineOperationFactory",
    "InlinePublication",
    "PublicationPolicy",
    "durable_operation",
    "inline_operation",
]
