"""Transport publication helpers for immutable operation declarations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, overload

from jacobian.contracts.results import ContractModel
from jacobian.operation_declarations import (
    DurableOperationFactory,
    DurablePublication,
    InlineOperation,
    InlineOperationFactory,
    InlinePublication,
    OperationDeclaration,
    PublicationPolicy,
)
from jacobian.operation_ports import InputPort, OutputPort


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
    spec: OperationDeclaration[RequestT, ResultT],
    *,
    input_ports: tuple[InputPort[Any], ...] = (),
    output_ports: tuple[OutputPort[Any], ...] = (),
) -> InlineOperation[RequestT, ResultT] | OperationDeclaration[RequestT, ResultT]:
    """Bind a semantic operation to inline publication."""

    if input_ports or output_ports:
        return replace(
            spec,
            publication=InlinePublication(),
            input_ports=input_ports,
            output_ports=output_ports,
        )
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
    "DurableOperationFactory",
    "DurablePublication",
    "InlineOperation",
    "InlineOperationFactory",
    "InlinePublication",
    "PublicationPolicy",
    "durable_operation",
    "inline_operation",
]
