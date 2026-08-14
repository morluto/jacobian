"""Transport publication helpers for immutable operation declarations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from jacobian.contracts.results import ContractModel
from jacobian.operation_declarations import (
    DurableOperationFactory,
    DurablePublication,
    InlineOperationFactory,
    InlinePublication,
    OperationDeclaration,
    PublicationPolicy,
)
from jacobian.operation_ports import InputPort, OutputPort


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


__all__ = [
    "DurableOperationFactory",
    "DurablePublication",
    "InlineOperationFactory",
    "InlinePublication",
    "PublicationPolicy",
    "durable_operation",
    "inline_operation",
]
