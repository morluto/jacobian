"""Runtime bindings for semantic mathematical operation specifications.

The semantic declaration lives in :mod:`jacobian.operations`.  This module
pairs it with provider selection and transport-only publication policy without
making either concern part of the mathematical function contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jacobian.contracts.capabilities import (
    CapabilityInvocationExample,
    CapabilityProviderRuntime,
)
from jacobian.contracts.results import ContractModel
from jacobian.operation_ports import InputPort, OutputPort, validate_ports
from jacobian.operations import (
    OperationFailure,
    OperationRefusalError,
    OperationSpec,
)


@dataclass(frozen=True, slots=True)
class ProviderBinding:
    """Optional operation-specific provider override.

    A missing runtime means that installation binds the operation to its
    owning domain bundle's declared provider.  The resolved provider is still
    recorded on the installed descriptor and invocation provenance.
    """

    runtime: CapabilityProviderRuntime | None = None


@dataclass(frozen=True, slots=True)
class InlinePublication:
    """Publish one small bounded mathematical value inline."""

    maximum_bytes: int = 10 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.maximum_bytes < 1:
            raise ValueError("inline publication byte limit must be positive")


@dataclass(frozen=True, slots=True)
class DurablePublication[
    RequestT: ContractModel,
    ResultT: ContractModel,
    PreviewT: ContractModel,
]:
    """Publish an input/result artifact pair with an optional typed preview."""

    resource_reason: str
    preview_type: type[PreviewT] | None = None
    preview: Callable[[ResultT], PreviewT] | None = None
    preview_complete: bool = False

    def __post_init__(self) -> None:
        if not self.resource_reason.strip():
            raise ValueError("durable publication requires an explicit resource reason")
        if self.preview_complete and self.preview is None:
            raise ValueError("a complete durable preview requires a preview")


type PublicationPolicy[
    RequestT: ContractModel,
    ResultT: ContractModel,
] = InlinePublication | DurablePublication[RequestT, ResultT, ContractModel]


@dataclass(frozen=True, slots=True)
class InstalledOperation[
    RequestT: ContractModel,
    ResultT: ContractModel,
]:
    """One semantic operation bound to publication and provider selection."""

    spec: OperationSpec[RequestT, ResultT]
    publication: PublicationPolicy[RequestT, ResultT]
    provider_binding: ProviderBinding = ProviderBinding()
    input_ports: tuple[InputPort[Any], ...] = ()
    output_ports: tuple[OutputPort[Any], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.publication, DurablePublication) and self.output_ports:
            raise ValueError(
                "durable operations cannot publish request-local output references"
            )
        validate_ports(
            self.spec.request_type,
            self.spec.result_type,
            self.input_ports,
            self.output_ports,
        )


def inline_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
](
    spec: OperationSpec[RequestT, ResultT],
    *,
    provider_runtime: CapabilityProviderRuntime | None = None,
    input_ports: tuple[InputPort[Any], ...] = (),
    output_ports: tuple[OutputPort[Any], ...] = (),
) -> InstalledOperation[RequestT, ResultT]:
    """Bind a semantic operation to inline publication."""

    return InstalledOperation(
        spec=spec,
        publication=InlinePublication(),
        provider_binding=ProviderBinding(provider_runtime),
        input_ports=input_ports,
        output_ports=output_ports,
    )


def durable_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
    PreviewT: ContractModel,
](
    spec: OperationSpec[RequestT, ResultT],
    *,
    resource_reason: str,
    preview_type: type[PreviewT] | None = None,
    preview: Callable[[ResultT], PreviewT] | None = None,
    preview_complete: bool = False,
    provider_runtime: CapabilityProviderRuntime | None = None,
    input_ports: tuple[InputPort[Any], ...] = (),
    output_ports: tuple[OutputPort[Any], ...] = (),
) -> InstalledOperation[RequestT, ResultT]:
    """Bind a semantic operation to durable artifact publication."""

    return InstalledOperation(
        spec=spec,
        publication=DurablePublication(
            resource_reason=resource_reason,
            preview_type=preview_type,
            preview=preview,
            preview_complete=preview_complete,
        ),
        provider_binding=ProviderBinding(provider_runtime),
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
        invocation_examples: tuple[CapabilityInvocationExample, ...] = (),
        provider_runtime: CapabilityProviderRuntime | None = None,
        version: str = "2",
    ) -> InstalledOperation[RequestT, ResultT]:
        def execute(request: RequestT) -> ResultT:
            try:
                return operation(request)
            except self.failure.exceptions as exc:
                raise OperationRefusalError(self.failure.diagnostic(exc)) from exc

        return inline_operation(
            OperationSpec(
                operation_id=operation_id,
                version=version,
                request_type=request_type,
                result_type=result_type,
                execute=execute,
                title=title,
                description=description,
                tags=tags,
                invocation_examples=invocation_examples,
            ),
            provider_runtime=provider_runtime,
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
        invocation_examples: tuple[CapabilityInvocationExample, ...] = (),
        resource_reason: str,
        provider_runtime: CapabilityProviderRuntime | None = None,
        preview: Callable[[ResultT], ResultT] | None = None,
        preview_complete: bool = False,
        version: str = "2",
    ) -> InstalledOperation[RequestT, ResultT]:
        def execute(request: RequestT) -> ResultT:
            try:
                return operation(request)
            except self.failure.exceptions as exc:
                raise OperationRefusalError(self.failure.diagnostic(exc)) from exc

        return durable_operation(
            OperationSpec(
                operation_id=operation_id,
                version=version,
                request_type=request_type,
                result_type=result_type,
                execute=execute,
                title=title,
                description=description,
                tags=tags,
                invocation_examples=invocation_examples,
            ),
            resource_reason=resource_reason,
            provider_runtime=provider_runtime,
            preview_type=result_type,
            preview=preview,
            preview_complete=preview_complete,
        )


__all__ = [
    "DurableOperationFactory",
    "DurablePublication",
    "InlineOperationFactory",
    "InlinePublication",
    "InstalledOperation",
    "ProviderBinding",
    "PublicationPolicy",
    "durable_operation",
    "inline_operation",
]
