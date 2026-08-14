"""Direct execution for ordinary inline mathematical operations.

This module is the serving path for ``InlineOperation`` IDs. It must not import
storage, checkers, SAT/SMT, Lean, MCP, or tenant isolation.
"""

from __future__ import annotations

import time
from typing import Any, cast

from pydantic import ValidationError

from jacobian import __version__
from jacobian.canonical import CanonicalizationError, encode_strict_json
from jacobian.contracts.domain_operations import InlineOperationOutput
from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationDiagnostic,
    OperationExample,
    OperationRequest,
)
from jacobian.contracts.results import ContractModel
from jacobian.operation_adapters import parse_operation_input
from jacobian.operation_declarations import InlineOperation
from jacobian.operation_declarations import (
    PreflightStatus as DeclarationPreflightStatus,
)
from jacobian.operation_errors import (
    OperationInvocationError,
    enriched_invalid_request,
)
from jacobian.operation_execution import OperationTerminal
from jacobian.operation_projection import OperationProjection, PublishedOperation
from jacobian.operations import (
    Completed,
    Failed,
    NonConclusion,
    OperationAbortError,
    OperationRefusalError,
    PreflightStatus,
)
from jacobian.schema_compiler import SCHEMA_COMPILER


def inline_output_schema(result_type: type[ContractModel]) -> dict[str, Any]:
    """Return the public inline envelope schema for one result type."""

    output_contract: Any = InlineOperationOutput
    return SCHEMA_COMPILER.compile_model(
        cast(type[ContractModel], output_contract[result_type])
    ).definition()


def inline_operation_descriptor(
    operation: InlineOperation[Any, Any],
) -> OperationDescriptor:
    """Describe one inline operation without constructing a binder adapter."""

    return OperationDescriptor(
        operation_id=operation.operation_id,
        version=operation.version,
        title=operation.title,
        description=operation.description,
        provider="built-in",
        input_schema=SCHEMA_COMPILER.compile_model(operation.request_type).definition(),
        output_schema=inline_output_schema(operation.result_type),
        read_only=True,
        tags=operation.tags,
        examples=tuple(
            OperationExample(
                name=example.name,
                description=example.description,
                input=dict(example.input),
            )
            for example in operation.examples
        ),
    )


def run_inline[RequestT: ContractModel, ResultT: ContractModel](
    operation: InlineOperation[RequestT, ResultT],
    request: RequestT,
) -> OperationTerminal[ResultT]:
    """Run preflight (if declared), execution, and result validation once."""

    if operation.preflight is not None:
        preflight = operation.preflight(request)
        if preflight.status not in {
            PreflightStatus.SUPPORTED,
            DeclarationPreflightStatus.SUPPORTED,
        }:
            return NonConclusion(
                OperationDiagnostic(
                    code=preflight.status.value,
                    stage="operation_preflight",
                    message=preflight.reason or "Operation preflight rejected.",
                )
            )

    started = time.monotonic()
    try:
        outcome = operation.run(request)
    except OperationRefusalError as exc:
        return NonConclusion(exc.diagnostic)
    except OperationAbortError as exc:
        return Failed(status=exc.status, diagnostic=exc.diagnostic)
    result = operation.result_type.model_validate(outcome)
    return Completed(
        value=result,
        runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
    )


class InlineOperationAdapter:
    """Execute one inline operation without publication, ports, or a binder."""

    def __init__(self, operation: InlineOperation[Any, Any]) -> None:
        self.operation = operation
        self._descriptor = inline_operation_descriptor(operation)

    @property
    def descriptor(self) -> OperationDescriptor:
        return self._descriptor

    def prepare(self, request: OperationRequest) -> ContractModel:
        try:
            encode_strict_json(request.input)
        except CanonicalizationError as exc:
            raise OperationInvocationError(
                OperationDiagnostic(
                    code="REQUEST_RESOURCE_LIMIT_EXCEEDED",
                    stage="operation_request",
                    message=str(exc),
                    hint="Reduce the request to the operation's published bound.",
                )
            ) from exc
        try:
            return cast(
                ContractModel,
                parse_operation_input(self.operation.request_type, request.input),
            )
        except ValidationError as exc:
            invalid_request = self.operation.invalid_request
            base = invalid_request or OperationDiagnostic(
                code="INVALID_REQUEST",
                stage="operation_input_validation",
                message="The operation request is invalid.",
                hint="Inspect the operation schema and provide a strictly valid request.",
            )
            raise OperationInvocationError(
                enriched_invalid_request(base, exc)
                if self.operation.enrich_invalid_request or invalid_request is None
                else base
            ) from exc

    def invoke(self, prepared: ContractModel) -> OperationProjection:
        try:
            terminal = run_inline(self.operation, prepared)
        except ValidationError as exc:
            raise OperationInvocationError(
                OperationDiagnostic(
                    code="ADAPTER_EXECUTION_FAILED",
                    stage="operation_result_validation",
                    message="The operation returned an invalid typed result.",
                    hint="Fix the operation implementation to satisfy its result contract.",
                )
            ) from exc
        publication = None
        if isinstance(terminal, Completed):
            output_contract: Any = InlineOperationOutput
            publication = PublishedOperation(
                output=cast(
                    type[ContractModel], output_contract[self.operation.result_type]
                ).model_validate(
                    {
                        "result": terminal.value,
                        "backend_version": __version__,
                    }
                )
            )
        return OperationProjection(
            operation_id=self.operation.operation_id,
            version=self.operation.version,
            terminal=terminal,
            publication=publication,
        )


__all__ = [
    "InlineOperationAdapter",
    "inline_operation_descriptor",
    "inline_output_schema",
    "run_inline",
]
