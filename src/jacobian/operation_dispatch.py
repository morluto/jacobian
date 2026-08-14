"""Request normalization, provider readiness, adapter invocation, and outcomes."""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import ValidationError

from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationDiagnostic,
    OperationRequest,
    OperationResult,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.operation_adapters import (
    OperationAdapter,
)
from jacobian.operation_errors import (
    OperationError,
    OperationInvocationError,
    enriched_invalid_request,
)
from jacobian.operation_projection import OperationProjection, project_operation_result
from jacobian.operation_validation import validate_payload, validator
from jacobian.operations import Failed
from jacobian.provider_runtime import (
    ProviderRuntimeError,
    require_provider_runtime_ready,
)

_LOGGER = logging.getLogger(__name__)


def register_operation(
    adapter: OperationAdapter[Any],
    adapters: dict[str, OperationAdapter[Any]],
    descriptors: dict[str, OperationDescriptor],
) -> None:
    """Record one adapter after validating its advertised schemas and examples."""

    descriptor = adapter.descriptor
    if descriptor.operation_id in adapters:
        raise OperationError(f"duplicate operation ID: {descriptor.operation_id}")
    validator(descriptor.input_schema)
    validator(descriptor.output_schema)
    for example in descriptor.examples:
        try:
            validate_payload(descriptor.input_schema, example.input)
        except OperationError as exc:
            raise OperationError(
                f"operation {descriptor.operation_id} invocation example "
                f"{example.name!r} does not match its input schema"
            ) from exc
    descriptors[descriptor.operation_id] = descriptor.model_copy(deep=True)
    adapters[descriptor.operation_id] = adapter


def log_invocation(result: OperationResult, started: float) -> None:
    elapsed_ms = round((time.monotonic() - started) * 1000)
    diagnostic_codes = (
        ",".join(diagnostic.code for diagnostic in result.diagnostics) or "-"
    )
    _LOGGER.info(
        (
            "operation invocation operation_id=%s version=%s "
            "status=%s elapsed_ms=%d diagnostics=%s"
        ),
        result.operation_id,
        result.operation_version,
        result.execution.status.value,
        elapsed_ms,
        diagnostic_codes,
        extra={
            "jacobian_operation_id": result.operation_id,
            "jacobian_operation_version": result.operation_version,
            "jacobian_execution_status": result.execution.status.value,
            "jacobian_elapsed_ms": elapsed_ms,
            "jacobian_diagnostic_codes": tuple(
                diagnostic.code for diagnostic in result.diagnostics
            ),
        },
    )


def dispatch_operation(dispatch: Any, request: OperationRequest) -> OperationResult:
    """Invoke one already-resolved operation through the typed state machine."""

    started = time.monotonic()
    try:
        adapter: OperationAdapter[Any] = dispatch._adapters[request.operation_id]
    except KeyError:
        result = _unknown_operation_failure(request)
        log_invocation(result, started)
        return result
    descriptor = dispatch._descriptors[request.operation_id]
    resolution = _operation_resolution_failure(dispatch, descriptor)
    if resolution is not None:
        log_invocation(resolution, started)
        return resolution
    try:
        prepared = adapter.prepare(request)
        result = invoke_ready_adapter(
            adapter=adapter, descriptor=descriptor, prepared=prepared
        )
    except OperationInvocationError as exc:
        result = failed_result(
            operation_id=descriptor.operation_id,
            version=descriptor.version,
            diagnostic=exc.diagnostic,
        )
    except OperationError as exc:
        result = failed_result(
            operation_id=descriptor.operation_id,
            version=descriptor.version,
            diagnostic=OperationDiagnostic(
                code="ADAPTER_CONFIGURATION_FAILED",
                stage="adapter_execution",
                message=str(exc),
                hint=(
                    "The operation is misconfigured. Check the provider "
                    "runtime identity and digest in the operation descriptor."
                ),
            ),
        )
    except Exception as exc:
        result = _adapter_execution_failure(descriptor, request, exc)
    invalid = _adapter_result_identity_failure(descriptor=descriptor, result=result)
    if invalid is not None:
        log_invocation(invalid, started)
        return invalid
    log_invocation(result, started)
    return result


def _unknown_operation_failure(request: OperationRequest) -> OperationResult:
    return failed_result(
        operation_id=request.operation_id,
        version="not-installed",
        diagnostic=OperationDiagnostic(
            code="UNKNOWN_OPERATION",
            stage="operation_resolution",
            message=(f"Operation {request.operation_id!r} is not installed."),
            hint=(
                "Call math.find without operation_id to list "
                "installed operations, then retry with one of those IDs."
            ),
        ),
    )


def _operation_resolution_failure(
    dispatch: Any,
    descriptor: Any,
) -> OperationResult | None:
    policy_reasons = dispatch.policy.denial_reasons(
        descriptor,
    )
    if policy_reasons:
        result = failed_result(
            operation_id=descriptor.operation_id,
            version=descriptor.version,
            diagnostic=OperationDiagnostic(
                code="OPERATION_POLICY_DENIED",
                stage="operation_policy",
                message=(
                    f"Operation {descriptor.operation_id!r} is denied by the "
                    "operator-controlled operation policy."
                ),
                hint=(
                    "Choose an operation visible in math.find, or ask "
                    "the operator to change the evaluation/runtime policy."
                ),
                details={
                    "policy_profile": dispatch.policy.profile,
                    "policy_digest": dispatch.policy.digest,
                    "reasons": list(policy_reasons),
                    "checker_authorization_affected": False,
                },
            ),
        )
        return result
    return None


def _adapter_execution_failure(
    descriptor: Any,
    request: OperationRequest,
    exc: Exception,
) -> OperationResult:
    if isinstance(exc, ValidationError):
        return failed_result(
            operation_id=descriptor.operation_id,
            version=descriptor.version,
            diagnostic=enriched_invalid_request(
                OperationDiagnostic(
                    code="INVALID_REQUEST",
                    stage="operation_input_validation",
                    message="The operation request is invalid.",
                ),
                exc,
            ),
        )
    _LOGGER.warning(
        "operation %s stopped during execution",
        request.operation_id,
        exc_info=exc,
    )
    return failed_result(
        operation_id=descriptor.operation_id,
        version=descriptor.version,
        diagnostic=OperationDiagnostic(
            code="ADAPTER_EXECUTION_FAILED",
            stage="adapter_execution",
            message="The operation stopped before returning a result.",
            hint=(
                "Retry once. If it fails again, inspect the local Jacobian "
                "log for this operation."
            ),
        ),
    )


def _adapter_result_identity_failure(
    *,
    descriptor: Any,
    result: OperationResult,
) -> OperationResult | None:
    if (
        result.operation_id != descriptor.operation_id
        or result.operation_version != descriptor.version
    ):
        return failed_result(
            operation_id=descriptor.operation_id,
            version=descriptor.version,
            diagnostic=OperationDiagnostic(
                code="ADAPTER_RESULT_INVALID",
                stage="adapter_execution",
                message="The adapter returned a result with a mismatched identity.",
                hint="The operation adapter produced a result for a different operation.",
            ),
        )
    return None


def failed_result(
    *,
    operation_id: str,
    version: str,
    diagnostic: OperationDiagnostic,
) -> OperationResult:
    return project_operation_result(
        OperationProjection(
            operation_id=operation_id,
            version=version,
            terminal=Failed(
                status=ExecutionStatus.ERROR,
                diagnostic=diagnostic,
            ),
        )
    )


def invoke_ready_adapter(
    *, adapter: OperationAdapter[Any], descriptor: Any, prepared: Any
) -> OperationResult:
    runtime = descriptor.provider_runtime
    if runtime is not None:
        try:
            require_provider_runtime_ready(runtime)
        except ProviderRuntimeError as exc:
            return failed_result(
                operation_id=descriptor.operation_id,
                version=descriptor.version,
                diagnostic=OperationDiagnostic(
                    code="PROVIDER_READINESS_FAILED",
                    stage="provider_readiness",
                    message="The selected external executable is not ready.",
                    hint="Run `jacobian update` after repairing the executable.",
                    details={"provider_failure_code": exc.code.value},
                ),
            )
    return project_operation_result(adapter.invoke(prepared))


__all__ = ["dispatch_operation", "log_invocation", "register_operation"]
