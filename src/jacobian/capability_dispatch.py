"""Request normalization, provider readiness, adapter invocation, and outcomes."""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import ValidationError

from jacobian.capability_adapters import CapabilityAdapter, TypedInputAdapter
from jacobian.capability_errors import (
    CapabilityError,
    CapabilityInvocationError,
    PayloadValidationError,
    enriched_invalid_request,
)
from jacobian.capability_telemetry import log_invocation
from jacobian.capability_validation import json_value_type, validate_payload
from jacobian.contracts.capabilities import (
    CapabilityDiagnostic,
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.contracts.results import Execution, ExecutionStatus
from jacobian.operation_projection import project_operation_result
from jacobian.provider_runtime import (
    ProviderRuntimeError,
    require_provider_runtime_ready,
)

_LOGGER = logging.getLogger(__name__)


class CapabilityDispatchMixin:
    """Own the invocation state machine after registry resolution."""

    def invoke(self: Any, request: CapabilityRequest) -> CapabilityResult:
        started = time.monotonic()
        try:
            adapter: CapabilityAdapter = self._adapters[request.capability_id]
        except KeyError:
            result = _unknown_capability_failure(self, request)
            log_invocation(result, started)
            return result
        descriptor = self._descriptors[request.capability_id]
        resolution = _capability_resolution_failure(self, request, descriptor)
        if resolution is not None:
            log_invocation(resolution, started)
            return resolution
        try:
            normalized_request = _normalize_request(adapter, descriptor, request)
        except CapabilityError as exc:
            result = _input_validation_failure(descriptor, request, exc)
            log_invocation(result, started)
            return result
        try:
            result = invoke_ready_adapter(
                adapter=adapter,
                descriptor=descriptor,
                request=normalized_request,
            )
        except CapabilityInvocationError as exc:
            result = failed_result(
                descriptor=descriptor,
                request=request,
                diagnostic=exc.diagnostic,
            )
        except CapabilityError as exc:
            result = failed_result(
                descriptor=descriptor,
                request=request,
                diagnostic=CapabilityDiagnostic(
                    code="ADAPTER_CONFIGURATION_FAILED",
                    stage="adapter_execution",
                    message=str(exc),
                    hint=(
                        "The capability is misconfigured. Check the provider "
                        "runtime identity and digest in the capability descriptor."
                    ),
                ),
            )
        except Exception as exc:
            result = _adapter_execution_failure(descriptor, request, exc)
        invalid = _adapter_result_identity_failure(
            descriptor=descriptor,
            request=request,
            result=result,
        )
        if invalid is not None:
            log_invocation(invalid, started)
            return invalid
        if result.execution.status is ExecutionStatus.COMPLETED and not isinstance(
            adapter, TypedInputAdapter
        ):
            result = _normalize_completed_adapter_output(
                descriptor=descriptor,
                request=request,
                result=result,
                started=started,
            )
            if result.execution.status is not ExecutionStatus.COMPLETED:
                return result
        self._validate_verified_result(result)
        log_invocation(result, started)
        return result


def _normalize_request(
    adapter: CapabilityAdapter,
    descriptor: Any,
    request: CapabilityRequest,
) -> CapabilityRequest:
    """Use the adapter's typed parser or the external schema-only boundary."""

    if request.inputs and not descriptor.input_ports:
        raise PayloadValidationError(
            "the selected capability declares no typed value inputs",
            path="inputs",
            actual_type="object",
            expected="no value references",
            details={"unknown_input_ports": sorted(request.inputs)},
        )
    if isinstance(adapter, TypedInputAdapter):
        return request
    normalized_input = validate_payload(descriptor.input_schema, request.input)
    return request.model_copy(update={"input": normalized_input})


def _unknown_capability_failure(
    dispatch: Any, request: CapabilityRequest
) -> CapabilityResult:
    return resolution_failure(
        request=request,
        capability_version="not-installed",
        diagnostic=CapabilityDiagnostic(
            code="UNKNOWN_CAPABILITY",
            stage="capability_resolution",
            message=(f"Capability {request.capability_id!r} is not installed."),
            hint=(
                "Call math.find without capability_id to list "
                "installed capabilities, then retry with one of those IDs."
            ),
        ),
        context={
            "available_capability_ids": [
                descriptor.capability_id
                for descriptor in dispatch.catalog().capabilities
            ],
        },
    )


def _capability_resolution_failure(
    dispatch: Any,
    request: CapabilityRequest,
    descriptor: Any,
) -> CapabilityResult | None:
    policy_reasons = dispatch.policy.denial_reasons(
        descriptor,
    )
    if policy_reasons:
        result = resolution_failure(
            request=request,
            capability_version=descriptor.version,
            diagnostic=CapabilityDiagnostic(
                code="CAPABILITY_POLICY_DENIED",
                stage="capability_policy",
                message=(
                    f"Capability {request.capability_id!r} is denied by the "
                    "operator-controlled capability policy."
                ),
                hint=(
                    "Choose a capability visible in math.find, or ask "
                    "the operator to change the evaluation/runtime policy."
                ),
                details={
                    "policy_profile": dispatch.policy.profile,
                    "policy_digest": dispatch.policy.digest,
                    "reasons": list(policy_reasons),
                    "checker_authorization_affected": False,
                },
            ),
            context={"capability_policy": dispatch.policy.definition},
        )
        return result
    return None


def _adapter_execution_failure(
    descriptor: Any,
    request: CapabilityRequest,
    exc: Exception,
) -> CapabilityResult:
    if isinstance(exc, ValidationError):
        return failed_result(
            descriptor=descriptor,
            request=request,
            diagnostic=enriched_invalid_request(
                CapabilityDiagnostic(
                    code="INVALID_REQUEST",
                    stage="capability_input_validation",
                    message="The capability request is invalid.",
                ),
                exc,
            ),
        )
    _LOGGER.warning(
        "capability %s stopped during execution",
        request.capability_id,
        exc_info=exc,
    )
    return failed_result(
        descriptor=descriptor,
        request=request,
        diagnostic=CapabilityDiagnostic(
            code="ADAPTER_EXECUTION_FAILED",
            stage="adapter_execution",
            message="The capability stopped before returning a result.",
            hint=(
                "Retry once. If it fails again, inspect the local Jacobian "
                "log for this capability."
            ),
        ),
    )


def _input_validation_failure(
    descriptor: Any,
    request: CapabilityRequest,
    exc: CapabilityError,
) -> CapabilityResult:
    path = exc.path if isinstance(exc, PayloadValidationError) else error_path(exc)
    return failed_result(
        descriptor=descriptor,
        request=request,
        diagnostic=CapabilityDiagnostic(
            code="INVALID_REQUEST",
            stage="capability_input_validation",
            message=(
                "The capability input does not match its advertised schema"
                + (f" at {path}." if path else ".")
            ),
            path=path,
            expected=(
                exc.expected
                if isinstance(exc, PayloadValidationError)
                else "input matching the capability descriptor JSON Schema"
            ),
            actual_type=(
                exc.actual_type
                if isinstance(exc, PayloadValidationError)
                else json_value_type(request.input)
            ),
            hint=(
                "Correct the reported field. The exact violated constraint "
                "and any required or missing top-level fields are included "
                "in diagnostic details."
            ),
            details={
                "required_fields": descriptor.input_schema.get("required", []),
                "missing_fields": sorted(
                    set(descriptor.input_schema.get("required", []))
                    - set(request.input)
                ),
                **(exc.details if isinstance(exc, PayloadValidationError) else {}),
            },
        ),
    )


def _adapter_result_identity_failure(
    *,
    descriptor: Any,
    request: CapabilityRequest,
    result: CapabilityResult,
) -> CapabilityResult | None:
    if (
        result.capability_id != descriptor.capability_id
        or result.capability_version != descriptor.version
    ):
        return failed_result(
            descriptor=descriptor,
            request=request,
            diagnostic=CapabilityDiagnostic(
                code="ADAPTER_RESULT_INVALID",
                stage="adapter_execution",
                message="The adapter returned a result with a mismatched identity.",
                hint="The capability adapter produced a result for a different capability.",
            ),
        )
    return None


def _normalize_completed_adapter_output(
    *,
    descriptor: Any,
    request: CapabilityRequest,
    result: CapabilityResult,
    started: float,
) -> CapabilityResult:
    try:
        normalized_output = validate_payload(descriptor.output_schema, result.output)
    except PayloadValidationError as exc:
        invalid = failed_result(
            descriptor=descriptor,
            request=request,
            diagnostic=CapabilityDiagnostic(
                code="ADAPTER_RESULT_INVALID",
                stage="adapter_execution",
                message=(
                    "The adapter output does not match its advertised "
                    f"schema at {exc.path}."
                ),
                path=exc.path,
                expected=exc.expected,
                actual_type=exc.actual_type,
                hint=(
                    "Fix the capability adapter to return output matching "
                    "its descriptor schema."
                ),
                details=exc.details,
            ),
        )
        log_invocation(invalid, started)
        return invalid
    return result.model_copy(update={"output": normalized_output})


def failed_result(
    *, descriptor: Any, request: CapabilityRequest, diagnostic: CapabilityDiagnostic
) -> CapabilityResult:
    return CapabilityResult(
        capability_id=descriptor.capability_id,
        capability_version=descriptor.version,
        execution=Execution(status=ExecutionStatus.ERROR, detail=diagnostic.message),
        output={"error": diagnostic.model_dump(mode="json", exclude_none=True)},
        diagnostics=(diagnostic,),
    )


def invoke_ready_adapter(
    *, adapter: CapabilityAdapter, descriptor: Any, request: CapabilityRequest
) -> CapabilityResult:
    runtime = descriptor.provider_runtime
    if runtime is None:
        raise CapabilityError(
            f"capability {descriptor.capability_id} has no provider runtime identity"
        )
    try:
        require_provider_runtime_ready(runtime)
    except ProviderRuntimeError as exc:
        return failed_result(
            descriptor=descriptor,
            request=request,
            diagnostic=CapabilityDiagnostic(
                code="PROVIDER_READINESS_FAILED",
                stage="provider_readiness",
                message="The declared capability provider is not ready for first use.",
                hint=(
                    "Repair or reinstall the declared provider, then retry the "
                    "capability invocation."
                ),
                details={"provider_failure_code": exc.code.value},
            ),
        )
    outcome = adapter.invoke(request)
    return project_operation_result(outcome)


def resolution_failure(
    *,
    request: CapabilityRequest,
    capability_version: str,
    diagnostic: CapabilityDiagnostic,
    context: dict[str, object],
) -> CapabilityResult:
    return CapabilityResult(
        capability_id=request.capability_id,
        capability_version=capability_version,
        execution=Execution(status=ExecutionStatus.ERROR, detail=diagnostic.message),
        output={
            "error": diagnostic.model_dump(mode="json", exclude_none=True),
            **context,
        },
        diagnostics=(diagnostic,),
    )


def error_path(exc: Exception) -> str | None:
    path, separator, _ = str(exc).partition(": ")
    return path if separator else None


__all__ = ["CapabilityDispatchMixin"]
