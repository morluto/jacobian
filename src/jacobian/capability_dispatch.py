"""Request normalization, provider readiness, adapter invocation, and outcomes."""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import ValidationError

from jacobian.capability_adapters import CapabilityAdapter
from jacobian.capability_errors import (
    CapabilityError,
    CapabilityInvocationError,
    enriched_invalid_request,
)
from jacobian.capability_telemetry import log_invocation
from jacobian.contracts.capabilities import (
    CapabilityDiagnostic,
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.operation_projection import OperationProjection, project_operation_result
from jacobian.operations import Failed
from jacobian.provider_runtime import (
    ProviderRuntimeError,
    require_provider_runtime_ready,
)
from jacobian.schema_registry import model_schema

_LOGGER = logging.getLogger(__name__)


class CapabilityDispatchMixin:
    """Own the invocation state machine after registry resolution."""

    def invoke(self: Any, request: CapabilityRequest) -> CapabilityResult:
        started = time.monotonic()
        try:
            adapter: CapabilityAdapter[Any] = self._adapters[request.capability_id]
        except KeyError:
            result = _unknown_capability_failure(request)
            log_invocation(result, started)
            return result
        descriptor = self._descriptors[request.capability_id]
        resolution = _capability_resolution_failure(self, descriptor)
        if resolution is not None:
            log_invocation(resolution, started)
            return resolution
        invalid_inputs = _undeclared_value_inputs_failure(descriptor, request)
        if invalid_inputs is not None:
            result = invalid_inputs
            log_invocation(result, started)
            return result
        try:
            prepared = adapter.prepare(request)
            result = invoke_ready_adapter(
                adapter=adapter,
                descriptor=descriptor,
                prepared=prepared,
            )
        except CapabilityInvocationError as exc:
            result = failed_result(
                operation_id=descriptor.capability_id,
                version=descriptor.version,
                diagnostic=exc.diagnostic,
            )
        except CapabilityError as exc:
            result = failed_result(
                operation_id=descriptor.capability_id,
                version=descriptor.version,
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
            result=result,
        )
        if invalid is not None:
            log_invocation(invalid, started)
            return invalid
        self._validate_verified_result(result)
        log_invocation(result, started)
        return result


def _undeclared_value_inputs_failure(
    descriptor: Any,
    request: CapabilityRequest,
) -> CapabilityResult | None:
    if not request.inputs or descriptor.input_ports:
        return None
    return failed_result(
        operation_id=descriptor.capability_id,
        version=descriptor.version,
        diagnostic=CapabilityDiagnostic(
            code="INVALID_REQUEST",
            stage="capability_input_validation",
            message="The selected capability declares no typed value inputs.",
            path="inputs",
            expected="no value references",
            actual_type="object",
            hint="Remove the undeclared value-reference inputs and retry.",
            details={"unknown_input_ports": sorted(request.inputs)},
        ),
    )


def _unknown_capability_failure(request: CapabilityRequest) -> CapabilityResult:
    return failed_result(
        operation_id=request.capability_id,
        version="not-installed",
        diagnostic=CapabilityDiagnostic(
            code="UNKNOWN_CAPABILITY",
            stage="capability_resolution",
            message=(f"Capability {request.capability_id!r} is not installed."),
            hint=(
                "Call math.find without capability_id to list "
                "installed capabilities, then retry with one of those IDs."
            ),
        ),
    )


def _capability_resolution_failure(
    dispatch: Any,
    descriptor: Any,
) -> CapabilityResult | None:
    policy_reasons = dispatch.policy.denial_reasons(
        descriptor,
    )
    if policy_reasons:
        result = failed_result(
            operation_id=descriptor.capability_id,
            version=descriptor.version,
            diagnostic=CapabilityDiagnostic(
                code="CAPABILITY_POLICY_DENIED",
                stage="capability_policy",
                message=(
                    f"Capability {descriptor.capability_id!r} is denied by the "
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
            operation_id=descriptor.capability_id,
            version=descriptor.version,
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
        operation_id=descriptor.capability_id,
        version=descriptor.version,
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


def _adapter_result_identity_failure(
    *,
    descriptor: Any,
    result: CapabilityResult,
) -> CapabilityResult | None:
    if (
        result.capability_id != descriptor.capability_id
        or result.capability_version != descriptor.version
    ):
        return failed_result(
            operation_id=descriptor.capability_id,
            version=descriptor.version,
            diagnostic=CapabilityDiagnostic(
                code="ADAPTER_RESULT_INVALID",
                stage="adapter_execution",
                message="The adapter returned a result with a mismatched identity.",
                hint="The capability adapter produced a result for a different capability.",
            ),
        )
    return None


def failed_result(
    *,
    operation_id: str,
    version: str,
    diagnostic: CapabilityDiagnostic,
) -> CapabilityResult:
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
    *, adapter: CapabilityAdapter[Any], descriptor: Any, prepared: Any
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
            operation_id=descriptor.capability_id,
            version=descriptor.version,
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
    outcome = adapter.invoke(prepared)
    invalid = _adapter_output_model_failure(descriptor, outcome)
    if invalid is not None:
        outcome = invalid
    return project_operation_result(outcome)


def _adapter_output_model_failure(
    descriptor: Any, outcome: OperationProjection
) -> OperationProjection | None:
    publication = outcome.publication
    output = publication.output if publication is not None else None
    if output is None:
        return None
    output_type = type(output)
    if model_schema(output_type) == descriptor.output_schema:
        try:
            # Validate the serialized contract rather than a Python-mode dump.
            # Python-mode dumps turn nested dataclasses (for example
            # PrimeFieldMatrix) into mappings, which strict model validation
            # quite correctly rejects even though the published output is
            # already a valid typed model.
            output_type.model_validate_json(
                output.model_dump_json(warnings="error"), strict=True
            )
        except (TypeError, ValueError, ValidationError):
            pass
        else:
            return None
    return OperationProjection(
        operation_id=descriptor.capability_id,
        version=descriptor.version,
        terminal=Failed(
            status=ExecutionStatus.ERROR,
            diagnostic=CapabilityDiagnostic(
                code="ADAPTER_RESULT_INVALID",
                stage="adapter_execution",
                message=(
                    "The adapter returned a typed result that does not match "
                    "its installed output model."
                ),
                hint="Fix the adapter output type or its capability descriptor.",
            ),
        ),
    )


__all__ = ["CapabilityDispatchMixin"]
