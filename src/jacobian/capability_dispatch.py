"""Request normalization, provider readiness, adapter invocation, and outcomes."""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from jacobian.capability_errors import (
    CapabilityError,
    CapabilityInvocationError,
    PayloadValidationError,
)
from jacobian.capability_telemetry import log_invocation
from jacobian.capability_validation import json_value_type, validate_payload
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityDiagnostic,
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.contracts.results import Execution, ExecutionStatus
from jacobian.provider_runtime import (
    ProviderRuntimeError,
    require_provider_runtime_ready,
)

_LOGGER = logging.getLogger(__name__)


class AdapterLike(Protocol):
    @property
    def descriptor(self) -> Any: ...

    def invoke(self, request: CapabilityRequest) -> CapabilityResult: ...


class CapabilityDispatchMixin:
    """Own the invocation state machine after registry resolution."""

    def invoke(self: Any, request: CapabilityRequest) -> CapabilityResult:
        started = time.monotonic()
        try:
            adapter: AdapterLike = self._adapters[request.capability_id]
        except KeyError:
            result = resolution_failure(
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
                        for descriptor in self.catalog().capabilities
                    ],
                },
            )
            log_invocation(result, started)
            return result
        descriptor = adapter.descriptor
        policy_reasons = self.policy.denial_reasons(
            descriptor,
            mode=request.mode,
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
                        "policy_profile": self.policy.profile,
                        "policy_digest": self.policy.digest,
                        "reasons": list(policy_reasons),
                        "checker_authorization_affected": False,
                    },
                ),
                context={"capability_policy": self.policy.definition},
            )
            result = result.model_copy(update=provider_provenance(descriptor))
            log_invocation(result, started)
            return result
        if request.mode not in descriptor.modes:
            result = resolution_failure(
                request=request,
                capability_version=descriptor.version,
                diagnostic=CapabilityDiagnostic(
                    code="UNSUPPORTED_MODE",
                    stage="capability_resolution",
                    message=(
                        f"Capability {request.capability_id!r} does not support "
                        f"{request.mode.value} mode."
                    ),
                    hint=(
                        "Call math.find for this capability, then retry "
                        "with one of its advertised modes."
                    ),
                ),
                context={"available_modes": [mode.value for mode in descriptor.modes]},
            )
            result = result.model_copy(update=provider_provenance(descriptor))
            log_invocation(result, started)
            return result
        try:
            normalized_input = validate_payload(descriptor.input_schema, request.input)
        except CapabilityError as exc:
            path = (
                exc.path if isinstance(exc, PayloadValidationError) else error_path(exc)
            )
            result = failed_result(
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
                        "Correct the reported field. The exact required and missing "
                        "top-level fields are included in diagnostic details."
                    ),
                    details={
                        "required_fields": descriptor.input_schema.get("required", []),
                        "missing_fields": sorted(
                            set(descriptor.input_schema.get("required", []))
                            - set(request.input)
                        ),
                    },
                ),
            )
            log_invocation(result, started)
            return result
        normalized_request = request.model_copy(update={"input": normalized_input})
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
            _LOGGER.warning(
                "capability %s stopped during execution",
                request.capability_id,
                exc_info=exc,
            )
            result = failed_result(
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
        if (
            result.capability_id != descriptor.capability_id
            or result.capability_version != descriptor.version
            or result.mode is not request.mode
        ):
            result = failed_result(
                descriptor=descriptor,
                request=request,
                diagnostic=CapabilityDiagnostic(
                    code="ADAPTER_RESULT_INVALID",
                    stage="adapter_execution",
                    message="The adapter returned a result with a mismatched identity.",
                    hint="The capability adapter produced a result for a different capability or mode.",
                ),
            )
            log_invocation(result, started)
            return result
        if result.provider is not None and result.provider != descriptor.provider:
            result = failed_result(
                descriptor=descriptor,
                request=request,
                diagnostic=CapabilityDiagnostic(
                    code="ADAPTER_RESULT_INVALID",
                    stage="adapter_execution",
                    message="The adapter returned a result from a different provider runtime.",
                    hint="The capability adapter produced a result from a provider that differs from its descriptor.",
                ),
            )
            log_invocation(result, started)
            return result
        provenance = provider_provenance(descriptor)
        if (
            result.provider_digest is not None
            and result.provider_digest != provenance["provider_digest"]
        ):
            result = failed_result(
                descriptor=descriptor,
                request=request,
                diagnostic=CapabilityDiagnostic(
                    code="ADAPTER_RESULT_INVALID",
                    stage="adapter_execution",
                    message="The adapter returned a result with a mismatched provider digest.",
                    hint="The capability adapter produced a result with a provider digest that differs from its descriptor.",
                ),
            )
            log_invocation(result, started)
            return result
        result = result.model_copy(update=provenance)
        if result.execution.status is ExecutionStatus.COMPLETED:
            normalized_output = validate_payload(
                descriptor.output_schema, result.output
            )
            result = result.model_copy(update={"output": normalized_output})
        self._validate_artifact_references(result)
        self._validate_verified_result(result)
        log_invocation(result, started)
        return result


def failed_result(
    *, descriptor: Any, request: CapabilityRequest, diagnostic: CapabilityDiagnostic
) -> CapabilityResult:
    provenance = provider_provenance(descriptor)
    return CapabilityResult(
        capability_id=descriptor.capability_id,
        capability_version=descriptor.version,
        mode=request.mode,
        execution=Execution(status=ExecutionStatus.ERROR, detail=diagnostic.message),
        output={"error": diagnostic.model_dump(mode="json", exclude_none=True)},
        diagnostics=(diagnostic,),
        assurance=CapabilityAssurance(
            level=CapabilityAssuranceLevel.HEURISTIC,
            basis="execution or input failure; no mathematical conclusion",
        ),
        provider=provenance["provider"],
        provider_digest=provenance["provider_digest"],
    )


def invoke_ready_adapter(
    *, adapter: AdapterLike, descriptor: Any, request: CapabilityRequest
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
    return CapabilityResult.model_validate(adapter.invoke(request))


def provider_provenance(descriptor: Any) -> dict[str, str]:
    if descriptor.provider_runtime is None:
        raise CapabilityError(
            f"capability {descriptor.capability_id} has no provider runtime identity"
        )
    if descriptor.provider_runtime.digest is None:
        raise CapabilityError(
            f"capability {descriptor.capability_id} has no provider runtime digest"
        )
    return {
        "provider": descriptor.provider,
        "provider_digest": descriptor.provider_runtime.digest,
    }


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
        mode=request.mode,
        execution=Execution(status=ExecutionStatus.ERROR, detail=diagnostic.message),
        output={
            "error": diagnostic.model_dump(mode="json", exclude_none=True),
            **context,
        },
        diagnostics=(diagnostic,),
        assurance=CapabilityAssurance(
            level=CapabilityAssuranceLevel.HEURISTIC,
            basis="capability resolution failed; no mathematical conclusion",
        ),
    )


def error_path(exc: Exception) -> str | None:
    path, separator, _ = str(exc).partition(": ")
    return path if separator else None


__all__ = ["CapabilityDispatchMixin", "provider_provenance"]
