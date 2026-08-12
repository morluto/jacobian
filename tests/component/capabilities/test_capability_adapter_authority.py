"""Tests for adapter authority and forgery boundaries.

Covers: invocation recording, provider readiness, error diagnostics, forged
verified status, and verification record binding.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from tests.component.capabilities.capability_service_support import (
    ComputedAdapter,
    CrashingAdapter,
    ForgedVerifiedAdapter,
    InvalidOutputAdapter,
    NotReadyProviderAdapter,
)
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.capability_errors import CapabilityError
from jacobian.contracts.capabilities import (
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus


@pytest.fixture
def capability_core_services(tmp_path) -> Iterator[DomainTestServices]:
    with open_domain_services(tmp_path / "state") as services:
        yield services


def test_external_adapter_invocation_is_recorded_and_retrievable(
    capability_core_services: DomainTestServices,
) -> None:
    core = capability_core_services.core
    capability_core_services.installation.register_capability(ComputedAdapter())

    result = core.capabilities.invoke(
        CapabilityRequest(
            capability_id="example.double",
            input={"value": 21},
        )
    )

    assert result.output == {"value": 42}


def test_provider_required_attributes_are_checked_before_first_use(
    capability_core_services: DomainTestServices,
) -> None:
    core = capability_core_services.core
    capability_core_services.installation.register_capability(NotReadyProviderAdapter())

    result = core.capabilities.invoke(
        CapabilityRequest(
            capability_id="example.not-ready-provider",
            input={},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "PROVIDER_READINESS_FAILED"
    assert result.diagnostics[0].stage == "provider_readiness"
    assert result.diagnostics[0].details == {
        "provider_failure_code": "READINESS_FAILED"
    }


def test_unknown_capability_returns_an_actionable_result(
    capability_core_services: DomainTestServices,
) -> None:
    core = capability_core_services.core
    capability_core_services.installation.register_capability(ComputedAdapter())

    result = core.capabilities.invoke(
        CapabilityRequest(
            capability_id="missing.capability",
            input={},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "UNKNOWN_CAPABILITY"
    assert result.diagnostics[0].stage == "capability_resolution"
    assert result.diagnostics[0].message == (
        "Capability 'missing.capability' is not installed."
    )
    assert "math.find" in (result.diagnostics[0].hint or "")
    assert result.output["available_capability_ids"]


def test_tool_id_owns_role(
    capability_core_services: DomainTestServices,
) -> None:
    """Tool identity determines role; no client mode switch."""
    core = capability_core_services.core
    capability_core_services.installation.register_capability(ComputedAdapter())

    result = core.capabilities.invoke(
        CapabilityRequest(
            capability_id="example.double",
            input={"value": 21},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["value"] == 42


def test_invalid_capability_input_does_not_echo_payload(
    capability_core_services: DomainTestServices,
) -> None:
    core = capability_core_services.core
    capability_core_services.installation.register_capability(ComputedAdapter())

    result = core.capabilities.invoke(
        CapabilityRequest(
            capability_id="example.double",
            input={"value": "fixture-secret-value"},
        )
    )

    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "INVALID_REQUEST"
    assert diagnostic.path == "value"
    assert diagnostic.message == (
        "The capability input does not match its advertised schema at value."
    )
    assert diagnostic.actual_type == "string"
    assert diagnostic.expected == "JSON type integer"
    assert "fixture-secret-value" not in diagnostic.message
    assert diagnostic.details == {
        "required_fields": ["value"],
        "missing_fields": [],
        "validator": "type",
        "constraint": "integer",
    }
    assert "fixture-secret-value" not in repr(diagnostic)


def test_adapter_failure_does_not_expose_internal_exception_text(
    capability_core_services: DomainTestServices,
) -> None:
    core = capability_core_services.core
    capability_core_services.installation.register_capability(CrashingAdapter())

    result = core.capabilities.invoke(
        CapabilityRequest(
            capability_id="example.crash",
            input={},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "ADAPTER_EXECUTION_FAILED"
    assert result.diagnostics[0].message == (
        "The capability stopped before returning a result."
    )
    assert result.diagnostics[0].hint == (
        "Retry once. If it fails again, inspect the local Jacobian log for this "
        "capability."
    )
    assert "fixture" not in result.execution.detail
    assert "RuntimeError" not in result.execution.detail


def test_schema_invalid_adapter_output_returns_a_typed_failure(
    capability_core_services: DomainTestServices,
) -> None:
    core = capability_core_services.core
    capability_core_services.installation.register_capability(InvalidOutputAdapter())

    result = core.capabilities.invoke(
        CapabilityRequest(
            capability_id="example.invalid-output",
            input={"value": 21},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "ADAPTER_RESULT_INVALID"
    assert result.diagnostics[0].stage == "adapter_execution"
    assert result.diagnostics[0].path == "value"
    assert result.diagnostics[0].actual_type == "string"
    assert result.diagnostics[0].expected == "JSON type integer"


def test_adapter_cannot_promote_without_a_local_verification_record(
    capability_core_services: DomainTestServices,
) -> None:
    core = capability_core_services.core
    capability_core_services.installation.register_capability(ForgedVerifiedAdapter())

    with pytest.raises(CapabilityError, match="verification record"):
        core.capabilities.invoke(
            CapabilityRequest(
                capability_id="example.forged",
                input={},
            )
        )
