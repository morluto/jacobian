"""Tests for adapter authority and forgery boundaries.

Covers: invocation recording, provider readiness, error diagnostics, forged
verified status, and verification record binding.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import pytest
from tests.component.capabilities.capability_service_support import (
    TEST_RUNTIME,
    ComputedAdapter,
    CrashingAdapter,
    ForgedVerifiedAdapter,
    InvalidOutputValueAdapter,
    MismatchedOutputAdapter,
    NotReadyProviderAdapter,
)
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.builtin_capabilities import LeanCheckAdapter
from jacobian.capability_errors import CapabilityError
from jacobian.capability_service import CapabilityPolicy
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


def test_invalid_request_precedes_provider_readiness(
    capability_core_services: DomainTestServices,
) -> None:
    core = capability_core_services.core
    capability_core_services.installation.register_capability(NotReadyProviderAdapter())

    result = core.capabilities.invoke(
        CapabilityRequest(
            capability_id="example.not-ready-provider",
            input={"unexpected": True},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_REQUEST"
    assert result.diagnostics[0].stage == "capability_input_validation"


def test_published_model_must_match_installed_output_contract(
    capability_core_services: DomainTestServices,
) -> None:
    core = capability_core_services.core
    capability_core_services.installation.register_capability(MismatchedOutputAdapter())

    result = core.capabilities.invoke(
        CapabilityRequest(capability_id="example.mismatched-output", input={})
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "ADAPTER_RESULT_INVALID"
    assert result.output == {
        "error": result.diagnostics[0].model_dump(mode="json", exclude_none=True)
    }


def test_published_model_instance_must_satisfy_its_contract(
    capability_core_services: DomainTestServices,
) -> None:
    core = capability_core_services.core
    capability_core_services.installation.register_capability(
        InvalidOutputValueAdapter()
    )

    result = core.capabilities.invoke(
        CapabilityRequest(capability_id="example.invalid-output-value", input={})
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "ADAPTER_RESULT_INVALID"


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
    assert result.output == {
        "error": result.diagnostics[0].model_dump(mode="json", exclude_none=True)
    }


def test_policy_denial_does_not_embed_the_policy_document(
    capability_core_services: DomainTestServices,
) -> None:
    core = capability_core_services.core
    capability_core_services.installation.register_capability(ComputedAdapter())
    core.capabilities.policy = CapabilityPolicy(
        denied_capability_ids=frozenset({"example.double"})
    )

    result = core.capabilities.invoke(
        CapabilityRequest(capability_id="example.double", input={"value": 21})
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "CAPABILITY_POLICY_DENIED"
    assert result.diagnostics[0].details == {
        "policy_profile": "DEFAULT",
        "policy_digest": core.capabilities.policy.digest,
        "reasons": ["capability_id_denied"],
        "checker_authorization_affected": False,
    }
    assert result.output == {
        "error": result.diagnostics[0].model_dump(mode="json", exclude_none=True)
    }


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
    assert diagnostic.message == "The capability request is invalid."
    assert diagnostic.actual_type is None
    assert diagnostic.expected is None
    assert "fixture-secret-value" not in diagnostic.message
    assert diagnostic.hint == (
        "1 validation error; first at value: "
        "Request value violates validation rule int_type"
    )
    assert diagnostic.details == {}
    assert "fixture-secret-value" not in repr(diagnostic)


def test_capability_input_does_not_coerce_numeric_strings(
    capability_core_services: DomainTestServices,
) -> None:
    core = capability_core_services.core
    capability_core_services.installation.register_capability(ComputedAdapter())

    result = core.capabilities.invoke(
        CapabilityRequest(capability_id="example.double", input={"value": "21"})
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_REQUEST"
    assert result.diagnostics[0].path == "value"


def test_non_json_capability_input_is_an_invalid_request(
    capability_core_services: DomainTestServices,
) -> None:
    core = capability_core_services.core
    capability_core_services.installation.register_capability(ComputedAdapter())

    result = core.capabilities.invoke(
        CapabilityRequest(capability_id="example.double", input={"value": object()})
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_REQUEST"
    assert result.diagnostics[0].stage == "capability_input_validation"
    assert result.diagnostics[0].path is None


def test_lean_check_parses_its_typed_request_before_execution(
    capability_core_services: DomainTestServices,
) -> None:
    class _UnexpectedLean:
        def verify(self, **_kwargs: object) -> None:
            raise AssertionError("invalid Lean input must not execute")

    adapter = LeanCheckAdapter(
        cast(Any, _UnexpectedLean()),
        TEST_RUNTIME.model_copy(update={"provider": "jacobian.lean4"}),
    )
    core = capability_core_services.core
    capability_core_services.installation.register_capability(adapter)

    result = core.capabilities.invoke(
        CapabilityRequest(
            capability_id="lean.check",
            input={"statement": "True"},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_REQUEST"
    assert result.diagnostics[0].path == "proof"


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
