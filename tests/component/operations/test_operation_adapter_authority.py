"""Tests for adapter authority and forgery boundaries.

Covers: invocation recording, provider readiness, error diagnostics, forged
verified status, and verification record binding.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import pytest
from tests.component.operations.operation_service_support import (
    TEST_RUNTIME,
    ComputedAdapter,
    CrashingAdapter,
    ForgedVerifiedAdapter,
    InvalidOutputValueAdapter,
    MismatchedOutputAdapter,
    NotReadyProviderAdapter,
)
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.builtin_operations import LeanCheckAdapter
from jacobian.contracts.operations import (
    OperationRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.operation_errors import OperationError
from jacobian.operation_visibility import OperationVisibilityPolicy


@pytest.fixture
def operation_core_services(tmp_path) -> Iterator[DomainTestServices]:
    with open_domain_services(tmp_path / "state") as services:
        yield services


def test_external_adapter_invocation_is_recorded_and_retrievable(
    operation_core_services: DomainTestServices,
) -> None:
    core = operation_core_services.core
    operation_core_services.installation.register_operation(ComputedAdapter())

    result = core.operations.invoke(
        OperationRequest(
            operation_id="example.double",
            input={"value": 21},
        )
    )

    assert result.output == {"value": 42}


def test_provider_required_attributes_are_checked_before_first_use(
    operation_core_services: DomainTestServices,
) -> None:
    core = operation_core_services.core
    operation_core_services.installation.register_operation(NotReadyProviderAdapter())

    result = core.operations.invoke(
        OperationRequest(
            operation_id="example.not-ready-provider",
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
    operation_core_services: DomainTestServices,
) -> None:
    core = operation_core_services.core
    operation_core_services.installation.register_operation(NotReadyProviderAdapter())

    result = core.operations.invoke(
        OperationRequest(
            operation_id="example.not-ready-provider",
            input={"unexpected": True},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_REQUEST"
    assert result.diagnostics[0].stage == "operation_input_validation"


def test_published_model_must_match_installed_output_contract(
    operation_core_services: DomainTestServices,
) -> None:
    core = operation_core_services.core
    operation_core_services.installation.register_operation(MismatchedOutputAdapter())

    result = core.operations.invoke(
        OperationRequest(operation_id="example.mismatched-output", input={})
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "ADAPTER_RESULT_INVALID"
    assert result.output == {
        "error": result.diagnostics[0].model_dump(mode="json", exclude_none=True)
    }


def test_published_model_instance_must_satisfy_its_contract(
    operation_core_services: DomainTestServices,
) -> None:
    core = operation_core_services.core
    operation_core_services.installation.register_operation(InvalidOutputValueAdapter())

    result = core.operations.invoke(
        OperationRequest(operation_id="example.invalid-output-value", input={})
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "ADAPTER_RESULT_INVALID"


def test_unknown_operation_returns_an_actionable_result(
    operation_core_services: DomainTestServices,
) -> None:
    core = operation_core_services.core
    operation_core_services.installation.register_operation(ComputedAdapter())

    result = core.operations.invoke(
        OperationRequest(
            operation_id="missing.operation",
            input={},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "UNKNOWN_OPERATION"
    assert result.diagnostics[0].stage == "operation_resolution"
    assert result.diagnostics[0].message == (
        "Operation 'missing.operation' is not installed."
    )
    assert "math.find" in (result.diagnostics[0].hint or "")
    assert result.output == {
        "error": result.diagnostics[0].model_dump(mode="json", exclude_none=True)
    }


def test_policy_denial_does_not_embed_the_policy_document(
    operation_core_services: DomainTestServices,
) -> None:
    core = operation_core_services.core
    operation_core_services.installation.register_operation(ComputedAdapter())
    core.operations.policy = OperationVisibilityPolicy(
        denied_operation_ids=frozenset({"example.double"})
    )

    result = core.operations.invoke(
        OperationRequest(operation_id="example.double", input={"value": 21})
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "OPERATION_POLICY_DENIED"
    assert result.diagnostics[0].details == {
        "policy_profile": "DEFAULT",
        "policy_digest": core.operations.policy.digest,
        "reasons": ["operation_id_denied"],
        "checker_authorization_affected": False,
    }
    assert result.output == {
        "error": result.diagnostics[0].model_dump(mode="json", exclude_none=True)
    }


def test_tool_id_owns_role(
    operation_core_services: DomainTestServices,
) -> None:
    """Tool identity determines role; no client mode switch."""
    core = operation_core_services.core
    operation_core_services.installation.register_operation(ComputedAdapter())

    result = core.operations.invoke(
        OperationRequest(
            operation_id="example.double",
            input={"value": 21},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["value"] == 42


def test_invalid_operation_input_does_not_echo_payload(
    operation_core_services: DomainTestServices,
) -> None:
    core = operation_core_services.core
    operation_core_services.installation.register_operation(ComputedAdapter())

    result = core.operations.invoke(
        OperationRequest(
            operation_id="example.double",
            input={"value": "fixture-secret-value"},
        )
    )

    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "INVALID_REQUEST"
    assert diagnostic.path == "value"
    assert diagnostic.message == "The operation request is invalid."
    assert diagnostic.actual_type is None
    assert diagnostic.expected is None
    assert "fixture-secret-value" not in diagnostic.message
    assert diagnostic.hint == (
        "1 validation error; first at value: "
        "Request value violates validation rule int_type"
    )
    assert diagnostic.details == {}
    assert "fixture-secret-value" not in repr(diagnostic)


def test_operation_input_does_not_coerce_numeric_strings(
    operation_core_services: DomainTestServices,
) -> None:
    core = operation_core_services.core
    operation_core_services.installation.register_operation(ComputedAdapter())

    result = core.operations.invoke(
        OperationRequest(operation_id="example.double", input={"value": "21"})
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_REQUEST"
    assert result.diagnostics[0].path == "value"


def test_non_json_operation_input_is_an_invalid_request(
    operation_core_services: DomainTestServices,
) -> None:
    core = operation_core_services.core
    operation_core_services.installation.register_operation(ComputedAdapter())

    result = core.operations.invoke(
        OperationRequest(operation_id="example.double", input={"value": object()})
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_REQUEST"
    assert result.diagnostics[0].stage == "operation_input_validation"
    assert result.diagnostics[0].path is None


def test_lean_check_parses_its_typed_request_before_execution(
    operation_core_services: DomainTestServices,
) -> None:
    class _UnexpectedLean:
        def verify(self, **_kwargs: object) -> None:
            raise AssertionError("invalid Lean input must not execute")

    adapter = LeanCheckAdapter(
        cast(Any, _UnexpectedLean()),
        TEST_RUNTIME.model_copy(update={"provider": "jacobian.lean4"}),
    )
    core = operation_core_services.core
    operation_core_services.installation.register_operation(adapter)

    result = core.operations.invoke(
        OperationRequest(
            operation_id="lean.check",
            input={"statement": "True"},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_REQUEST"
    assert result.diagnostics[0].path == "proof"


def test_adapter_failure_does_not_expose_internal_exception_text(
    operation_core_services: DomainTestServices,
) -> None:
    core = operation_core_services.core
    operation_core_services.installation.register_operation(CrashingAdapter())

    result = core.operations.invoke(
        OperationRequest(
            operation_id="example.crash",
            input={},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "ADAPTER_EXECUTION_FAILED"
    assert result.diagnostics[0].message == (
        "The operation stopped before returning a result."
    )
    assert result.diagnostics[0].hint == (
        "Retry once. If it fails again, inspect the local Jacobian log for this "
        "operation."
    )
    assert "fixture" not in result.execution.detail
    assert "RuntimeError" not in result.execution.detail


def test_adapter_cannot_promote_without_a_local_verification_record(
    operation_core_services: DomainTestServices,
) -> None:
    core = operation_core_services.core
    operation_core_services.installation.register_operation(ForgedVerifiedAdapter())

    with pytest.raises(OperationError, match="verification record"):
        core.operations.invoke(
            OperationRequest(
                operation_id="example.forged",
                input={},
            )
        )
