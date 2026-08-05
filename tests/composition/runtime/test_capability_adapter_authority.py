"""Tests for adapter authority and forgery boundaries.

Covers: invocation recording, provider readiness, error diagnostics, forged
provenance, forged VERIFIED assurance, relationship endpoint exposure, and
relationship verification record binding.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.composition.runtime.capability_service_support import (
    ComputedAdapter,
    CrashingAdapter,
    ForgedProviderAdapter,
    ForgedRelationshipVerificationAdapter,
    ForgedVerifiedAdapter,
    NotReadyProviderAdapter,
    OmittedRelationshipArtifactAdapter,
)
from tests.support.services import DomainTestServices

from jacobian.capability_service import CapabilityError
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.runtime import create_runtime
from jacobian.runtime.model import JacobianRuntime


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
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED


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
    assert "capability.describe" in (result.diagnostics[0].hint or "")
    assert result.output["available_capability_ids"]


def test_unsupported_capability_mode_lists_available_modes(
    capability_core_services: DomainTestServices,
) -> None:
    core = capability_core_services.core
    capability_core_services.installation.register_capability(ComputedAdapter())

    result = core.capabilities.invoke(
        CapabilityRequest(
            capability_id="example.double",
            mode=CapabilityMode.VERIFY,
            input={"value": 21},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "UNSUPPORTED_MODE"
    assert "capability.describe" in (result.diagnostics[0].hint or "")
    assert result.output["available_modes"] == ["EXPLORE"]


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


def test_adapter_cannot_forge_provider_provenance(
    capability_core_services: DomainTestServices,
) -> None:
    core = capability_core_services.core
    capability_core_services.installation.register_capability(ForgedProviderAdapter())

    with pytest.raises(
        CapabilityError,
        match="provider runtime differs from its descriptor",
    ):
        core.capabilities.invoke(
            CapabilityRequest(
                capability_id="example.forged-provider",
                input={},
            )
        )


def test_external_adapter_loads_from_an_operator_entrypoint(
    tmp_path: Path,
    attached_complete_runtime: None,
) -> None:
    _ = attached_complete_runtime
    runtime = create_runtime(
        tmp_path,
        capability_adapter_entrypoints=(
            "tests.component.capabilities._fixture_capabilities:create_adapter",
        ),
    )

    try:
        result = runtime.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="fixture.increment",
                input={"value": 4},
            )
        )
        assert result.output == {"value": 5}
    finally:
        runtime.close()


def test_adapter_cannot_promote_without_a_local_verification_record(
    capability_core_services: DomainTestServices,
) -> None:
    core = capability_core_services.core
    capability_core_services.installation.register_capability(ForgedVerifiedAdapter())

    with pytest.raises(CapabilityError, match="verification record"):
        core.capabilities.invoke(
            CapabilityRequest(
                capability_id="example.forged",
                mode=CapabilityMode.VERIFY,
                input={},
            )
        )


def test_first_class_relationship_endpoints_must_be_exposed(
    capability_core_services: DomainTestServices,
) -> None:
    core = capability_core_services.core
    capability_core_services.installation.register_capability(
        OmittedRelationshipArtifactAdapter()
    )

    with pytest.raises(CapabilityError, match="missing from artifact_uris"):
        core.capabilities.invoke(
            CapabilityRequest(
                capability_id="example.relationship",
                input={},
            )
        )


def test_verified_relationship_must_match_checker_selected_endpoints(
    authorized_complete_runtime: JacobianRuntime,
) -> None:
    runtime = authorized_complete_runtime
    verified = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="case.partition.finite",
            mode=CapabilityMode.VERIFY,
            input={
                "universe": ["a", "b"],
                "cases": [
                    {"case_id": "left", "members": ["a"]},
                    {"case_id": "right", "members": ["b"]},
                ],
                "require_disjoint": True,
            },
        )
    )
    record_uri = verified.assurance.verification_record_uri
    assert record_uri is not None
    record = runtime.core.store.get(record_uri)
    forged = ForgedRelationshipVerificationAdapter(
        verification_record_uri=record_uri,
        artifact_uris=(*record.manifest.parents, record_uri),
        relation_id="case.relation.partitions",
        source_uri=verified.output["claim_uri"],
        target_uri=verified.output["partition_uri"],
    )
    runtime.core.capabilities.register(forged)

    with pytest.raises(CapabilityError, match="endpoints differ"):
        runtime.core.capabilities.invoke(
            CapabilityRequest(
                capability_id=forged.descriptor.capability_id,
                mode=CapabilityMode.VERIFY,
                input={},
            )
        )


def test_verified_relationship_must_match_checker_selected_obligation(
    authorized_complete_runtime: JacobianRuntime,
) -> None:
    runtime = authorized_complete_runtime
    verified = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="case.partition.finite",
            mode=CapabilityMode.VERIFY,
            input={
                "universe": ["a"],
                "cases": [{"case_id": "only", "members": ["a"]}],
                "require_disjoint": True,
            },
        )
    )
    record_uri = verified.assurance.verification_record_uri
    assert record_uri is not None
    record = runtime.core.store.get(record_uri)
    forged = ForgedRelationshipVerificationAdapter(
        verification_record_uri=record_uri,
        artifact_uris=(*record.manifest.parents, record_uri),
        relation_id="case.relation.partitions",
        source_uri=verified.output["scope_uri"],
        target_uri=verified.output["partition_uri"],
        obligation_uris=(verified.output["certificate_uri"],),
    )
    runtime.core.capabilities.register(forged)

    with pytest.raises(CapabilityError, match="obligations differ"):
        runtime.core.capabilities.invoke(
            CapabilityRequest(
                capability_id=forged.descriptor.capability_id,
                mode=CapabilityMode.VERIFY,
                input={},
            )
        )
