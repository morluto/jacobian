"""Shared adapter stubs and runtime constants for capability-service tests.

Each stub is minimal: it only exercises the boundary relevant to the test cluster
that registers it.
"""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.contracts.results import (
    Execution,
    ExecutionStatus,
)

TEST_RUNTIME = CapabilityProviderRuntime(
    provider="tests",
    availability=CapabilityProviderAvailability.AVAILABLE,
    version="1",
    digest="sha256:" + "a" * 64,
    digest_kind=CapabilityProviderDigestKind.SOURCE_TREE,
    platform="any",
    install_tier=CapabilityInstallTier.T0,
    license_id="MIT",
)

NOT_READY_RUNTIME = CapabilityProviderRuntime(
    provider="tests-python",
    availability=CapabilityProviderAvailability.AVAILABLE,
    version="1",
    digest="sha256:" + "b" * 64,
    digest_kind=CapabilityProviderDigestKind.PYTHON_DISTRIBUTION_RECORD,
    platform="any",
    install_tier=CapabilityInstallTier.T0,
    license_id="MIT",
    configuration={"distribution": "tests-fixture"},
    distribution_import_name="tests.support.process_entrypoints",
    distribution_required_attributes=("missing_first_use_attribute",),
)


@dataclass(frozen=True)
class InvalidEvidenceValue:
    evidence: object
    verification_record_uri: str | None = None

    def model_dump(self, *, mode: str) -> dict[str, object]:
        if mode != "json":
            raise ValueError("fixture supports only JSON projection")
        return {
            "kind": "SUFFICIENT",
            "conditions": {},
            "evidence": self.evidence,
            "sample_uris": [],
            "subject_uri": None,
            "verification_record_uri": self.verification_record_uri,
        }


@dataclass(frozen=True)
class ComputedAdapter:
    descriptor = CapabilityDescriptor(
        capability_id="example.double",
        version="1",
        title="Double an integer",
        description="Small adapter used to prove no MCP or runtime edit is required.",
        provider="tests",
        provider_runtime=TEST_RUNTIME,
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        tags=("test",),
    )

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            execution=Execution(status=ExecutionStatus.COMPLETED),
            output={"value": int(request.input["value"]) * 2},
        )


@dataclass(frozen=True)
class InvalidOutputAdapter:
    descriptor = ComputedAdapter.descriptor.model_copy(
        update={
            "capability_id": "example.invalid-output",
            "title": "Return schema-invalid output",
            "description": "Fixture for the adapter result validation boundary.",
        }
    )

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            execution=Execution(status=ExecutionStatus.COMPLETED),
            output={"value": "not-an-integer"},
        )


@dataclass(frozen=True)
class NotReadyProviderAdapter:
    descriptor = CapabilityDescriptor(
        capability_id="example.not-ready-provider",
        version="1",
        title="Provider readiness fixture",
        description="Fixture for the first-use provider readiness boundary.",
        provider="tests-python",
        provider_runtime=NOT_READY_RUNTIME,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )

    def invoke(self, _request: CapabilityRequest) -> CapabilityResult:
        raise AssertionError("provider must be rejected before adapter invocation")


@dataclass(frozen=True)
class DiscoveryAdapter:
    descriptor: CapabilityDescriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            execution=Execution(status=ExecutionStatus.COMPLETED),
            output={"value": request.input["value"]},
        )


@dataclass(frozen=True)
class CrashingAdapter:
    descriptor = CapabilityDescriptor(
        capability_id="example.crash",
        version="1",
        title="Crash during execution",
        description="Fixture for testing public adapter-failure diagnostics.",
        provider="tests",
        provider_runtime=TEST_RUNTIME,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )

    def invoke(self, _request: CapabilityRequest) -> CapabilityResult:
        raise RuntimeError("provider=fixture internal-adapter-id=secret")


@dataclass(frozen=True)
class ForgedVerifiedAdapter:
    descriptor = CapabilityDescriptor(
        capability_id="example.forged",
        version="1",
        title="Forge a result",
        description="Adversarial adapter used to test the verification boundary.",
        provider="tests",
        provider_runtime=TEST_RUNTIME,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        record_uri = "artifact://sha256/" + "f" * 64
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            execution=Execution(status=ExecutionStatus.COMPLETED),
            verification_record_uri=record_uri,
            artifact_uris=(record_uri,),
        )
