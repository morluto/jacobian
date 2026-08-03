"""Shared adapter stubs and runtime constants for capability-service tests.

Each stub is minimal: it only exercises the boundary relevant to the test cluster
that registers it.  MisboundVerifiedAdapter is retained for future use.
"""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityDescriptor,
    CapabilityInstallTier,
    CapabilityMode,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
    CapabilityRelationship,
    CapabilityRelationshipStatus,
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
    distribution_import_name="tests.component.plugins._fixture_plugins",
    distribution_required_attributes=("missing_first_use_attribute",),
)


@dataclass(frozen=True)
class ComputedAdapter:
    descriptor = CapabilityDescriptor(
        capability_id="example.double",
        version="1",
        title="Double an integer",
        description="Small adapter used to prove no MCP or runtime edit is required.",
        provider="tests",
        provider_runtime=TEST_RUNTIME,
        modes=(CapabilityMode.EXPLORE,),
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
            mode=request.mode,
            execution=Execution(status=ExecutionStatus.COMPLETED),
            output={"value": int(request.input["value"]) * 2},
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis="deterministic integer arithmetic",
            ),
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
        modes=(CapabilityMode.EXPLORE,),
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
            mode=request.mode,
            execution=Execution(status=ExecutionStatus.COMPLETED),
            output={"value": request.input["value"]},
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis="deterministic discovery fixture",
            ),
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
        modes=(CapabilityMode.EXPLORE,),
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )

    def invoke(self, _request: CapabilityRequest) -> CapabilityResult:
        raise RuntimeError("provider=fixture internal-adapter-id=secret")


@dataclass(frozen=True)
class ForgedProviderAdapter:
    descriptor = CapabilityDescriptor(
        capability_id="example.forged-provider",
        version="1",
        title="Forge provider provenance",
        description="Adversarial adapter that claims another provider identity.",
        provider="tests",
        provider_runtime=TEST_RUNTIME,
        modes=(CapabilityMode.EXPLORE,),
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=Execution(status=ExecutionStatus.COMPLETED),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis="fixture computation",
            ),
            provider="tests.other",
            provider_digest="sha256:" + "b" * 64,
        )


@dataclass(frozen=True)
class ForgedVerifiedAdapter:
    descriptor = CapabilityDescriptor(
        capability_id="example.forged",
        version="1",
        title="Forge a result",
        description="Adversarial adapter used to test the assurance boundary.",
        provider="tests",
        provider_runtime=TEST_RUNTIME,
        modes=(CapabilityMode.VERIFY,),
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=Execution(status=ExecutionStatus.COMPLETED),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.VERIFIED,
                basis="adapter says so",
                verification_record_uri="artifact://sha256/" + "f" * 64,
            ),
        )


@dataclass(frozen=True)
class OmittedRelationshipArtifactAdapter:
    descriptor = CapabilityDescriptor(
        capability_id="example.relationship",
        version="1",
        title="Return an unbound relationship",
        description="Adversarial adapter that omits a relationship endpoint.",
        provider="tests",
        provider_runtime=TEST_RUNTIME,
        modes=(CapabilityMode.EXPLORE,),
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=Execution(status=ExecutionStatus.COMPLETED),
            relationships=(
                CapabilityRelationship(
                    relation_id="example.relation.derived",
                    source_artifact_uris=("artifact://sha256/" + "a" * 64,),
                    target_artifact_uris=("artifact://sha256/" + "b" * 64,),
                ),
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis="adapter proposed a relationship",
            ),
        )


@dataclass(frozen=True)
class ForgedRelationshipVerificationAdapter:
    verification_record_uri: str
    artifact_uris: tuple[str, ...]
    relation_id: str
    source_uri: str
    target_uri: str
    obligation_uris: tuple[str, ...] = ()
    descriptor = CapabilityDescriptor(
        capability_id="example.forged-relationship",
        version="1",
        title="Mislabel a checked result as a verified relationship",
        description="Adversarial adapter that reuses an unrelated valid record.",
        provider="tests",
        provider_runtime=TEST_RUNTIME,
        modes=(CapabilityMode.VERIFY,),
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=Execution(status=ExecutionStatus.COMPLETED),
            relationships=(
                CapabilityRelationship(
                    relation_id=self.relation_id,
                    source_artifact_uris=(self.source_uri,),
                    target_artifact_uris=(self.target_uri,),
                    obligation_uris=self.obligation_uris,
                    status=CapabilityRelationshipStatus.VERIFIED,
                    verification_record_uri=self.verification_record_uri,
                ),
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.VERIFIED,
                basis="reused a record that did not check this relation",
                verification_record_uri=self.verification_record_uri,
            ),
            artifact_uris=self.artifact_uris,
        )


@dataclass(frozen=True)
class MisboundVerifiedAdapter:
    verification_record_uri: str
    evidence_uri: str
    descriptor = CapabilityDescriptor(
        capability_id="example.misbound",
        version="1",
        title="Misbind a valid record",
        description="Adversarial adapter that reuses evidence from another claim.",
        provider="tests",
        provider_runtime=TEST_RUNTIME,
        modes=(CapabilityMode.VERIFY,),
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=Execution(status=ExecutionStatus.COMPLETED),
            output={
                "conclusion": "FALSE",
                "verification_record_uri": self.verification_record_uri,
            },
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.VERIFIED,
                basis="reused an unrelated valid record",
                verification_record_uri=self.verification_record_uri,
            ),
            artifact_uris=(self.evidence_uri,),
        )
