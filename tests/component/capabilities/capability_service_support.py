"""Shared adapter stubs and runtime constants for capability-service tests.

Each stub is minimal: it only exercises the boundary relevant to the test cluster
that registers it.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ConfigDict

from jacobian.capability_adapters import parse_capability_input
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
    CapabilityRequest,
)
from jacobian.contracts.results import (
    ContractModel,
)
from jacobian.operation_projection import OperationProjection
from jacobian.operation_publication import PublishedOperation
from jacobian.operations import Completed
from jacobian.schema_registry import model_schema

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


class _Value(ContractModel):
    value: int


class _ValueRequest(ContractModel):
    value: int


class _EmptyRequest(ContractModel):
    model_config = ConfigDict(extra="forbid")


class _WrongValue(ContractModel):
    value: str


@dataclass(frozen=True)
class ComputedAdapter:
    descriptor = CapabilityDescriptor(
        capability_id="example.double",
        version="1",
        title="Double an integer",
        description="Small adapter used to prove no MCP or runtime edit is required.",
        provider="tests",
        provider_runtime=TEST_RUNTIME,
        input_schema=model_schema(_ValueRequest),
        output_schema=model_schema(_Value),
        tags=("test",),
    )

    def prepare(self, request: CapabilityRequest) -> _ValueRequest:
        return parse_capability_input(_ValueRequest, request.input)

    def invoke(self, parsed: _ValueRequest) -> OperationProjection:
        value = _Value(value=parsed.value * 2)
        return OperationProjection(
            operation_id=self.descriptor.capability_id,
            version=self.descriptor.version,
            terminal=Completed(value=value),
            publication=PublishedOperation(output=value),
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

    def prepare(self, request: CapabilityRequest) -> _EmptyRequest:
        return parse_capability_input(_EmptyRequest, request.input)

    def invoke(self, _request: _EmptyRequest) -> OperationProjection:
        raise AssertionError("provider must be rejected before adapter invocation")


@dataclass(frozen=True)
class MismatchedOutputAdapter:
    descriptor = CapabilityDescriptor(
        capability_id="example.mismatched-output",
        version="1",
        title="Return the wrong typed value",
        description="Fixture for the installed output-model boundary.",
        provider="tests",
        provider_runtime=TEST_RUNTIME,
        input_schema=model_schema(_EmptyRequest),
        output_schema=model_schema(_Value),
    )

    def prepare(self, request: CapabilityRequest) -> _EmptyRequest:
        return parse_capability_input(_EmptyRequest, request.input)

    def invoke(self, _request: _EmptyRequest) -> OperationProjection:
        value = _WrongValue(value="not-an-integer")
        return OperationProjection(
            operation_id=self.descriptor.capability_id,
            version=self.descriptor.version,
            terminal=Completed(value=value),
            publication=PublishedOperation(output=value),
        )


@dataclass(frozen=True)
class InvalidOutputValueAdapter:
    descriptor = CapabilityDescriptor(
        capability_id="example.invalid-output-value",
        version="1",
        title="Return an invalid typed value",
        description="Fixture for unchecked model construction at publication.",
        provider="tests",
        provider_runtime=TEST_RUNTIME,
        input_schema=model_schema(_EmptyRequest),
        output_schema=model_schema(_Value),
    )

    def prepare(self, request: CapabilityRequest) -> _EmptyRequest:
        return parse_capability_input(_EmptyRequest, request.input)

    def invoke(self, _request: _EmptyRequest) -> OperationProjection:
        value = _Value.model_construct(value="not-an-integer")
        return OperationProjection(
            operation_id=self.descriptor.capability_id,
            version=self.descriptor.version,
            terminal=Completed(value=value),
            publication=PublishedOperation(output=value),
        )


@dataclass(frozen=True)
class DiscoveryAdapter:
    descriptor: CapabilityDescriptor

    def prepare(self, request: CapabilityRequest) -> _ValueRequest:
        return parse_capability_input(_ValueRequest, request.input)

    def invoke(self, request: _ValueRequest) -> OperationProjection:
        value = _Value(value=request.value)
        return OperationProjection(
            operation_id=self.descriptor.capability_id,
            version=self.descriptor.version,
            terminal=Completed(value=value),
            publication=PublishedOperation(output=value),
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

    def prepare(self, request: CapabilityRequest) -> _EmptyRequest:
        return parse_capability_input(_EmptyRequest, request.input)

    def invoke(self, _request: _EmptyRequest) -> OperationProjection:
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
        output_schema=model_schema(_Value),
    )

    def prepare(self, request: CapabilityRequest) -> _EmptyRequest:
        return parse_capability_input(_EmptyRequest, request.input)

    def invoke(self, request: _EmptyRequest) -> OperationProjection:
        del request
        record_uri = "artifact://sha256/" + "f" * 64
        value = _Value(value=0)
        return OperationProjection(
            operation_id=self.descriptor.capability_id,
            version=self.descriptor.version,
            terminal=Completed(value=value),
            publication=PublishedOperation(
                output=value,
                artifact_uris=(record_uri,),
            ),
            verification_record_uri=record_uri,
        )
