from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.support.provider_lean import (
    PINNED_MATHLIB_RUNTIME_UNAVAILABLE_REASON,
    pinned_mathlib_runtime_available,
)
from tests.support.services import DomainTestServices

from jacobian.capabilities import CapabilityError
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityDescriptor,
    CapabilityDiscoveryRequest,
    CapabilityInstallTier,
    CapabilityInvocationExample,
    CapabilityMode,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
    CapabilityRelationship,
    CapabilityRelationshipStatus,
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.contracts.memory import ResearchEpisode
from jacobian.contracts.results import (
    Execution,
    ExecutionStatus,
)
from jacobian.runtime import create_runtime
from jacobian.runtime.model import JacobianRuntime

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
    assert result.episode_uri is not None
    episode = core.store.get(result.episode_uri)
    assert episode.payload["result"]["response_version"] == "2"
    assert episode.payload["result"]["completeness"]["status"] == "NOT_APPLICABLE"
    hits = core.memory.search(query="double computed").hits
    assert [hit.episode_uri for hit in hits] == [result.episode_uri]


def test_installed_capability_discovery_is_compact_deterministic_and_transparent(
    capability_core_services: DomainTestServices,
) -> None:
    core = capability_core_services.core
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    capability_core_services.installation.register_capability(
        DiscoveryAdapter(
            CapabilityDescriptor(
                capability_id="fixture_algebra.search.countermodel",
                version="1",
                title="Search finite countermodels",
                description="Find a finite algebra that falsifies a target law.",
                provider="tests",
                provider_runtime=TEST_RUNTIME,
                modes=(CapabilityMode.EXPLORE,),
                input_schema=schema,
                output_schema=schema,
                tags=("counterexample", "bounded-search"),
                invocation_examples=(
                    CapabilityInvocationExample(
                        name="small",
                        description="Use a small integer fixture.",
                        mode=CapabilityMode.EXPLORE,
                        input={"value": 2},
                    ),
                ),
            )
        )
    )
    capability_core_services.installation.register_capability(
        DiscoveryAdapter(
            CapabilityDescriptor(
                capability_id="fixture_graph.verify.coloring",
                version="1",
                title="Verify a graph coloring",
                description="Independently check a proposed graph coloring.",
                provider="tests",
                provider_runtime=TEST_RUNTIME,
                modes=(CapabilityMode.VERIFY,),
                input_schema=schema,
                output_schema=schema,
                tags=("graph", "checker"),
            )
        )
    )

    request = CapabilityDiscoveryRequest(
        query="find a counterexample to associativity",
        domain="fixture-algebra",
        mode=CapabilityMode.EXPLORE,
        limit=10,
    )
    first = core.capabilities.discover(request)
    second = core.capabilities.discover(request)

    assert first == second
    assert [match.capability_id for match in first.matches] == [
        "fixture_algebra.search.countermodel"
    ]
    assert first.matches[0].matched_on == ("tags",)
    assert first.matches[0].matched_terms == ("counterexample",)
    assert first.matches[0].has_invocation_examples is True
    assert first.matches[0].lexical_fit == "WEAK_LEXICAL_MATCH"
    assert first.portfolio_fit == "ONLY_WEAK_LEXICAL_MATCHES"
    assert first.domain == "fixture_algebra"
    assert "fixture_algebra" in first.available_domains
    assert "fixture_graph" in first.available_domains


def test_discovery_distinguishes_strong_weak_and_absent_lexical_fit(
    authorized_complete_runtime: JacobianRuntime,
) -> None:
    runtime = authorized_complete_runtime

    strong = runtime.core.capabilities.discover(
        CapabilityDiscoveryRequest(
            query="graded Jacobian syzygy minimum degree",
            limit=3,
        )
    )
    assert strong.portfolio_fit == "STRONG_CANDIDATES_FOUND"
    assert strong.matches[0].capability_id == (
        "polynomial.jacobian_syzygy.minimum_degree.compute"
    )
    assert strong.matches[0].lexical_fit == "STRONG_CANDIDATE"
    assert strong.matches[0].query_coverage_milli == 1000

    gaussian = runtime.core.capabilities.discover(
        CapabilityDiscoveryRequest(
            query="exact fixed order Gaussian polynomial moment Wick contraction",
            limit=3,
        )
    )
    assert gaussian.portfolio_fit == "STRONG_CANDIDATES_FOUND"
    assert gaussian.matches[0].capability_id == (
        "probability.gaussian_polynomial.moment.compute"
    )
    assert gaussian.matches[0].lexical_fit == "STRONG_CANDIDATE"
    assert "does not establish an identity for every order" in (
        gaussian.matches[0].description
    )

    reliability = runtime.core.capabilities.discover(
        CapabilityDiscoveryRequest(
            query="exact small graph reliability terminal connection probability",
            limit=3,
        )
    )
    assert reliability.portfolio_fit == "STRONG_CANDIDATES_FOUND"
    assert reliability.matches[0].capability_id == (
        "probability.graph_reliability.connection_probability.compute"
    )

    symmetry = runtime.core.capabilities.discover(
        CapabilityDiscoveryRequest(
            query=(
                "declared graph automorphism generators vertex edge orbit "
                "symmetry compression"
            ),
            limit=3,
        )
    )
    assert symmetry.portfolio_fit == "STRONG_CANDIDATES_FOUND"
    assert symmetry.matches[0].capability_id == (
        "graph.symmetry.generator_orbits.compute"
    )

    absent = runtime.core.capabilities.discover(
        CapabilityDiscoveryRequest(
            query="quuxonium frobnicator",
            limit=3,
        )
    )
    assert absent.matches == ()
    assert absent.portfolio_fit == "NO_LEXICAL_MATCHES"


def test_capability_registration_rejects_an_invalid_invocation_example(
    capability_core_services: DomainTestServices,
) -> None:
    adapter = DiscoveryAdapter(
        CapabilityDescriptor(
            capability_id="example.invalid-example",
            version="1",
            title="Invalid example fixture",
            description="Advertises an example that violates its input schema.",
            provider="tests",
            provider_runtime=TEST_RUNTIME,
            modes=(CapabilityMode.EXPLORE,),
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            invocation_examples=(
                CapabilityInvocationExample(
                    name="invalid",
                    description="This value has the wrong type.",
                    mode=CapabilityMode.EXPLORE,
                    input={"value": "not-an-integer"},
                ),
            ),
        )
    )

    with pytest.raises(CapabilityError, match="invocation example"):
        capability_core_services.installation.register_capability(adapter)


def test_knowledge_search_filters_episode_domain_tags_and_failures(
    authorized_complete_runtime: JacobianRuntime,
) -> None:
    runtime = authorized_complete_runtime
    graph_episode = ResearchEpisode(
        capability_id="graph.compute.properties",
        capability_version="1",
        mode=CapabilityMode.EXPLORE,
        request={"graph": "K5"},
        result={
            "output": {"failure_classifications": ["nonplanar_obstruction"]},
            "diagnostics": [],
        },
        assurance_level=CapabilityAssuranceLevel.COMPUTED,
        summary="K5 counterexample with a nonplanar obstruction",
        tags=("graph", "counterexample", "failure"),
    )
    graph_uri = runtime.core.memory.record(graph_episode)
    runtime.core.memory.record(
        ResearchEpisode(
            capability_id="lean.check",
            capability_version="1",
            mode=CapabilityMode.VERIFY,
            request={"statement": "True"},
            result={"output": {"conclusion": "TRUE"}, "diagnostics": []},
            assurance_level=CapabilityAssuranceLevel.VERIFIED,
            summary="Lean replay succeeded",
            tags=("lean", "proof"),
        )
    )

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="knowledge.search",
            input={
                "query": "counterexample",
                "domains": ["graph"],
                "tags_all": ["failure"],
                "tags_any": ["counterexample", "proof"],
                "failure_stages": ["mathematical_evaluation"],
                "failure_classifications": ["nonplanar_obstruction"],
                "limit": 10,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.capability_version == "2"
    assert result.completeness.status.value == "COMPLETE"
    assert result.scope is not None
    assert [hit["episode_uri"] for hit in result.output["hits"]] == [graph_uri]
    assert result.output["hits"][0]["matched_query_terms"] == ["counterexample"]
    assert result.output["hits"][0]["matched_filters"] == [
        "domains",
        "tags_all",
        "tags_any",
        "failure_stages",
        "failure_classifications",
    ]
    assert result.output["indexed_episode_count"] == 2
    assert result.scope.parameters["index_snapshot"] == result.output["index_snapshot"]
    assert result.output["total_matches"] == 1
    assert result.output["returned_count"] == 1
    assert result.output["truncated"] is False
    assert result.output["completeness"] == "COMPLETE"
    assert result.output["index_snapshot"].startswith("sha256:")


def test_knowledge_search_reports_snapshot_bounded_partial_results(
    authorized_complete_runtime: JacobianRuntime,
) -> None:
    runtime = authorized_complete_runtime
    for value in (1, 2):
        runtime.core.memory.record(
            ResearchEpisode(
                capability_id="polynomial.factor.compute",
                capability_version="1",
                mode=CapabilityMode.EXPLORE,
                request={"value": value},
                result={"output": {"value": value}, "diagnostics": []},
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
                summary=f"factor episode {value}",
                tags=("polynomial",),
            )
        )

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="knowledge.search",
            input={"domains": ["polynomial"], "limit": 1},
        )
    )

    assert result.output["indexed_episode_count"] == 2
    assert result.completeness.status.value == "PARTIAL"
    assert result.scope is not None
    assert result.scope.parameters["index_snapshot"] == result.output["index_snapshot"]
    assert result.output["total_matches"] == 2
    assert result.output["returned_count"] == 1
    assert result.output["truncated"] is True
    assert result.output["completeness"] == "PARTIAL"


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
    assert result.episode_uri is None
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


@pytest.mark.skipif(
    not pinned_mathlib_runtime_available(),
    reason=PINNED_MATHLIB_RUNTIME_UNAVAILABLE_REASON,
)
def test_lean_capability_returns_bound_verified_result(
    authorized_complete_runtime: JacobianRuntime,
) -> None:
    runtime = authorized_complete_runtime

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="lean.check",
            mode=CapabilityMode.VERIFY,
            input={
                "statement": "1 + 1 = 2",
                "proof": "rfl",
                "environment": "CORE",
            },
        )
    )

    assert result.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert result.assurance.verification_record_uri is not None
    assert result.output["conclusion"] == "TRUE"


@pytest.mark.skipif(
    not pinned_mathlib_runtime_available(),
    reason=PINNED_MATHLIB_RUNTIME_UNAVAILABLE_REASON,
)
def test_lean_capability_projects_repairable_checker_diagnostics(
    authorized_complete_runtime: JacobianRuntime,
) -> None:
    runtime = authorized_complete_runtime

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="lean.check",
            mode=CapabilityMode.VERIFY,
            input={
                "statement": "1 + 1 = 2",
                "proof": "sorry",
                "environment": "CORE",
            },
        )
    )

    assert result.output["input"]["status"] == "REJECTED"
    assert "forbidden Lean command" in result.output["input"]["errors"][0]
    assert result.output["diagnostics"] == result.output["input"]["errors"]
    assert result.assurance.verification_record_uri is None
