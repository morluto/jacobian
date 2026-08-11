"""Tests for capability discovery: routing, lexical fit, input contracts, and registration."""

from __future__ import annotations

import pytest
from tests.composition.runtime.capability_service_support import (
    TEST_RUNTIME,
    DiscoveryAdapter,
)
from tests.support.services import DomainTestServices

from jacobian.capability_service import CapabilityError
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiscoveryRequest,
    CapabilityInputKind,
    CapabilityInvocationExample,
    CapabilityMode,
)
from jacobian.runtime.model import JacobianRuntime


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
    assert first.domain_filter_status == "MATCHED"
    assert "matches at least one installed capability" in first.domain_filter_basis
    assert "fixture_algebra" in first.available_domains
    assert "fixture_graph" in first.available_domains


def test_discovery_distinguishes_unknown_domain_from_lexical_absence(
    capability_core_services: DomainTestServices,
) -> None:
    schema = {"type": "object"}
    capability_core_services.installation.register_capability(
        DiscoveryAdapter(
            CapabilityDescriptor(
                capability_id="fixture_probability.event.compute",
                version="1",
                title="Compute event probability",
                description="Compute one exact finite event probability.",
                provider="tests",
                provider_runtime=TEST_RUNTIME,
                modes=(CapabilityMode.EXPLORE,),
                input_schema=schema,
                output_schema=schema,
                tags=("probability", "exact"),
            )
        )
    )

    discovered = capability_core_services.core.capabilities.discover(
        CapabilityDiscoveryRequest(
            query="compute exact event probability",
            domain="arithmetic",
        )
    )

    assert discovered.matches == ()
    assert discovered.domain == "arithmetic"
    assert discovered.domain_filter_status == "UNKNOWN"
    assert "matches no installed capability" in discovered.domain_filter_basis
    assert "lexical fit outside that filter was not assessed" in (
        discovered.portfolio_fit_basis
    )

    browsed = capability_core_services.core.capabilities.discover(
        CapabilityDiscoveryRequest(domain="arithmetic")
    )
    assert browsed.matches == ()
    assert browsed.domain_filter_status == "UNKNOWN"
    assert browsed.portfolio_fit == "UNFILTERED"
    assert browsed.routing_status == "UNFILTERED"


def test_discovery_recognizes_hidden_installed_domains_without_returning_them(
    authorized_complete_runtime: JacobianRuntime,
) -> None:
    discovered = authorized_complete_runtime.core.capabilities.discover(
        CapabilityDiscoveryRequest(domain="artifact")
    )

    assert discovered.domain == "artifact"
    assert discovered.domain_filter_status == "MATCHED"
    assert "matches at least one installed capability" in discovered.domain_filter_basis
    assert discovered.matches == ()
    assert "artifact" not in discovered.available_domains


def test_storage_primitive_is_catalogued_but_not_discovered(
    authorized_complete_runtime: JacobianRuntime,
) -> None:
    capabilities = authorized_complete_runtime.core.capabilities
    catalog_ids = {
        descriptor.capability_id for descriptor in capabilities.catalog().capabilities
    }
    discovered_ids = {
        match.capability_id
        for match in capabilities.discover(CapabilityDiscoveryRequest(limit=20)).matches
    }

    assert "artifact.put" in catalog_ids
    assert "artifact.put" not in discovered_ids


def test_bounded_search_producers_advertise_produced_artifact_types(
    authorized_complete_runtime: JacobianRuntime,
) -> None:
    descriptors = {
        descriptor.capability_id: descriptor
        for descriptor in (
            authorized_complete_runtime.core.capabilities.catalog().capabilities
        )
    }
    induced_tree = descriptors["graph.induced_tree.maximum.compute"]
    assert induced_tree.produced_artifact_types, (
        "bounded-search producers must advertise their materialized result schema"
    )


def test_materialize_to_width_produced_types_are_symmetric_and_discoverable(
    authorized_complete_runtime: JacobianRuntime,
) -> None:
    descriptors = {
        descriptor.capability_id: descriptor
        for descriptor in (
            authorized_complete_runtime.core.capabilities.catalog().capabilities
        )
    }
    assert descriptors["poset.finite.compute"].produced_artifact_types == ()
    assert descriptors["poset.width.compute"].accepted_artifact_types == ()


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


def test_discovery_rejects_unsupported_natural_language_proof_routes(
    capability_core_services: DomainTestServices,
) -> None:
    schema = {"type": "object"}
    capability_core_services.installation.register_capability(
        DiscoveryAdapter(
            CapabilityDescriptor(
                capability_id="fixture_sat.proof.verify",
                version="1",
                title="Verify a formal UNSAT proof",
                description="Replay one structured formal proof certificate.",
                provider="tests",
                provider_runtime=TEST_RUNTIME,
                modes=(CapabilityMode.VERIFY,),
                input_schema=schema,
                output_schema=schema,
                tags=("proof", "verify"),
            )
        )
    )
    discovered = capability_core_services.core.capabilities.discover(
        CapabilityDiscoveryRequest(
            query=(
                "Independently verify this natural-language proof trace: "
                "suppose n is even, so n = 2k."
            ),
            limit=20,
        )
    )

    assert discovered.resolved_input_kind == (
        CapabilityInputKind.NATURAL_LANGUAGE_PROOF
    )
    assert discovered.routing_status == "NO_ROUTE"
    assert discovered.matches == ()
    assert "No installed capability accepts" in discovered.routing_basis

    formal_method = capability_core_services.core.capabilities.discover(
        CapabilityDiscoveryRequest(
            query="check a formal Lean proof by contradiction",
            limit=20,
        )
    )
    assert formal_method.resolved_input_kind is None
    assert formal_method.routing_status == "UNFILTERED"

    written_formal_proof = capability_core_services.core.capabilities.discover(
        CapabilityDiscoveryRequest(
            query="verify the written proof in Lean",
            limit=20,
        )
    )
    assert written_formal_proof.resolved_input_kind is None
    assert written_formal_proof.routing_status == "UNFILTERED"

    explicitly_structured = capability_core_services.core.capabilities.discover(
        CapabilityDiscoveryRequest(
            query="formal UNSAT proof",
            mode=CapabilityMode.VERIFY,
            input_kind=CapabilityInputKind.STRUCTURED_REQUEST,
            limit=20,
        )
    )
    assert [match.capability_id for match in explicitly_structured.matches] == [
        "fixture_sat.proof.verify"
    ]
    assert explicitly_structured.routing_status == "ROUTES_FOUND"

    formal_intent = capability_core_services.core.capabilities.discover(
        CapabilityDiscoveryRequest(
            query="formal UNSAT proof",
            mode=CapabilityMode.VERIFY,
        )
    )
    assert formal_intent.resolved_input_kind is None
    assert [match.capability_id for match in formal_intent.matches] == [
        "fixture_sat.proof.verify"
    ]

    formal_trace = capability_core_services.core.capabilities.discover(
        CapabilityDiscoveryRequest(
            query="verify an LRAT proof trace",
            mode=CapabilityMode.VERIFY,
        )
    )
    assert formal_trace.resolved_input_kind is None


def test_discovery_routes_only_declared_input_and_artifact_contracts(
    capability_core_services: DomainTestServices,
) -> None:
    proof_schema_uri = "artifact://sha256/" + ("1" * 64)
    other_schema_uri = "artifact://sha256/" + ("2" * 64)
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    capability_core_services.installation.register_capability(
        DiscoveryAdapter(
            CapabilityDescriptor(
                capability_id="fixture_claim.elaborate",
                version="1",
                title="Elaborate a formal proposition",
                description="Accept formal proposition syntax.",
                provider="tests",
                provider_runtime=TEST_RUNTIME,
                modes=(CapabilityMode.EXPLORE,),
                input_schema=schema,
                output_schema=schema,
                tags=("formal", "proposition"),
                accepted_input_kinds=(CapabilityInputKind.FORMAL_PROPOSITION,),
            )
        )
    )
    capability_core_services.installation.register_capability(
        DiscoveryAdapter(
            CapabilityDescriptor(
                capability_id="fixture_claim.replay",
                version="1",
                title="Replay a typed proof artifact",
                description="Accept one exact bound proof artifact.",
                provider="tests",
                provider_runtime=TEST_RUNTIME,
                modes=(CapabilityMode.VERIFY,),
                input_schema=schema,
                output_schema=schema,
                tags=("proof", "artifact"),
                accepted_input_kinds=(CapabilityInputKind.TYPED_ARTIFACT,),
                accepted_artifact_types=(proof_schema_uri,),
            )
        )
    )
    service = capability_core_services.core.capabilities

    formal = service.discover(
        CapabilityDiscoveryRequest(
            query="formal proposition",
            input_kind=CapabilityInputKind.FORMAL_PROPOSITION,
        )
    )
    assert [match.capability_id for match in formal.matches] == [
        "fixture_claim.elaborate"
    ]
    assert formal.routing_status == "ROUTES_FOUND"

    typed = service.discover(
        CapabilityDiscoveryRequest(
            query="proof artifact",
            mode=CapabilityMode.VERIFY,
            input_kind=CapabilityInputKind.TYPED_ARTIFACT,
            artifact_type=proof_schema_uri,
        )
    )
    assert [match.capability_id for match in typed.matches] == ["fixture_claim.replay"]
    assert typed.routing_status == "ROUTES_FOUND"

    mismatched = service.discover(
        CapabilityDiscoveryRequest(
            query="proof artifact",
            input_kind=CapabilityInputKind.TYPED_ARTIFACT,
            artifact_type=other_schema_uri,
        )
    )
    assert mismatched.matches == ()
    assert mismatched.routing_status == "NO_ROUTE"

    lexically_absent = service.discover(
        CapabilityDiscoveryRequest(
            query="quuxonium",
            input_kind=CapabilityInputKind.FORMAL_PROPOSITION,
        )
    )
    assert lexically_absent.matches == ()
    assert lexically_absent.routing_status == "ROUTES_FOUND"
    assert lexically_absent.portfolio_fit == "NO_LEXICAL_MATCHES"

    incompatible_lexical_match = service.discover(
        CapabilityDiscoveryRequest(
            query="formal proposition",
            input_kind=CapabilityInputKind.STRUCTURED_REQUEST,
        )
    )
    assert incompatible_lexical_match.matches == ()
    assert incompatible_lexical_match.routing_status == "NO_ROUTE"
    assert incompatible_lexical_match.portfolio_fit == "STRONG_CANDIDATES_FOUND"


def test_discovery_artifact_type_requires_typed_artifact_input() -> None:
    proof_schema_uri = "artifact://sha256/" + ("1" * 64)
    with pytest.raises(
        ValueError,
        match="artifact_type requires input_kind=TYPED_ARTIFACT",
    ):
        CapabilityDiscoveryRequest(
            query="proof artifact",
            input_kind=CapabilityInputKind.STRUCTURED_REQUEST,
            artifact_type=proof_schema_uri,
        )
    with pytest.raises(
        ValueError,
        match="TYPED_ARTIFACT input requires artifact_type",
    ):
        CapabilityDiscoveryRequest(
            query="proof artifact",
            input_kind=CapabilityInputKind.TYPED_ARTIFACT,
        )


def test_descriptor_artifact_contract_requires_typed_artifact_input() -> None:
    proof_schema_uri = "artifact://sha256/" + ("1" * 64)
    with pytest.raises(
        ValueError,
        match="accepted artifact types require TYPED_ARTIFACT input",
    ):
        CapabilityDescriptor(
            capability_id="fixture.invalid.artifact",
            version="1",
            title="Invalid artifact contract",
            description="Invalid routing metadata fixture.",
            provider="tests",
            provider_runtime=TEST_RUNTIME,
            modes=(CapabilityMode.EXPLORE,),
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            accepted_artifact_types=(proof_schema_uri,),
        )

    with pytest.raises(
        ValueError,
        match="TYPED_ARTIFACT input requires accepted artifact types",
    ):
        CapabilityDescriptor(
            capability_id="fixture.invalid.typed-artifact",
            version="1",
            title="Invalid typed artifact contract",
            description="Typed artifact routing requires an exact stored schema.",
            provider="tests",
            provider_runtime=TEST_RUNTIME,
            modes=(CapabilityMode.EXPLORE,),
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            accepted_input_kinds=(CapabilityInputKind.TYPED_ARTIFACT,),
        )


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
