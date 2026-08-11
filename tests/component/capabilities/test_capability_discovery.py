"""Tests for capability discovery: routing, lexical fit, input contracts, and registration."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from tests.component.capabilities.capability_service_support import (
    TEST_RUNTIME,
    DiscoveryAdapter,
)
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.capability_service import CapabilityError
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiscoveryRequest,
    CapabilityInputKind,
    CapabilityInvocationExample,
)


@pytest.fixture
def capability_core_services(tmp_path) -> Iterator[DomainTestServices]:
    with open_domain_services(tmp_path / "state") as services:
        yield services


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
                input_schema=schema,
                output_schema=schema,
                tags=("counterexample", "bounded-search"),
                invocation_examples=(
                    CapabilityInvocationExample(
                        name="small",
                        description="Use a small integer fixture.",
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
                input_schema=schema,
                output_schema=schema,
                tags=("graph", "checker"),
            )
        )
    )

    request = CapabilityDiscoveryRequest(
        query="find a counterexample to associativity",
        domain="fixture-algebra",
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
        )
    )
    assert formal_intent.resolved_input_kind is None
    assert [match.capability_id for match in formal_intent.matches] == [
        "fixture_sat.proof.verify"
    ]

    formal_trace = capability_core_services.core.capabilities.discover(
        CapabilityDiscoveryRequest(
            query="verify an LRAT proof trace",
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
                    input={"value": "not-an-integer"},
                ),
            ),
        )
    )

    with pytest.raises(CapabilityError, match="invocation example"):
        capability_core_services.installation.register_capability(adapter)
