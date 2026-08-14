"""Tests for operation discovery: routing, lexical fit, input contracts, and registration."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from tests.component.operations.operation_service_support import (
    TEST_RUNTIME,
    DiscoveryAdapter,
)
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationDiscoveryRequest,
    OperationExample,
    OperationInputKind,
)
from jacobian.operation_errors import OperationError


@pytest.fixture
def operation_core_services(tmp_path) -> Iterator[DomainTestServices]:
    with open_domain_services(tmp_path / "state") as services:
        yield services


def test_installed_operation_discovery_is_compact_deterministic_and_transparent(
    operation_core_services: DomainTestServices,
) -> None:
    core = operation_core_services.core
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    operation_core_services.installation.register_operation(
        DiscoveryAdapter(
            OperationDescriptor(
                operation_id="fixture_algebra.search.countermodel",
                version="1",
                title="Search finite countermodels",
                description="Find a finite algebra that falsifies a target law.",
                provider="tests",
                provider_runtime=TEST_RUNTIME,
                input_schema=schema,
                output_schema=schema,
                tags=("counterexample", "bounded-search"),
                examples=(
                    OperationExample(
                        name="small",
                        description="Use a small integer fixture.",
                        input={"value": 2},
                    ),
                ),
            )
        )
    )
    operation_core_services.installation.register_operation(
        DiscoveryAdapter(
            OperationDescriptor(
                operation_id="fixture_graph.verify.coloring",
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

    request = OperationDiscoveryRequest(
        query="find a counterexample to associativity",
        domain="fixture-algebra",
        limit=10,
    )
    first = core.operations.search(request)
    second = core.operations.search(request)

    assert first == second
    assert [match.operation_id for match in first.matches] == [
        "fixture_algebra.search.countermodel"
    ]
    assert first.matches[0].relevance_score > 0
    assert first.matches[0].applicability == "NEEDS_MORE_TYPED_REQUIREMENTS"
    assert first.matches[0].applicability_code == "FULL_REQUEST_REQUIRED"
    assert first.domain == "fixture_algebra"


def test_discovery_applies_domain_filter_without_extra_status_prose(
    operation_core_services: DomainTestServices,
) -> None:
    schema = {"type": "object"}
    operation_core_services.installation.register_operation(
        DiscoveryAdapter(
            OperationDescriptor(
                operation_id="fixture_probability.event.compute",
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

    discovered = operation_core_services.core.operations.search(
        OperationDiscoveryRequest(
            query="compute exact event probability",
            domain="arithmetic",
        )
    )

    assert discovered.matches == ()
    assert discovered.domain == "arithmetic"


def test_discovery_does_not_infer_input_types_from_query_wording(
    operation_core_services: DomainTestServices,
) -> None:
    schema = {"type": "object"}
    operation_core_services.installation.register_operation(
        DiscoveryAdapter(
            OperationDescriptor(
                operation_id="fixture_sat.proof.verify",
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
    discovered = operation_core_services.core.operations.search(
        OperationDiscoveryRequest(
            query=(
                "Independently verify this natural-language proof trace: "
                "suppose n is even, so n = 2k."
            ),
            limit=20,
        )
    )

    assert discovered.input_kind is None
    assert discovered.matches[0].operation_id == "fixture_sat.proof.verify"
    assert discovered.matches[0].applicability == "NEEDS_MORE_TYPED_REQUIREMENTS"

    formal_method = operation_core_services.core.operations.search(
        OperationDiscoveryRequest(
            query="check a formal Lean proof by contradiction",
            limit=20,
        )
    )
    assert formal_method.input_kind is None

    written_formal_proof = operation_core_services.core.operations.search(
        OperationDiscoveryRequest(
            query="verify the written proof in Lean",
            limit=20,
        )
    )
    assert written_formal_proof.input_kind is None

    explicitly_structured = operation_core_services.core.operations.search(
        OperationDiscoveryRequest(
            query="formal UNSAT proof",
            input_kind=OperationInputKind.STRUCTURED_REQUEST,
            limit=20,
        )
    )
    assert [match.operation_id for match in explicitly_structured.matches] == [
        "fixture_sat.proof.verify"
    ]
    assert explicitly_structured.matches[0].applicability == (
        "NEEDS_MORE_TYPED_REQUIREMENTS"
    )

    formal_intent = operation_core_services.core.operations.search(
        OperationDiscoveryRequest(
            query="formal UNSAT proof",
        )
    )
    assert formal_intent.input_kind is None
    assert [match.operation_id for match in formal_intent.matches] == [
        "fixture_sat.proof.verify"
    ]

    formal_trace = operation_core_services.core.operations.search(
        OperationDiscoveryRequest(
            query="verify an LRAT proof trace",
        )
    )
    assert formal_trace.input_kind is None


def test_discovery_routes_only_declared_input_and_artifact_contracts(
    operation_core_services: DomainTestServices,
) -> None:
    proof_schema_uri = "artifact://sha256/" + ("1" * 64)
    other_schema_uri = "artifact://sha256/" + ("2" * 64)
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    operation_core_services.installation.register_operation(
        DiscoveryAdapter(
            OperationDescriptor(
                operation_id="fixture_claim.elaborate",
                version="1",
                title="Elaborate a formal proposition",
                description="Accept formal proposition syntax.",
                provider="tests",
                provider_runtime=TEST_RUNTIME,
                input_schema=schema,
                output_schema=schema,
                tags=("formal", "proposition"),
                accepted_input_kinds=(OperationInputKind.FORMAL_PROPOSITION,),
            )
        )
    )
    operation_core_services.installation.register_operation(
        DiscoveryAdapter(
            OperationDescriptor(
                operation_id="fixture_claim.replay",
                version="1",
                title="Replay a typed proof artifact",
                description="Accept one exact bound proof artifact.",
                provider="tests",
                provider_runtime=TEST_RUNTIME,
                input_schema=schema,
                output_schema=schema,
                tags=("proof", "artifact"),
                accepted_input_kinds=(OperationInputKind.TYPED_ARTIFACT,),
                accepted_artifact_types=(proof_schema_uri,),
            )
        )
    )
    service = operation_core_services.core.operations

    formal = service.search(
        OperationDiscoveryRequest(
            query="formal proposition",
            input_kind=OperationInputKind.FORMAL_PROPOSITION,
        )
    )
    assert [match.operation_id for match in formal.matches] == [
        "fixture_claim.elaborate"
    ]
    assert formal.matches[0].applicability == "NEEDS_MORE_TYPED_REQUIREMENTS"

    typed = service.search(
        OperationDiscoveryRequest(
            query="proof artifact",
            input_kind=OperationInputKind.TYPED_ARTIFACT,
            artifact_type=proof_schema_uri,
        )
    )
    assert [match.operation_id for match in typed.matches] == ["fixture_claim.replay"]
    assert typed.matches[0].applicability == "NEEDS_MORE_TYPED_REQUIREMENTS"

    mismatched = service.search(
        OperationDiscoveryRequest(
            query="proof artifact",
            input_kind=OperationInputKind.TYPED_ARTIFACT,
            artifact_type=other_schema_uri,
        )
    )
    assert [match.operation_id for match in mismatched.matches] == [
        "fixture_claim.replay"
    ]
    assert mismatched.matches[0].applicability == "INCOMPATIBLE"
    assert mismatched.matches[0].applicability_code == "ARTIFACT_TYPE_MISMATCH"

    lexically_absent = service.search(
        OperationDiscoveryRequest(
            query="quuxonium",
            input_kind=OperationInputKind.FORMAL_PROPOSITION,
        )
    )
    assert lexically_absent.matches == ()

    incompatible_lexical_match = service.search(
        OperationDiscoveryRequest(
            query="formal proposition",
            input_kind=OperationInputKind.STRUCTURED_REQUEST,
        )
    )
    assert incompatible_lexical_match.matches[0].operation_id == (
        "fixture_claim.elaborate"
    )
    assert incompatible_lexical_match.matches[0].applicability == "INCOMPATIBLE"
    assert incompatible_lexical_match.matches[0].applicability_code == (
        "INPUT_KIND_MISMATCH"
    )


def test_discovery_artifact_type_requires_typed_artifact_input() -> None:
    proof_schema_uri = "artifact://sha256/" + ("1" * 64)
    with pytest.raises(
        ValueError,
        match="artifact_type requires input_kind=TYPED_ARTIFACT",
    ):
        OperationDiscoveryRequest(
            query="proof artifact",
            input_kind=OperationInputKind.STRUCTURED_REQUEST,
            artifact_type=proof_schema_uri,
        )
    with pytest.raises(
        ValueError,
        match="TYPED_ARTIFACT input requires artifact_type",
    ):
        OperationDiscoveryRequest(
            query="proof artifact",
            input_kind=OperationInputKind.TYPED_ARTIFACT,
        )


def test_descriptor_artifact_contract_requires_typed_artifact_input() -> None:
    proof_schema_uri = "artifact://sha256/" + ("1" * 64)
    with pytest.raises(
        ValueError,
        match="accepted artifact types require TYPED_ARTIFACT input",
    ):
        OperationDescriptor(
            operation_id="fixture.invalid.artifact",
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
        OperationDescriptor(
            operation_id="fixture.invalid.typed-artifact",
            version="1",
            title="Invalid typed artifact contract",
            description="Typed artifact routing requires an exact stored schema.",
            provider="tests",
            provider_runtime=TEST_RUNTIME,
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            accepted_input_kinds=(OperationInputKind.TYPED_ARTIFACT,),
        )


def test_operation_registration_rejects_an_invalid_invocation_example(
    operation_core_services: DomainTestServices,
) -> None:
    adapter = DiscoveryAdapter(
        OperationDescriptor(
            operation_id="example.invalid-example",
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
            examples=(
                OperationExample(
                    name="invalid",
                    description="This value has the wrong type.",
                    input={"value": "not-an-integer"},
                ),
            ),
        )
    )

    with pytest.raises(OperationError, match="invocation example"):
        operation_core_services.installation.register_operation(adapter)
