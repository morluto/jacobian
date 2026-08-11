from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.support.services import (
    DomainTestServices,
    atomic_installation,
    open_domain_services,
)

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityRelationshipStatus,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.graphs.installation import GraphInstallation, install_graph_capabilities
from jacobian.graphs.isomorphism import (
    GraphIsomorphismInstallation,
    install_graph_isomorphism,
)
from jacobian.runtime import CheckerAuthorityMode


@dataclass(frozen=True, slots=True)
class GraphIsomorphismTestServices(DomainTestServices):
    graph: GraphInstallation
    isomorphism: GraphIsomorphismInstallation


@contextmanager
def _open_graph_isomorphism_services(
    root: Path,
    *,
    authorize_checker: bool,
) -> Iterator[GraphIsomorphismTestServices]:
    authority = (
        CheckerAuthorityMode.INSTALL_BUNDLED
        if authorize_checker
        else CheckerAuthorityMode.NONE
    )
    with open_domain_services(root, checker_authority=authority) as services:
        with atomic_installation(services.core):
            graph_adapters, graph = install_graph_capabilities(
                services.core.store,
                services.core.schemas,
                services.core.artifacts,
                services.application.verification,
                services.core.checkers,
                authorize_checker=services.installation.authorizes_bundled_checkers,
            )
            for adapter in graph_adapters:
                services.installation.register_capability(adapter)
            adapter, isomorphism = install_graph_isomorphism(
                services.core.store,
                services.core.schemas,
                services.core.artifacts,
                services.application.verification,
                services.core.checkers,
                graph,
                authorize_checker=services.installation.authorizes_bundled_checkers,
            )
            if adapter is not None:
                services.installation.register_capability(adapter)
        yield GraphIsomorphismTestServices(
            core=services.core,
            application=services.application,
            installation=services.installation,
            graph=graph,
            isomorphism=isomorphism,
        )


@pytest.fixture
def graph_isomorphism_services(
    tmp_path: Path,
) -> Iterator[GraphIsomorphismTestServices]:
    with _open_graph_isomorphism_services(
        tmp_path / "state", authorize_checker=True
    ) as services:
        yield services


@pytest.fixture
def unauthorized_graph_isomorphism_services(
    tmp_path: Path,
) -> Iterator[GraphIsomorphismTestServices]:
    with _open_graph_isomorphism_services(
        tmp_path / "state", authorize_checker=False
    ) as services:
        yield services


def _graph_uri(
    runtime: GraphIsomorphismTestServices,
    *,
    vertices: list[str],
    edges: list[list[str]],
) -> str:
    return runtime.core.artifacts.put(
        schema_uri=runtime.graph.graph_schema_uri,
        semantics_uri=runtime.graph.semantics_uri,
        payload={
            "graph_schema_version": "1",
            "vertices": vertices,
            "edges": edges,
        },
        summary="graph-isomorphism test input",
    ).artifact_uri


def _input(
    runtime: GraphIsomorphismTestServices,
    mapping: dict[str, str],
) -> dict[str, object]:
    left_graph_uri = _graph_uri(
        runtime,
        vertices=["a", "b", "c"],
        edges=[["a", "b"], ["b", "c"]],
    )
    right_graph_uri = _graph_uri(
        runtime,
        vertices=["x", "y", "z"],
        edges=[["x", "z"], ["y", "z"]],
    )
    return {
        "left_graph_uri": left_graph_uri,
        "right_graph_uri": right_graph_uri,
        "mapping": mapping,
    }


def test_graph_isomorphism_verifies_a_valid_bijection(
    graph_isomorphism_services,
) -> None:

    result = graph_isomorphism_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.isomorphism.verify",
            input=_input(graph_isomorphism_services, {"a": "x", "b": "z", "c": "y"}),
        )
    )

    assert result.output["is_isomorphism"] is True
    assert result.output["conclusion"] == "TRUE"
    assert result.output["coverage"] == "EXHAUSTIVE"
    assert result.assurance.level is CapabilityAssuranceLevel.VERIFIED
    verified_relationships = [
        relationship
        for relationship in result.relationships
        if relationship.status is CapabilityRelationshipStatus.VERIFIED
    ]
    assert len(verified_relationships) == 1
    assert verified_relationships[0].relation_id == "graph.relation.isomorphic-via"
    assert result.output["verification_record_uri"] in result.artifact_uris


def test_graph_isomorphism_verifies_a_negative_result(
    graph_isomorphism_services,
) -> None:

    result = graph_isomorphism_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.isomorphism.verify",
            input=_input(graph_isomorphism_services, {"a": "x", "b": "y", "c": "z"}),
        )
    )

    assert result.output["is_isomorphism"] is False
    assert result.output["conclusion"] == "FALSE"
    assert result.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert not any(
        relationship.relation_id == "graph.relation.isomorphic-via"
        for relationship in result.relationships
    )


def test_graph_isomorphism_keeps_checker_rejection_unknown(
    graph_isomorphism_services,
) -> None:
    checker_id = graph_isomorphism_services.isomorphism.checker_id
    assert checker_id is not None
    request_input = _input(graph_isomorphism_services, {"a": "x", "b": "z", "c": "y"})
    graph_isomorphism_services.core.checkers.revoke(
        checker_id, reason="force fail-closed integration case"
    )

    result = graph_isomorphism_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.isomorphism.verify",
            input=request_input,
        )
    )

    assert result.output["is_isomorphism"] is None
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["coverage"] == "UNKNOWN"
    assert result.output["verification_record_uri"] is None
    assert result.assurance.level is CapabilityAssuranceLevel.HEURISTIC
    assert not any(
        relationship.relation_id == "graph.relation.isomorphic-via"
        for relationship in result.relationships
    )


def test_graph_isomorphism_accepts_graph_atlas_artifact_handoff(
    graph_isomorphism_services,
) -> None:
    searched = graph_isomorphism_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.search.atlas",
            input={"order": 3, "constraints": {"connected": True}, "limit": 1},
        )
    )
    candidate = searched.output["candidates"][0]
    graph_uri = candidate["graph_uri"]
    vertices = candidate["graph"]["vertices"]

    result = graph_isomorphism_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.isomorphism.verify",
            input={
                "left_graph_uri": graph_uri,
                "right_graph_uri": graph_uri,
                "mapping": {vertex: vertex for vertex in vertices},
            },
        )
    )

    assert result.output["conclusion"] == "TRUE"
    assert result.output["left_graph_uri"] == graph_uri
    assert result.output["right_graph_uri"] == graph_uri
    assert graph_uri in result.artifact_uris
    pair = graph_isomorphism_services.core.store.get(result.output["graph_pair_uri"])
    assert pair.manifest.parents == (graph_uri,)
    assert any(
        relationship.relation_id == "graph.relation.pair-scope"
        and relationship.source_artifact_uris == (graph_uri,)
        for relationship in result.relationships
    )


def test_graph_isomorphism_accepts_valid_unsorted_graph_artifacts(
    graph_isomorphism_services,
) -> None:
    left_graph_uri = _graph_uri(
        graph_isomorphism_services,
        vertices=["c", "a", "b"],
        edges=[["b", "c"], ["a", "b"]],
    )
    right_graph_uri = _graph_uri(
        graph_isomorphism_services,
        vertices=["z", "x", "y"],
        edges=[["y", "z"], ["x", "y"]],
    )

    result = graph_isomorphism_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.isomorphism.verify",
            input={
                "left_graph_uri": left_graph_uri,
                "right_graph_uri": right_graph_uri,
                "mapping": {"a": "x", "b": "y", "c": "z"},
            },
        )
    )

    assert result.output["conclusion"] == "TRUE"
    record = graph_isomorphism_services.core.store.get(
        result.output["verification_record_uri"]
    )
    assert left_graph_uri in record.manifest.parents
    assert right_graph_uri in record.manifest.parents


def test_graph_isomorphism_rejects_incompatible_graph_artifact(
    graph_isomorphism_services,
) -> None:
    wrong_artifact = graph_isomorphism_services.core.artifacts.put(
        schema_uri=graph_isomorphism_services.graph.scope_schema_uri,
        semantics_uri=graph_isomorphism_services.graph.semantics_uri,
        payload={
            "scope_schema_version": "1",
            "source": "networkx.graph_atlas_g",
            "backend_version": "test",
            "order": 3,
            "enumerated_count": 2,
        },
    )
    right_graph_uri = _graph_uri(
        graph_isomorphism_services,
        vertices=["x"],
        edges=[],
    )

    result = graph_isomorphism_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.isomorphism.verify",
            input={
                "left_graph_uri": wrong_artifact.artifact_uri,
                "right_graph_uri": right_graph_uri,
                "mapping": {"x": "x"},
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INCOMPATIBLE_GRAPH_ARTIFACT"
    assert result.diagnostics[0].path == "left_graph_uri"


def test_graph_isomorphism_is_unavailable_without_reference_checkers(
    unauthorized_graph_isomorphism_services,
) -> None:
    runtime = unauthorized_graph_isomorphism_services

    assert "graph.isomorphism.verify" not in {
        descriptor.capability_id
        for descriptor in runtime.core.capabilities.catalog().capabilities
    }
