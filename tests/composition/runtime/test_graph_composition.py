"""Integration tests for graph composition and bounded enumeration capabilities."""

from __future__ import annotations

from pathlib import Path

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityCompletenessStatus,
    CapabilityRelationshipStatus,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.runtime import create_runtime
from jacobian.runtime.model import JacobianRuntime


def _runtime_with_composition(tmp_path: Path) -> JacobianRuntime:
    """Build a runtime with the bundled composition adapters installed."""
    return create_runtime(tmp_path)


def _atlas_graph_uri(runtime: JacobianRuntime, order: int, limit: int = 2) -> list[str]:
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.search.atlas",
            input={
                "order": order,
                "constraints": {"connected": True},
                "limit": limit,
            },
        )
    )
    assert result.execution.status is ExecutionStatus.COMPLETED
    return [c["graph_uri"] for c in result.output["candidates"]]


# ---------------------------------------------------------------------------
# graph.construct.compose
# ---------------------------------------------------------------------------


def test_compose_complement_returns_computed_graph_artifact(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_composition(tmp_path)
    graph_uris = _atlas_graph_uri(runtime, order=3, limit=1)
    left_uri = graph_uris[0]

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.construct.compose",
            input={
                "operation": "COMPLEMENT",
                "left_graph_uri": left_uri,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.completeness.status is CapabilityCompletenessStatus.COMPLETE
    assert result.completeness.assurance_level is CapabilityAssuranceLevel.COMPUTED
    assert result.scope is not None
    assert result.scope.artifact_uri == result.output["result_graph_uri"]
    assert "conclusion" not in result.output

    payload = result.output["result_graph"]
    assert payload["graph_schema_version"] == "1"
    assert len(payload["vertices"]) == 3

    # The complement of a connected 3-vertex graph has at most one edge.
    assert len(payload["edges"]) <= 1

    # The result graph artifact is retrievable from the store.
    stored = runtime.core.store.get(result.output["result_graph_uri"])
    assert stored.payload == payload

    # A composition-record artifact was materialized.
    composition = runtime.core.store.get(result.output["composition_artifact_uri"])
    assert composition.payload["operation"] == "COMPLEMENT"
    assert composition.payload["left_graph_uri"] == left_uri
    assert composition.payload["right_graph_uri"] is None
    assert composition.payload["result_graph_uri"] == result.output["result_graph_uri"]
    assert composition.payload["backend"] == "networkx.complement"

    # Relationship links the result back to its source.
    assert len(result.relationships) == 1
    rel = result.relationships[0]
    assert rel.relation_id == "graph.relation.composed-from"
    assert rel.status is CapabilityRelationshipStatus.PROPOSED
    assert left_uri in rel.target_artifact_uris

    # Both the source, result, and composition record appear in artifact_uris.
    assert left_uri in result.artifact_uris
    assert result.output["result_graph_uri"] in result.artifact_uris
    assert result.output["composition_artifact_uri"] in result.artifact_uris


def test_compose_disjoint_union_combines_two_graphs(tmp_path: Path) -> None:
    runtime = _runtime_with_composition(tmp_path)
    graph_uris = _atlas_graph_uri(runtime, order=3, limit=2)
    left_uri, right_uri = graph_uris[0], graph_uris[1]

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.construct.compose",
            input={
                "operation": "DISJOINT_UNION",
                "left_graph_uri": left_uri,
                "right_graph_uri": right_uri,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    payload = result.output["result_graph"]
    assert len(payload["vertices"]) == 6  # 3 + 3
    assert len(payload["edges"]) >= 2  # at least one edge from each

    composition = runtime.core.store.get(result.output["composition_artifact_uri"])
    assert composition.payload["operation"] == "DISJOINT_UNION"
    assert composition.payload["right_graph_uri"] == right_uri
    assert composition.payload["backend"] == "networkx.disjoint_union"

    rel = result.relationships[0]
    assert left_uri in rel.target_artifact_uris
    assert right_uri in rel.target_artifact_uris


def test_compose_join_adds_all_cross_edges(tmp_path: Path) -> None:
    runtime = _runtime_with_composition(tmp_path)
    graph_uris = _atlas_graph_uri(runtime, order=3, limit=2)
    left_uri, right_uri = graph_uris[0], graph_uris[1]

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.construct.compose",
            input={
                "operation": "JOIN",
                "left_graph_uri": left_uri,
                "right_graph_uri": right_uri,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    payload = result.output["result_graph"]
    assert len(payload["vertices"]) == 6
    # Join = disjoint union + all 3*3 = 9 cross edges, plus original edges.
    assert len(payload["edges"]) >= 9

    composition = runtime.core.store.get(result.output["composition_artifact_uri"])
    assert composition.payload["backend"] == "networkx.full_join"


def test_compose_lexicographic_product_doubles_vertex_count(tmp_path: Path) -> None:
    runtime = _runtime_with_composition(tmp_path)
    graph_uris = _atlas_graph_uri(runtime, order=3, limit=2)
    left_uri, right_uri = graph_uris[0], graph_uris[1]

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.construct.compose",
            input={
                "operation": "LEXICOGRAPHIC_PRODUCT",
                "left_graph_uri": left_uri,
                "right_graph_uri": right_uri,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    payload = result.output["result_graph"]
    assert len(payload["vertices"]) == 9  # 3 * 3

    composition = runtime.core.store.get(result.output["composition_artifact_uri"])
    assert composition.payload["backend"] == "networkx.lexicographic_product"


def test_compose_rejects_missing_right_graph_for_binary_operation(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_composition(tmp_path)
    graph_uris = _atlas_graph_uri(runtime, order=3, limit=1)
    left_uri = graph_uris[0]

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.construct.compose",
            input={
                "operation": "DISJOINT_UNION",
                "left_graph_uri": left_uri,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics
    assert result.diagnostics[0].code == "INVALID_COMPOSITION_REQUEST"


def test_compose_rejects_right_graph_for_unary_complement(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_composition(tmp_path)
    graph_uris = _atlas_graph_uri(runtime, order=3, limit=2)
    left_uri, right_uri = graph_uris[0], graph_uris[1]

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.construct.compose",
            input={
                "operation": "COMPLEMENT",
                "left_graph_uri": left_uri,
                "right_graph_uri": right_uri,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics
    assert result.diagnostics[0].code == "INVALID_COMPOSITION_REQUEST"


def test_compose_rejects_nonexistent_graph_artifact(tmp_path: Path) -> None:
    runtime = _runtime_with_composition(tmp_path)
    fake_uri = "artifact://sha256/" + "0" * 64

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.construct.compose",
            input={
                "operation": "COMPLEMENT",
                "left_graph_uri": fake_uri,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics
    assert result.diagnostics[0].code == "GRAPH_ARTIFACT_NOT_FOUND"


# ---------------------------------------------------------------------------
# graph.enumerate.nonisomorphic
# ---------------------------------------------------------------------------


def test_enumerate_returns_complete_atlas_catalog_with_boundary(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_composition(tmp_path)

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.enumerate.nonisomorphic",
            input={"order": 4, "limit": 100},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.completeness.status is CapabilityCompletenessStatus.COMPLETE
    assert result.completeness.assurance_level is CapabilityAssuranceLevel.COMPUTED
    assert result.scope is not None
    assert result.scope.artifact_uri == result.output["scope_uri"]
    assert "conclusion" not in result.output

    # The Graph Atlas has exactly 11 nonisomorphic graphs of order 4.
    assert result.output["total_count"] == 11
    assert result.output["returned_count"] == 11
    assert result.output["truncated"] is False
    assert result.output["backend"] == "networkx.graph_atlas_g"

    # The backend boundary is explicit so agents do not over-read the scope.
    assert "Graph Atlas" in result.output["backend_boundary"]
    assert "not all nonisomorphic" in result.output["backend_boundary"]

    # Every returned entry is a valid graph artifact with order and size.
    for entry in result.output["graphs"]:
        payload = entry["graph"]
        assert payload["graph_schema_version"] == "1"
        assert len(payload["vertices"]) == 4
        assert entry["order"] == 4
        assert entry["size"] == len(payload["edges"])
        stored = runtime.core.store.get(entry["graph_uri"])
        assert stored.payload == payload

    # The scope artifact records the backend boundary.
    scope = runtime.core.store.get(result.output["scope_uri"])
    assert scope.payload["source"] == "networkx.graph_atlas_g"
    assert scope.payload["order"] == 4
    assert scope.payload["enumerated_count"] == 11
    assert "not all nonisomorphic" in scope.payload["backend_boundary"]

    # Every graph is linked to the scope.
    assert len(result.relationships) == 11
    for rel in result.relationships:
        assert rel.relation_id == "graph.relation.enumerated-in"
        assert rel.source_artifact_uris == (result.output["scope_uri"],)

    # Scope + all graphs appear in artifact_uris.
    assert result.output["scope_uri"] in result.artifact_uris
    for entry in result.output["graphs"]:
        assert entry["graph_uri"] in result.artifact_uris


def test_enumerate_paginates_with_limit_and_offset(tmp_path: Path) -> None:
    runtime = _runtime_with_composition(tmp_path)

    first = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.enumerate.nonisomorphic",
            input={"order": 4, "limit": 3, "offset": 0},
        )
    )
    second = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.enumerate.nonisomorphic",
            input={"order": 4, "limit": 3, "offset": 3},
        )
    )

    assert first.output["total_count"] == 11
    assert first.output["returned_count"] == 3
    assert first.output["truncated"] is True

    assert second.output["total_count"] == 11
    assert second.output["returned_count"] == 3
    assert second.output["truncated"] is True

    # No overlap between the two windows.
    first_uris = {g["graph_uri"] for g in first.output["graphs"]}
    second_uris = {g["graph_uri"] for g in second.output["graphs"]}
    assert first_uris.isdisjoint(second_uris)


def test_enumerate_rejects_order_outside_backend_boundary(
    tmp_path: Path,
) -> None:
    runtime = _runtime_with_composition(tmp_path)

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.enumerate.nonisomorphic",
            input={"order": 10},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics
    assert result.diagnostics[0].code == "INVALID_REQUEST"


def test_enumerate_order_zero_returns_single_empty_graph(tmp_path: Path) -> None:
    runtime = _runtime_with_composition(tmp_path)

    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.enumerate.nonisomorphic",
            input={"order": 0},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["total_count"] == 1
    assert result.output["returned_count"] == 1
    entry = result.output["graphs"][0]
    assert entry["order"] == 0
    assert entry["size"] == 0
    assert entry["graph"]["vertices"] == []
    assert entry["graph"]["edges"] == []
