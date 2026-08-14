"""Graph composition and bounded enumeration behavior contracts."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.services import (
    DomainTestServices,
    atomic_installation,
    open_domain_services,
)

from jacobian.contracts.operations import OperationRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.graphs.composition import build_graph_composition_operations
from jacobian.graphs.operation_resources import build_graph_operations


@pytest.fixture
def graph_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    """Install only the production graph and graph-composition adapters."""

    with open_domain_services(tmp_path / "state") as services:
        with atomic_installation(services.core):
            graph_adapters, graph = build_graph_operations(
                services.core.store,
                services.core.schemas,
                services.core.artifacts,
                services.verification,
                services.core.checkers,
                authorize_checker=False,
            )
            for adapter in graph_adapters:
                services.installation.register_operation(adapter)
            composition_adapters = build_graph_composition_operations(
                services.core.store,
                services.core.schemas,
                services.core.artifacts,
                semantics_uri=graph.semantics_uri,
                graph_schema_uri=graph.graph_schema_uri,
            )
            for adapter in composition_adapters:
                services.installation.register_operation(adapter)
        yield services


def _atlas_graph_uri(
    runtime: DomainTestServices, order: int, limit: int = 2
) -> list[str]:
    result = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="graph.search.atlas",
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
    graph_services: DomainTestServices,
) -> None:
    runtime = graph_services
    graph_uris = _atlas_graph_uri(runtime, order=3, limit=1)
    left_uri = graph_uris[0]

    result = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="graph.construct.compose",
            input={
                "operation": "COMPLEMENT",
                "left_graph_uri": left_uri,
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
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

    # Both the source, result, and composition record appear in artifact_uris.
    assert left_uri in result.artifact_uris
    assert result.output["result_graph_uri"] in result.artifact_uris
    assert result.output["composition_artifact_uri"] in result.artifact_uris


def test_compose_binary_operations_preserve_their_graph_contracts(
    graph_services: DomainTestServices,
) -> None:
    runtime = graph_services
    graph_uris = _atlas_graph_uri(runtime, order=3, limit=2)
    left_uri, right_uri = graph_uris[0], graph_uris[1]
    cases = (
        ("DISJOINT_UNION", 6, 2, "networkx.disjoint_union"),
        ("JOIN", 6, 9, "networkx.full_join"),
        ("LEXICOGRAPHIC_PRODUCT", 9, 0, "networkx.lexicographic_product"),
    )
    for operation, vertex_count, minimum_edges, backend in cases:
        result = runtime.core.operations.invoke(
            OperationRequest(
                operation_id="graph.construct.compose",
                input={
                    "operation": operation,
                    "left_graph_uri": left_uri,
                    "right_graph_uri": right_uri,
                },
            )
        )

        assert result.execution.status is ExecutionStatus.COMPLETED, operation
        payload = result.output["result_graph"]
        assert len(payload["vertices"]) == vertex_count, operation
        assert len(payload["edges"]) >= minimum_edges, operation
        composition = runtime.core.store.get(result.output["composition_artifact_uri"])
        assert composition.payload["operation"] == operation
        assert composition.payload["right_graph_uri"] == right_uri
        assert composition.payload["backend"] == backend


def test_compose_rejects_invalid_inputs(graph_services: DomainTestServices) -> None:
    runtime = graph_services
    left_uri, right_uri = _atlas_graph_uri(runtime, order=3, limit=2)
    fake_uri = "artifact://sha256/" + "0" * 64
    cases = (
        (
            {
                "operation": "DISJOINT_UNION",
                "left_graph_uri": left_uri,
            },
            "INVALID_COMPOSITION_REQUEST",
        ),
        (
            {
                "operation": "COMPLEMENT",
                "left_graph_uri": left_uri,
                "right_graph_uri": right_uri,
            },
            "INVALID_COMPOSITION_REQUEST",
        ),
        (
            {
                "operation": "COMPLEMENT",
                "left_graph_uri": fake_uri,
            },
            "GRAPH_ARTIFACT_NOT_FOUND",
        ),
    )
    for request_input, diagnostic_code in cases:
        result = runtime.core.operations.invoke(
            OperationRequest(
                operation_id="graph.construct.compose",
                input=request_input,
            )
        )

        assert result.execution.status is ExecutionStatus.ERROR, request_input
        assert result.diagnostics, request_input
        assert result.diagnostics[0].code == diagnostic_code, request_input


# ---------------------------------------------------------------------------
# graph.enumerate.nonisomorphic
# ---------------------------------------------------------------------------


def test_enumerate_returns_complete_atlas_catalog_with_boundary(
    graph_services: DomainTestServices,
) -> None:
    runtime = graph_services

    result = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="graph.enumerate.nonisomorphic",
            input={"order": 4, "limit": 100},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
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

    # Scope + all graphs appear in artifact_uris.
    assert result.output["scope_uri"] in result.artifact_uris
    for entry in result.output["graphs"]:
        assert entry["graph_uri"] in result.artifact_uris


def test_enumerate_paginates_with_limit_and_offset(
    graph_services: DomainTestServices,
) -> None:
    runtime = graph_services

    first = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="graph.enumerate.nonisomorphic",
            input={"order": 4, "limit": 3, "offset": 0},
        )
    )
    second = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="graph.enumerate.nonisomorphic",
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


def test_enumerate_rejects_invalid_order_before_provider_or_publication(
    graph_services: DomainTestServices, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = graph_services

    def unexpected_provider_load() -> None:
        pytest.fail("invalid input must not load the NetworkX provider")

    def unexpected_artifact_publication(*_args: object, **_kwargs: object) -> None:
        pytest.fail("invalid input must not publish graph artifacts")

    class UnavailableProvider:
        def get(self) -> None:
            unexpected_provider_load()

    monkeypatch.setattr(
        "jacobian.graphs.composition.networkx_loader",
        UnavailableProvider(),
    )
    monkeypatch.setattr(
        runtime.core.artifacts,
        "put",
        unexpected_artifact_publication,
    )

    result = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="graph.enumerate.nonisomorphic",
            input={"order": "4"},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics
    assert result.diagnostics[0].code == "INVALID_ENUMERATION_REQUEST"


def test_enumerate_order_zero_returns_single_empty_graph(
    graph_services: DomainTestServices,
) -> None:
    runtime = graph_services

    result = runtime.core.operations.invoke(
        OperationRequest(
            operation_id="graph.enumerate.nonisomorphic",
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
