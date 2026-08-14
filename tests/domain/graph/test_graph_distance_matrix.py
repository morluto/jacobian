from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.support.graph_distance_cases import (
    c7_strong_c7_distance,
    c7_strong_c7_graph,
    hoffman_singleton_graph,
)
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.graph_coloring import GraphChromaticNumberRequest
from jacobian.contracts.graph_distance_matrix import GraphDistanceMatrixResult
from jacobian.contracts.graph_optimization import (
    GraphHamiltonianPathRequest,
    GraphOptimizationRequest,
)
from jacobian.contracts.operations import (
    OperationRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.graph_optimization import graph_invariant_operations


@pytest.fixture
def domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(
        tmp_path / "state", graph_invariant_operations()
    ) as services:
        yield services


def _invoke(domain_services, vertices: list[str], edges: list[list[str]]):
    return domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="graph.distance_matrix.compute",
            input={"graph": {"vertices": vertices, "edges": edges}},
        )
    )


def _invoke_graph(domain_services, graph: dict[str, object]):
    return domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="graph.distance_matrix.compute",
            input={"graph": graph},
        )
    )


def _result(**changes: object) -> GraphDistanceMatrixResult:
    values: dict[str, object] = {
        "semantics_version": "unweighted-shortest-path-distance-matrix.v1",
        "vertex_ordering": "LEXICOGRAPHIC_ASCENDING",
        "pair_coverage": "ALL_ORDERED_VERTEX_PAIRS",
        "unreachable_representation": "JSON_NULL",
        "vertices": ("a", "b", "c"),
        "distances": ((0, 1, 2), (1, 0, 1), (2, 1, 0)),
        "connected": True,
    }
    values.update(changes)
    return GraphDistanceMatrixResult.model_validate(values)


def test_distance_matrix_is_complete_canonical_and_lineage_bound(
    domain_services,
) -> None:
    result = _invoke(
        domain_services,
        ["c", "a", "b"],
        [["a", "b"], ["b", "c"]],
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {
        "semantics_version": "unweighted-shortest-path-distance-matrix.v1",
        "vertex_ordering": "LEXICOGRAPHIC_ASCENDING",
        "pair_coverage": "ALL_ORDERED_VERTEX_PAIRS",
        "unreachable_representation": "JSON_NULL",
        "vertices": ["a", "b", "c"],
        "distances": [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
        "connected": True,
    }
    assert result.artifact_uris == ()


def test_distance_matrix_represents_disconnected_pairs_with_null(
    domain_services,
) -> None:
    result = _invoke(domain_services, ["c", "a", "b"], [["a", "b"]])

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"]["vertices"] == ["a", "b", "c"]
    assert result.output["result"]["distances"] == [
        [0, 1, None],
        [1, 0, None],
        [None, None, 0],
    ]
    assert result.output["result"]["connected"] is False


def test_distance_matrix_empty_and_singleton_conventions(domain_services) -> None:
    empty = _invoke(domain_services, [], [])
    singleton = _invoke(domain_services, ["only"], [])

    assert empty.output["result"]["vertices"] == []
    assert empty.output["result"]["distances"] == []
    assert empty.output["result"]["connected"] is False
    assert singleton.output["result"]["vertices"] == ["only"]
    assert singleton.output["result"]["distances"] == [[0]]
    assert singleton.output["result"]["connected"] is True


def test_distance_matrix_rejects_graph_above_dedicated_order_bound(
    domain_services,
) -> None:
    result = _invoke(domain_services, [f"v{index:02d}" for index in range(65)], [])

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.artifact_uris == ()
    assert result.diagnostics[0].code == "INVALID_GRAPH_DISTANCE_MATRIX_REQUEST"


def test_distance_matrix_accepts_connected_order_64_boundary(domain_services) -> None:
    vertices = [f"v{index:02d}" for index in range(64)]
    result = _invoke(
        domain_services,
        vertices,
        [[vertices[index], vertices[index + 1]] for index in range(63)],
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"]["vertices"] == vertices
    assert result.output["result"]["distances"][0][-1] == 63
    assert result.output["result"]["connected"] is True


def test_distance_matrix_accepts_disconnected_order_64_boundary(
    domain_services,
) -> None:
    vertices = [f"v{index:02d}" for index in range(64)]
    result = _invoke(domain_services, vertices, [])

    assert result.execution.status is ExecutionStatus.COMPLETED
    matrix = result.output["result"]["distances"]
    assert len(matrix) == 64
    assert matrix[0][0] == 0
    assert matrix[0][1] is None
    assert result.output["result"]["connected"] is False


def test_unrelated_np_hard_graph_contract_bounds_are_unchanged() -> None:
    graph_33 = {
        "vertices": [f"v{index:02d}" for index in range(33)],
        "edges": [],
    }
    graph_19 = {
        "vertices": [f"v{index:02d}" for index in range(19)],
        "edges": [],
    }

    with pytest.raises(ValidationError):
        GraphChromaticNumberRequest.model_validate({"graph": graph_33})
    with pytest.raises(ValidationError):
        GraphOptimizationRequest.model_validate({"graph": graph_33})
    with pytest.raises(ValidationError):
        GraphHamiltonianPathRequest.model_validate({"graph": graph_19})


def test_distance_matrix_computes_hoffman_singleton_case(domain_services) -> None:
    graph = hoffman_singleton_graph()
    result = _invoke_graph(domain_services, graph)

    assert result.execution.status is ExecutionStatus.COMPLETED
    output = result.output["result"]
    assert len(output["vertices"]) == 50
    assert len(graph["edges"]) == 175
    edge_set = {tuple(edge) for edge in graph["edges"]}
    expected = [
        [
            0
            if source == target
            else 1
            if tuple(sorted((source, target))) in edge_set
            else 2
            for target in output["vertices"]
        ]
        for source in output["vertices"]
    ]
    assert output["distances"] == expected
    assert output["connected"] is True


def test_distance_matrix_computes_c7_strong_c7_case(domain_services) -> None:
    graph = c7_strong_c7_graph()
    result = _invoke_graph(domain_services, graph)

    assert result.execution.status is ExecutionStatus.COMPLETED
    output = result.output["result"]
    assert len(output["vertices"]) == 49
    assert len(graph["edges"]) == 196
    expected = [
        [c7_strong_c7_distance(source, target) for target in output["vertices"]]
        for source in output["vertices"]
    ]
    assert output["distances"] == expected
    assert output["connected"] is True


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"vertices": ("b", "a", "c")}, "unique and sorted"),
        ({"distances": ((0, 1), (1, 0))}, "square"),
        (
            {"distances": ((1, 1, 2), (1, 0, 1), (2, 1, 0))},
            "diagonal",
        ),
        (
            {"distances": ((0, 0, 2), (0, 0, 1), (2, 1, 0))},
            "off-diagonal",
        ),
        (
            {"distances": ((0, 1, 2), (2, 0, 1), (2, 1, 0))},
            "symmetric",
        ),
        (
            {"distances": ((0, 1, 3), (1, 0, 1), (3, 1, 0))},
            "triangle inequality",
        ),
        (
            {"distances": ((0, 1, None), (1, 0, 1), (None, 1, 0))},
            "component closure",
        ),
        ({"connected": False}, "connected"),
    ),
)
def test_distance_matrix_result_rejects_inconsistent_claims(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _result(**changes)


def test_distance_matrix_public_postdoc_graph(domain_services) -> None:
    result = _invoke(
        domain_services,
        ["0", "1", "2", "3", "4", "5"],
        [["0", "3"], ["0", "4"], ["1", "4"], ["2", "4"], ["3", "4"], ["3", "5"]],
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"]["distances"] == [
        [0, 2, 2, 1, 1, 2],
        [2, 0, 2, 2, 1, 3],
        [2, 2, 0, 2, 1, 3],
        [1, 2, 2, 0, 1, 1],
        [1, 1, 1, 1, 0, 2],
        [2, 3, 3, 1, 2, 0],
    ]
    assert result.output["result"]["connected"] is True
