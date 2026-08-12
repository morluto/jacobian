from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.capabilities import (
    CapabilityRequest,
)
from jacobian.contracts.graph_invariant_operations import GraphDistanceMatrixResult
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.graph_optimization import build_graph_invariant_bundle


@pytest.fixture
def domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(
        tmp_path / "state", build_graph_invariant_bundle()
    ) as services:
        yield services


def _invoke(domain_services, vertices: list[str], edges: list[list[str]]):
    return domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.distance_matrix.compute",
            input={"graph": {"vertices": vertices, "edges": edges}},
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


def test_distance_matrix_rejects_graph_above_existing_order_bound(
    domain_services,
) -> None:
    result = _invoke(domain_services, [f"v{index:02d}" for index in range(33)], [])

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.artifact_uris == ()
    assert result.diagnostics[0].code == "INVALID_GRAPH_INVARIANT_REQUEST"


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
