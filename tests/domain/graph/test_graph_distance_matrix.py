from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.capabilities import CapabilityRequest
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


def _row(
    source_vertex: str,
    distances: tuple[int | None, ...] | list[int | None],
    *,
    targets: tuple[str, ...] = ("a", "b", "c"),
) -> dict[str, object]:
    return {
        "source_vertex": source_vertex,
        "distances_by_target": dict(zip(targets, distances, strict=True)),
    }


def _result(**changes: object) -> GraphDistanceMatrixResult:
    values: dict[str, object] = {
        "semantics_version": "unweighted-shortest-path-distance-matrix.v3",
        "row_ordering": "SOURCE_VERTEX_LEXICOGRAPHIC_ASCENDING",
        "target_ordering": "TARGET_VERTEX_LEXICOGRAPHIC_ASCENDING",
        "pair_coverage": "ALL_ORDERED_VERTEX_PAIRS",
        "unreachable_representation": "JSON_NULL",
        "target_vertices": ("a", "b", "c"),
        "rows": (
            _row("a", (0, 1, 2)),
            _row("b", (1, 0, 1)),
            _row("c", (2, 1, 0)),
        ),
        "connected": True,
    }
    values.update(changes)
    return GraphDistanceMatrixResult.model_validate(values)


def test_distance_matrix_is_complete_and_label_bound(domain_services) -> None:
    result = _invoke(
        domain_services,
        ["c", "a", "b"],
        [["a", "b"], ["b", "c"]],
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {
        "semantics_version": "unweighted-shortest-path-distance-matrix.v3",
        "row_ordering": "SOURCE_VERTEX_LEXICOGRAPHIC_ASCENDING",
        "target_ordering": "TARGET_VERTEX_LEXICOGRAPHIC_ASCENDING",
        "pair_coverage": "ALL_ORDERED_VERTEX_PAIRS",
        "unreachable_representation": "JSON_NULL",
        "target_vertices": ["a", "b", "c"],
        "rows": [
            _row("a", [0, 1, 2]),
            _row("b", [1, 0, 1]),
            _row("c", [2, 1, 0]),
        ],
        "connected": True,
    }
    assert result.artifact_uris == ()


def test_distance_matrix_rows_bind_numeric_looking_vertex_labels(
    domain_services,
) -> None:
    result = _invoke(
        domain_services,
        ["0", "2", "10"],
        [["0", "2"], ["2", "10"]],
    )

    assert "distances" not in result.output["result"]
    assert result.output["result"]["target_vertices"] == ["0", "10", "2"]
    assert result.output["result"]["rows"] == [
        _row("0", [0, 2, 1], targets=("0", "10", "2")),
        _row("10", [2, 0, 1], targets=("0", "10", "2")),
        _row("2", [1, 1, 0], targets=("0", "10", "2")),
    ]


def test_distance_matrix_represents_disconnected_pairs_with_null(
    domain_services,
) -> None:
    result = _invoke(domain_services, ["c", "a", "b"], [["a", "b"]])

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"]["target_vertices"] == ["a", "b", "c"]
    assert result.output["result"]["rows"] == [
        _row("a", [0, 1, None]),
        _row("b", [1, 0, None]),
        _row("c", [None, None, 0]),
    ]
    assert result.output["result"]["connected"] is False


def test_distance_matrix_empty_and_singleton_conventions(domain_services) -> None:
    empty = _invoke(domain_services, [], [])
    singleton = _invoke(domain_services, ["only"], [])

    assert empty.output["result"]["target_vertices"] == []
    assert empty.output["result"]["rows"] == []
    assert empty.output["result"]["connected"] is False
    assert singleton.output["result"]["target_vertices"] == ["only"]
    assert singleton.output["result"]["rows"] == [
        _row("only", [0], targets=("only",))
    ]
    assert singleton.output["result"]["connected"] is True


def test_distance_matrix_rejects_graph_above_existing_order_bound(
    domain_services,
) -> None:
    result = _invoke(domain_services, [f"v{index:02d}" for index in range(33)], [])

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.artifact_uris == ()
    assert result.diagnostics[0].code == "INVALID_REQUEST"


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"target_vertices": ("b", "a", "c")}, "unique and sorted"),
        (
            {
                "rows": (
                    _row("b", (0, 1, 2)),
                    _row("a", (1, 0, 1)),
                    _row("c", (2, 1, 0)),
                )
            },
            "bind every source vertex",
        ),
        (
            {
                "rows": (
                    _row("a", (0, 1, 2)),
                    _row("b", (1, 0, 1)),
                )
            },
            "bind every source vertex",
        ),
        (
            {
                "rows": (
                    _row("a", (1, 1, 2)),
                    _row("b", (1, 0, 1)),
                    _row("c", (2, 1, 0)),
                )
            },
            "diagonal",
        ),
        (
            {
                "rows": (
                    _row("a", (0, 0, 2)),
                    _row("b", (0, 0, 1)),
                    _row("c", (2, 1, 0)),
                )
            },
            "off-diagonal",
        ),
        (
            {
                "rows": (
                    _row("a", (0, 1, 2)),
                    _row("b", (2, 0, 1)),
                    _row("c", (2, 1, 0)),
                )
            },
            "symmetric",
        ),
        (
            {
                "rows": (
                    _row("a", (0, 1, 3)),
                    _row("b", (1, 0, 1)),
                    _row("c", (3, 1, 0)),
                )
            },
            "triangle inequality",
        ),
        (
            {
                "rows": (
                    _row("a", (0, 1, None)),
                    _row("b", (1, 0, 1)),
                    _row("c", (None, 1, 0)),
                )
            },
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
        [
            ["0", "3"],
            ["0", "4"],
            ["1", "4"],
            ["2", "4"],
            ["3", "4"],
            ["3", "5"],
        ],
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    targets = ("0", "1", "2", "3", "4", "5")
    assert result.output["result"]["rows"] == [
        _row("0", [0, 2, 2, 1, 1, 2], targets=targets),
        _row("1", [2, 0, 2, 2, 1, 3], targets=targets),
        _row("2", [2, 2, 0, 2, 1, 3], targets=targets),
        _row("3", [1, 2, 2, 0, 1, 1], targets=targets),
        _row("4", [1, 1, 1, 1, 0, 2], targets=targets),
        _row("5", [2, 3, 3, 1, 2, 0], targets=targets),
    ]
    assert result.output["result"]["connected"] is True
