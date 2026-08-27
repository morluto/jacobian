"""Focused tests for the exact graph distance-matrix operation.

Covers the labelled-row result contract: numeric-looking string identifiers
stay lexicographically canonical while every row carries its own source
label, so a dense matrix cannot be detached from the authoritative vertex
axis and relabelled.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.optimization._distance_matrix import (
    DISTANCE_MATRIX_OPERATION,
    compute_distance_matrix,
)
from jacobian.math.graphs.optimization._distance_models import (
    GraphDistanceMatrixRequest,
    GraphDistanceMatrixResult,
    GraphDistanceRow,
)


def _request(vertices: list[str], edges: list[list[str]]) -> GraphDistanceMatrixRequest:
    return GraphDistanceMatrixRequest.model_validate(
        {"graph": {"vertices": vertices, "edges": edges}}
    )


def _row(source: str, distances: list[int | None]) -> GraphDistanceRow:
    return GraphDistanceRow(source=source, distances=tuple(distances))


def _complete_rows(vertices: list[str]) -> list[GraphDistanceRow]:
    return [
        _row(
            source,
            [0 if target == source else 1 for target in vertices],
        )
        for source in vertices
    ]


def test_numeric_looking_labels_stay_lexicographic_and_bound_to_rows() -> None:
    result = compute_distance_matrix(
        _request(["1", "2", "10"], [["1", "2"], ["2", "10"]])
    )

    assert "semantics_version" not in result.model_dump(mode="json")
    assert result.vertex_ordering == "LEXICOGRAPHIC_ASCENDING"
    assert result.vertices == ("1", "10", "2")
    assert [row.source for row in result.rows] == list(result.vertices)
    assert result.connected is True
    assert result.rows == (
        _row("1", [0, 2, 1]),
        _row("10", [2, 0, 1]),
        _row("2", [1, 1, 0]),
    )


def test_numeric_looking_label_set_locks_lexicographic_order() -> None:
    vertices = ["0", "1", "2", "10", "11", "16"]
    result = compute_distance_matrix(_request(vertices, []))

    assert result.vertices == ("0", "1", "10", "11", "16", "2")
    assert [row.source for row in result.rows] == list(result.vertices)
    assert all(
        row.distances[index] == 0
        and all(
            distance is None
            for column, distance in enumerate(row.distances)
            if column != index
        )
        for index, row in enumerate(result.rows)
    )
    assert result.connected is False


def test_known_answer_path_with_unsorted_string_labels() -> None:
    result = compute_distance_matrix(
        _request(["c", "a", "b"], [["a", "b"], ["b", "c"]])
    )

    assert result.vertices == ("a", "b", "c")
    assert result.rows == (
        _row("a", [0, 1, 2]),
        _row("b", [1, 0, 1]),
        _row("c", [2, 1, 0]),
    )
    assert result.connected is True


def test_disconnected_graph_uses_null_for_unreachable_pairs() -> None:
    result = compute_distance_matrix(_request(["a", "b", "c"], [["a", "b"]]))

    assert result.vertices == ("a", "b", "c")
    assert result.connected is False
    assert result.rows[0].distances == (0, 1, None)
    assert result.rows[2].distances == (None, None, 0)


def test_empty_graph_returns_empty_labelled_matrix() -> None:
    result = compute_distance_matrix(_request([], []))

    assert result.vertices == ()
    assert result.rows == ()
    assert result.connected is False


def test_arbitrary_string_identifiers_remain_deterministic() -> None:
    result = compute_distance_matrix(
        _request(["root", "leaf-2", "10", "1"], [["1", "10"]])
    )

    assert result.vertices == ("1", "10", "leaf-2", "root")
    assert [row.source for row in result.rows] == list(result.vertices)


def test_operation_example_runs_through_the_catalog_wrapper() -> None:
    example_input = DISTANCE_MATRIX_OPERATION.examples[0].input
    request = DISTANCE_MATRIX_OPERATION.request_type.model_validate(example_input)
    result = DISTANCE_MATRIX_OPERATION.run(request)

    assert isinstance(result, DISTANCE_MATRIX_OPERATION.result_type)
    assert result.vertices == ("a", "b", "c")
    assert [row.source for row in result.rows] == list(result.vertices)


def test_producer_payload_revalidates_with_rows_bound_to_the_same_order() -> None:
    result = compute_distance_matrix(
        _request(["1", "2", "10"], [["1", "2"], ["2", "10"]])
    )
    rebound = GraphDistanceMatrixResult.model_validate(result.model_dump())

    assert rebound == result


def test_result_rejects_rows_reordered_away_from_the_canonical_axis() -> None:
    # Natural-order rows ("1", "2", "10") over the lexicographic vertex axis
    # ("1", "10", "2") must not look schema-correct: row labels are bound.
    with pytest.raises(ValidationError):
        GraphDistanceMatrixResult(
            vertex_ordering="LEXICOGRAPHIC_ASCENDING",
            pair_coverage="ALL_ORDERED_VERTEX_PAIRS",
            unreachable_representation="JSON_NULL",
            vertices=("1", "10", "2"),
            rows=(
                _row("1", [0, 1, 1]),
                _row("2", [1, 0, 1]),
                _row("10", [1, 1, 0]),
            ),
            connected=True,
        )


def test_result_rejects_relabelled_rows_over_the_same_positions() -> None:
    # Swapping row labels without touching the matrix cells must also fail:
    # every row must carry the canonical vertex label at its position.
    with pytest.raises(ValidationError):
        GraphDistanceMatrixResult(
            vertex_ordering="LEXICOGRAPHIC_ASCENDING",
            pair_coverage="ALL_ORDERED_VERTEX_PAIRS",
            unreachable_representation="JSON_NULL",
            vertices=("1", "10", "2"),
            rows=(
                _row("10", [0, 1, 1]),
                _row("1", [1, 0, 1]),
                _row("2", [1, 1, 0]),
            ),
            connected=True,
        )


@pytest.mark.parametrize(
    ("vertices", "message"),
    [
        (("2", "10", "1"), "unique and sorted"),
        (("1", "1", "2"), "unique and sorted"),
    ],
)
def test_result_rejects_unsorted_or_duplicate_vertices(
    vertices: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError):
        GraphDistanceMatrixResult(
            vertex_ordering="LEXICOGRAPHIC_ASCENDING",
            pair_coverage="ALL_ORDERED_VERTEX_PAIRS",
            unreachable_representation="JSON_NULL",
            vertices=vertices,
            rows=tuple(_complete_rows(list(vertices))),
            connected=True,
        )


def test_result_rejects_missing_rows_and_nonsquare_rows() -> None:
    vertices = ("1", "10", "2")
    with pytest.raises(ValidationError):
        GraphDistanceMatrixResult(
            vertex_ordering="LEXICOGRAPHIC_ASCENDING",
            pair_coverage="ALL_ORDERED_VERTEX_PAIRS",
            unreachable_representation="JSON_NULL",
            vertices=vertices,
            rows=tuple(_complete_rows(list(vertices))[:2]),
            connected=True,
        )
    with pytest.raises(ValidationError):
        GraphDistanceMatrixResult(
            vertex_ordering="LEXICOGRAPHIC_ASCENDING",
            pair_coverage="ALL_ORDERED_VERTEX_PAIRS",
            unreachable_representation="JSON_NULL",
            vertices=vertices,
            rows=(
                _row("1", [0, 1]),
                _row("10", [1, 0]),
                _row("2", [0, 1]),
            ),
            connected=True,
        )


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            (
                _row("1", [1, 1, 1]),
                _row("10", [1, 0, 1]),
                _row("2", [1, 1, 0]),
            ),
            "diagonal must be zero",
        ),
        (
            (
                _row("1", [0, 0, 1]),
                _row("10", [0, 0, 1]),
                _row("2", [1, 1, 0]),
            ),
            "positive or null",
        ),
        (
            (
                _row("1", [0, 1, 1]),
                _row("10", [2, 0, 1]),
                _row("2", [1, 1, 0]),
            ),
            "must be symmetric",
        ),
        (
            (
                _row("1", [0, 1, 3]),
                _row("10", [1, 0, 1]),
                _row("2", [3, 1, 0]),
            ),
            "triangle inequality",
        ),
    ],
)
def test_result_rejects_broken_metric_invariants(
    rows: tuple[GraphDistanceRow, ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError):
        GraphDistanceMatrixResult(
            vertex_ordering="LEXICOGRAPHIC_ASCENDING",
            pair_coverage="ALL_ORDERED_VERTEX_PAIRS",
            unreachable_representation="JSON_NULL",
            vertices=("1", "10", "2"),
            rows=rows,
            connected=True,
        )


def test_result_rejects_connected_mismatch() -> None:
    with pytest.raises(ValidationError):
        GraphDistanceMatrixResult(
            vertex_ordering="LEXICOGRAPHIC_ASCENDING",
            pair_coverage="ALL_ORDERED_VERTEX_PAIRS",
            unreachable_representation="JSON_NULL",
            vertices=("1", "10", "2"),
            rows=tuple(_complete_rows(["1", "10", "2"])),
            connected=False,
        )
