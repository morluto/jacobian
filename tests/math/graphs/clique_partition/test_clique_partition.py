"""Defining-invariant tests for edge-clique partition checking."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.clique_partition._models import (
    EdgeCliquePartitionRequest,
    EdgeCliquePartitionResult,
)
from jacobian.math.graphs.clique_partition.operations import (
    check_edge_clique_partition,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

DIAMOND = {
    "vertices": ["a", "b", "c", "d"],
    "edges": [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"], ["b", "d"]],
}


def _check(
    graph: dict[str, object], parts: list[list[str]]
) -> EdgeCliquePartitionResult:
    request = EdgeCliquePartitionRequest.model_validate(
        {"graph": graph, "parts": [tuple(part) for part in parts]}
    )
    return check_edge_clique_partition(request.graph, request.parts)


class TestDiamondFixtures:
    def test_valid_partition(self) -> None:
        result = _check(DIAMOND, [["a", "b", "c"], ["a", "d"], ["b", "d"]])
        assert result.verdict == "VALID"
        assert result.is_partition is True

    def test_double_covered_edge(self) -> None:
        result = _check(DIAMOND, [["a", "b", "c"], ["a", "b", "d"]])
        assert result.verdict == "INVALID"
        assert result.overcovered_edge == ("a", "b")
        assert result.overcovering_parts == (0, 1)

    def test_uncovered_edge(self) -> None:
        result = _check(DIAMOND, [["a", "b", "c"], ["a", "d"]])
        assert result.verdict == "INVALID"
        assert result.uncovered_edge == ("b", "d")

    def test_non_clique_part(self) -> None:
        result = _check(DIAMOND, [["a", "b", "c", "d"]])
        assert result.verdict == "INVALID"
        assert result.failing_part == 0
        assert result.failing_nonedge == ("c", "d")


class TestStructuralRejections:
    def test_singleton_part_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _check(DIAMOND, [["a", "b", "c"], ["d"]])

    def test_empty_part_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _check(DIAMOND, [["a", "b", "c"], []])

    def test_duplicate_members_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _check(DIAMOND, [["a", "a", "b"]])

    def test_foreign_vertex_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _check(DIAMOND, [["a", "b", "z"]])

    def test_duplicate_parts_fail_as_overcoverage(self) -> None:
        result = _check(
            {
                "vertices": ["a", "b"],
                "edges": [["a", "b"]],
            },
            [["a", "b"], ["a", "b"]],
        )
        assert result.verdict == "INVALID"
        assert result.overcovered_edge == ("a", "b")


class TestSemantics:
    def test_part_order_does_not_matter(self) -> None:
        first = _check(DIAMOND, [["a", "d"], ["a", "b", "c"], ["b", "d"]])
        assert first.verdict == "VALID"

    def test_empty_graph_no_parts_valid(self) -> None:
        result = _check({"vertices": ["a", "b"], "edges": []}, [])
        assert result.verdict == "VALID"

    def test_k2_parts(self) -> None:
        result = _check(
            {
                "vertices": ["a", "b", "c"],
                "edges": [["a", "b"], ["b", "c"]],
            },
            [["a", "b"], ["b", "c"]],
        )
        assert result.verdict == "VALID"

    def test_overlapping_vertices_disjoint_edges_valid(self) -> None:
        result = _check(
            {
                "vertices": ["a", "b", "c"],
                "edges": [["a", "b"], ["a", "c"]],
            },
            [["a", "b"], ["a", "c"]],
        )
        assert result.verdict == "VALID"

    def test_result_reparses(self) -> None:
        result = _check(DIAMOND, [["a", "b", "c"], ["a", "d"], ["b", "d"]])
        assert (
            EdgeCliquePartitionResult.model_validate_json(result.model_dump_json())
            == result
        )

    def test_independent_pair_reconstruction(self) -> None:
        graph = SimpleUndirectedGraph.model_validate(DIAMOND)
        parts = (("a", "b", "c"), ("a", "d"), ("b", "d"))
        result = check_edge_clique_partition(graph, parts)
        assert result.verdict == "VALID"
        edge_set = set(graph.edges)
        seen: dict[tuple[str, str], int] = dict.fromkeys(edge_set, 0)
        for part in parts:
            for left_index in range(len(part)):
                for right_index in range(left_index + 1, len(part)):
                    left, right = part[left_index], part[right_index]
                    edge = (left, right) if left < right else (right, left)
                    assert edge in edge_set
                    seen[edge] += 1
        assert all(count == 1 for count in seen.values())
