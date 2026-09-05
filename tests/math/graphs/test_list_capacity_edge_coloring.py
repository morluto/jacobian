"""Tests for prescribed-list edge coloring with per-color capacities."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.coloring._models import (
    ListCapacityEdgeColoringRequest,
    ListCapacityEdgeColoringResult,
)
from jacobian.math.graphs.coloring.operations import list_capacity_edge_coloring
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _run(
    vertices: list[str],
    edges: list[tuple[str, str]],
    palette: list[str],
    lists: list[tuple[tuple[str, str], list[str]]],
    capacities: list[tuple[str, int]],
):
    graph = SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple((a, b) if a < b else (b, a) for a, b in edges),
    )
    request = ListCapacityEdgeColoringRequest.model_validate(
        {
            "graph": graph.model_dump(mode="json"),
            "palette": palette,
            "lists": [{"edge": list(edge), "colors": colors} for edge, colors in lists],
            "capacities": [
                {"color": color, "capacity": capacity} for color, capacity in capacities
            ],
        }
    )
    return list_capacity_edge_coloring(
        request.graph, request.palette, request.lists, request.capacities
    )


def _assert_valid_assignment(result) -> None:
    assert result.status == "FEASIBLE"
    assert result.assignment is not None
    graph = result.graph
    assert len(result.assignment) == len(graph.edges)
    list_of = {tuple(entry.edge): set(entry.colors) for entry in result.lists}
    capacity_of = {entry.color: entry.capacity for entry in result.capacities}
    counts: dict[str, int] = {}
    incident: dict[str, set[str]] = {vertex: set() for vertex in graph.vertices}
    for edge, color in zip(graph.edges, result.assignment, strict=True):
        assert color in list_of[tuple(edge)]
        left, right = edge
        assert color not in incident[left] and color not in incident[right]
        incident[left].add(color)
        incident[right].add(color)
        counts[color] = counts.get(color, 0) + 1
    for color, count in counts.items():
        assert count <= capacity_of[color]


class TestDistinguishingFixtures:
    def test_forced_incident_conflict_infeasible(self) -> None:
        result = _run(
            ["a", "b", "c"],
            [("a", "b"), ("b", "c")],
            ["x", "y"],
            [((("a", "b")), ["x"]), ((("b", "c")), ["x"])],
            [("x", 5), ("y", 5)],
        )
        assert result.status == "INFEASIBLE"
        assert result.assignment is None

    def test_capacities_make_list_feasible_instance_infeasible(self) -> None:
        vertices = ["a", "b", "c", "d", "e", "f"]
        edges = [("a", "b"), ("c", "d"), ("e", "f")]
        lists = [
            ((("a", "b")), ["x"]),
            ((("c", "d")), ["x", "y"]),
            ((("e", "f")), ["y"]),
        ]
        result = _run(vertices, edges, ["x", "y"], lists, [("x", 1), ("y", 1)])
        assert result.status == "INFEASIBLE"

    def test_relaxed_capacities_feasible(self) -> None:
        vertices = ["a", "b", "c", "d", "e", "f"]
        edges = [("a", "b"), ("c", "d"), ("e", "f")]
        lists = [
            ((("a", "b")), ["x"]),
            ((("c", "d")), ["x", "y"]),
            ((("e", "f")), ["y"]),
        ]
        result = _run(vertices, edges, ["x", "y"], lists, [("x", 2), ("y", 1)])
        _assert_valid_assignment(result)


class TestBoundaries:
    def test_empty_list_forces_infeasibility(self) -> None:
        result = _run(
            ["a", "b"],
            [("a", "b")],
            ["x"],
            [((("a", "b")), [])],
            [("x", 1)],
        )
        assert result.status == "INFEASIBLE"

    def test_zero_capacities_infeasible(self) -> None:
        result = _run(
            ["a", "b"],
            [("a", "b")],
            ["x"],
            [((("a", "b")), ["x"])],
            [("x", 0)],
        )
        assert result.status == "INFEASIBLE"

    def test_no_edges_feasible(self) -> None:
        result = _run(["a"], [], ["x"], [], [("x", 0)])
        assert result.status == "FEASIBLE"
        assert result.assignment == ()

    def test_large_capacity_clamps(self) -> None:
        result = _run(
            ["a", "b"],
            [("a", "b")],
            ["x"],
            [((("a", "b")), ["x"])],
            [("x", 10**9)],
        )
        _assert_valid_assignment(result)

    def test_outside_palette_color_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _run(
                ["a", "b"],
                [("a", "b")],
                ["x"],
                [((("a", "b")), ["z"])],
                [("x", 1)],
            )

    def test_partial_coverage_rejected(self) -> None:
        graph = SimpleUndirectedGraph(
            vertices=("a", "b", "c"), edges=(("a", "b"), ("b", "c"))
        )
        with pytest.raises(ValidationError):
            ListCapacityEdgeColoringRequest.model_validate(
                {
                    "graph": graph.model_dump(mode="json"),
                    "palette": ["x"],
                    "lists": [{"edge": ["a", "b"], "colors": ["x"]}],
                    "capacities": [{"color": "x", "capacity": 1}],
                }
            )

    def test_result_reparses(self) -> None:
        result = _run(
            ["a", "b", "c"],
            [("a", "b"), ("b", "c")],
            ["x", "y"],
            [((("a", "b")), ["x"]), ((("b", "c")), ["y"])],
            [("x", 1), ("y", 1)],
        )
        _assert_valid_assignment(result)
        assert (
            ListCapacityEdgeColoringResult.model_validate(
                result.model_dump(mode="json")
            )
            == result
        )

    def test_brute_force_agreement_on_small_instances(self) -> None:
        from itertools import product

        vertices = ["a", "b", "c"]
        edges = [("a", "b"), ("a", "c"), ("b", "c")]
        palette = ["x", "y"]
        list_options = [[], ["x"], ["y"], ["x", "y"]]
        checked = 0
        for lists_choice in product(list_options, repeat=len(edges)):
            for cap_x in (0, 1, 2):
                for cap_y in (0, 1, 2):
                    lists = [
                        ((edge), list(colors))
                        for edge, colors in zip(edges, lists_choice, strict=True)
                    ]
                    result = _run(
                        vertices, edges, palette, lists, [("x", cap_x), ("y", cap_y)]
                    )
                    feasible = any(
                        all(
                            assignment[edge_index] in lists_choice[edge_index]
                            for edge_index in range(len(edges))
                        )
                        and all(
                            assignment[left] != assignment[right]
                            for left in range(len(edges))
                            for right in range(left + 1, len(edges))
                            if set(edges[left]) & set(edges[right])
                        )
                        and sum(1 for color in assignment if color == "x") <= cap_x
                        and sum(1 for color in assignment if color == "y") <= cap_y
                        for assignment in product(palette, repeat=len(edges))
                    )
                    assert (result.status == "FEASIBLE") == feasible
                    if feasible:
                        _assert_valid_assignment(result)
                    checked += 1
        assert checked == 4**3 * 3 * 3
