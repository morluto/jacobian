"""Tests for graph flow, cut, and edge-disjoint path operations."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.flows._models import (
    EdgeDisjointPathsRequest,
    EdgeDisjointPathsResult,
    MaxFlowRequest,
    MaxFlowResult,
    MinCutRequest,
    MinCutResult,
)
from jacobian.math.graphs.flows._tools import (
    compute_edge_disjoint_paths,
    compute_max_flow,
    compute_min_cut,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _max_flow(graph: dict[str, object], source: int, sink: int) -> MaxFlowResult:
    return compute_max_flow(
        MaxFlowRequest.model_validate({"graph": graph, "source": source, "sink": sink})
    )


def _min_cut(graph: dict[str, object], source: int, sink: int) -> MinCutResult:
    return compute_min_cut(
        MinCutRequest.model_validate({"graph": graph, "source": source, "sink": sink})
    )


def _edge_disjoint(
    graph: dict[str, object], source: int, sink: int
) -> EdgeDisjointPathsResult:
    return compute_edge_disjoint_paths(
        EdgeDisjointPathsRequest.model_validate(
            {"graph": graph, "source": source, "sink": sink}
        )
    )


# ---------------------------------------------------------------------------
# Max-flow
# ---------------------------------------------------------------------------


class TestMaxFlow:
    def test_simple_path_max_flow(self) -> None:
        result = _max_flow(
            {
                "vertex_count": 3,
                "edges": [
                    {"source": 0, "target": 1, "capacity": {"num": "3", "den": "1"}},
                    {"source": 1, "target": 2, "capacity": {"num": "2", "den": "1"}},
                ],
            },
            0,
            2,
        )
        assert result.flow_value.num == "2"
        assert result.flow_value.den == "1"

    def test_flow_decomposition_returns_per_edge_flow(self) -> None:
        """The flow_edges field should contain per-edge flow values."""
        result = _max_flow(
            {
                "vertex_count": 3,
                "edges": [
                    {"source": 0, "target": 1, "capacity": {"num": "3", "den": "1"}},
                    {"source": 1, "target": 2, "capacity": {"num": "2", "den": "1"}},
                ],
            },
            0,
            2,
        )
        assert len(result.flow_edges) == 2
        for edge in result.flow_edges:
            assert edge.flow.as_fraction() > 0

    def test_flow_decomposition_satisfies_capacity_constraint(self) -> None:
        """Each edge flow must not exceed its capacity."""
        graph = {
            "vertex_count": 4,
            "edges": [
                {"source": 0, "target": 1, "capacity": {"num": "5", "den": "1"}},
                {"source": 0, "target": 2, "capacity": {"num": "3", "den": "1"}},
                {"source": 1, "target": 3, "capacity": {"num": "4", "den": "1"}},
                {"source": 2, "target": 3, "capacity": {"num": "6", "den": "1"}},
            ],
        }
        result = _max_flow(graph, 0, 3)
        capacities = {(0, 1): 5, (0, 2): 3, (1, 3): 4, (2, 3): 6}
        for edge in result.flow_edges:
            assert edge.flow.as_fraction() <= capacities[(edge.source, edge.target)]

    def test_flow_value_matches_sum_of_outgoing_source_flows(self) -> None:
        """Conservation: the flow value equals the total outflow from source."""
        result = _max_flow(
            {
                "vertex_count": 4,
                "edges": [
                    {"source": 0, "target": 1, "capacity": {"num": "5", "den": "1"}},
                    {"source": 0, "target": 2, "capacity": {"num": "3", "den": "1"}},
                    {"source": 1, "target": 3, "capacity": {"num": "4", "den": "1"}},
                    {"source": 2, "target": 3, "capacity": {"num": "6", "den": "1"}},
                ],
            },
            0,
            3,
        )
        source_outflow = sum(
            edge.flow.as_fraction()
            for edge in result.flow_edges
            if edge.source == result.source
        )
        assert source_outflow == result.flow_value.as_fraction()

    def test_direct_edge_max_flow(self) -> None:
        result = _max_flow(
            {
                "vertex_count": 2,
                "edges": [
                    {"source": 0, "target": 1, "capacity": {"num": "10", "den": "1"}},
                ],
            },
            0,
            1,
        )
        assert result.flow_value.num == "10"
        assert result.flow_value.den == "1"

    def test_disconnected_graph_max_flow_is_zero(self) -> None:
        """When there is no path from source to sink, max flow is 0."""
        result = _max_flow(
            {
                "vertex_count": 4,
                "edges": [
                    {"source": 0, "target": 1, "capacity": {"num": "5", "den": "1"}},
                    {"source": 2, "target": 3, "capacity": {"num": "7", "den": "1"}},
                ],
            },
            0,
            3,
        )
        assert result.flow_value.num == "0"
        assert result.flow_value.den == "1"

    def test_rational_capacities_max_flow(self) -> None:
        """Max flow with rational (non-integer) capacities."""
        result = _max_flow(
            {
                "vertex_count": 3,
                "edges": [
                    {"source": 0, "target": 1, "capacity": {"num": "1", "den": "2"}},
                    {"source": 1, "target": 2, "capacity": {"num": "1", "den": "3"}},
                ],
            },
            0,
            2,
        )
        assert result.flow_value.as_fraction() == Fraction(1, 3)

    def test_contract_rejects_source_equals_sink(self) -> None:
        request = MaxFlowRequest.model_validate(
            {
                "graph": {
                    "vertex_count": 2,
                    "edges": [
                        {
                            "source": 0,
                            "target": 1,
                            "capacity": {"num": "1", "den": "1"},
                        },
                    ],
                },
                "source": 0,
                "sink": 0,
            }
        )
        with pytest.raises(OperationDomainValidationError):
            compute_max_flow(request)

    def test_contract_rejects_out_of_range_source(self) -> None:
        request = MaxFlowRequest.model_validate(
            {
                "graph": {
                    "vertex_count": 2,
                    "edges": [
                        {
                            "source": 0,
                            "target": 1,
                            "capacity": {"num": "1", "den": "1"},
                        },
                    ],
                },
                "source": 5,
                "sink": 1,
            }
        )
        with pytest.raises(OperationDomainValidationError):
            compute_max_flow(request)


# ---------------------------------------------------------------------------
# Min-cut
# ---------------------------------------------------------------------------


class TestMinCut:
    def test_simple_path_min_cut(self) -> None:
        result = _min_cut(
            {
                "vertex_count": 3,
                "edges": [
                    {"source": 0, "target": 1, "capacity": {"num": "3", "den": "1"}},
                    {"source": 1, "target": 2, "capacity": {"num": "2", "den": "1"}},
                ],
            },
            0,
            2,
        )
        assert result.cut_value.num == "2"
        assert result.cut_value.den == "1"

    def test_cut_partition_covers_all_vertices(self) -> None:
        result = _min_cut(
            {
                "vertex_count": 4,
                "edges": [
                    {"source": 0, "target": 1, "capacity": {"num": "5", "den": "1"}},
                    {"source": 0, "target": 2, "capacity": {"num": "3", "den": "1"}},
                    {"source": 1, "target": 3, "capacity": {"num": "4", "den": "1"}},
                    {"source": 2, "target": 3, "capacity": {"num": "6", "den": "1"}},
                ],
            },
            0,
            3,
        )
        all_vertices = set(result.reachable) | set(result.unreachable)
        assert all_vertices == {0, 1, 2, 3}
        assert len(set(result.reachable) & set(result.unreachable)) == 0

    def test_source_in_reachable_sink_in_unreachable(self) -> None:
        result = _min_cut(
            {
                "vertex_count": 3,
                "edges": [
                    {"source": 0, "target": 1, "capacity": {"num": "3", "den": "1"}},
                    {"source": 1, "target": 2, "capacity": {"num": "2", "den": "1"}},
                ],
            },
            0,
            2,
        )
        assert 0 in result.reachable
        assert 2 in result.unreachable

    def test_min_cut_equals_max_flow(self) -> None:
        """Max-flow min-cut theorem: the max flow equals the min cut."""
        graph = {
            "vertex_count": 4,
            "edges": [
                {"source": 0, "target": 1, "capacity": {"num": "5", "den": "1"}},
                {"source": 0, "target": 2, "capacity": {"num": "3", "den": "1"}},
                {"source": 1, "target": 3, "capacity": {"num": "4", "den": "1"}},
                {"source": 2, "target": 3, "capacity": {"num": "6", "den": "1"}},
            ],
        }
        flow = _max_flow(graph, 0, 3)
        cut = _min_cut(graph, 0, 3)
        assert flow.flow_value.as_fraction() == cut.cut_value.as_fraction()

    def test_disconnected_graph_min_cut_is_zero(self) -> None:
        result = _min_cut(
            {
                "vertex_count": 4,
                "edges": [
                    {"source": 0, "target": 1, "capacity": {"num": "5", "den": "1"}},
                    {"source": 2, "target": 3, "capacity": {"num": "7", "den": "1"}},
                ],
            },
            0,
            3,
        )
        assert result.cut_value.num == "0"
        assert result.cut_value.den == "1"

    def test_rational_capacities_min_cut(self) -> None:
        result = _min_cut(
            {
                "vertex_count": 3,
                "edges": [
                    {"source": 0, "target": 1, "capacity": {"num": "1", "den": "2"}},
                    {"source": 1, "target": 2, "capacity": {"num": "1", "den": "3"}},
                ],
            },
            0,
            2,
        )
        assert result.cut_value.as_fraction() == Fraction(1, 3)

    def test_contract_rejects_duplicate_edges(self) -> None:
        with pytest.raises(ValidationError):
            MinCutRequest.model_validate(
                {
                    "graph": {
                        "vertex_count": 3,
                        "edges": [
                            {
                                "source": 0,
                                "target": 1,
                                "capacity": {"num": "1", "den": "1"},
                            },
                            {
                                "source": 0,
                                "target": 1,
                                "capacity": {"num": "2", "den": "1"},
                            },
                        ],
                    },
                    "source": 0,
                    "sink": 2,
                }
            )

    def test_contract_rejects_negative_capacity(self) -> None:
        with pytest.raises(ValidationError):
            MinCutRequest.model_validate(
                {
                    "graph": {
                        "vertex_count": 2,
                        "edges": [
                            {
                                "source": 0,
                                "target": 1,
                                "capacity": {"num": "-1", "den": "1"},
                            },
                        ],
                    },
                    "source": 0,
                    "sink": 1,
                }
            )


# ---------------------------------------------------------------------------
# Edge-disjoint paths (Menger)
# ---------------------------------------------------------------------------


class TestEdgeDisjointPaths:
    def test_diamond_graph_has_two_edge_disjoint_paths(self) -> None:
        result = _edge_disjoint(
            {
                "vertex_count": 4,
                "edges": [[0, 1], [0, 2], [1, 3], [2, 3]],
            },
            0,
            3,
        )
        assert result.path_count == 2
        assert len(result.paths) == 2
        for path in result.paths:
            assert path[0] == 0
            assert path[-1] == 3

    def test_single_path_has_one_edge_disjoint_path(self) -> None:
        result = _edge_disjoint(
            {
                "vertex_count": 3,
                "edges": [[0, 1], [1, 2]],
            },
            0,
            2,
        )
        assert result.path_count == 1
        assert len(result.paths) == 1
        assert result.paths[0] == (0, 1, 2)

    def test_no_path_returns_zero(self) -> None:
        result = _edge_disjoint(
            {
                "vertex_count": 2,
                "edges": [[0, 1]],
            },
            1,
            0,
        )
        assert result.path_count == 0
        assert result.paths == ()

    def test_disconnected_graph_returns_zero(self) -> None:
        result = _edge_disjoint(
            {
                "vertex_count": 4,
                "edges": [[0, 1], [2, 3]],
            },
            0,
            3,
        )
        assert result.path_count == 0
        assert result.paths == ()

    def test_complete_graph_edge_disjoint_paths(self) -> None:
        """In K_n, there are n-1 edge-disjoint paths from 0 to n-1."""
        n = 5
        edges = [[i, j] for i in range(n) for j in range(n) if i != j]
        result = _edge_disjoint(
            {"vertex_count": n, "edges": edges},
            0,
            n - 1,
        )
        assert result.path_count == n - 1

    def test_paths_are_edge_disjoint(self) -> None:
        """Verify that returned paths share no edges."""
        result = _edge_disjoint(
            {
                "vertex_count": 6,
                "edges": [
                    [0, 1],
                    [0, 2],
                    [1, 3],
                    [2, 3],
                    [3, 4],
                    [3, 5],
                    [4, 5],
                ],
            },
            0,
            5,
        )
        used_edges: set[tuple[int, int]] = set()
        for path in result.paths:
            for u, v in zip(path[:-1], path[1:]):  # noqa: B905, RUF007
                edge = (u, v)
                assert edge not in used_edges, f"Edge {edge} used in multiple paths"
                used_edges.add(edge)

    def test_direct_edge_has_one_path(self) -> None:
        result = _edge_disjoint(
            {
                "vertex_count": 2,
                "edges": [[0, 1]],
            },
            0,
            1,
        )
        assert result.path_count == 1
        assert result.paths[0] == (0, 1)

    def test_contract_rejects_self_loop(self) -> None:
        with pytest.raises(ValidationError):
            EdgeDisjointPathsRequest.model_validate(
                {
                    "graph": {
                        "vertex_count": 3,
                        "edges": [[0, 0], [0, 1]],
                    },
                    "source": 0,
                    "sink": 1,
                }
            )

    def test_contract_rejects_duplicate_edges(self) -> None:
        with pytest.raises(ValidationError):
            EdgeDisjointPathsRequest.model_validate(
                {
                    "graph": {
                        "vertex_count": 3,
                        "edges": [[0, 1], [0, 1]],
                    },
                    "source": 0,
                    "sink": 1,
                }
            )

    def test_contract_rejects_source_equals_sink(self) -> None:
        request = EdgeDisjointPathsRequest.model_validate(
            {
                "graph": {
                    "vertex_count": 2,
                    "edges": [[0, 1]],
                },
                "source": 0,
                "sink": 0,
            }
        )
        with pytest.raises(OperationDomainValidationError):
            compute_edge_disjoint_paths(request)

    def test_contract_rejects_out_of_range_vertex_in_edge(self) -> None:
        with pytest.raises(ValidationError):
            EdgeDisjointPathsRequest.model_validate(
                {
                    "graph": {
                        "vertex_count": 2,
                        "edges": [[0, 3]],
                    },
                    "source": 0,
                    "sink": 1,
                }
            )


# ---------------------------------------------------------------------------
# Cross-consistency: max-flow vs min-cut vs edge-disjoint
# ---------------------------------------------------------------------------


class TestCrossConsistency:
    def test_unit_capacity_flow_equals_edge_disjoint_count(self) -> None:
        """For a unit-capacity graph, max flow = edge-disjoint path count."""
        vertex_count = 4
        edges_list = [[0, 1], [0, 2], [1, 3], [2, 3]]
        # Build a capacitated graph with unit capacities
        cap_graph = {
            "vertex_count": vertex_count,
            "edges": [
                {"source": s, "target": t, "capacity": {"num": "1", "den": "1"}}
                for s, t in edges_list
            ],
        }
        uncap_graph = {
            "vertex_count": vertex_count,
            "edges": [tuple(e) for e in edges_list],
        }
        flow = _max_flow(cap_graph, 0, 3)
        paths = _edge_disjoint(uncap_graph, 0, 3)
        assert flow.flow_value.as_fraction() == paths.path_count

    def test_max_flow_min_cut_equality(self) -> None:
        """Max-flow min-cut theorem."""
        graph = {
            "vertex_count": 5,
            "edges": [
                {"source": 0, "target": 1, "capacity": {"num": "3", "den": "1"}},
                {"source": 0, "target": 2, "capacity": {"num": "2", "den": "1"}},
                {"source": 1, "target": 2, "capacity": {"num": "1", "den": "1"}},
                {"source": 1, "target": 3, "capacity": {"num": "2", "den": "1"}},
                {"source": 2, "target": 3, "capacity": {"num": "3", "den": "1"}},
                {"source": 3, "target": 4, "capacity": {"num": "5", "den": "1"}},
            ],
        }
        flow = _max_flow(graph, 0, 4)
        cut = _min_cut(graph, 0, 4)
        assert flow.flow_value.as_fraction() == cut.cut_value.as_fraction()
