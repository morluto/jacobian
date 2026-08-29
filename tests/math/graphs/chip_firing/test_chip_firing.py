"""Tests for chip-firing operations."""

from typing import TypedDict

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.chip_firing._models import (
    CriticalGroupRequest,
    CriticalGroupResult,
    FireVectorRequest,
    FiringRequest,
    ReducedLaplacianRequest,
    SinkConfiguration,
)
from jacobian.math.graphs.chip_firing._tools import compute_critical_group
from jacobian.math.graphs.chip_firing.operations import (
    abel_jacobi,
    canonical_divisor,
    critical_group,
    degree,
    fire_vector,
    firing,
    laplacian,
    parallel_step,
    q_reduced,
    reduced_laplacian,
    stabilize,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


class GraphWire(TypedDict):
    vertices: list[str]
    edges: list[list[str]]


def _graph(wire: GraphWire) -> SimpleUndirectedGraph:
    """Validate a JSON-shaped graph fixture at the typed model boundary."""
    return SimpleUndirectedGraph.model_validate(wire)


GRAPH_WIRE: GraphWire = {
    "vertices": ["a", "b", "c"],
    "edges": [["a", "b"], ["b", "c"]],
}
C3_WIRE: GraphWire = {
    "vertices": ["a", "b", "c"],
    "edges": [["a", "b"], ["b", "c"], ["a", "c"]],
}
GRAPH = _graph(GRAPH_WIRE)
C3 = _graph(C3_WIRE)


class TestLaplacian:
    def test_path_graph(self) -> None:
        result = laplacian(GRAPH)
        assert result.vertices == ("a", "b", "c")
        assert result.degrees == (1, 2, 1)
        assert result.laplacian == ((1, -1, 0), (-1, 2, -1), (0, -1, 1))

    def test_single_vertex(self) -> None:
        result = laplacian(_graph({"vertices": ["x"], "edges": []}))
        assert result.laplacian == ((0,),)

    def test_triangle(self) -> None:
        result = laplacian(C3)
        assert result.degrees == (2, 2, 2)
        assert result.laplacian == ((2, -1, -1), (-1, 2, -1), (-1, -1, 2))


class TestReducedLaplacian:
    def test_path_graph_sink_a(self) -> None:
        result = reduced_laplacian(GRAPH, "a")
        assert result.sink == "a"
        assert result.reduced_laplacian == ((2, -1), (-1, 1))

    def test_triangle_sink_b(self) -> None:
        result = reduced_laplacian(C3, "b")
        assert result.reduced_laplacian == ((2, -1), (-1, 2))

    def test_invalid_sink(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ReducedLaplacianRequest(graph=GRAPH, sink="x")
        assert exc_info.value.errors()[0]["type"] == "chip_firing.sink_not_in_graph"


class TestFiring:
    def test_fire_middle_vertex(self) -> None:
        result = firing(GRAPH, (3, 0, 1), "b")
        assert result.fired_divisor == (4, -2, 2)

    def test_fire_leaf(self) -> None:
        result = firing(GRAPH, (3, 0, 1), "a")
        assert result.fired_divisor == (2, 1, 1)

    def test_invalid_vertex(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FiringRequest(graph=GRAPH, divisor=(0, 0, 0), firing_vertex="x")
        assert (
            exc_info.value.errors()[0]["type"]
            == "chip_firing.firing_vertex_not_in_graph"
        )

    def test_wrong_divisor_length(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FiringRequest(graph=GRAPH, divisor=(0, 0), firing_vertex="a")
        assert exc_info.value.errors()[0]["type"] == "chip_firing.divisor_length"


class TestFireVector:
    def test_fire_unit_vector_e_a(self) -> None:
        result = fire_vector(GRAPH, (3, 0, 1), (1, 0, 0))
        assert result.fired_divisor == (2, 1, 1)
        assert result.degree_preserved is True

    def test_fire_vertex_b_is_unit_vector(self) -> None:
        fire_vertex = firing(GRAPH, (3, 0, 1), "b")
        fire_vector_result = fire_vector(GRAPH, (3, 0, 1), (0, 1, 0))
        assert fire_vertex.fired_divisor == fire_vector_result.fired_divisor

    def test_degree_preservation(self) -> None:
        result = fire_vector(GRAPH, (5, 3, 2), (2, 1, 3))
        assert result.degree_preserved is True
        assert sum(result.fired_divisor) == sum([5, 3, 2])

    def test_firing_vector_composition(self) -> None:
        div = (5, 3, 2)
        f1 = (1, 0, 0)
        f2 = (0, 1, 0)
        r1 = fire_vector(GRAPH, div, f1)
        r2 = fire_vector(GRAPH, r1.fired_divisor, f2)
        composed = fire_vector(GRAPH, div, tuple(f1[i] + f2[i] for i in range(3)))
        assert r2.fired_divisor == composed.fired_divisor

    def test_wrong_vector_length(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FireVectorRequest(graph=GRAPH, divisor=(0, 0, 0), firing_vector=(0, 0))
        assert exc_info.value.errors()[0]["type"] == "chip_firing.firing_vector_length"


class TestStabilize:
    def test_stabilize_path_graph(self) -> None:
        sc = SinkConfiguration(graph=GRAPH, sink="a", configuration=(0, 3, 0))
        result = stabilize(sc.graph, sc.sink, sc.configuration)
        assert result.stable == (2, 1, 0)
        assert result.odometer == (0, 2, 2)
        assert result.total_firings == 4

    def test_stabilization_is_stable(self) -> None:
        sc = SinkConfiguration(graph=GRAPH, sink="a", configuration=(0, 5, 0))
        result = stabilize(sc.graph, sc.sink, sc.configuration)
        degrees = laplacian(GRAPH).degrees
        for i, v in enumerate(GRAPH.vertices):
            if v != sc.sink:
                assert 0 <= result.stable[i] < degrees[i]

    def test_stabilization_idempotence(self) -> None:
        sc = SinkConfiguration(graph=GRAPH, sink="a", configuration=(0, 5, 0))
        r1 = stabilize(sc.graph, sc.sink, sc.configuration)
        sc2 = SinkConfiguration(graph=GRAPH, sink="a", configuration=r1.stable)
        r2 = stabilize(sc2.graph, sc2.sink, sc2.configuration)
        assert r1.stable == r2.stable
        assert r2.total_firings == 0

    def test_nontrivial_odometer(self) -> None:
        sc = SinkConfiguration(graph=GRAPH, sink="a", configuration=(0, 10, 0))
        result = stabilize(sc.graph, sc.sink, sc.configuration)
        assert sum(result.odometer) > 0

    def test_negative_nonsink_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            SinkConfiguration(graph=GRAPH, sink="a", configuration=(0, -1, 0))
        assert exc_info.value.errors()[0]["type"] == "chip_firing.nonsink_negative"


class TestParallelStep:
    def test_one_step(self) -> None:
        sc = SinkConfiguration(graph=GRAPH, sink="a", configuration=(0, 3, 0))
        result = parallel_step(sc.graph, sc.sink, sc.configuration)
        assert result.fired_vertices == ("b",)
        assert result.next_configuration[1] == 3 - 2

    def test_parallel_step_agrees_with_stabilization(self) -> None:
        sc = SinkConfiguration(graph=GRAPH, sink="a", configuration=(0, 5, 0))
        config = list(sc.configuration)
        for _ in range(20):
            result = parallel_step(GRAPH, "a", tuple(config))
            config = list(result.next_configuration)
        stable = stabilize(sc.graph, sc.sink, sc.configuration)
        assert tuple(config) == stable.stable


class TestQReduced:
    def test_triangle_q_reduced(self) -> None:
        result = q_reduced(C3, (5, 0, 0), "a")
        assert result.reduced_divisor[0] == 5
        assert all(c >= 0 for i, c in enumerate(result.reduced_divisor) if i > 0)

    def test_q_reduced_idempotence(self) -> None:
        r1 = q_reduced(C3, (10, 5, 3), "a")
        r2 = q_reduced(C3, r1.reduced_divisor, "a")
        assert r1.reduced_divisor == r2.reduced_divisor

    def test_q_reduced_nonnegative_nonsink(self) -> None:
        result = q_reduced(C3, (10, 5, 3), "a")
        assert result.reduced_divisor[1] >= 0
        assert result.reduced_divisor[2] >= 0

    def test_q_reduced_firing_vector(self) -> None:
        result = q_reduced(C3, (10, 5, 3), "a")
        lap = laplacian(C3).laplacian
        f = result.firing_vector
        reconstructed = []
        for i in range(3):
            delta = sum(lap[i][j] * f[j] for j in range(3))
            reconstructed.append([10, 5, 3][i] - delta)
        assert tuple(reconstructed) == result.reduced_divisor


class TestDegree:
    def test_degree(self) -> None:
        result = degree((3, 0, 1))
        assert result.degree == 4

    def test_degree_negative(self) -> None:
        result = degree((-1, 2, -3))
        assert result.degree == -2


class TestCanonicalDivisor:
    def test_path_graph(self) -> None:
        result = canonical_divisor(GRAPH)
        assert result.divisor == (-1, 0, -1)
        assert result.degree == -2

    def test_triangle(self) -> None:
        result = canonical_divisor(C3)
        assert result.divisor == (0, 0, 0)
        assert result.degree == 0

    def test_degree_formula(self) -> None:
        graph: GraphWire = {
            "vertices": ["a", "b", "c", "d"],
            "edges": [["a", "b"], ["b", "c"], ["c", "d"], ["a", "d"]],
        }
        result = canonical_divisor(_graph(graph))
        assert result.degree == 2 * len(graph["edges"]) - 2 * len(graph["vertices"])


class TestCriticalGroup:
    def test_triangle(self) -> None:
        result = critical_group(C3, "a")
        assert result.invariant_factors == (1, 3)
        assert result.order == 3

    def test_cycle_c4(self) -> None:
        c4: GraphWire = {
            "vertices": ["a", "b", "c", "d"],
            "edges": [["a", "b"], ["b", "c"], ["c", "d"], ["a", "d"]],
        }
        result = critical_group(_graph(c4), "a")
        assert result.invariant_factors == (1, 1, 4)
        assert result.order == 4

    def test_tree_is_trivial(self) -> None:
        tree: GraphWire = {"vertices": ["a", "b"], "edges": [["a", "b"]]}
        result = critical_group(_graph(tree), "a")
        assert result.order == 1

    def test_complete_k4(self) -> None:
        k4: GraphWire = {
            "vertices": ["a", "b", "c", "d"],
            "edges": [
                ["a", "b"],
                ["a", "c"],
                ["a", "d"],
                ["b", "c"],
                ["b", "d"],
                ["c", "d"],
            ],
        }
        result = critical_group(_graph(k4), "a")
        assert result.order == 16

    def test_order_matches_spanning_tree_count(self) -> None:
        from sympy import Matrix

        def count_spanning_trees(
            vertices: list[str], edges: list[tuple[str, str]]
        ) -> tuple[CriticalGroupResult, int]:
            n = len(vertices)
            idx = {v: i for i, v in enumerate(vertices)}
            lap = [[0] * n for _ in range(n)]
            for u, v in edges:
                lap[idx[u]][idx[v]] -= 1
                lap[idx[v]][idx[u]] -= 1
            for i in range(n):
                deg = -sum(lap[i][j] for j in range(n) if j != i)
                lap[i][i] = deg
            minor = Matrix([row[: n - 1] for row in lap[: n - 1]])
            graph: GraphWire = {
                "vertices": vertices,
                "edges": [list(e) for e in edges],
            }
            result = critical_group(_graph(graph), vertices[0])
            return result, int(minor.det())

        for vertices, edges in [
            (["a", "b", "c"], [("a", "b"), ("b", "c"), ("a", "c")]),
            (
                ["a", "b", "c", "d"],
                [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")],
            ),
        ]:
            res, trees = count_spanning_trees(vertices, edges)
            assert res.order == trees

    def test_sink_change_preserves_order(self) -> None:
        r1 = critical_group(C3, "a")
        r2 = critical_group(C3, "b")
        r3 = critical_group(C3, "c")
        assert r1.order == r2.order == r3.order
        assert r1.invariant_factors == r2.invariant_factors == r3.invariant_factors

    def test_flint_diagonal_snf_exceeds_the_previous_vertex_ceiling(self) -> None:
        order = 96
        labels = [f"v{index:03d}" for index in range(order)]
        graph = _graph(
            {
                "vertices": labels,
                "edges": [
                    [labels[index], labels[index + 1]] for index in range(order - 1)
                ]
                + [[labels[0], labels[-1]]],
            }
        )

        result = compute_critical_group(
            CriticalGroupRequest(graph=graph, sink=labels[0])
        )

        assert result.order == order
        assert result.invariant_factors == (1,) * (order - 2) + (order,)
        assert all(
            right % left == 0
            for left, right in zip(
                result.invariant_factors,
                result.invariant_factors[1:],
                strict=False,
            )
        )

    def test_expanded_snf_binds_the_request_deadline(self) -> None:
        import time

        from jacobian._execution import current_request_execution, request_execution
        from jacobian.math.graphs.chip_firing._snf_process import (
            _SNF_WALL_SECONDS,
            smith_normal_form_diagonal,
        )

        matrix = [[2, -1], [-1, 2]]
        started = time.monotonic()
        with request_execution(started_at=started):
            diagonal = smith_normal_form_diagonal(matrix)
            execution = current_request_execution()
            assert execution is not None
            assert execution.deadline == pytest.approx(
                started + _SNF_WALL_SECONDS, abs=1e-6
            )

        assert diagonal == (1, 3)

    def test_critical_group_rejects_disconnected_graph(self) -> None:
        graph = _graph(
            {
                "vertices": ["a", "b", "c", "d"],
                "edges": [["a", "b"], ["c", "d"]],
            }
        )

        with pytest.raises(
            OperationDomainValidationError, match="requires a connected graph"
        ):
            critical_group(graph, "a")

    def test_critical_group_rejects_the_snf_work_boundary(self) -> None:
        order = 116
        labels = [f"v{index:03d}" for index in range(order)]
        graph = _graph(
            {
                "vertices": labels,
                "edges": [
                    [labels[index], labels[index + 1]] for index in range(order - 1)
                ]
                + [[labels[0], labels[-1]]],
            }
        )

        with pytest.raises(OperationDomainValidationError, match="SNF exceeds"):
            critical_group(graph, labels[0])


class TestAbelJacobi:
    def test_degree_zero_mapping(self) -> None:
        result = abel_jacobi(C3, (1, -1, 0), "a")
        assert len(result.coordinates) > 0
        assert result.invariant_factors == (1, 3)

    def test_zero_divisor(self) -> None:
        result = abel_jacobi(C3, (0, 0, 0), "a")
        assert all(c == 0 for c in result.coordinates)

    def test_coordinates_in_range(self) -> None:
        result = abel_jacobi(C3, (5, -3, -2), "a")
        for c in result.coordinates:
            assert c >= 0


class TestVertexRelabelling:
    def test_laplacian_equivariance(self) -> None:
        graph1: GraphWire = {
            "vertices": ["a", "b", "c"],
            "edges": [["a", "b"], ["b", "c"]],
        }
        graph2: GraphWire = {
            "vertices": ["x", "y", "z"],
            "edges": [["x", "y"], ["y", "z"]],
        }
        r1 = laplacian(_graph(graph1))
        r2 = laplacian(_graph(graph2))
        assert r1.laplacian == r2.laplacian

    def test_critical_group_relabelling(self) -> None:
        graph1: GraphWire = {
            "vertices": ["a", "b", "c"],
            "edges": [["a", "b"], ["b", "c"], ["a", "c"]],
        }
        graph2: GraphWire = {
            "vertices": ["x", "y", "z"],
            "edges": [["x", "y"], ["y", "z"], ["x", "z"]],
        }
        r1 = critical_group(_graph(graph1), "a")
        r2 = critical_group(_graph(graph2), "x")
        assert r1.invariant_factors == r2.invariant_factors
        assert r1.order == r2.order
