"""Tests for bounded minor/subdivision search and the subdivision checker.

Covers ``graph.minor_model.find``, ``graph.topological_minor.check``, and
``graph.topological_minor.find`` (#3754 follow-up / #1803). Owner-local: no
imports from jacobian.catalog.catalog, jacobian.dispatch, jacobian.cli, or
jacobian.mcp; examples parse via strict-JSON round trip plus the owning
request model and execute via the owner-local TOOLS run adapter.
"""

from __future__ import annotations

import itertools
import json
from itertools import pairwise

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.minors._models import (
    BranchSet,
    BranchVertex,
    EdgeWitness,
    MinorModelFindBudget,
    MinorModelFindRequest,
    MinorModelFindResult,
    SubdivisionPath,
    TopologicalMinorCheckRequest,
    TopologicalMinorCheckResult,
    TopologicalMinorFindBudget,
    TopologicalMinorFindRequest,
    TopologicalMinorFindResult,
)
from jacobian.math.graphs.minors._tools import TOOLS
from jacobian.math.graphs.minors.operations import (
    check_minor_model,
    check_topological_minor,
    find_minor_model,
    find_topological_minor,
    verify_topological_minor,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(vertices: list[str], edges: list[list[str]]) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph.model_validate({"vertices": vertices, "edges": edges})


def _branch(target: str, members: list[str]) -> BranchSet:
    return BranchSet(target=target, members=tuple(sorted(members)))


def _witness(targets: list[str], edge: list[str]) -> EdgeWitness:
    return EdgeWitness(targets=(targets[0], targets[1]), source_edge=(edge[0], edge[1]))


def _branch_vertex(target: str, source: str) -> BranchVertex:
    return BranchVertex(target=target, source=source)


def _path(targets: list[str], vertices: list[str]) -> SubdivisionPath:
    return SubdivisionPath(targets=(targets[0], targets[1]), vertices=tuple(vertices))


def _run(operation_id: str, request):  # type: ignore[no-untyped-def]
    for tool in TOOLS:
        if tool.operation_id == operation_id:
            return tool.run(request)
    raise AssertionError(f"operation {operation_id!r} not declared")


def _json_request(model, payload: dict):  # type: ignore[no-untyped-def]
    return model.model_validate(json.loads(json.dumps(payload)))


EDGE = _graph(["x", "y"], [["x", "y"]])
TRIANGLE = _graph(["x", "y", "z"], [["x", "y"], ["x", "z"], ["y", "z"]])
K4 = _graph(
    ["a", "b", "c", "d"],
    [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"], ["b", "d"], ["c", "d"]],
)
PATH4 = _graph(["a", "b", "c", "d"], [["a", "b"], ["b", "c"], ["c", "d"]])
PATH6 = _graph(
    ["v0", "v1", "v2", "v3", "v4", "v5"],
    [["v0", "v1"], ["v1", "v2"], ["v2", "v3"], ["v3", "v4"], ["v4", "v5"]],
)
WHEEL5 = _graph(
    ["hub", "c0", "c1", "c2", "c3", "c4"],
    [
        ["c0", "c1"],
        ["c1", "c2"],
        ["c2", "c3"],
        ["c3", "c4"],
        ["c0", "c4"],
        ["c0", "hub"],
        ["c1", "hub"],
        ["c2", "hub"],
        ["c3", "hub"],
        ["c4", "hub"],
    ],
)
K4_PLUS_PENDANT = _graph(
    ["a", "b", "c", "d", "e"],
    [
        ["a", "b"],
        ["a", "c"],
        ["a", "d"],
        ["b", "c"],
        ["b", "d"],
        ["c", "d"],
        ["a", "e"],
    ],
)
K2_3 = _graph(
    ["p0", "p1", "q0", "q1", "q2"],
    [
        ["p0", "q0"],
        ["p0", "q1"],
        ["p0", "q2"],
        ["p1", "q0"],
        ["p1", "q1"],
        ["p1", "q2"],
    ],
)
K3_3 = _graph(
    ["a0", "a1", "a2", "b0", "b1", "b2"],
    [
        ["a0", "b0"],
        ["a0", "b1"],
        ["a0", "b2"],
        ["a1", "b0"],
        ["a1", "b1"],
        ["a1", "b2"],
        ["a2", "b0"],
        ["a2", "b1"],
        ["a2", "b2"],
    ],
)


class TestMinorFindKnownAnswer:
    def test_triangle_minor_of_wheel(self) -> None:
        result = find_minor_model(WHEEL5, TRIANGLE, MinorModelFindBudget())
        assert result.status == "FOUND"
        assert result.termination_reason == "WITNESS_FOUND"
        assert len(result.branch_sets) == 3
        assert len(result.witnesses) == 3

    def test_triangle_minor_of_k4(self) -> None:
        result = find_minor_model(K4, TRIANGLE, MinorModelFindBudget())
        assert result.status == "FOUND"
        replay = check_minor_model(K4, TRIANGLE, result.branch_sets, result.witnesses)
        assert replay.status == "VALID_MINOR_MODEL"

    def test_single_edge_minor_of_path(self) -> None:
        result = find_minor_model(PATH4, EDGE, MinorModelFindBudget())
        assert result.status == "FOUND"
        assert result.candidates_enumerated >= 1

    def test_triangle_not_a_minor_of_a_tree(self) -> None:
        result = find_minor_model(PATH4, TRIANGLE, MinorModelFindBudget())
        assert result.status == "EXHAUSTED"
        # Complete assignment space: (3 branches + deletion) ** 4 vertices.
        assert result.candidates_enumerated == 4**4
        assert result.branch_sets == ()
        assert result.witnesses == ()

    def test_k2_3_not_a_minor_of_small_planar_graph(self) -> None:
        # K4 plus a pendant vertex is planar, yet K2,3 needs five branch
        # sets from five vertices while the degree-1 pendant vertex can
        # never serve a degree>=2 branch alone; the search proves absence
        # over the complete (5 + 1) ** 5 assignment space.
        result = find_minor_model(K4_PLUS_PENDANT, K2_3, MinorModelFindBudget())
        assert result.status == "EXHAUSTED"
        assert result.candidates_enumerated == 6**5

    def test_k4_minor_of_itself(self) -> None:
        k4_target = _graph(
            ["x", "y", "z", "w"],
            [["w", "x"], ["w", "y"], ["w", "z"], ["x", "y"], ["x", "z"], ["y", "z"]],
        )
        result = find_minor_model(K4, k4_target, MinorModelFindBudget())
        assert result.status == "FOUND"


class _UnionFind:
    """Tiny union-find for the independent contraction oracle."""

    def __init__(self, vertices: list[str]) -> None:
        self.parent = {vertex: vertex for vertex in vertices}

    def find(self, vertex: str) -> str:
        while self.parent[vertex] != vertex:
            self.parent[vertex] = self.parent[self.parent[vertex]]
            vertex = self.parent[vertex]
        return vertex

    def union(self, left: str, right: str) -> None:
        self.parent[self.find(left)] = self.find(right)


def _contraction_embeds(
    kept: tuple[str, ...],
    induced: list[tuple[str, str]],
    contracted: tuple[tuple[str, str], ...],
    target_vertices: list[str],
    target_edges: set[tuple[str, str]],
) -> bool:
    """Decide whether H embeds as a subgraph in one contraction."""

    components = _UnionFind(list(kept))
    for left, right in contracted:
        components.union(left, right)
    blocks: dict[str, int] = {}
    for vertex in kept:
        root = components.find(vertex)
        if root not in blocks:
            blocks[root] = len(blocks)
    if len(blocks) < len(target_vertices):
        return False
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(blocks))}
    for left, right in induced:
        first, second = blocks[components.find(left)], blocks[components.find(right)]
        if first != second:
            adjacency[first].add(second)
            adjacency[second].add(first)
    positions = {vertex: index for index, vertex in enumerate(target_vertices)}
    for image in itertools.permutations(range(len(blocks)), len(target_vertices)):
        if all(
            image[positions[right]] in adjacency[image[positions[left]]]
            for left, right in target_edges
        ):
            return True
    return False


def _oracle_has_minor(
    source: SimpleUndirectedGraph, target: SimpleUndirectedGraph
) -> bool:
    """Independent deletion/contraction/subgraph oracle for tiny graphs.

    Distinct algorithm from the kernel's branch-assignment backtracking: H
    is a minor of G iff H embeds as a subgraph in some contraction of some
    induced subgraph of G.
    """

    kept_vertices = list(source.vertices)
    kept_edges = [tuple(sorted(edge)) for edge in source.edges]
    target_vertices = list(target.vertices)
    target_edges = {tuple(sorted(edge)) for edge in target.edges}
    order = len(target_vertices)
    if order > len(kept_vertices) or len(target_edges) > len(kept_edges):
        return False
    for size in range(order, len(kept_vertices) + 1):
        for kept in itertools.combinations(kept_vertices, size):
            kept_set = set(kept)
            induced = [
                (left, right)
                for left, right in kept_edges
                if left in kept_set and right in kept_set
            ]
            for contract_size in range(len(induced) + 1):
                for contracted in itertools.combinations(induced, contract_size):
                    if _contraction_embeds(
                        kept, induced, contracted, target_vertices, target_edges
                    ):
                        return True
    return False


_ORACLE_SOURCES = {
    "edge": _graph(["a", "b"], [["a", "b"]]),
    "path3": _graph(["a", "b", "c"], [["a", "b"], ["b", "c"]]),
    "triangle": _graph(["a", "b", "c"], [["a", "b"], ["a", "c"], ["b", "c"]]),
    "path4": PATH4,
    "star4": _graph(["o", "a", "b", "c"], [["a", "o"], ["b", "o"], ["c", "o"]]),
    "diamond": _graph(
        ["a", "b", "c", "d"],
        [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"], ["c", "d"]],
    ),
    "k4": K4,
}
_ORACLE_TARGETS = {
    "edge": EDGE,
    "vee": _graph(["x", "y", "z"], [["x", "y"], ["x", "z"]]),
    "triangle": TRIANGLE,
}


class TestMinorFindOracle:
    @pytest.mark.parametrize("source_name", sorted(_ORACLE_SOURCES))
    @pytest.mark.parametrize("target_name", sorted(_ORACLE_TARGETS))
    def test_kernel_matches_deletion_contraction_oracle(
        self, source_name: str, target_name: str
    ) -> None:
        source = _ORACLE_SOURCES[source_name]
        target = _ORACLE_TARGETS[target_name]
        expected = _oracle_has_minor(source, target)
        result = find_minor_model(source, target, MinorModelFindBudget())
        assert (result.status == "FOUND") is expected
        if expected:
            replay = check_minor_model(
                source, target, result.branch_sets, result.witnesses
            )
            assert replay.status == "VALID_MINOR_MODEL"
        else:
            assert result.status == "EXHAUSTED"


class TestMinorFindBoundaries:
    def test_over_cap_source_order_rejected(self) -> None:
        big = _graph(
            [f"v{i}" for i in range(9)],
            [[f"v{i}", f"v{i + 1}"] for i in range(8)],
        )
        with pytest.raises(OperationResourceAdmissionError):
            find_minor_model(big, EDGE, MinorModelFindBudget())

    def test_over_cap_target_order_rejected(self) -> None:
        big_target = _graph(
            [f"t{i}" for i in range(6)],
            [[f"t{i}", f"t{i + 1}"] for i in range(5)],
        )
        with pytest.raises(OperationResourceAdmissionError):
            find_minor_model(PATH6, big_target, MinorModelFindBudget())

    def test_k3_3_target_exceeds_envelope(self) -> None:
        with pytest.raises(OperationResourceAdmissionError):
            find_minor_model(WHEEL5, K3_3, MinorModelFindBudget())

    def test_empty_target_rejected(self) -> None:
        empty = SimpleUndirectedGraph.model_validate({"vertices": [], "edges": []})
        with pytest.raises(OperationDomainValidationError):
            find_minor_model(PATH4, empty, MinorModelFindBudget())

    def test_candidate_budget_truncation_is_unknown(self) -> None:
        # The wheel/triangle search needs more than one candidate, so a
        # single-candidate budget truncates instead of concluding absence.
        result = find_minor_model(
            WHEEL5, TRIANGLE, MinorModelFindBudget(max_candidates=1)
        )
        assert result.status == "UNKNOWN"
        assert result.termination_reason == "CANDIDATE_BUDGET_EXCEEDED"
        assert result.candidates_enumerated == 1
        assert result.branch_sets == ()

    def test_truncated_tree_search_never_exhausted(self) -> None:
        result = find_minor_model(
            PATH4, TRIANGLE, MinorModelFindBudget(max_candidates=10)
        )
        assert result.status == "UNKNOWN"
        assert result.candidates_enumerated == 10

    def test_presolve_vertex_overflow_is_exhausted(self) -> None:
        result = find_minor_model(EDGE, TRIANGLE, MinorModelFindBudget())
        assert result.status == "EXHAUSTED"
        assert result.candidates_enumerated == 0

    def test_at_envelope_boundary_accepted(self) -> None:
        source = _graph(
            [f"v{i}" for i in range(8)],
            [[f"v{i}", f"v{i + 1}"] for i in range(7)],
        )
        target = _graph(
            [f"t{i}" for i in range(5)],
            [[f"t{i}", f"t{i + 1}"] for i in range(4)],
        )
        result = find_minor_model(
            source, target, MinorModelFindBudget(max_candidates=1)
        )
        assert result.status == "UNKNOWN"


class TestMinorFindConsumerComposition:
    def test_found_model_feeds_check_kernel_unchanged(self) -> None:
        result = find_minor_model(WHEEL5, TRIANGLE, MinorModelFindBudget())
        assert result.status == "FOUND"
        replay = check_minor_model(
            result.source, result.target, result.branch_sets, result.witnesses
        )
        assert replay.status == "VALID_MINOR_MODEL"
        assert replay.branch_sets == result.branch_sets
        assert replay.witnesses == result.witnesses

    def test_serialization_round_trip(self) -> None:
        for outcome in (
            find_minor_model(WHEEL5, TRIANGLE, MinorModelFindBudget()),
            find_minor_model(PATH4, TRIANGLE, MinorModelFindBudget()),
            find_minor_model(WHEEL5, TRIANGLE, MinorModelFindBudget(max_candidates=1)),
        ):
            assert (
                MinorModelFindResult.model_validate_json(outcome.model_dump_json())
                == outcome
            )

    def test_request_strict_json_round_trip(self) -> None:
        payload = {
            "source": {
                "vertices": ["a", "b", "c"],
                "edges": [["a", "b"], ["a", "c"], ["b", "c"]],
            },
            "target": {"vertices": ["x", "y"], "edges": [["x", "y"]]},
            "resource_budget": {"max_candidates": 1000},
        }
        request = _json_request(MinorModelFindRequest, payload)
        assert request.resource_budget.max_candidates == 1000
        assert _run("graph.minor_model.find", request).status == "FOUND"


class TestTopologicalCheckKnownAnswer:
    def test_subdivided_edge_in_path(self) -> None:
        result = check_topological_minor(
            PATH4,
            EDGE,
            (_branch_vertex("x", "a"), _branch_vertex("y", "d")),
            (_path(["x", "y"], ["a", "b", "c", "d"]),),
        )
        assert result.status == "VALID_SUBDIVISION"
        assert result.obstruction_code is None
        assert result.used_source_vertices == ("a", "b", "c", "d")
        assert result.deleted_source_vertices == ()
        assert result.used_source_edges == (("a", "b"), ("b", "c"), ("c", "d"))

    def test_triangle_subdivision(self) -> None:
        subdivided = _graph(
            ["a", "b", "c", "ab", "bc", "ca"],
            [
                ["a", "ab"],
                ["a", "ca"],
                ["ab", "b"],
                ["b", "bc"],
                ["bc", "c"],
                ["c", "ca"],
            ],
        )
        result = check_topological_minor(
            subdivided,
            TRIANGLE,
            (
                _branch_vertex("x", "a"),
                _branch_vertex("y", "b"),
                _branch_vertex("z", "c"),
            ),
            (
                _path(["x", "y"], ["a", "ab", "b"]),
                _path(["x", "z"], ["a", "ca", "c"]),
                _path(["y", "z"], ["b", "bc", "c"]),
            ),
        )
        assert result.status == "VALID_SUBDIVISION"

    def test_k4_subdivision_of_itself(self) -> None:
        k4_target = _graph(
            ["x", "y", "z", "w"],
            [["w", "x"], ["w", "y"], ["w", "z"], ["x", "y"], ["x", "z"], ["y", "z"]],
        )
        result = check_topological_minor(
            K4,
            k4_target,
            (
                _branch_vertex("x", "a"),
                _branch_vertex("y", "b"),
                _branch_vertex("z", "c"),
                _branch_vertex("w", "d"),
            ),
            (
                _path(["x", "y"], ["a", "b"]),
                _path(["x", "z"], ["a", "c"]),
                _path(["w", "x"], ["d", "a"]),
                _path(["y", "z"], ["b", "c"]),
                _path(["w", "y"], ["d", "b"]),
                _path(["w", "z"], ["d", "c"]),
            ),
        )
        assert result.status == "VALID_SUBDIVISION"


def _oracle_subdivision(
    source: SimpleUndirectedGraph,
    target: SimpleUndirectedGraph,
    branch_vertices: tuple[BranchVertex, ...],
    paths: tuple[SubdivisionPath, ...],
) -> bool:
    """Independent plain-set replay of the subdivision predicate."""

    bound = {row.target: row.source for row in branch_vertices}
    if set(bound) != set(target.vertices) or len(bound) != len(branch_vertices):
        return False
    if len(set(bound.values())) != len(bound):
        return False
    if {tuple(sorted(path.targets)) for path in paths} != {
        tuple(sorted(edge)) for edge in target.edges
    }:
        return False
    source_edges = {tuple(sorted(edge)) for edge in source.edges}
    seen_internals: set[str] = set()
    for edge in target.edges:
        key = tuple(sorted(edge))
        candidates = [path for path in paths if tuple(sorted(path.targets)) == key]
        if len(candidates) != 1:
            return False
        trail = list(candidates[0].vertices)
        if len(set(trail)) != len(trail):
            return False
        if (
            trail[0] != bound[candidates[0].targets[0]]
            or trail[-1] != bound[candidates[0].targets[1]]
        ):
            return False
        if any(tuple(sorted(step)) not in source_edges for step in pairwise(trail)):
            return False
        internals = set(trail[1:-1])
        if internals & set(bound.values()) or internals & seen_internals:
            return False
        seen_internals |= internals
    return True


class TestTopologicalCheckObstructions:
    def test_duplicate_branch_vertex(self) -> None:
        result = check_topological_minor(
            PATH4,
            EDGE,
            (_branch_vertex("x", "a"), _branch_vertex("y", "a")),
            (_path(["x", "y"], ["a", "b"]),),
        )
        assert result.status == "INVALID_SUBDIVISION"
        assert result.obstruction_code == "DUPLICATE_BRANCH_VERTEX"

    def test_non_simple_path(self) -> None:
        result = check_topological_minor(
            PATH4,
            EDGE,
            (_branch_vertex("x", "a"), _branch_vertex("y", "c")),
            (_path(["x", "y"], ["a", "b", "c", "b", "c"]),),
        )
        assert result.status == "INVALID_SUBDIVISION"
        assert result.obstruction_code == "NON_SIMPLE_PATH"

    def test_repeated_vertex_path_reports_non_simple(self) -> None:
        cyclic = _graph(["a", "b", "c"], [["a", "b"], ["a", "c"], ["b", "c"]])
        result = check_topological_minor(
            cyclic,
            EDGE,
            (_branch_vertex("x", "a"), _branch_vertex("y", "b")),
            (_path(["x", "y"], ["a", "c", "b", "c", "b"]),),
        )
        assert result.status == "INVALID_SUBDIVISION"
        assert result.obstruction_code == "NON_SIMPLE_PATH"

    def test_false_witness_edge_absent(self) -> None:
        result = check_topological_minor(
            PATH4,
            EDGE,
            (_branch_vertex("x", "a"), _branch_vertex("y", "d")),
            (_path(["x", "y"], ["a", "b", "d"]),),
        )
        assert result.status == "INVALID_SUBDIVISION"
        assert result.obstruction_code == "PATH_EDGE_ABSENT"

    def test_endpoint_mismatch(self) -> None:
        result = check_topological_minor(
            PATH4,
            EDGE,
            (_branch_vertex("x", "a"), _branch_vertex("y", "d")),
            (_path(["x", "y"], ["d", "c", "b", "a"]),),
        )
        assert result.status == "INVALID_SUBDIVISION"
        assert result.obstruction_code == "PATH_ENDPOINT_MISMATCH"

    def test_path_through_other_branch_vertex(self) -> None:
        result = check_topological_minor(
            PATH4,
            TRIANGLE,
            (
                _branch_vertex("x", "a"),
                _branch_vertex("y", "b"),
                _branch_vertex("z", "d"),
            ),
            (
                _path(["x", "y"], ["a", "b"]),
                _path(["x", "z"], ["a", "b", "c", "d"]),
                _path(["y", "z"], ["b", "c", "d"]),
            ),
        )
        assert result.status == "INVALID_SUBDIVISION"
        assert result.obstruction_code == "PATH_INTERNAL_INTERSECTION"

    def test_shared_internal_vertex(self) -> None:
        fan = _graph(
            ["a", "b", "c", "d", "m"],
            [["a", "m"], ["b", "m"], ["c", "m"], ["c", "d"]],
        )
        result = check_topological_minor(
            fan,
            TRIANGLE,
            (
                _branch_vertex("x", "a"),
                _branch_vertex("y", "b"),
                _branch_vertex("z", "d"),
            ),
            (
                _path(["x", "y"], ["a", "m", "b"]),
                _path(["x", "z"], ["a", "m", "c", "d"]),
                _path(["y", "z"], ["b", "m", "c", "d"]),
            ),
        )
        assert result.status == "INVALID_SUBDIVISION"
        assert result.obstruction_code == "PATH_INTERNAL_INTERSECTION"

    def test_missing_path_witness(self) -> None:
        result = check_topological_minor(
            PATH4,
            EDGE,
            (_branch_vertex("x", "a"), _branch_vertex("y", "b")),
            (),
        )
        assert result.status == "INVALID_SUBDIVISION"
        assert result.obstruction_code == "MISSING_PATH_WITNESS"

    def test_undeclared_path_vertex(self) -> None:
        result = check_topological_minor(
            PATH4,
            EDGE,
            (_branch_vertex("x", "a"), _branch_vertex("y", "b")),
            (_path(["x", "y"], ["a", "zzz"]),),
        )
        assert result.status == "INVALID_SUBDIVISION"
        assert result.obstruction_code == "UNDECLARED_PATH_VERTEX"

    def test_oracle_agrees_with_checker(self) -> None:
        valid = check_topological_minor(
            PATH4,
            EDGE,
            (_branch_vertex("x", "a"), _branch_vertex("y", "d")),
            (_path(["x", "y"], ["a", "b", "c", "d"]),),
        )
        assert valid.status == "VALID_SUBDIVISION"
        assert _oracle_subdivision(PATH4, EDGE, valid.branch_vertices, valid.paths)
        assert verify_topological_minor(valid)


class TestTopologicalFind:
    def test_edge_subdivision_in_path(self) -> None:
        result = find_topological_minor(PATH4, EDGE, TopologicalMinorFindBudget())
        assert result.status == "FOUND"
        replay = check_topological_minor(
            PATH4, EDGE, result.branch_vertices, result.paths
        )
        assert replay.status == "VALID_SUBDIVISION"

    def test_k4_subdivision_of_itself(self) -> None:
        k4_target = _graph(
            ["x", "y", "z", "w"],
            [["w", "x"], ["w", "y"], ["w", "z"], ["x", "y"], ["x", "z"], ["y", "z"]],
        )
        result = find_topological_minor(K4, k4_target, TopologicalMinorFindBudget())
        assert result.status == "FOUND"
        assert len(result.paths) == 6

    def test_triangle_subdivision_in_wheel(self) -> None:
        result = find_topological_minor(WHEEL5, TRIANGLE, TopologicalMinorFindBudget())
        assert result.status == "FOUND"

    def test_triangle_subdivision_absent_in_tree(self) -> None:
        result = find_topological_minor(PATH4, TRIANGLE, TopologicalMinorFindBudget())
        assert result.status == "EXHAUSTED"
        assert result.branch_vertices == ()
        assert result.paths == ()

    def test_probe_budget_truncation_is_unknown(self) -> None:
        result = find_topological_minor(
            WHEEL5, TRIANGLE, TopologicalMinorFindBudget(max_candidates=1)
        )
        assert result.status == "UNKNOWN"
        assert result.termination_reason == "CANDIDATE_BUDGET_EXCEEDED"

    def test_over_cap_source_order_rejected(self) -> None:
        big = _graph(
            [f"v{i}" for i in range(9)],
            [[f"v{i}", f"v{i + 1}"] for i in range(8)],
        )
        with pytest.raises(OperationResourceAdmissionError):
            find_topological_minor(big, EDGE, TopologicalMinorFindBudget())

    def test_empty_target_rejected(self) -> None:
        empty = SimpleUndirectedGraph.model_validate({"vertices": [], "edges": []})
        with pytest.raises(OperationDomainValidationError):
            find_topological_minor(PATH4, empty, TopologicalMinorFindBudget())

    def test_found_routing_feeds_check_kernel_unchanged(self) -> None:
        result = find_topological_minor(K4, TRIANGLE, TopologicalMinorFindBudget())
        assert result.status == "FOUND"
        replay = check_topological_minor(
            result.source, result.target, result.branch_vertices, result.paths
        )
        assert replay.status == "VALID_SUBDIVISION"
        assert replay.branch_vertices == result.branch_vertices
        assert replay.paths == result.paths
        assert _oracle_subdivision(
            result.source, result.target, result.branch_vertices, result.paths
        )

    def test_serialization_round_trip(self) -> None:
        for outcome in (
            find_topological_minor(K4, TRIANGLE, TopologicalMinorFindBudget()),
            find_topological_minor(PATH4, TRIANGLE, TopologicalMinorFindBudget()),
            find_topological_minor(
                K4, TRIANGLE, TopologicalMinorFindBudget(max_candidates=1)
            ),
        ):
            assert (
                TopologicalMinorFindResult.model_validate_json(
                    outcome.model_dump_json()
                )
                == outcome
            )


class TestNativeCatalogParity:
    @pytest.mark.parametrize(
        "operation_id",
        [
            "graph.minor_model.check",
            "graph.minor_model.find",
            "graph.topological_minor.check",
            "graph.topological_minor.find",
        ],
    )
    def test_operation_is_published(self, operation_id: str) -> None:
        assert operation_id in {tool.operation_id for tool in TOOLS}

    def test_minor_find_parity(self) -> None:
        request = _json_request(
            MinorModelFindRequest,
            {
                "source": {
                    "vertices": ["a", "b", "c", "d"],
                    "edges": [["a", "b"], ["b", "c"], ["c", "d"]],
                },
                "target": {"vertices": ["x", "y"], "edges": [["x", "y"]]},
                "resource_budget": {"max_candidates": 1000},
            },
        )
        assert _run("graph.minor_model.find", request) == find_minor_model(
            request.source, request.target, request.resource_budget
        )

    def test_topo_check_parity(self) -> None:
        request = _json_request(
            TopologicalMinorCheckRequest,
            {
                "source": {
                    "vertices": ["a", "b", "c", "d"],
                    "edges": [["a", "b"], ["b", "c"], ["c", "d"]],
                },
                "target": {"vertices": ["x", "y"], "edges": [["x", "y"]]},
                "branch_vertices": [
                    {"target": "x", "source": "a"},
                    {"target": "y", "source": "d"},
                ],
                "paths": [{"targets": ["x", "y"], "vertices": ["a", "b", "c", "d"]}],
            },
        )
        assert _run("graph.topological_minor.check", request) == (
            check_topological_minor(
                request.source,
                request.target,
                request.branch_vertices,
                request.paths,
            )
        )

    def test_topo_find_parity(self) -> None:
        request = _json_request(
            TopologicalMinorFindRequest,
            {
                "source": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["a", "c"], ["b", "c"]],
                },
                "target": {"vertices": ["x", "y"], "edges": [["x", "y"]]},
                "resource_budget": {"max_candidates": 1000},
            },
        )
        assert _run("graph.topological_minor.find", request) == (
            find_topological_minor(
                request.source, request.target, request.resource_budget
            )
        )

    def test_topo_check_result_round_trip(self) -> None:
        verdict = check_topological_minor(
            PATH4,
            EDGE,
            (_branch_vertex("x", "a"), _branch_vertex("y", "d")),
            (_path(["x", "y"], ["a", "b", "c", "d"]),),
        )
        assert (
            TopologicalMinorCheckResult.model_validate_json(verdict.model_dump_json())
            == verdict
        )


class TestToolExamplesExecute:
    @pytest.mark.parametrize(
        ("operation_id", "request_model"),
        [
            ("graph.minor_model.find", MinorModelFindRequest),
            ("graph.topological_minor.check", TopologicalMinorCheckRequest),
            ("graph.topological_minor.find", TopologicalMinorFindRequest),
        ],
    )
    def test_declared_examples_parse_and_run(
        self,
        operation_id: str,
        request_model,  # type: ignore[no-untyped-def]
    ) -> None:
        tools = [tool for tool in TOOLS if tool.operation_id == operation_id]
        assert len(tools) == 1
        for example in tools[0].examples:
            request = _json_request(request_model, example.input)
            outcome = tools[0].run(request)
            assert outcome.status in ("FOUND", "VALID_SUBDIVISION")
