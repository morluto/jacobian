"""Tests for the deterministic H-minor-model checker (#1803)."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.minors._models import (
    BranchSet,
    EdgeWitness,
    MinorModelCheckRequest,
    MinorModelCheckResult,
)
from jacobian.math.graphs.minors._tools import _check
from jacobian.math.graphs.minors.operations import check_minor_model
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(vertices: list[str], edges: list[list[str]]) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph.model_validate({"vertices": vertices, "edges": edges})


def _branch(target: str, members: list[str]) -> BranchSet:
    return BranchSet(target=target, members=tuple(sorted(members)))


def _witness(targets: list[str], edge: list[str]) -> EdgeWitness:
    return EdgeWitness(targets=(targets[0], targets[1]), source_edge=(edge[0], edge[1]))


TRIANGLE = _graph(["a", "b", "c"], [["a", "b"], ["a", "c"], ["b", "c"]])
EDGE = _graph(["x", "y"], [["x", "y"]])
PATH4 = _graph(["a", "b", "c", "d"], [["a", "b"], ["b", "c"], ["c", "d"]])


class TestKnownAnswer:
    def test_singleton_subgraph_model(self) -> None:
        result = check_minor_model(
            TRIANGLE,
            EDGE,
            (_branch("x", ["a"]), _branch("y", ["b"])),
            (_witness(["x", "y"], ["a", "b"]),),
        )
        assert isinstance(result, MinorModelCheckResult)
        assert result.status == "VALID_MINOR_MODEL"
        assert result.obstruction_code is None
        assert {ledger.target for ledger in result.connectivity} == {"x", "y"}
        assert result.used_source_vertices == ("a", "b")
        assert result.deleted_source_vertices == ("c",)

    def test_nontrivial_connected_branch_sets(self) -> None:
        result = check_minor_model(
            PATH4,
            EDGE,
            (_branch("x", ["a", "b"]), _branch("y", ["c", "d"])),
            (_witness(["x", "y"], ["b", "c"]),),
        )
        assert result.status == "VALID_MINOR_MODEL"
        assert result.used_source_vertices == ("a", "b", "c", "d")
        assert result.deleted_source_vertices == ()

    def test_extra_source_edges_stay_legal(self) -> None:
        # The triangle's third edge is extra structure, not an induced-minor violation.
        result = check_minor_model(
            TRIANGLE,
            EDGE,
            (_branch("x", ["a"]), _branch("y", ["b"])),
            (_witness(["x", "y"], ["a", "b"]),),
        )
        assert result.status == "VALID_MINOR_MODEL"
        assert result.deleted_source_edges == (("a", "c"), ("b", "c"))


class TestObstructions:
    def test_overlapping_branch_sets(self) -> None:
        result = check_minor_model(
            TRIANGLE,
            EDGE,
            (_branch("x", ["a", "b"]), _branch("y", ["b", "c"])),
            (_witness(["x", "y"], ["b", "c"]),),
        )
        assert result.status == "INVALID_MINOR_MODEL"
        assert result.obstruction_code == "OVERLAPPING_BRANCH_SETS"

    def test_disconnected_branch_set(self) -> None:
        result = check_minor_model(
            PATH4,
            EDGE,
            (_branch("x", ["a", "c"]), _branch("y", ["d"])),
            (_witness(["x", "y"], ["c", "d"]),),
        )
        assert result.status == "INVALID_MINOR_MODEL"
        assert result.obstruction_code == "DISCONNECTED_BRANCH_SET"
        assert result.obstruction_detail is not None

    def test_missing_witness(self) -> None:
        result = check_minor_model(
            TRIANGLE, EDGE, (_branch("x", ["a"]), _branch("y", ["b"])), ()
        )
        assert result.status == "INVALID_MINOR_MODEL"
        assert result.obstruction_code == "MISSING_WITNESS"

    def test_witness_wrong_branch_sets(self) -> None:
        result = check_minor_model(
            TRIANGLE,
            EDGE,
            (_branch("x", ["a"]), _branch("y", ["b"])),
            (_witness(["x", "y"], ["a", "c"]),),
        )
        assert result.status == "INVALID_MINOR_MODEL"
        assert result.obstruction_code == "WITNESS_ENDPOINT_MISMATCH"

    def test_witness_edge_absent(self) -> None:
        line = _graph(["a", "b", "c"], [["a", "b"], ["b", "c"]])
        result = check_minor_model(
            line,
            EDGE,
            (_branch("x", ["a"]), _branch("y", ["c"])),
            (_witness(["x", "y"], ["a", "c"]),),
        )
        assert result.status == "INVALID_MINOR_MODEL"
        assert result.obstruction_code == "WITNESS_EDGE_ABSENT"

    def test_missing_branch_set(self) -> None:
        result = check_minor_model(
            TRIANGLE, EDGE, (_branch("x", ["a"]),), (_witness(["x", "y"], ["a", "b"]),)
        )
        assert result.status == "INVALID_MINOR_MODEL"
        assert result.obstruction_code == "MISSING_BRANCH_SET"

    def test_undeclared_source_vertex(self) -> None:
        result = check_minor_model(
            TRIANGLE,
            EDGE,
            (_branch("x", ["a"]), _branch("y", ["zzz"])),
            (_witness(["x", "y"], ["a", "b"]),),
        )
        assert result.status == "INVALID_MINOR_MODEL"
        assert result.obstruction_code == "UNDECLARED_SOURCE_VERTEX"


class TestAdversarial:
    def test_empty_target_rejected(self) -> None:
        empty = SimpleUndirectedGraph.model_validate({"vertices": [], "edges": []})
        with pytest.raises(OperationDomainValidationError):
            check_minor_model(TRIANGLE, empty, (), ())

    def test_duplicate_witness_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            check_minor_model(
                TRIANGLE,
                EDGE,
                (_branch("x", ["a"]), _branch("y", ["b"])),
                (
                    _witness(["x", "y"], ["a", "b"]),
                    _witness(["x", "y"], ["a", "c"]),
                ),
            )


class TestDefiningInvariant:
    def test_valid_model_replays_independently(self) -> None:
        branch_sets = (_branch("x", ["a", "b"]), _branch("y", ["c", "d"]))
        witnesses = (_witness(["x", "y"], ["b", "c"]),)
        result = check_minor_model(PATH4, EDGE, branch_sets, witnesses)
        assert result.status == "VALID_MINOR_MODEL"
        adjacency = {v: set() for v in PATH4.vertices}
        for left, right in PATH4.edges:
            adjacency[left].add(right)
            adjacency[right].add(left)
        owner = {
            member: branch.target for branch in branch_sets for member in branch.members
        }
        assert len(set(owner)) == len(owner)  # disjoint
        for branch in branch_sets:  # connected
            members = set(branch.members)
            seen, queue = set(), [next(iter(members))]
            seen.add(queue[0])
            while queue:
                for neighbor in adjacency[queue.pop()] & members - seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
            assert seen == members
        for witness in witnesses:  # witness crosses the right branch sets
            left, right = witness.source_edge
            assert tuple(sorted((left, right))) in {
                tuple(sorted(edge)) for edge in PATH4.edges
            }
            assert {owner[left], owner[right]} == set(witness.targets)


class TestNativeCatalogParity:
    def test_native_matches_catalog(self) -> None:
        request = MinorModelCheckRequest(
            source=TRIANGLE,
            target=EDGE,
            branch_sets=(_branch("x", ["a"]), _branch("y", ["b"])),
            witnesses=(_witness(["x", "y"], ["a", "b"]),),
        )
        assert _check(request) == check_minor_model(
            request.source, request.target, request.branch_sets, request.witnesses
        )
