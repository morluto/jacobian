"""Tests for clique candidate hypergraphs over graph-edge resources."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.hypergraphs.operations import (
    maximum_edge_matching,
)
from jacobian.math.graphs.clique_candidate_hypergraph._models import (
    AllCliqueCandidatesRequest,
    CliqueCandidateHypergraphResult,
)
from jacobian.math.graphs.clique_candidate_hypergraph.operations import (
    construct_all_clique_candidate_hypergraph,
    convert_candidate_cliques,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

BOWTIE = {
    "vertices": ["a", "b", "c", "d", "e"],
    "edges": [
        ["a", "b"],
        ["a", "c"],
        ["a", "d"],
        ["a", "e"],
        ["b", "c"],
        ["d", "e"],
    ],
}


def _graph(source: object) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph.model_validate(source)


class TestCompleteConstructor:
    def test_bowtie_has_eight_candidates(self) -> None:
        result = construct_all_clique_candidate_hypergraph(_graph(BOWTIE))
        assert result.candidate_count == 8
        members = [set(entry.members) for entry in result.candidate_map]
        assert {frozenset({"a", "b", "c"}), frozenset({"a", "d", "e"})} <= {
            frozenset(entry) for entry in members
        }
        assert sum(1 for entry in members if len(entry) == 2) == 6

    def test_resource_map_covers_source_edges(self) -> None:
        result = construct_all_clique_candidate_hypergraph(_graph(BOWTIE))
        assert {entry.endpoints for entry in result.resource_map} == {
            ("a", "b"),
            ("a", "c"),
            ("a", "d"),
            ("a", "e"),
            ("b", "c"),
            ("d", "e"),
        }

    def test_hyperedges_hold_exactly_internal_resources(self) -> None:
        graph = _graph(BOWTIE)
        result = construct_all_clique_candidate_hypergraph(graph)
        resource_of = {entry.endpoints: entry.resource for entry in result.resource_map}
        by_candidate = {
            entry.candidate: entry.members for entry in result.candidate_map
        }
        for edge_id, members in result.hypergraph.edges:
            expected = {
                resource_of[tuple(sorted((left, right)))]
                for left in by_candidate[edge_id]
                for right in by_candidate[edge_id]
                if left < right
            }
            assert set(members) == expected

    def test_result_reparses(self) -> None:
        result = construct_all_clique_candidate_hypergraph(_graph(BOWTIE))
        assert (
            CliqueCandidateHypergraphResult.model_validate(
                result.model_dump(mode="json")
            )
            == result
        )

    def test_request_path_matches_native(self) -> None:
        from jacobian.math.graphs.clique_candidate_hypergraph._tools import (
            _compute_all_clique_candidates,
        )

        request = AllCliqueCandidatesRequest(graph=_graph(BOWTIE))
        assert _compute_all_clique_candidates(request).candidate_count == 8


class TestConversion:
    def test_bowtie_triangles_are_edge_disjoint(self) -> None:
        graph = _graph(BOWTIE)
        result = convert_candidate_cliques(graph, (("a", "b", "c"), ("a", "d", "e")))
        first, second = (set(members) for _, members in result.hypergraph.edges)
        assert first.isdisjoint(second)
        matching = maximum_edge_matching(result.hypergraph)
        assert matching.count == 2

    def test_duplicate_member_sets_keep_identities(self) -> None:
        graph = _graph(BOWTIE)
        result = convert_candidate_cliques(graph, (("a", "b", "c"), ("a", "b", "c")))
        assert result.candidate_count == 2
        assert [edge for edge, _ in result.hypergraph.edges] == ["q0", "q1"]

    def test_non_clique_rejected_with_index(self) -> None:
        graph = _graph(BOWTIE)
        with pytest.raises(OperationDomainValidationError) as error:
            convert_candidate_cliques(graph, (("a", "b", "d"),))
        assert error.value.errors()[0]["type"] == (
            "graph.clique_candidate.candidate_not_complete"
        )

    def test_small_and_foreign_candidates_rejected(self) -> None:
        graph = _graph(BOWTIE)
        with pytest.raises(OperationDomainValidationError):
            convert_candidate_cliques(graph, (("a",),))
        with pytest.raises(OperationDomainValidationError):
            convert_candidate_cliques(graph, (("a", "z"),))
