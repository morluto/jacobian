"""Defining-invariant and boundary tests for maximum edge matchings."""

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
    MaximumEdgeMatchingRequest,
    MaximumEdgeMatchingResult,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._operations import (
    compute_maximum_edge_matching,
)

HYPERGRAPH = {
    "vertices": ["a", "b", "c", "d"],
    "edges": [
        ["e1", ["a", "b", "c"]],
        ["e2", ["b", "c", "d"]],
        ["e3", ["a", "d"]],
    ],
}


def _matching(source: object) -> MaximumEdgeMatchingResult:
    return compute_maximum_edge_matching(
        MaximumEdgeMatchingRequest(hypergraph=FiniteHypergraph.model_validate(source))
    )


class TestMaximumEdgeMatching:
    def test_known_maximum_all_intersect(self) -> None:
        # All three edges pairwise intersect -> max matching size 1.
        result = _matching(HYPERGRAPH)
        assert result.count == 1
        assert result.matching == ("e1",)

    def test_empty_edge_family_empty_matching(self) -> None:
        result = _matching({"vertices": ["a", "b"], "edges": []})
        assert result.matching == ()
        assert result.count == 0

    def test_disjoint_edges_all_matched(self) -> None:
        result = _matching(
            {
                "vertices": ["a", "b", "c", "d"],
                "edges": [
                    ["e1", ["a", "b"]],
                    ["e2", ["c", "d"]],
                ],
            }
        )
        assert result.count == 2
        assert result.matching == ("e1", "e2")

    def test_partial_overlap(self) -> None:
        # e1∩e2={b}, e1∩e3={a}, e2∩e3={c} -> all intersect -> max 1
        result = _matching(
            {
                "vertices": ["a", "b", "c", "d"],
                "edges": [
                    ["e1", ["a", "b"]],
                    ["e2", ["b", "c"]],
                    ["e3", ["a", "c"]],
                ],
            }
        )
        assert result.count == 1

    def test_two_disjoint_one_overlapping(self) -> None:
        # e1={a,b}, e2={c,d}, e3={b,c}
        # e1∩e2=∅ (disjoint), e1∩e3={b}, e2∩e3={c}
        # max matching: {e1, e2}
        result = _matching(
            {
                "vertices": ["a", "b", "c", "d"],
                "edges": [
                    ["e1", ["a", "b"]],
                    ["e2", ["c", "d"]],
                    ["e3", ["b", "c"]],
                ],
            }
        )
        assert result.count == 2
        assert result.matching == ("e1", "e2")

    def test_matching_in_declared_edge_order(self) -> None:
        result = _matching(
            {
                "vertices": ["a", "b", "c", "d"],
                "edges": [
                    ["e1", ["a", "b"]],
                    ["e2", ["c", "d"]],
                ],
            }
        )
        assert result.matching == ("e1", "e2")

    def test_empty_edge_is_matchable(self) -> None:
        result = _matching({"vertices": ["a"], "edges": [["e", []]]})
        assert result.matching == ("e",)
        assert result.count == 1

    def test_all_empty_edges_above_search_cap_are_admitted(self) -> None:
        result = _matching(
            {
                "vertices": ["a"],
                "edges": [[f"e{i}", []] for i in range(21)],
            }
        )
        assert result.matching == tuple(f"e{i}" for i in range(21))
        assert result.count == 21

    def test_empty_edge_prefix_is_presolved_before_search_cap(self) -> None:
        result = _matching(
            {
                "vertices": ["a", "b"],
                "edges": [
                    *[[f"empty{i}", []] for i in range(21)],
                    ["nonempty", ["a"]],
                ],
            }
        )
        assert result.matching == tuple([f"empty{i}" for i in range(21)] + ["nonempty"])
        assert result.count == 22

    def test_edge_bound_exceeded(self) -> None:
        edges = [[f"e{i}", [f"v{i}"]] for i in range(21)]
        vertices = [f"v{i}" for i in range(21)]
        request = MaximumEdgeMatchingRequest(
            hypergraph=FiniteHypergraph.model_validate(
                {"vertices": vertices, "edges": edges}
            )
        )
        with pytest.raises(ValueError, match="search exceeds"):
            compute_maximum_edge_matching(request)

    def test_rejects_wrong_count(self) -> None:
        with pytest.raises(ValidationError):
            MaximumEdgeMatchingResult.model_validate(
                {
                    "hypergraph": HYPERGRAPH,
                    "matching": ["e1"],
                    "count": 2,
                }
            )

    def test_rejects_unknown_edge_id(self) -> None:
        with pytest.raises(ValidationError):
            MaximumEdgeMatchingResult.model_validate(
                {
                    "hypergraph": HYPERGRAPH,
                    "matching": ["eX"],
                    "count": 1,
                }
            )
