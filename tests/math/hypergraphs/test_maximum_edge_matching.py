"""Defining-invariant and boundary tests for maximum edge matchings."""

import pytest
from pydantic import ValidationError

from jacobian.math.hypergraphs._models import (
    MaximumEdgeMatchingRequest,
    MaximumEdgeMatchingResult,
    FiniteHypergraph,
)
from jacobian.math.hypergraphs._operations import (
    compute_maximum_edge_matching,
    verify_maximum_edge_matching_result,
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
    return compute_maximum_edge_matching(MaximumEdgeMatchingRequest(hypergraph=source))


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

    def test_empty_edge_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MaximumEdgeMatchingRequest(
                hypergraph={"vertices": ["a"], "edges": [["e", []]]}
            )

    def test_edge_bound_exceeded(self) -> None:
        edges = [[f"e{i}", [f"v{i}"]] for i in range(21)]
        vertices = [f"v{i}" for i in range(21)]
        with pytest.raises(ValidationError):
            MaximumEdgeMatchingRequest(hypergraph={"vertices": vertices, "edges": edges})

    def test_verify_round_trip(self) -> None:
        result = _matching(HYPERGRAPH)
        assert verify_maximum_edge_matching_result(result)

    def test_rejects_intersecting_matching(self) -> None:
        with pytest.raises(ValidationError):
            MaximumEdgeMatchingResult.model_validate(
                {
                    "hypergraph": HYPERGRAPH,
                    "matching": ["e1", "e2"],
                    "count": 2,
                }
            )

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
