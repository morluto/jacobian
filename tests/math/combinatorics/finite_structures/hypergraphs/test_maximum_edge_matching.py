"""Defining-invariant and boundary tests for maximum edge matchings."""

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
    MaximumEdgeMatchingRequest,
    MaximumEdgeMatchingResult,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs.operations import (
    maximum_edge_matching,
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
    return maximum_edge_matching(FiniteHypergraph.model_validate(source))


def _undecomposed_maximum_matching(
    hypergraph: FiniteHypergraph,
) -> tuple[tuple[str, ...], int]:
    """Reference global decreasing-subset search without component presolve."""

    from itertools import combinations

    edges = tuple(
        (edge_id, tuple(sorted(members))) for edge_id, members in hypergraph.edges
    )
    edge_ids = tuple(edge_id for edge_id, _ in edges)
    empty_edge_ids = tuple(edge_id for edge_id, members in edges if not members)
    search_edges = tuple((edge_id, members) for edge_id, members in edges if members)
    search_edge_ids = tuple(edge_id for edge_id, _ in search_edges)
    edge_sets = tuple(frozenset(members) for _, members in search_edges)
    if not search_edges:
        return edge_ids, len(edge_ids)
    for size in range(len(search_edges), 0, -1):
        for combo in combinations(range(len(search_edges)), size):
            picked = [edge_sets[i] for i in combo]
            disjoint = True
            for i in range(len(picked)):
                for j in range(i + 1, len(picked)):
                    if picked[i] & picked[j]:
                        disjoint = False
                        break
                if not disjoint:
                    break
            if disjoint:
                selected_ids = set(empty_edge_ids)
                selected_ids.update(search_edge_ids[i] for i in combo)
                ordered = tuple(
                    edge_id for edge_id in edge_ids if edge_id in selected_ids
                )
                return ordered, len(empty_edge_ids) + size
    return (), 0


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
        edges = [[f"e{i}", [f"v{i}", f"v{(i + 1) % 21}"]] for i in range(21)]
        vertices = [f"v{i}" for i in range(21)]
        request = MaximumEdgeMatchingRequest(
            hypergraph=FiniteHypergraph.model_validate(
                {"vertices": vertices, "edges": edges}
            )
        )
        with pytest.raises(ValueError, match="search exceeds"):
            maximum_edge_matching(request.hypergraph)

    def test_many_disjoint_singletons_are_admitted(self) -> None:
        result = _matching(
            {
                "vertices": [f"v{i}" for i in range(21)],
                "edges": [[f"e{i}", [f"v{i}"]] for i in range(21)],
            }
        )
        assert result.count == 21
        assert result.matching == tuple(f"e{i}" for i in range(21))

    def test_independent_components_compose(self) -> None:
        # Two disjoint conflicting pairs plus one isolated candidate.
        result = _matching(
            {
                "vertices": ["a", "b", "c", "d", "e", "f"],
                "edges": [
                    ["e1", ["a", "b"]],
                    ["e2", ["b", "c"]],
                    ["e3", ["d", "e"]],
                    ["e4", ["f"]],
                ],
            }
        )
        assert result.count == 3
        assert result.matching == ("e1", "e3", "e4")

    def test_total_search_work_bound_still_rejects(self) -> None:
        # 21 pairwise-overlapping pair edges form one 21-candidate
        # component beyond the per-component envelope.
        result_edges = [[f"e{i}", [f"v{i}", f"v{(i + 1) % 21}"]] for i in range(21)]
        with pytest.raises(ValueError, match="search exceeds"):
            _matching(
                {
                    "vertices": [f"v{i}" for i in range(21)],
                    "edges": result_edges,
                }
            )

    def test_decomposed_search_matches_undecomposed_search(self) -> None:
        from itertools import combinations

        vertices = ("a", "b", "c")
        pool = [
            frozenset(combo)
            for width in range(1, 4)
            for combo in combinations(vertices, width)
        ]
        for mask in range(1 << len(pool)):
            family = [pool[index] for index in range(len(pool)) if mask & (1 << index)]
            source = FiniteHypergraph.model_validate(
                {
                    "vertices": list(vertices),
                    "edges": [
                        [f"e{index}", sorted(members)]
                        for index, members in enumerate(family)
                    ],
                }
            )
            expected = _undecomposed_maximum_matching(source)
            actual = maximum_edge_matching(source)
            assert (actual.matching, actual.count) == expected

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
