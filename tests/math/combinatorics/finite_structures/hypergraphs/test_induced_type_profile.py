"""Defining-invariant and boundary tests for induced type profiles."""

from itertools import combinations

import pytest

from jacobian.canonical import CanonicalLimits, canonicalize_json
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
    InducedTypeProfileResult,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs.operations import (
    induced_type_profile,
)

# The Fano-plane-like hypergraph used across the domain.
HYPERGRAPH = {
    "vertices": ["a", "b", "c", "d"],
    "edges": [
        ["e1", ["a", "b", "c"]],
        ["e2", ["b", "c", "d"]],
        ["e3", ["a", "d"]],
    ],
}


def _profile(source: object, subset_size: int) -> InducedTypeProfileResult:
    return induced_type_profile(FiniteHypergraph.model_validate(source), subset_size)


class TestInducedTypeProfile:
    def test_known_2_subset_profile(self) -> None:
        result = _profile(HYPERGRAPH, 2)
        # For each 2-subset, the distinct nonempty induced edges:
        # {a,b}: e1∩={a,b}, e2∩={b}, e3∩={a}        -> 3
        # {a,c}: e1∩={a,c}, e2∩={c}, e3∩={a}        -> 3
        # {a,d}: e1∩={a}, e2∩={d}, e3∩={a,d}        -> 3
        # {b,c}: e1∩={b,c}, e2∩={b,c}, e3∩={}        -> 1
        # {b,d}: e1∩={b}, e2∩={b,d}, e3∩={d}        -> 3
        # {c,d}: e1∩={c}, e2∩={c,d}, e3∩={d}        -> 3
        assert tuple(
            (entry.vertex_subset, entry.induced_edge_count) for entry in result.entries
        ) == (
            (("a", "b"), 3),
            (("a", "c"), 3),
            (("a", "d"), 3),
            (("b", "c"), 1),
            (("b", "d"), 3),
            (("c", "d"), 3),
        )
        assert result.subset_size == 2

    def test_subset_size_0_yields_single_empty_entry(self) -> None:
        result = _profile(HYPERGRAPH, 0)
        assert len(result.entries) == 1
        assert result.entries[0].vertex_subset == ()
        assert result.entries[0].induced_edge_count == 0

    def test_subset_size_equals_vertex_count(self) -> None:
        # Every vertex: induced edges are the original edges, deduplicated.
        result = _profile(HYPERGRAPH, 4)
        assert len(result.entries) == 1
        entry = result.entries[0]
        assert entry.vertex_subset == ("a", "b", "c", "d")
        assert entry.induced_edge_count == 3

    def test_induced_edge_count_uses_edge_bound(self) -> None:
        vertices = [f"v{i}" for i in range(9)]
        edges = [
            [f"e{i}", list(subset)]
            for i, subset in enumerate(
                subset
                for size in range(1, len(vertices) + 1)
                for subset in combinations(vertices, size)
            )
        ]

        result = _profile({"vertices": vertices, "edges": edges}, len(vertices))

        assert result.entries[0].induced_edge_count == 2 ** len(vertices) - 1

    def test_no_edges_profile_is_all_zero(self) -> None:
        result = _profile({"vertices": ["a", "b", "c"], "edges": []}, 2)
        assert all(entry.induced_edge_count == 0 for entry in result.entries)
        assert len(result.entries) == 3

    def test_decomposed_vertex_labels_use_original_lookup_keys(self) -> None:
        decomposed = "e\u0301"

        result = _profile(
            {
                "vertices": [decomposed, "b"],
                "edges": [["e1", [decomposed]]],
            },
            1,
        )

        assert tuple(entry.vertex_subset for entry in result.entries) == (
            ("b",),
            (decomposed,),
        )

    def test_deduplication_of_identical_induced_edges(self) -> None:
        # Two edges with identical members collapse to one induced edge.
        result = _profile(
            {
                "vertices": ["a", "b", "c"],
                "edges": [
                    ["e1", ["a", "b"]],
                    ["e2", ["a", "b"]],
                ],
            },
            2,
        )
        # For {a,b}: both induce {a,b}, dedup -> 1
        counts = {
            entry.vertex_subset: entry.induced_edge_count for entry in result.entries
        }
        assert counts[("a", "b")] == 1

    def test_subset_size_exceeds_vertex_count_rejected(self) -> None:
        with pytest.raises(ValueError):
            _profile(HYPERGRAPH, 5)

    def test_subset_exceeds_profile_bound_rejected(self) -> None:
        # C(20, 10) = 184_756 > 4096 bound
        hg = {"vertices": [f"v{i}" for i in range(20)], "edges": []}
        with pytest.raises(ValueError):
            _profile(hg, 10)

    def test_profile_bound_accounts_for_actual_subset_label_bytes(self) -> None:
        wide_vertices = [f"{index:03d}" + "😀" * 61 for index in range(256)]
        with pytest.raises(ValueError, match="canonical output limit"):
            _profile(
                {"vertices": wide_vertices, "edges": []},
                255,
            )

        compact = _profile(
            {"vertices": [f"v{index}" for index in range(256)], "edges": []},
            255,
        )
        assert len(canonicalize_json(compact.model_dump(mode="json"))) <= (
            CanonicalLimits().max_output_bytes
        )

    def test_entries_in_lexicographic_order(self) -> None:
        result = _profile(
            {
                "vertices": ["z", "a", "m"],
                "edges": [["e1", ["z", "a"]]],
            },
            2,
        )
        subsets = [entry.vertex_subset for entry in result.entries]
        # Both each subset and the complete profile use lexicographic order.
        assert subsets == [("a", "m"), ("a", "z"), ("m", "z")]
