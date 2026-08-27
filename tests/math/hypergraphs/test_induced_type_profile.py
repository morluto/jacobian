"""Defining-invariant and boundary tests for induced type profiles."""

import pytest
from pydantic import ValidationError

from jacobian.math.hypergraphs._models import (
    InducedTypeProfileRequest,
    InducedTypeProfileResult,
    FiniteHypergraph,
)
from jacobian.math.hypergraphs._operations import (
    compute_induced_type_profile,
    verify_induced_type_profile_result,
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
    return compute_induced_type_profile(
        InducedTypeProfileRequest(hypergraph=source, subset_size=subset_size)
    )


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
            (entry.vertex_subset, entry.induced_edge_count)
            for entry in result.entries
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

    def test_no_edges_profile_is_all_zero(self) -> None:
        result = _profile({"vertices": ["a", "b", "c"], "edges": []}, 2)
        assert all(entry.induced_edge_count == 0 for entry in result.entries)
        assert len(result.entries) == 3

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
        counts = {entry.vertex_subset: entry.induced_edge_count for entry in result.entries}
        assert counts[("a", "b")] == 1

    def test_subset_size_exceeds_vertex_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            InducedTypeProfileRequest(hypergraph=HYPERGRAPH, subset_size=5)

    def test_subset_exceeds_profile_bound_rejected(self) -> None:
        # C(20, 10) = 184_756 > 4096 bound
        hg = {"vertices": [f"v{i}" for i in range(20)], "edges": []}
        with pytest.raises(ValidationError):
            InducedTypeProfileRequest(hypergraph=hg, subset_size=10)

    def test_verify_round_trip(self) -> None:
        result = _profile(HYPERGRAPH, 3)
        assert verify_induced_type_profile_result(result)

    def test_verify_rejects_tampered_count(self) -> None:
        result = _profile(HYPERGRAPH, 2)
        tampered = result.model_copy(
            update={
                "entries": tuple(
                    result.entries[:1]
                    + (
                        result.entries[1].model_copy(
                            update={"induced_edge_count": 99}
                        ),
                    )
                    + result.entries[2:]
                )
            }
        )
        with pytest.raises(ValidationError):
            InducedTypeProfileResult.model_validate(tampered.model_dump())

    def test_entries_in_lexicographic_order(self) -> None:
        result = _profile(
            {
                "vertices": ["z", "a", "m"],
                "edges": [["e1", ["z", "a"]]],
            },
            2,
        )
        subsets = [entry.vertex_subset for entry in result.entries]
        # Each subset is internally lexicographically sorted; the subsets
        # appear in the order produced by combinations over the declared
        # vertex order: (z,a), (z,m), (a,m) -> sorted: (a,z), (m,z), (a,m).
        assert subsets == [("a", "z"), ("m", "z"), ("a", "m")]
