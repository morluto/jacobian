from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations, product

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.coloring.induced_edge_deletion_profile._models import (
    InducedEdgeDeletionProfileRequest,
    InducedEdgeDeletionProfileResult,
)
from jacobian.math.graphs.coloring.induced_edge_deletion_profile.operations import (
    compute_induced_edge_deletion_profile,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(
    vertices: Sequence[str], edges: Sequence[Sequence[str]]
) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple((a, b) if a < b else (b, a) for a, b in edges),
    )


def _brute_is_r_colorable(
    vertices: list[str], edges: list[tuple[str, str]], r: int
) -> bool:
    if not edges:
        return True
    if r >= len(vertices):
        return True
    if r == 1:
        return False
    idx = {v: i for i, v in enumerate(vertices)}
    for coloring in product(range(r), repeat=len(vertices)):
        if all(coloring[idx[a]] != coloring[idx[b]] for a, b in edges):
            return True
    return False


def _brute_min_deletions(
    vertices: list[str], edges: list[tuple[str, str]], r: int
) -> tuple[int, tuple[tuple[str, str], ...]]:
    sorted_edges = sorted(edges)
    m = len(sorted_edges)
    for k in range(m + 1):
        for combo in combinations(range(m), k):
            remaining = [e for i, e in enumerate(sorted_edges) if i not in combo]
            if _brute_is_r_colorable(vertices, remaining, r):
                deleted = tuple(sorted_edges[i] for i in combo)
                return k, deleted
    return m, tuple(sorted_edges)


def _exhaustive_profile_brute(
    graph: SimpleUndirectedGraph, r: int
) -> list[tuple[tuple[str, ...], int, tuple[tuple[str, str], ...]]]:
    sorted_vertices = sorted(graph.vertices)
    sorted_edges = tuple(sorted(graph.edges))
    rows: list[tuple[tuple[str, ...], int, tuple[tuple[str, str], ...]]] = []
    n = len(sorted_vertices)
    for size in range(n + 1):
        for subset in combinations(sorted_vertices, size):
            subset_set = set(subset)
            induced = [
                e for e in sorted_edges if e[0] in subset_set and e[1] in subset_set
            ]
            k, deleted = _brute_min_deletions(list(subset), induced, r)
            rows.append((tuple(subset), k, deleted))
    return rows


def test_empty_graph() -> None:
    g = _graph([], [])
    result = compute_induced_edge_deletion_profile(g, 1)
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.vertex_subset == ()
    assert row.min_deletions == 0
    assert row.deleted_edges == ()
    assert row.induced_edge_count == 0
    assert result.max_deletions_by_size[0].maximum_min_deletions == 0


def test_singleton_graph() -> None:
    g = _graph(["x"], [])
    result = compute_induced_edge_deletion_profile(g, 1)
    assert len(result.rows) == 2
    for row in result.rows:
        assert row.min_deletions == 0
        assert row.deleted_edges == ()
        assert row.induced_edge_count == 0
    assert result.max_deletions_by_size[0].maximum_min_deletions == 0
    assert result.max_deletions_by_size[1].maximum_min_deletions == 0


def test_triangle_r2_complete_profile() -> None:
    # Triangle K3 at r=2: whole set needs 1 deletion, proper subsets need 0
    g = _graph(["a", "b", "c"], [("a", "b"), ("a", "c"), ("b", "c")])
    result = compute_induced_edge_deletion_profile(g, 2)
    assert len(result.rows) == 8
    # check whole set
    whole = next(r for r in result.rows if set(r.vertex_subset) == {"a", "b", "c"})
    assert whole.min_deletions == 1
    assert whole.deleted_edges == (("a", "b"),)
    assert whole.induced_edge_count == 3
    # proper subsets
    for row in result.rows:
        if len(row.vertex_subset) < 3:
            assert row.min_deletions == 0
            assert row.deleted_edges == ()
    # per-size maxima
    assert result.max_deletions_by_size[0].maximum_min_deletions == 0
    assert result.max_deletions_by_size[1].maximum_min_deletions == 0
    assert result.max_deletions_by_size[2].maximum_min_deletions == 0
    assert result.max_deletions_by_size[3].maximum_min_deletions == 1
    assert result.max_deletions_by_size[3].attaining_subset_count == 1


def test_path4_r2_already_bipartite() -> None:
    g = _graph(["0", "1", "2", "3"], [("0", "1"), ("1", "2"), ("2", "3")])
    result = compute_induced_edge_deletion_profile(g, 2)
    for row in result.rows:
        assert row.min_deletions == 0, (
            f"path induced {row.vertex_subset} should be bipartite"
        )
        assert row.deleted_edges == ()
    for entry in result.max_deletions_by_size:
        assert entry.maximum_min_deletions == 0


def test_odd_cycle_r2() -> None:
    # C5 at r=2 needs 1 deletion only for the full cycle
    verts = ["0", "1", "2", "3", "4"]
    edges = [("0", "1"), ("1", "2"), ("2", "3"), ("3", "4"), ("0", "4")]
    g = _graph(verts, edges)
    result = compute_induced_edge_deletion_profile(g, 2)
    whole = next(r for r in result.rows if len(r.vertex_subset) == 5)
    assert whole.min_deletions == 1
    assert whole.deleted_edges == (("0", "1"),)
    # all smaller induced subgraphs are forests -> 0
    for row in result.rows:
        if len(row.vertex_subset) < 5:
            assert row.min_deletions == 0
    assert result.max_deletions_by_size[5].maximum_min_deletions == 1


def test_complete_graph_k4_r2_and_r3() -> None:
    verts = ["a", "b", "c", "d"]
    edges = list(combinations(verts, 2))
    g = _graph(verts, edges)
    # r=2: Turan(4,2)=4 edges, so need 2 deletions; triangles need 1
    result2 = compute_induced_edge_deletion_profile(g, 2)
    whole2 = next(r for r in result2.rows if len(r.vertex_subset) == 4)
    assert whole2.min_deletions == 2
    assert whole2.deleted_edges == (("a", "b"), ("c", "d"))
    # size 3 subsets (triangles) need 1
    for row in result2.rows:
        if len(row.vertex_subset) == 3:
            assert row.min_deletions == 1
    assert result2.max_deletions_by_size[3].maximum_min_deletions == 1
    assert result2.max_deletions_by_size[4].maximum_min_deletions == 2
    # r=3: K4 needs 1, triangles need 0
    result3 = compute_induced_edge_deletion_profile(g, 3)
    whole3 = next(r for r in result3.rows if len(r.vertex_subset) == 4)
    assert whole3.min_deletions == 1
    assert whole3.deleted_edges == (("a", "b"),)
    for row in result3.rows:
        if len(row.vertex_subset) == 3:
            assert row.min_deletions == 0


def test_disconnected_two_triangles() -> None:
    g = _graph(
        ["a", "b", "c", "d", "e", "f"],
        [("a", "b"), ("a", "c"), ("b", "c"), ("d", "e"), ("d", "f"), ("e", "f")],
    )
    result = compute_induced_edge_deletion_profile(g, 2)
    whole = next(r for r in result.rows if len(r.vertex_subset) == 6)
    assert whole.min_deletions == 2
    assert whole.deleted_edges == (("a", "b"), ("d", "e"))
    # subsets containing one triangle only need 1, empty/bipartite need 0
    for row in result.rows:
        # row with vertices exactly one triangle
        if set(row.vertex_subset) == {"a", "b", "c"} or set(row.vertex_subset) == {
            "d",
            "e",
            "f",
        }:
            assert row.min_deletions == 1


def test_relabelling_invariance() -> None:
    g1 = _graph(["a", "b", "c"], [("a", "b"), ("a", "c"), ("b", "c")])
    g2 = _graph(["x", "y", "z"], [("x", "y"), ("x", "z"), ("y", "z")])
    r = 2
    res1 = compute_induced_edge_deletion_profile(g1, r)
    res2 = compute_induced_edge_deletion_profile(g2, r)
    # per-size maxima must coincide
    assert [
        (e.subset_size, e.maximum_min_deletions, e.attaining_subset_count)
        for e in res1.max_deletions_by_size
    ] == [
        (e.subset_size, e.maximum_min_deletions, e.attaining_subset_count)
        for e in res2.max_deletions_by_size
    ]
    # multiset of D values must coincide
    from collections import Counter

    c1 = Counter(rw.min_deletions for rw in res1.rows)
    c2 = Counter(rw.min_deletions for rw in res2.rows)
    assert c1 == c2


def test_tied_optima_canonical_tie_break() -> None:
    # K3 has 3 equally optimal single-edge deletions; canonical must be lexicographically smallest edge
    g = _graph(["a", "b", "c"], [("a", "b"), ("a", "c"), ("b", "c")])
    result = compute_induced_edge_deletion_profile(g, 2)
    whole = next(r for r in result.rows if len(r.vertex_subset) == 3)
    assert whole.deleted_edges == (("a", "b"),)
    # K4 at r=2 has 3 optimal deletions of size 2; canonical is (a,b)(c,d)
    verts = ["a", "b", "c", "d"]
    edges = list(combinations(verts, 2))
    g4 = _graph(verts, edges)
    res4 = compute_induced_edge_deletion_profile(g4, 2)
    whole4 = next(r for r in res4.rows if len(r.vertex_subset) == 4)
    assert whole4.deleted_edges == (("a", "b"), ("c", "d"))
    # ensure deleted edges are sorted
    for row in res4.rows:
        assert row.deleted_edges == tuple(sorted(row.deleted_edges))


def test_reconstruction_invariant() -> None:
    # For each S, deleted edges subset of induced, and G[S]-F is r-colourable, and no smaller suffices
    g = _graph(
        ["a", "b", "c", "d"],
        [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("a", "c")],
    )
    for r in [1, 2, 3]:
        result = compute_induced_edge_deletion_profile(g, r)
        sorted_edges = tuple(sorted(g.edges))
        for row in result.rows:
            subset_set = set(row.vertex_subset)
            induced = [
                e for e in sorted_edges if e[0] in subset_set and e[1] in subset_set
            ]
            assert set(row.deleted_edges) <= set(induced)
            assert len(row.deleted_edges) == row.min_deletions
            remaining = [e for e in induced if e not in set(row.deleted_edges)]
            assert _brute_is_r_colorable(list(row.vertex_subset), remaining, r), (
                f"remaining not r-colorable for {row}"
            )
            # no smaller deletion works: try all subsets of size min_deletions-1 if >0
            if row.min_deletions > 0:
                k = row.min_deletions - 1
                for combo in combinations(sorted(induced), k):
                    remaining2 = [e for e in induced if e not in set(combo)]
                    assert not _brute_is_r_colorable(
                        list(row.vertex_subset), remaining2, r
                    ), (
                        f"found smaller feasible deletion {combo} for {row.vertex_subset} r={r}"
                    )


def test_exhaustive_small_graphs_vs_oracle() -> None:
    # Exhaustive check over all graphs on 4 vertices for r=2,3 against brute oracle
    verts = ["0", "1", "2", "3"]
    all_edges = list(combinations(verts, 2))
    # test a deterministic sample of graphs (all 64 graphs would be heavy; sample 10)
    import random

    random.seed(42)
    sampled_masks = random.sample(range(1 << len(all_edges)), 10)
    for mask in sampled_masks:
        edges = [all_edges[i] for i in range(len(all_edges)) if (mask >> i) & 1]
        g = _graph(verts, edges)
        for r in [2, 3]:
            result = compute_induced_edge_deletion_profile(g, r)
            brute_rows = _exhaustive_profile_brute(g, r)
            for row, (bs, bk, bd) in zip(result.rows, brute_rows, strict=True):
                assert row.vertex_subset == bs
                assert row.min_deletions == bk, (
                    f"mismatch min_deletions for graph {edges} r={r} subset {bs}: got {row.min_deletions} expected {bk}"
                )
                assert row.deleted_edges == bd, (
                    f"mismatch deleted_edges for {bs}: got {row.deleted_edges} expected {bd}"
                )
            # per-size maxima must be max of brute
            for entry in result.max_deletions_by_size:
                vals = [k for (s, k, _) in brute_rows if len(s) == entry.subset_size]
                if vals:
                    assert entry.maximum_min_deletions == max(vals)
                else:
                    assert entry.maximum_min_deletions == 0


def test_aggregate_bound_rejection() -> None:
    # 9 vertices exceeds vertex envelope (8)
    verts9 = [f"v{i}" for i in range(9)]
    edges9 = [(verts9[i], verts9[j]) for i in range(9) for j in range(i + 1, 9)]
    g9 = _graph(verts9, edges9)
    with pytest.raises(OperationDomainValidationError, match="at most 8 vertices"):
        compute_induced_edge_deletion_profile(g9, 2)
    # solver-call / ledger bound: dense 8-vertex graph with huge conflict budget exceeds ledger
    verts8 = [f"v{i}" for i in range(8)]
    edges8 = [(verts8[i], verts8[j]) for i in range(8) for j in range(i + 1, 8)]
    g8 = _graph(verts8, edges8)
    with pytest.raises(OperationDomainValidationError, match="ledger"):
        compute_induced_edge_deletion_profile(g8, 2, solver_conflicts=1_000_000)
    # retained label characters bound
    long_label = "x" * 200_000
    g_long = _graph([long_label, "y"], [(long_label, "y")])
    # This may exceed retained characters due to rows * witness
    with pytest.raises(OperationDomainValidationError):
        compute_induced_edge_deletion_profile(g_long, 2)


def test_cross_check_s_equals_v_vs_brute() -> None:
    # For S=V, min_deletions should equal brute whole-graph optimum
    g = _graph(
        ["a", "b", "c", "d"],
        [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("a", "c")],
    )
    for r in [2, 3]:
        result = compute_induced_edge_deletion_profile(g, r)
        whole = next(
            row for row in result.rows if len(row.vertex_subset) == len(g.vertices)
        )
        # brute whole
        induced = list(g.edges)
        k, _ = _brute_min_deletions(list(g.vertices), induced, r)
        assert whole.min_deletions == k


def test_native_mcp_parity() -> None:
    g = _graph(["a", "b", "c"], [("a", "b"), ("a", "c"), ("b", "c")])
    native = compute_induced_edge_deletion_profile(g, 2)
    request = InducedEdgeDeletionProfileRequest(graph=g, r=2)
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    tool = next(
        t
        for t in BUILTIN_TOOLS
        if t.operation_id == "graph.coloring.induced_edge_deletion_profile.compute"
    )
    via_tool = tool.run(request)
    assert native == via_tool
    # also test via wire round-trip
    dumped = InducedEdgeDeletionProfileResult.model_validate(native.model_dump())
    assert dumped == native


def test_serialized_round_trip() -> None:
    g = _graph(["0", "1", "2"], [("0", "1"), ("1", "2")])
    result = compute_induced_edge_deletion_profile(g, 2)
    serialized = result.model_dump()
    restored = InducedEdgeDeletionProfileResult.model_validate(serialized)
    assert restored == result
    # rows remain sorted
    assert restored.rows == tuple(
        sorted(restored.rows, key=lambda r: (len(r.vertex_subset), r.vertex_subset))
    )


def test_per_size_maximum_derivation() -> None:
    g = _graph(
        ["a", "b", "c", "d"],
        [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d"), ("a", "c"), ("b", "d")],
    )
    result = compute_induced_edge_deletion_profile(g, 2)
    # verify derived
    from collections import defaultdict

    grouped: dict[int, list[int]] = defaultdict(list)
    for row in result.rows:
        grouped[len(row.vertex_subset)].append(row.min_deletions)
    for entry in result.max_deletions_by_size:
        assert (
            entry.maximum_min_deletions == max(grouped[entry.subset_size])
            if grouped[entry.subset_size]
            else 0
        )
        assert entry.attaining_subset_count == sum(
            1 for v in grouped[entry.subset_size] if v == entry.maximum_min_deletions
        )


def test_r1_and_large_r_shortcuts() -> None:
    # r=1 needs to delete all induced edges
    g = _graph(["a", "b", "c"], [("a", "b"), ("b", "c")])
    res1 = compute_induced_edge_deletion_profile(g, 1)
    for row in res1.rows:
        induced = [
            e
            for e in g.edges
            if e[0] in set(row.vertex_subset) and e[1] in set(row.vertex_subset)
        ]
        assert row.min_deletions == len(induced)
        assert set(row.deleted_edges) == set(induced)
    # r >= n needs zero
    g2 = _graph(["a", "b", "c"], [("a", "b"), ("a", "c"), ("b", "c")])
    res_big = compute_induced_edge_deletion_profile(g2, 5)
    for row in res_big.rows:
        assert row.min_deletions == 0


def test_request_validation_and_large_trivial_colour_count() -> None:
    from pydantic import ValidationError

    g = _graph(["a"], [])
    with pytest.raises(ValidationError):
        InducedEdgeDeletionProfileRequest(graph=g, r=0)
    assert InducedEdgeDeletionProfileRequest(graph=g, r=100).r == 100
    assert all(
        row.min_deletions == 0
        for row in compute_induced_edge_deletion_profile(g, 100).rows
    )
