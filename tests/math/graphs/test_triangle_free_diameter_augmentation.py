"""Tests for triangle-free diameter augmentation operation."""

from __future__ import annotations

import json
from typing import Any, cast

import networkx as nx
import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.triangle_free_diameter_augmentation._augmentation_z3 import (
    _derive_candidates,
    _solve_augmentation_kernel,
)
from jacobian.math.graphs.triangle_free_diameter_augmentation._models import (
    TriangleFreeDiameterAugmentationBudget,
    TriangleFreeDiameterAugmentationRequest,
    TriangleFreeDiameterAugmentationResult,
)
from jacobian.math.graphs.triangle_free_diameter_augmentation._tools import TOOLS
from jacobian.math.graphs.triangle_free_diameter_augmentation.operations import (
    triangle_free_diameter_augmentation,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(
    vertices: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(vertices=vertices, edges=edges)


def _edge(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _triangle_count(graph: nx.Graph[str]) -> int:
    triangles = nx.triangles(graph)
    assert isinstance(triangles, dict)
    return sum(triangles.values()) // 3


def _path(n: int) -> SimpleUndirectedGraph:
    verts = tuple(str(i) for i in range(n))
    edges = tuple(_edge(str(i), str(i + 1)) for i in range(n - 1))
    # canonical sort
    verts = tuple(sorted(verts))
    edges = tuple(sorted(_edge(*e) for e in edges))
    # For n>=10 lexicographic weirdness, keep numeric string order but ensure edges canonical left<right string cmp
    # Rebuild with proper canonical edges via sorted string compare
    edges = tuple(sorted(_edge(*e) for e in edges))
    return SimpleUndirectedGraph(vertices=verts, edges=edges)


def _path_padded(n: int) -> SimpleUndirectedGraph:
    verts = tuple(f"{i:02d}" for i in range(n))
    edges = tuple(_edge(f"{i:02d}", f"{(i + 1):02d}") for i in range(n - 1))
    return SimpleUndirectedGraph(vertices=verts, edges=edges)


def test_path_four_target_two_unique_one_edge_solution() -> None:
    g = _graph(("0", "1", "2", "3"), (("0", "1"), ("1", "2"), ("2", "3")))
    result = triangle_free_diameter_augmentation(g, 2)
    assert result.status == "EXACT"
    assert result.added_edge_count == 1
    assert result.added_edges == (("0", "3"),)
    assert result.augmented_diameter == 2
    # validate union triangle-free
    aug: nx.Graph[str] = nx.Graph()
    aug.add_nodes_from(g.vertices)
    aug.add_edges_from(g.edges)
    aug.add_edges_from(result.added_edges)
    assert _triangle_count(aug) == 0
    assert nx.diameter(aug) == 2


def test_zero_case_c4_already_meets_target() -> None:
    g = _graph(("0", "1", "2", "3"), (("0", "1"), ("1", "2"), ("2", "3"), ("0", "3")))
    result = triangle_free_diameter_augmentation(g, 2)
    assert result.status == "EXACT"
    assert result.added_edge_count == 0
    assert result.added_edges == ()
    assert result.augmented_diameter == 2


def test_forbidden_triangle_shortcut_is_not_used() -> None:
    # Path 0-1-2, target 1 requires edge (0,2) but it has common neighbor 1 => illegal => infeasible
    g = _graph(("0", "1", "2"), (("0", "1"), ("1", "2")))
    result = triangle_free_diameter_augmentation(g, 1)
    assert result.status == "INFEASIBLE"
    assert result.added_edge_count is None
    assert result.added_edges == ()

    # Also direct check that candidate derivation excludes triangle-closing edge
    _, cands, _ = _derive_candidates(g)
    assert ("0", "2") not in cands
    assert cands == []


def test_exhaustive_differential_small_graphs() -> None:
    # Enumerate all small graphs up to order 5 for several targets and compare to brute force

    def brute(graph: SimpleUndirectedGraph, target: int) -> int | None:
        verts = tuple(sorted(graph.vertices))
        _, cands, _ = _derive_candidates(graph)
        best = None
        for mask in range(1 << len(cands)):
            added = tuple(
                sorted(cands[i] for i in range(len(cands)) if (mask >> i) & 1)
            )
            if best is not None and len(added) >= best:
                continue
            aug: nx.Graph[str] = nx.Graph()
            aug.add_nodes_from(verts)
            aug.add_edges_from(graph.edges)
            aug.add_edges_from(added)
            if _triangle_count(aug) != 0:
                continue
            if not nx.is_connected(aug):
                continue
            if nx.diameter(aug) > target:
                continue
            if best is None or len(added) < best:
                best = len(added)
        return best

    labels = ["a", "b", "c", "d"]
    # Exhaustive over all 2^6=64 graphs
    verts = tuple(labels)
    all_edges = [(labels[i], labels[j]) for i in range(4) for j in range(i + 1, 4)]
    for mask in range(1 << len(all_edges)):
        edges = tuple(
            sorted(
                _edge(*all_edges[i]) for i in range(len(all_edges)) if (mask >> i) & 1
            )
        )
        if not edges:
            continue
        try:
            g = SimpleUndirectedGraph(vertices=verts, edges=edges)
        except Exception:
            continue
        # filter triangle-free and connected
        aug0: nx.Graph[str] = nx.Graph()
        aug0.add_nodes_from(g.vertices)
        aug0.add_edges_from(g.edges)
        if _triangle_count(aug0) != 0:
            continue
        if not nx.is_connected(aug0):
            continue
        for target in [2, 3]:
            budget = TriangleFreeDiameterAugmentationBudget(
                wall_seconds=5, max_order=10
            )
            res = triangle_free_diameter_augmentation(g, target, resource_budget=budget)
            brute_best = brute(g, target)
            if brute_best is None:
                assert res.status == "INFEASIBLE"
            else:
                assert res.status == "EXACT"
                assert res.added_edge_count == brute_best
                # verify diameter
                aug: nx.Graph[str] = nx.Graph()
                aug.add_nodes_from(g.vertices)
                aug.add_edges_from(g.edges)
                aug.add_edges_from(res.added_edges)
                assert _triangle_count(aug) == 0
                assert nx.is_connected(aug)
                assert nx.diameter(aug) <= target


def test_multiple_additions_minimality() -> None:
    # Path on 6 vertices target 2 requires at least 2 edges (from earlier brute)
    g = _path_padded(6)
    result = triangle_free_diameter_augmentation(g, 2)
    assert result.status == "EXACT"
    assert result.added_edge_count == 2
    # verify no single edge suffices
    _, cands, _ = _derive_candidates(g)
    for edge in cands:
        aug: nx.Graph[str] = nx.Graph()
        aug.add_nodes_from(g.vertices)
        aug.add_edges_from(g.edges)
        aug.add_edge(*edge)
        if _triangle_count(aug) != 0:
            continue
        if not nx.is_connected(aug):
            continue
        assert nx.diameter(aug) > 2

    # Cycle 6 target 2 needs 3 edges
    verts = tuple(f"{i:02d}" for i in range(6))
    edges = tuple(_edge(f"{i:02d}", f"{(i + 1) % 6:02d}") for i in range(6))
    g6 = SimpleUndirectedGraph(vertices=verts, edges=tuple(sorted(edges)))
    res6 = triangle_free_diameter_augmentation(g6, 2)
    assert res6.status == "EXACT"
    assert res6.added_edge_count == 3


def test_invalid_disconnected_rejected() -> None:
    g = _graph(("0", "1", "2"), (("0", "1"),))
    with pytest.raises(OperationDomainValidationError, match="connected"):
        triangle_free_diameter_augmentation(g, 2)


def test_invalid_triangle_rejected() -> None:
    g = _graph(("0", "1", "2"), (("0", "1"), ("1", "2"), ("0", "2")))
    with pytest.raises(OperationDomainValidationError, match="triangle-free"):
        triangle_free_diameter_augmentation(g, 2)


def test_invalid_target_boundary() -> None:
    g = _graph(("0", "1", "2", "3"), (("0", "1"), ("1", "2"), ("2", "3")))
    with pytest.raises(ValidationError):
        TriangleFreeDiameterAugmentationRequest.model_validate(
            {"graph": g.model_dump(), "target_diameter": 0}
        )
    with pytest.raises(ValidationError):
        TriangleFreeDiameterAugmentationRequest.model_validate(
            {"graph": g.model_dump(), "target_diameter": 13}
        )


def test_solver_budget_exhaustion() -> None:
    # Force wall-clock expiry via monkeypatching time
    g = _graph(("0", "1", "2", "3"), (("0", "1"), ("1", "2"), ("2", "3")))
    budget = TriangleFreeDiameterAugmentationBudget(wall_seconds=1, max_order=10)
    # monkeypatch time.monotonic inside kernel to simulate expiry
    import jacobian.math.graphs.triangle_free_diameter_augmentation._augmentation_z3 as _augmentation_z3

    mod: Any = cast(Any, _augmentation_z3)

    orig = mod.time.monotonic
    try:
        # make remaining_ms negative by advancing clock
        it = iter([0.0, 10.0])
        mod.time.monotonic = lambda: next(it, 10.0)
        res = _solve_augmentation_kernel(g, 2, budget)
        assert res.status == "SOLVER_BUDGET_EXCEEDED"
        assert res.added_edge_count is None
        assert res.added_edges == ()
        assert res.augmented_diameter is None
    finally:
        mod.time.monotonic = orig


def test_sparse_triangle_free_family_native_mcp_replay() -> None:
    # Sparse triangle-free family: bipartite-like and path variants, padded labels for lexicographic
    families = [
        _path_padded(5),
        _path_padded(6),
        SimpleUndirectedGraph(
            vertices=tuple(f"{i:02d}" for i in range(6)),
            edges=tuple(
                sorted(
                    [
                        _edge("00", "01"),
                        _edge("00", "02"),
                        _edge("01", "03"),
                        _edge("02", "04"),
                        _edge("03", "05"),
                    ]
                )
            ),
        ),
        SimpleUndirectedGraph(
            vertices=tuple(f"{i:02d}" for i in range(8)),
            edges=tuple(
                sorted(
                    [
                        _edge("00", "01"),
                        _edge("01", "02"),
                        _edge("02", "03"),
                        _edge("03", "04"),
                        _edge("04", "05"),
                        _edge("05", "06"),
                        _edge("06", "07"),
                    ]
                )
            ),
        ),
    ]
    tool = TOOLS[0]
    for g in families:
        # ensure triangle-free and connected
        aug: nx.Graph[str] = nx.Graph()
        aug.add_nodes_from(g.vertices)
        aug.add_edges_from(g.edges)
        assert _triangle_count(aug) == 0
        assert nx.is_connected(aug)
        for target in [2, 3]:
            budget = TriangleFreeDiameterAugmentationBudget(
                wall_seconds=5, max_order=10
            )
            native = triangle_free_diameter_augmentation(
                g, target, resource_budget=budget
            )
            req = TriangleFreeDiameterAugmentationRequest(
                graph=g, target_diameter=target, resource_budget=budget
            )
            mcp = tool.run(req)
            assert native == mcp
            # replay via serialization
            payload = native.model_dump(mode="json")
            reparsed = TriangleFreeDiameterAugmentationResult.model_validate_json(
                json.dumps(payload)
            )
            assert reparsed == native
            if native.status == "EXACT":
                aug2: nx.Graph[str] = nx.Graph()
                aug2.add_nodes_from(g.vertices)
                aug2.add_edges_from(g.edges)
                aug2.add_edges_from(native.added_edges)
                assert _triangle_count(aug2) == 0
                assert nx.is_connected(aug2)
                assert nx.diameter(aug2) <= target
                # check canonical sorted
                assert native.added_edges == tuple(sorted(native.added_edges))
                # check not in original
                assert all(e not in set(g.edges) for e in native.added_edges)


def test_catalog_example() -> None:
    tool = TOOLS[0]
    example_input = tool.examples[0].input
    req = tool.request_type.model_validate(example_input)
    res = tool.run(req)
    assert res.status == "EXACT"
    assert res.added_edge_count == 1
    assert res.added_edges == (("0", "3"),)


def test_result_rejects_forged_ledger() -> None:
    g = _graph(("0", "1", "2", "3"), (("0", "1"), ("1", "2"), ("2", "3")))
    res = triangle_free_diameter_augmentation(g, 2)
    payload = res.model_dump(mode="json")
    # forge: add extra edge that creates triangle or wrong count
    payload["added_edges"].append(["0", "2"])
    payload["added_edge_count"] = 2
    with pytest.raises(ValidationError):
        TriangleFreeDiameterAugmentationResult.model_validate_json(json.dumps(payload))

    # Result deserialization validates structural claims only; the kernel owns
    # the semantic diameter computation.
    payload2 = res.model_dump(mode="json")
    payload2["augmented_diameter"] = 1
    assert (
        TriangleFreeDiameterAugmentationResult.model_validate(
            payload2
        ).augmented_diameter
        == 1
    )


def test_admission_bounds_candidate_and_reachability() -> None:
    # Candidate bound: use n=12 path which has 45 candidates within 55, but we can artificially test exceeding by lowering hard limit via monkeypatch
    g = _path_padded(12)
    # Normal should pass
    budget = TriangleFreeDiameterAugmentationBudget(wall_seconds=5, max_order=12)
    res = triangle_free_diameter_augmentation(g, 2, resource_budget=budget)
    assert res.status in ("EXACT", "INFEASIBLE", "SOLVER_BUDGET_EXCEEDED")

    # Force candidate bound exceed by temporarily lowering limit
    import jacobian.math.graphs.triangle_free_diameter_augmentation._augmentation_z3 as _augmentation_z3

    mod: Any = cast(Any, _augmentation_z3)

    orig_cand = mod.HARD_MAX_CANDIDATES
    try:
        mod.HARD_MAX_CANDIDATES = 1
        with pytest.raises(OperationDomainValidationError, match="candidate"):
            triangle_free_diameter_augmentation(g, 2, resource_budget=budget)
    finally:
        mod.HARD_MAX_CANDIDATES = orig_cand

    # A target already met by the source returns before Z3-only reachability
    # admission, even though the backend encoding would exceed its cap.
    assert (
        triangle_free_diameter_augmentation(g, 12, resource_budget=budget).status
        == "EXACT"
    )


def test_empty_graph_rejected() -> None:
    g = SimpleUndirectedGraph(vertices=(), edges=())
    with pytest.raises(OperationDomainValidationError, match="nonempty"):
        triangle_free_diameter_augmentation(g, 2)
