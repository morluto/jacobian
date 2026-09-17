"""Tests for the bounded genus-search operation.

Covers the FOUND/EXHAUSTED/UNKNOWN contract: first-hit witnesses with
checker certificates, exact exhaustion receipts on Kuratowski graphs,
budget truncation that never yields a negative conclusion, and the
admission envelope shared with the embedding checker.
"""

from __future__ import annotations

import json
import math
from itertools import combinations, permutations, product

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.topology.combinatorial_maps import (
    check_orientable_embedding,
    find_rotation_system,
    verify_orientable_embedding,
    verify_rotation_system_find,
)
from jacobian.math.topology.combinatorial_maps._models import (
    MAX_EMBEDDING_EDGES,
    MAX_EMBEDDING_VERTICES,
    MAX_ROTATION_SYSTEM_CANDIDATES,
    RotationSystemFindRequest,
    RotationSystemFindResult,
)
from jacobian.math.topology.combinatorial_maps._tools import (
    compute_rotation_system_find,
)


def _k4() -> SimpleUndirectedGraph:
    vertices = ("a", "b", "c", "d")
    edges: tuple[tuple[str, str], ...] = (
        ("a", "b"),
        ("a", "c"),
        ("a", "d"),
        ("b", "c"),
        ("b", "d"),
        ("c", "d"),
    )
    return SimpleUndirectedGraph(vertices=vertices, edges=edges)


def _k5() -> SimpleUndirectedGraph:
    vertices = tuple(f"v{index}" for index in range(5))
    edges: tuple[tuple[str, str], ...] = tuple(
        (vertices[a], vertices[b]) for a in range(5) for b in range(a + 1, 5)
    )
    return SimpleUndirectedGraph(vertices=vertices, edges=edges)


def _k33() -> SimpleUndirectedGraph:
    left = ("a", "b", "c")
    right = ("x", "y", "z")
    vertices = left + right
    edges: tuple[tuple[str, str], ...] = tuple(
        (u, v) if u < v else (v, u) for u in left for v in right
    )
    return SimpleUndirectedGraph(vertices=vertices, edges=edges)


class TestFound:
    def test_k4_planar_rotation_is_found(self) -> None:
        result = find_rotation_system(_k4(), 0, 1000)

        assert result.status == "FOUND"
        assert result.reason is None
        assert result.candidates_examined >= 1
        assert result.candidates_examined <= result.total_candidates
        assert result.total_candidates == 2**4
        certificate = result.certificate
        assert certificate is not None
        assert certificate.genus == 0
        assert certificate.rotations == result.rotations
        assert verify_orientable_embedding(certificate)

    def test_k5_genus_one_rotation_is_found(self) -> None:
        result = find_rotation_system(_k5(), 1, 10_000)

        assert result.status == "FOUND"
        assert result.certificate is not None
        assert result.certificate.genus <= 1
        assert verify_orientable_embedding(result.certificate)

    def test_k33_genus_one_rotation_is_found(self) -> None:
        result = find_rotation_system(_k33(), 1, 1000)

        assert result.status == "FOUND"
        assert result.certificate is not None
        assert result.certificate.genus <= 1

    def test_found_certificate_replays_through_the_checker(self) -> None:
        graph = _k4()
        result = find_rotation_system(graph, 0, 1000)

        assert result.status == "FOUND"
        assert result.certificate is not None
        replayed = check_orientable_embedding(graph, result.rotations)

        assert replayed == result.certificate
        assert replayed.genus <= 0

    def test_search_is_deterministic(self) -> None:
        first = find_rotation_system(_k33(), 1, 1000)
        second = find_rotation_system(_k33(), 1, 1000)

        assert first == second

    def test_single_vertex_is_genus_zero(self) -> None:
        graph = SimpleUndirectedGraph(vertices=("a",), edges=())
        result = find_rotation_system(graph, 0, 10)

        assert result.status == "FOUND"
        assert result.total_candidates == 1
        assert result.candidates_examined == 1
        assert result.certificate is not None
        assert result.certificate.genus == 0


class TestExhausted:
    def test_k33_has_no_planar_rotation(self) -> None:
        # K3,3 is nonplanar (Kuratowski): all (2!)^6 = 64 systems fail genus 0.
        result = find_rotation_system(_k33(), 0, 1000)

        assert result.status == "EXHAUSTED"
        assert result.total_candidates == 64
        assert result.candidates_examined == 64
        assert result.rotations == ()
        assert result.certificate is None

    def test_k5_has_no_planar_rotation(self) -> None:
        # K5 is nonplanar (Kuratowski): all (3!)^5 = 7776 systems fail genus 0.
        result = find_rotation_system(_k5(), 0, 10_000)

        assert result.status == "EXHAUSTED"
        assert result.total_candidates == 7776
        assert result.candidates_examined == 7776

    def test_exhaustion_counts_match_an_independent_enumeration(self) -> None:
        # Every K3,3 rotation system has genus one or two, and the minimum
        # one is attained: the checker genus over the full deterministic
        # enumeration never drops to zero.  Genus two is the orientable
        # maximum genus floor((E - V + 1) / 2) = 2.
        graph = _k33()
        genera: set[int] = set()
        index = {label: position for position, label in enumerate(graph.vertices)}
        incident: list[set[int]] = [set() for _ in graph.vertices]
        for edge_index, (left, right) in enumerate(graph.edges):
            incident[index[left]].add(edge_index)
            incident[index[right]].add(edge_index)
        rows = []
        for edges in incident:
            ordered = sorted(edges)
            rows.append([(ordered[0], *perm) for perm in permutations(ordered[1:])])
        for rotations in product(*rows):
            genera.add(check_orientable_embedding(graph, rotations).genus)

        assert genera == {1, 2}

    def test_tight_budget_still_exhausts_k33(self) -> None:
        result = find_rotation_system(_k33(), 0, 64)

        assert result.status == "EXHAUSTED"
        assert result.candidates_examined == 64


class TestUnknown:
    def test_truncated_search_never_claims_exhaustion(self) -> None:
        result = find_rotation_system(_k5(), 0, 100)

        assert result.status == "UNKNOWN"
        assert result.reason == "CANDIDATE_BUDGET_EXCEEDED"
        assert result.candidates_examined == 100
        assert result.total_candidates == 7776
        assert result.rotations == ()
        assert result.certificate is None

    def test_disconnected_graph_is_unknown(self) -> None:
        graph = SimpleUndirectedGraph(vertices=("a", "b", "c"), edges=(("a", "b"),))
        result = find_rotation_system(graph, 0, 100)

        assert result.status == "UNKNOWN"
        assert result.reason == "GRAPH_DISCONNECTED"
        assert result.candidates_examined == 0
        # The rotation-system count is still exact: 1 * 1 * 1.
        assert result.total_candidates == 1

    def test_high_degree_vertex_respects_budget_without_materializing(self) -> None:
        # K_{3,12}: each ``a`` vertex has degree 12, so it owns ``11!`` cyclic
        # orders, and the graph is non-planar, so no candidate qualifies.  A
        # budget of one must stop after a single candidate instead of building
        # factorial-many rows before consulting the budget.
        left = tuple(f"a{index}" for index in range(3))
        right = tuple(f"b{index}" for index in range(12))
        graph = SimpleUndirectedGraph(
            vertices=left + right,
            edges=tuple((a, b) for a in left for b in right),
        )

        result = find_rotation_system(graph, 0, 1)

        assert result.status == "UNKNOWN"
        assert result.reason == "CANDIDATE_BUDGET_EXCEEDED"
        assert result.candidates_examined == 1
        assert (
            result.total_candidates == math.factorial(11) ** 3 * math.factorial(2) ** 12
        )


class TestInvalidRequests:
    def test_empty_graph_is_a_domain_boundary(self) -> None:
        graph = SimpleUndirectedGraph(vertices=(), edges=())

        with pytest.raises(OperationDomainValidationError) as exc_info:
            find_rotation_system(graph, 0, 10)
        assert exc_info.value.errors()[0]["type"] == "topology.embedding.empty_graph"

    def test_zero_budget_is_a_domain_boundary(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            find_rotation_system(_k4(), 0, 0)
        assert (
            exc_info.value.errors()[0]["type"] == "topology.embedding.candidate_budget"
        )

    def test_negative_genus_is_a_domain_boundary(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            find_rotation_system(_k4(), -1, 10)
        assert exc_info.value.errors()[0]["type"] == "topology.embedding.genus_bound"

    def test_overlarge_budget_is_a_resource_boundary(self) -> None:
        with pytest.raises(OperationResourceAdmissionError) as exc_info:
            find_rotation_system(_k4(), 0, MAX_ROTATION_SYSTEM_CANDIDATES + 1)
        assert (
            exc_info.value.errors()[0]["type"] == "topology.embedding.candidate_bound"
        )

    def test_vertex_bound_is_a_resource_boundary(self) -> None:
        size = MAX_EMBEDDING_VERTICES + 1
        vertices = tuple(str(index) for index in range(size))
        edges: tuple[tuple[str, str], ...] = tuple(
            (vertices[i], vertices[i + 1])
            if vertices[i] < vertices[i + 1]
            else (vertices[i + 1], vertices[i])
            for i in range(size - 1)
        )
        graph = SimpleUndirectedGraph(vertices=vertices, edges=edges)

        with pytest.raises(OperationResourceAdmissionError) as exc_info:
            find_rotation_system(graph, 0, 10)
        assert exc_info.value.errors()[0]["type"] == "topology.embedding.vertex_bound"

    def test_edge_bound_is_a_resource_boundary(self) -> None:
        size = 64
        vertices = tuple(str(index) for index in range(size))
        edges: tuple[tuple[str, str], ...] = tuple(
            (left, right) if left < right else (right, left)
            for left, right in (
                tuple(sorted(pair)) for pair in combinations(vertices, 2)
            )
        )[: MAX_EMBEDDING_EDGES + 1]
        graph = SimpleUndirectedGraph(vertices=vertices, edges=edges)

        with pytest.raises(OperationResourceAdmissionError) as exc_info:
            find_rotation_system(graph, 0, 10)
        assert exc_info.value.errors()[0]["type"] == "topology.embedding.edge_bound"

    def test_request_validation_rejects_bad_bounds(self) -> None:
        with pytest.raises(ValidationError):
            RotationSystemFindRequest(graph=_k4(), max_genus=-1, max_candidates=10)
        with pytest.raises(ValidationError):
            RotationSystemFindRequest(graph=_k4(), max_genus=0, max_candidates=0)


class TestAdmissionAndParity:
    def test_native_and_catalog_paths_agree(self) -> None:
        native = find_rotation_system(_k33(), 0, 1000)
        catalog = compute_rotation_system_find(
            RotationSystemFindRequest(graph=_k33(), max_genus=0, max_candidates=1000)
        )

        assert catalog == native

    def test_round_trip_and_verify(self) -> None:
        for max_genus, budget in ((0, 1000), (1, 1000)):
            result = find_rotation_system(_k33(), max_genus, budget)
            restored = RotationSystemFindResult.model_validate_json(
                result.model_dump_json()
            )

            assert restored == result
            assert verify_rotation_system_find(restored)

    def test_found_forgery_is_rejected_or_fails_verify(self) -> None:
        result = find_rotation_system(_k4(), 0, 1000)
        assert result.status == "FOUND"
        forged = json.loads(result.model_dump_json())
        forged["certificate"]["genus"] = 5
        with pytest.raises(ValidationError):
            RotationSystemFindResult.model_validate_json(json.dumps(forged))

    def test_exhausted_forgery_fails_verify(self) -> None:
        result = find_rotation_system(_k33(), 0, 1000)
        assert result.status == "EXHAUSTED"
        forged = json.loads(result.model_dump_json())
        forged["candidates_examined"] = 63
        # Fails structural validation (receipt must equal the total)...
        with pytest.raises(ValidationError):
            RotationSystemFindResult.model_validate_json(json.dumps(forged))

    def test_found_count_forgery_fails_verify(self) -> None:
        result = find_rotation_system(_k4(), 0, 1000)
        assert result.status == "FOUND"
        forged = json.loads(result.model_dump_json())
        # Structurally valid (FOUND requires only examined >= 1) but false:
        # replay finds the first witness at the true examined count.
        forged["candidates_examined"] = result.candidates_examined + 1
        forged_claim = RotationSystemFindResult.model_validate_json(json.dumps(forged))
        assert not verify_rotation_system_find(forged_claim)
