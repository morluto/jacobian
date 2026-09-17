"""Tests for the signed cellular embedding checker.

Covers the orientability decision (exact balance test, never face parity),
the double-cover face projection (every edge twice), the odd-twist witness,
and the admission envelope shared with the unsigned checker.
"""

from __future__ import annotations

import json
from collections import deque
from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.topology.combinatorial_maps import (
    check_orientable_embedding,
    check_signed_embedding,
    verify_signed_embedding,
)
from jacobian.math.topology.combinatorial_maps._models import (
    MAX_EMBEDDING_EDGES,
    MAX_EMBEDDING_VERTICES,
    SignedEmbeddingCheckRequest,
    SignedEmbeddingCheckResult,
)
from jacobian.math.topology.combinatorial_maps._tools import (
    compute_signed_embedding_check,
)


def _k4() -> SimpleUndirectedGraph:
    vertices = ("a", "b", "c", "d")
    edges = (
        ("a", "b"),
        ("a", "c"),
        ("a", "d"),
        ("b", "c"),
        ("b", "d"),
        ("c", "d"),
    )
    return SimpleUndirectedGraph(vertices=vertices, edges=edges)


# A planar (sphere) rotation system of K4 / the tetrahedron graph.
_K4_SPHERE = ((0, 1, 2), (0, 4, 3), (1, 3, 5), (2, 5, 4))


def _k7() -> SimpleUndirectedGraph:
    vertices = tuple(str(index) for index in range(7))
    edges: tuple[tuple[str, str], ...] = tuple(
        (left, right) if left < right else (right, left)
        for left, right in (tuple(sorted(pair)) for pair in combinations(vertices, 2))
    )
    return SimpleUndirectedGraph(vertices=vertices, edges=edges)


# A triangular (genus-one torus) rotation system of K7.
_K7_TORUS = (
    (0, 2, 1, 5, 3, 4),
    (6, 8, 7, 0, 9, 10),
    (11, 13, 12, 6, 14, 1),
    (15, 17, 16, 11, 2, 7),
    (18, 3, 19, 15, 8, 12),
    (20, 9, 4, 18, 13, 16),
    (5, 14, 10, 20, 17, 19),
)


def _cycle(length: int = 5) -> SimpleUndirectedGraph:
    vertices = tuple(f"c{index}" for index in range(length))
    edges: tuple[tuple[str, str], ...] = tuple(
        (vertices[index], vertices[(index + 1) % length])
        if vertices[index] < vertices[(index + 1) % length]
        else (vertices[(index + 1) % length], vertices[index])
        for index in range(length)
    )
    return SimpleUndirectedGraph(vertices=vertices, edges=edges)


def _cycle_rotations(length: int = 5) -> tuple[tuple[int, ...], ...]:
    # Edge i joins vertex i to vertex (i+1) % length.
    return tuple(
        tuple(sorted(((index - 1) % length, index))) for index in range(length)
    )


def _path() -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=("a", "b", "c"), edges=(("a", "b"), ("b", "c"))
    )


def _directed_face_set(
    walks: tuple[tuple[int, ...], ...],
) -> set[tuple[int, ...]]:
    """Faces as undirected cycles: canonical rotation-minimal forms."""

    forms = set()
    for walk in walks:
        forward = min(walk[offset:] + walk[:offset] for offset in range(len(walk)))
        reversed_walk = tuple(dart ^ 1 for dart in reversed(walk))
        backward = min(
            reversed_walk[offset:] + reversed_walk[:offset]
            for offset in range(len(reversed_walk))
        )
        forms.add(min(forward, backward))
    return forms


def _endpoints(graph: SimpleUndirectedGraph) -> list[tuple[int, int]]:
    index = {label: position for position, label in enumerate(graph.vertices)}
    return [(index[left], index[right]) for left, right in graph.edges]


def _is_closed_odd_walk(
    graph: SimpleUndirectedGraph,
    signs: tuple[int, ...],
    walk: tuple[int, ...],
) -> bool:
    endpoints = _endpoints(graph)
    dart_count = 2 * len(endpoints)

    def tail(dart: int) -> int:
        left, right = endpoints[dart // 2]
        return left if dart % 2 == 0 else right

    def head(dart: int) -> int:
        return tail(dart ^ 1)

    if not walk or any(not 0 <= dart < dart_count for dart in walk):
        return False
    if any(
        head(walk[position]) != tail(walk[(position + 1) % len(walk)])
        for position in range(len(walk))
    ):
        return False
    return sum(1 for dart in walk if signs[dart // 2] == 0) % 2 == 1


def _is_balanced(graph: SimpleUndirectedGraph, signs: tuple[int, ...]) -> bool:
    endpoints = _endpoints(graph)
    adjacency: list[list[tuple[int, int]]] = [[] for _ in graph.vertices]
    for edge_index, (left, right) in enumerate(endpoints):
        parity = 1 if signs[edge_index] == 0 else 0
        adjacency[left].append((right, parity))
        adjacency[right].append((left, parity))
    potential: dict[int, int] = {}
    for root in range(len(graph.vertices)):
        if root in potential:
            continue
        potential[root] = 0
        queue = deque([root])
        while queue:
            node = queue.popleft()
            for target, parity in adjacency[node]:
                if target not in potential:
                    potential[target] = potential[node] ^ parity
                    queue.append(target)
                elif potential[target] != potential[node] ^ parity:
                    return False
    return True


class TestKnownAnswers:
    def test_k4_untwisted_is_the_sphere(self) -> None:
        result = check_signed_embedding(_k4(), _K4_SPHERE)

        assert result.status == "ORIENTABLE_EMBEDDING"
        assert result.orientable
        assert (result.vertices, result.edges, result.faces) == (4, 6, 4)
        assert result.euler_characteristic == 2
        assert result.genus == 0
        assert result.witness_dart_walk is None
        assert result.signs == (1,) * 6

    def test_k4_one_twist_is_the_projective_plane(self) -> None:
        result = check_signed_embedding(_k4(), _K4_SPHERE, twisted_edges=(0,))

        assert result.status == "NONORIENTABLE_EMBEDDING"
        assert not result.orientable
        assert (result.vertices, result.edges, result.faces) == (4, 6, 3)
        assert result.euler_characteristic == 1
        assert result.genus == 1
        assert result.signs == (0, 1, 1, 1, 1, 1)
        assert result.witness_dart_walk is not None
        assert _is_closed_odd_walk(_k4(), result.signs, result.witness_dart_walk)

    def test_k4_two_twists_is_the_klein_bottle(self) -> None:
        result = check_signed_embedding(_k4(), _K4_SPHERE, twisted_edges=(0, 1))

        assert result.status == "NONORIENTABLE_EMBEDDING"
        assert (result.vertices, result.edges, result.faces) == (4, 6, 2)
        assert result.euler_characteristic == 0
        assert result.genus == 2
        assert result.witness_dart_walk is not None
        assert _is_closed_odd_walk(_k4(), result.signs, result.witness_dart_walk)

    def test_k7_torus_untwisted_is_genus_one(self) -> None:
        result = check_signed_embedding(_k7(), _K7_TORUS)

        assert result.status == "ORIENTABLE_EMBEDDING"
        assert (result.vertices, result.edges, result.faces) == (7, 21, 14)
        assert result.euler_characteristic == 0
        assert result.genus == 1

    def test_k7_twisted_is_nonorientable_with_witness(self) -> None:
        result = check_signed_embedding(_k7(), _K7_TORUS, twisted_edges=(0,))

        assert result.status == "NONORIENTABLE_EMBEDDING"
        assert result.euler_characteristic <= 1
        assert result.genus == 2 - result.euler_characteristic
        assert result.witness_dart_walk is not None
        assert _is_closed_odd_walk(_k7(), result.signs, result.witness_dart_walk)

    def test_tree_is_orientable_for_any_signs(self) -> None:
        # A tree has no cycles, so every signature is balanced.
        result = check_signed_embedding(_path(), ((0,), (0, 1), (1,)), signs=(0, 0))

        assert result.status == "ORIENTABLE_EMBEDDING"
        assert (result.vertices, result.edges, result.faces) == (3, 2, 1)
        assert result.genus == 0

    def test_cycle_one_twist_is_the_projective_plane(self) -> None:
        graph = _cycle(5)
        result = check_signed_embedding(graph, _cycle_rotations(5), twisted_edges=(0,))

        assert result.status == "NONORIENTABLE_EMBEDDING"
        assert result.euler_characteristic == 1
        assert result.genus == 1
        assert result.witness_dart_walk is not None
        assert _is_closed_odd_walk(graph, result.signs, result.witness_dart_walk)

    def test_cycle_untwisted_is_the_sphere(self) -> None:
        graph = _cycle(5)
        result = check_signed_embedding(graph, _cycle_rotations(5))

        assert result.status == "ORIENTABLE_EMBEDDING"
        assert result.euler_characteristic == 2
        assert result.genus == 0

    def test_edgeless_single_vertex_is_the_trivial_sphere(self) -> None:
        graph = SimpleUndirectedGraph(vertices=("a",), edges=())
        result = check_signed_embedding(graph, ((),))

        assert result.status == "ORIENTABLE_EMBEDDING"
        assert (result.vertices, result.edges, result.faces) == (1, 0, 1)
        assert result.euler_characteristic == 2
        assert result.genus == 0

    def test_signs_and_twisted_edges_agree(self) -> None:
        via_signs = check_signed_embedding(_k4(), _K4_SPHERE, signs=(0, 1, 1, 1, 1, 1))
        via_twisted = check_signed_embedding(_k4(), _K4_SPHERE, twisted_edges=(0,))

        assert via_signs == via_twisted


class TestDefiningInvariants:
    @pytest.mark.parametrize("twisted", [(), (0,), (0, 1), (0, 1, 2, 3, 4, 5)])
    def test_every_edge_occurs_twice(self, twisted: tuple[int, ...]) -> None:
        result = check_signed_embedding(_k4(), _K4_SPHERE, twisted_edges=twisted)
        occurrences = [0] * 6
        for walk in result.face_walks:
            for dart in walk:
                assert 0 <= dart < len(result.darts)
                occurrences[dart // 2] += 1

        assert occurrences == [2] * 6

    @pytest.mark.parametrize("twisted", [(), (0,), (2, 4)])
    def test_euler_characteristic_binds_cells(self, twisted: tuple[int, ...]) -> None:
        result = check_signed_embedding(_k4(), _K4_SPHERE, twisted_edges=twisted)

        assert result.euler_characteristic == (
            result.vertices - result.edges + result.faces
        )
        assert len(result.face_walks) == result.faces

    def test_alpha_sigma_are_permutations(self) -> None:
        result = check_signed_embedding(_k4(), _K4_SPHERE, twisted_edges=(0,))
        darts = range(len(result.darts))

        assert sorted(result.alpha) == list(darts)
        assert sorted(result.sigma) == list(darts)
        for dart in darts:
            assert result.alpha[result.alpha[dart]] == dart

    def test_orientability_matches_balance_not_face_parity(self) -> None:
        # K4 with one twisted edge has all-even faces yet is nonorientable:
        # orientability follows the balance test, never face parity.
        result = check_signed_embedding(_k4(), _K4_SPHERE, twisted_edges=(0,))
        parities = [
            sum(1 for dart in walk if result.signs[dart // 2] == 0) % 2
            for walk in result.face_walks
        ]

        assert parities == [0] * len(parities)
        assert not _is_balanced(_k4(), result.signs)
        assert result.status == "NONORIENTABLE_EMBEDDING"

    def test_untwisted_matches_unsigned_checker(self) -> None:
        for graph, rotations in (
            (_k4(), _K4_SPHERE),
            (_k7(), _K7_TORUS),
            (_cycle(5), _cycle_rotations(5)),
            (_path(), ((0,), (0, 1), (1,))),
        ):
            signed = check_signed_embedding(graph, rotations)
            unsigned = check_orientable_embedding(graph, rotations)

            assert signed.faces == unsigned.faces
            assert signed.euler_characteristic == unsigned.euler_characteristic
            assert signed.genus == unsigned.genus
            assert signed.darts == unsigned.darts
            assert signed.alpha == unsigned.alpha
            assert signed.sigma == unsigned.sigma
            assert _directed_face_set(signed.face_walks) == _directed_face_set(
                unsigned.face_walks
            )

    def test_cyclic_shifts_change_presentation_not_embedding(self) -> None:
        shifted = tuple((row[-1], *row[:-1]) for row in _K4_SPHERE)
        original = check_signed_embedding(_k4(), _K4_SPHERE, twisted_edges=(0,))
        rotated = check_signed_embedding(_k4(), shifted, twisted_edges=(0,))

        assert original.rotations == rotated.rotations
        assert (original.faces, original.genus) == (rotated.faces, rotated.genus)

    def test_global_reversal_preserves_genus(self) -> None:
        reversed_rotations = tuple(tuple(reversed(row)) for row in _K4_SPHERE)
        result = check_signed_embedding(_k4(), reversed_rotations, twisted_edges=(0,))

        assert result.status == "NONORIENTABLE_EMBEDDING"
        assert result.genus == 1

    def test_witness_is_closed_and_odd(self) -> None:
        graph = _k7()
        result = check_signed_embedding(graph, _K7_TORUS, twisted_edges=(3, 7))

        assert result.status == "NONORIENTABLE_EMBEDDING"
        assert result.witness_dart_walk is not None
        assert _is_closed_odd_walk(graph, result.signs, result.witness_dart_walk)
        assert not _is_balanced(graph, result.signs)

    def test_deterministic_replay(self) -> None:
        first = check_signed_embedding(_k4(), _K4_SPHERE, twisted_edges=(0, 5))
        second = check_signed_embedding(_k4(), _K4_SPHERE, twisted_edges=(0, 5))

        assert first == second


class TestInvalidCandidates:
    def test_omitted_incidence_is_rejected(self) -> None:
        rotations = ((0, 1), (0, 4, 3), (1, 3, 5), (2, 5, 4))
        result = check_signed_embedding(_k4(), rotations, twisted_edges=(0,))

        assert result.status == "INVALID_EMBEDDING"
        assert result.obstruction_code == "ROTATION_DEGREE_MISMATCH"

    def test_duplicate_incidence_is_rejected(self) -> None:
        rotations = ((0, 0, 2), (0, 4, 3), (1, 3, 5), (2, 5, 4))
        result = check_signed_embedding(_k4(), rotations)

        assert result.status == "INVALID_EMBEDDING"
        assert result.obstruction_code == "DUPLICATE_INCIDENCE"

    def test_foreign_incidence_is_rejected(self) -> None:
        rotations = ((0, 1, 3), (0, 4, 3), (1, 3, 5), (2, 5, 4))
        result = check_signed_embedding(_k4(), rotations)

        assert result.status == "INVALID_EMBEDDING"
        assert result.obstruction_code == "FOREIGN_INCIDENCE"

    def test_out_of_range_edge_index_is_rejected(self) -> None:
        rotations = ((0, 1, 6), (0, 4, 3), (1, 3, 5), (2, 5, 4))
        result = check_signed_embedding(_k4(), rotations)

        assert result.status == "INVALID_EMBEDDING"
        assert result.obstruction_code == "EDGE_INDEX_OUT_OF_RANGE"

    def test_wrong_row_count_is_rejected(self) -> None:
        result = check_signed_embedding(_k4(), _K4_SPHERE[:3])

        assert result.status == "INVALID_EMBEDDING"
        assert result.obstruction_code == "ROTATION_ROW_COUNT"

    def test_disconnected_graph_is_rejected(self) -> None:
        graph = SimpleUndirectedGraph(vertices=("a", "b", "c"), edges=(("a", "b"),))
        result = check_signed_embedding(graph, ((0,), (0,), ()))

        assert result.status == "INVALID_EMBEDDING"
        assert result.obstruction_code == "GRAPH_DISCONNECTED"

    def test_bad_sign_value_is_rejected(self) -> None:
        result = check_signed_embedding(_k4(), _K4_SPHERE, signs=(2, 1, 1, 1, 1, 1))

        assert result.status == "INVALID_EMBEDDING"
        assert result.obstruction_code == "SIGN_INDEX_OUT_OF_RANGE"

    def test_short_signs_are_rejected(self) -> None:
        result = check_signed_embedding(_k4(), _K4_SPHERE, signs=(1, 1, 1))

        assert result.status == "INVALID_EMBEDDING"
        assert result.obstruction_code == "SIGN_INDEX_OUT_OF_RANGE"

    def test_out_of_range_twisted_edge_is_rejected(self) -> None:
        result = check_signed_embedding(_k4(), _K4_SPHERE, twisted_edges=(6,))

        assert result.status == "INVALID_EMBEDDING"
        assert result.obstruction_code == "SIGN_INDEX_OUT_OF_RANGE"

    def test_duplicate_twisted_edge_is_rejected(self) -> None:
        result = check_signed_embedding(_k4(), _K4_SPHERE, twisted_edges=(0, 0))

        assert result.status == "INVALID_EMBEDDING"
        assert result.obstruction_code == "SIGN_INDEX_OUT_OF_RANGE"

    def test_both_sign_encodings_are_rejected(self) -> None:
        result = check_signed_embedding(
            _k4(), _K4_SPHERE, signs=(1,) * 6, twisted_edges=(0,)
        )

        assert result.status == "INVALID_EMBEDDING"
        assert result.obstruction_code == "SIGN_INDEX_OUT_OF_RANGE"

    def test_request_rejects_both_sign_encodings(self) -> None:
        with pytest.raises(ValidationError):
            SignedEmbeddingCheckRequest(
                graph=_k4(),
                rotations=_K4_SPHERE,
                signs=(1,) * 6,
                twisted_edges=(0,),
            )


class TestAdmissionAndParity:
    def test_native_and_catalog_paths_agree(self) -> None:
        native = check_signed_embedding(_k4(), _K4_SPHERE, twisted_edges=(0,))
        catalog = compute_signed_embedding_check(
            SignedEmbeddingCheckRequest(
                graph=_k4(), rotations=_K4_SPHERE, twisted_edges=(0,)
            )
        )

        assert catalog == native

    def test_round_trip_and_forgery(self) -> None:
        result = check_signed_embedding(_k4(), _K4_SPHERE, twisted_edges=(0,))
        restored = SignedEmbeddingCheckResult.model_validate_json(
            result.model_dump_json()
        )

        assert restored == result
        assert verify_signed_embedding(restored)
        forged = json.loads(restored.model_dump_json())
        forged["genus"] = 2
        with pytest.raises(ValidationError):
            SignedEmbeddingCheckResult.model_validate_json(json.dumps(forged))
        forged_witness = json.loads(restored.model_dump_json())
        forged_witness["witness_dart_walk"] = [0]
        with pytest.raises(ValidationError):
            SignedEmbeddingCheckResult.model_validate_json(json.dumps(forged_witness))
        forged_rotations = json.loads(restored.model_dump_json())
        forged_rotations["rotations"][0] = [0, 1]
        forged_claim = SignedEmbeddingCheckResult.model_validate_json(
            json.dumps(forged_rotations)
        )
        assert not verify_signed_embedding(forged_claim)

    def test_orientable_round_trip_and_verify(self) -> None:
        result = check_signed_embedding(_k4(), _K4_SPHERE)
        restored = SignedEmbeddingCheckResult.model_validate_json(
            result.model_dump_json()
        )

        assert restored == result
        assert verify_signed_embedding(restored)

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
            check_signed_embedding(graph, tuple(() for _ in range(size)))
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
            check_signed_embedding(graph, tuple(() for _ in range(size)))
        assert exc_info.value.errors()[0]["type"] == "topology.embedding.edge_bound"

    def test_rotation_entry_bound_is_a_resource_boundary(self) -> None:
        size = 20
        vertices = tuple(str(index) for index in range(size))
        graph = SimpleUndirectedGraph(vertices=vertices, edges=())
        rotations = tuple((0,) * 64 for _ in range(size))

        with pytest.raises(OperationResourceAdmissionError) as exc_info:
            check_signed_embedding(graph, rotations)
        assert exc_info.value.errors()[0]["type"] == "topology.embedding.rotation_bound"

    def test_empty_graph_is_a_domain_boundary(self) -> None:
        graph = SimpleUndirectedGraph(vertices=(), edges=())

        with pytest.raises(OperationDomainValidationError) as exc_info:
            check_signed_embedding(graph, ())
        assert exc_info.value.errors()[0]["type"] == "topology.embedding.empty_graph"
