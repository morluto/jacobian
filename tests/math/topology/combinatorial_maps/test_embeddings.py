"""Tests for the orientable cellular embedding checker."""

from __future__ import annotations

import json
from itertools import combinations
from typing import cast

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.multigraph._models import LooplessMultigraph, MultigraphEdge
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.topology.combinatorial_maps import (
    FiniteCombinatorialMap,
    check_multigraph_embedding,
    check_orientable_embedding,
    euler_characteristic,
    orientable_genus,
    verify_orientable_embedding,
)
from jacobian.math.topology.combinatorial_maps._models import (
    MAX_EMBEDDING_EDGES,
    MAX_EMBEDDING_VERTICES,
    MultigraphEmbeddingResult,
    OrientableEmbeddingCheckRequest,
    OrientableEmbeddingCheckResult,
)
from jacobian.math.topology.combinatorial_maps._tools import (
    compute_orientable_embedding_check,
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

# An orientation-reversed copy of the same sphere embedding.
_K4_SPHERE_REVERSED = (
    (2, 1, 0),
    (3, 4, 0),
    (5, 3, 1),
    (4, 5, 2),
)


def _k7() -> SimpleUndirectedGraph:
    vertices = tuple(str(index) for index in range(7))
    edges = cast(
        tuple[tuple[str, str], ...],
        tuple(sorted(tuple(sorted(pair)) for pair in combinations(vertices, 2))),
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


def _cycle() -> SimpleUndirectedGraph:
    vertices = ("a", "b", "c", "d")
    edges = (("a", "b"), ("a", "d"), ("b", "c"), ("c", "d"))
    return SimpleUndirectedGraph(vertices=vertices, edges=edges)


def _path() -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=("a", "b", "c"), edges=(("a", "b"), ("b", "c"))
    )


def _combinatorial_map(
    result: OrientableEmbeddingCheckResult,
) -> FiniteCombinatorialMap:
    return FiniteCombinatorialMap(
        vertex_count=result.vertices,
        darts=result.darts,
        rotations=result.dart_rotations,
    )


class TestKnownAnswers:
    def test_edgeless_multigraph_with_multiple_vertices_is_disconnected(self) -> None:
        graph = LooplessMultigraph(vertex_count=2, edges=())

        result = check_multigraph_embedding(graph, ((), ()))

        assert result.status == "INVALID"
        assert result.obstruction == "source multigraph is disconnected"

    def test_null_multigraph_is_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            check_multigraph_embedding(LooplessMultigraph(vertex_count=0, edges=()), ())
        assert (
            exc_info.value.errors()[0]["type"]
            == "topology.multigraph_embedding.zero_vertices"
        )

    def test_native_graph_rotation_shape_is_admitted_before_indexing(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            check_orientable_embedding("not-a-graph", ())  # type: ignore[arg-type]
        with pytest.raises(OperationDomainValidationError):
            check_orientable_embedding(_k4(), None)  # type: ignore[arg-type]

    def test_native_rotation_shape_is_admitted_before_indexing(self) -> None:
        graph = LooplessMultigraph(
            vertex_count=2,
            edges=(
                MultigraphEdge(edge_id="e0", left=0, right=1),
                MultigraphEdge(edge_id="e1", left=0, right=1),
            ),
        )
        for rotations in (None, [["e0", "e1"], ["e1", "e0"]]):
            with pytest.raises(OperationDomainValidationError) as exc_info:
                check_multigraph_embedding(graph, rotations)  # type: ignore[arg-type]
            assert (
                exc_info.value.errors()[0]["type"]
                == "topology.multigraph_embedding.rotations_shape"
            )

    def test_multigraph_result_round_trip_rejects_forged_source_ledger(self) -> None:
        graph = LooplessMultigraph(
            vertex_count=2,
            edges=(
                MultigraphEdge(edge_id="e0", left=0, right=1),
                MultigraphEdge(edge_id="e1", left=0, right=1),
            ),
        )
        result = check_multigraph_embedding(graph, (("e0", "e1"), ("e1", "e0")))
        payload = json.loads(result.model_dump_json())
        payload["darts"][0][0] = 1
        with pytest.raises(ValidationError):
            MultigraphEmbeddingResult.model_validate_json(json.dumps(payload))

    def test_k4_planar_is_genus_zero(self) -> None:
        result = check_orientable_embedding(_k4(), _K4_SPHERE)

        assert result.status == "ORIENTABLE_CELLULAR_EMBEDDING"
        assert (result.vertices, result.edges, result.faces) == (4, 6, 4)
        assert result.euler_characteristic == 2
        assert result.genus == 0

    def test_tetrahedron_sphere_is_genus_zero(self) -> None:
        # K4 is the tetrahedron graph; the sphere embedding is genus zero.
        result = check_orientable_embedding(_k4(), _K4_SPHERE_REVERSED)

        assert result.status == "ORIENTABLE_CELLULAR_EMBEDDING"
        assert result.genus == 0

    def test_k7_torus_is_genus_one(self) -> None:
        result = check_orientable_embedding(_k7(), _K7_TORUS)

        assert result.status == "ORIENTABLE_CELLULAR_EMBEDDING"
        assert (result.vertices, result.edges, result.faces) == (7, 21, 14)
        assert result.euler_characteristic == 0
        assert result.genus == 1

    def test_cycle_sphere(self) -> None:
        result = check_orientable_embedding(_cycle(), ((0, 1), (0, 2), (2, 3), (1, 3)))

        assert result.status == "ORIENTABLE_CELLULAR_EMBEDDING"
        assert (result.vertices, result.edges, result.faces) == (4, 4, 2)
        assert result.genus == 0

    def test_tree_has_one_face_with_repeated_bridges(self) -> None:
        result = check_orientable_embedding(_path(), ((0,), (0, 1), (1,)))

        assert (result.vertices, result.edges, result.faces) == (3, 2, 1)
        assert result.genus == 0
        walk = result.face_walks[0]
        assert len(walk) == 4
        # A bridge appears twice in the single face boundary.
        assert sorted(walk) == sorted((0, 1, 2, 3))

    def test_edgeless_single_vertex_is_the_trivial_sphere(self) -> None:
        graph = SimpleUndirectedGraph(vertices=("a",), edges=())
        result = check_orientable_embedding(graph, ((),))

        assert result.status == "ORIENTABLE_CELLULAR_EMBEDDING"
        assert (result.vertices, result.edges, result.faces) == (1, 0, 1)
        assert result.euler_characteristic == 2
        assert result.genus == 0


class TestDefiningInvariants:
    @pytest.mark.parametrize(
        ("graph", "rotations"),
        [
            (_k4(), _K4_SPHERE),
            (_k7(), _K7_TORUS),
            (_cycle(), ((0, 1), (0, 2), (2, 3), (1, 3))),
            (_path(), ((0,), (0, 1), (1,))),
        ],
    )
    def test_face_cycles_partition_every_dart_once(
        self,
        graph: SimpleUndirectedGraph,
        rotations: tuple[tuple[int, ...], ...],
    ) -> None:
        result = check_orientable_embedding(graph, rotations)
        covered = [dart for walk in result.face_walks for dart in walk]

        assert sorted(covered) == list(range(len(result.darts)))
        assert len(set(covered)) == len(covered)
        assert len(result.face_of_dart) == len(result.darts)

    @pytest.mark.parametrize(
        ("graph", "rotations"),
        [(_k4(), _K4_SPHERE), (_k7(), _K7_TORUS)],
    )
    def test_euler_parity_is_nonnegative_and_even(
        self,
        graph: SimpleUndirectedGraph,
        rotations: tuple[tuple[int, ...], ...],
    ) -> None:
        result = check_orientable_embedding(graph, rotations)
        excess = 2 - result.euler_characteristic

        assert excess >= 0
        assert excess % 2 == 0
        assert result.genus == excess // 2

    @pytest.mark.parametrize(
        ("graph", "rotations"),
        [
            (_k4(), _K4_SPHERE),
            (_k7(), _K7_TORUS),
            (_cycle(), ((0, 1), (0, 2), (2, 3), (1, 3))),
        ],
    )
    def test_alpha_sigma_phi_are_permutations(
        self,
        graph: SimpleUndirectedGraph,
        rotations: tuple[tuple[int, ...], ...],
    ) -> None:
        result = check_orientable_embedding(graph, rotations)
        darts = range(len(result.darts))

        assert sorted(result.alpha) == list(darts)
        assert sorted(result.sigma) == list(darts)
        assert sorted(result.phi) == list(darts)
        for dart in darts:
            assert result.alpha[result.alpha[dart]] == dart

    def test_cyclic_shifts_change_presentation_not_embedding(self) -> None:
        shifted = tuple((row[-1], *row[:-1]) for row in _K4_SPHERE)
        original = check_orientable_embedding(_k4(), _K4_SPHERE)
        rotated = check_orientable_embedding(_k4(), shifted)

        assert original.rotations == rotated.rotations
        assert original.genus == rotated.genus == 0

    def test_global_reversal_preserves_genus(self) -> None:
        result = check_orientable_embedding(_k4(), _K4_SPHERE_REVERSED)

        assert result.genus == check_orientable_embedding(_k4(), _K4_SPHERE).genus

    def test_cross_check_against_existing_map_operations(self) -> None:
        result = check_orientable_embedding(_k7(), _K7_TORUS)
        map_ = _combinatorial_map(result)

        euler = euler_characteristic(map_).total
        genus = orientable_genus(map_).total
        assert (euler.vertices, euler.edges, euler.faces) == (
            result.vertices,
            result.edges,
            result.faces,
        )
        assert euler.characteristic == result.euler_characteristic
        assert genus == result.genus


class TestInvalidCandidates:
    def test_omitted_incidence_is_rejected(self) -> None:
        rotations = ((0, 1), (0, 4, 3), (1, 3, 5), (2, 5, 4))
        result = check_orientable_embedding(_k4(), rotations)

        assert result.status == "INVALID_EMBEDDING"
        assert result.obstruction_code == "ROTATION_DEGREE_MISMATCH"

    def test_duplicate_incidence_is_rejected(self) -> None:
        rotations = ((0, 0, 2), (0, 4, 3), (1, 3, 5), (2, 5, 4))
        result = check_orientable_embedding(_k4(), rotations)

        assert result.status == "INVALID_EMBEDDING"
        assert result.obstruction_code == "DUPLICATE_INCIDENCE"

    def test_foreign_incidence_is_rejected(self) -> None:
        rotations = ((0, 1, 3), (0, 4, 3), (1, 3, 5), (2, 5, 4))
        result = check_orientable_embedding(_k4(), rotations)

        assert result.status == "INVALID_EMBEDDING"
        assert result.obstruction_code == "FOREIGN_INCIDENCE"

    def test_out_of_range_edge_index_is_rejected(self) -> None:
        rotations = ((0, 1, 6), (0, 4, 3), (1, 3, 5), (2, 5, 4))
        result = check_orientable_embedding(_k4(), rotations)

        assert result.status == "INVALID_EMBEDDING"
        assert result.obstruction_code == "EDGE_INDEX_OUT_OF_RANGE"

    def test_wrong_row_count_is_rejected(self) -> None:
        result = check_orientable_embedding(_k4(), _K4_SPHERE[:3])

        assert result.status == "INVALID_EMBEDDING"
        assert result.obstruction_code == "ROTATION_ROW_COUNT"

    def test_disconnected_graph_is_rejected(self) -> None:
        graph = SimpleUndirectedGraph(vertices=("a", "b", "c"), edges=(("a", "b"),))
        result = check_orientable_embedding(graph, ((0,), (0,), ()))

        assert result.status == "INVALID_EMBEDDING"
        assert result.obstruction_code == "GRAPH_DISCONNECTED"

    def test_serialized_invalid_result_defers_semantics_to_verifier(self) -> None:
        result = check_orientable_embedding(_k4(), _K4_SPHERE[:3])
        payload = result.model_dump(mode="json")
        payload["obstruction_code"] = "GRAPH_DISCONNECTED"
        payload["obstruction_detail"] = "the supplied graph is not connected"
        forged_obstruction = OrientableEmbeddingCheckResult.model_validate(payload)
        assert not verify_orientable_embedding(forged_obstruction)

        payload = result.model_dump(mode="json")
        payload["rotations"] = _K4_SPHERE
        forged_source = OrientableEmbeddingCheckResult.model_validate(payload)
        assert not verify_orientable_embedding(forged_source)

    def test_corrupted_rotation_changes_the_face_partition(self) -> None:
        rotations = ((0, 2, 1), (0, 4, 3), (1, 3, 5), (2, 5, 4))
        result = check_orientable_embedding(_k4(), rotations)

        assert result.status == "ORIENTABLE_CELLULAR_EMBEDDING"
        assert result.genus != 0


class TestAdmissionAndParity:
    def test_native_and_catalog_paths_agree(self) -> None:
        native = check_orientable_embedding(_k4(), _K4_SPHERE)
        catalog = compute_orientable_embedding_check(
            OrientableEmbeddingCheckRequest(graph=_k4(), rotations=_K4_SPHERE)
        )

        assert catalog == native

    def test_round_trip_and_forgery(self) -> None:
        result = check_orientable_embedding(_k7(), _K7_TORUS)
        restored = OrientableEmbeddingCheckResult.model_validate_json(
            result.model_dump_json()
        )

        assert restored == result
        assert verify_orientable_embedding(restored)
        forged = json.loads(restored.model_dump_json())
        forged["genus"] = 2
        with pytest.raises(ValidationError):
            OrientableEmbeddingCheckResult.model_validate_json(json.dumps(forged))
        forged_rotations = json.loads(restored.model_dump_json())
        forged_rotations["rotations"][0] = [0, 1]
        with pytest.raises(ValidationError):
            OrientableEmbeddingCheckResult.model_validate_json(
                json.dumps(forged_rotations)
            )

        forged_dart = json.loads(restored.model_dump_json())
        forged_dart["darts"][0][0] = 1
        with pytest.raises(ValidationError):
            OrientableEmbeddingCheckResult.model_validate_json(json.dumps(forged_dart))

        forged_faces = json.loads(restored.model_dump_json())
        forged_faces["face_walks"] = [
            [0, 1, 2],
            [3, 4, 5],
            *forged_faces["face_walks"][2:],
        ]
        with pytest.raises(ValidationError):
            OrientableEmbeddingCheckResult.model_validate_json(json.dumps(forged_faces))

    def test_vertex_bound_is_a_resource_boundary(self) -> None:
        size = MAX_EMBEDDING_VERTICES + 1
        vertices = tuple(str(index) for index in range(size))
        edges = cast(
            tuple[tuple[str, str], ...],
            tuple(
                tuple(sorted((vertices[i], vertices[i + 1]))) for i in range(size - 1)
            ),
        )
        graph = SimpleUndirectedGraph(vertices=vertices, edges=edges)

        with pytest.raises(OperationResourceAdmissionError) as exc_info:
            check_orientable_embedding(graph, tuple(() for _ in range(size)))
        assert exc_info.value.errors()[0]["type"] == "topology.embedding.vertex_bound"

    def test_edge_bound_is_a_resource_boundary(self) -> None:
        size = 64
        vertices = tuple(str(index) for index in range(size))
        edges = cast(
            tuple[tuple[str, str], ...],
            tuple(sorted(tuple(sorted(pair)) for pair in combinations(vertices, 2)))[
                : MAX_EMBEDDING_EDGES + 1
            ],
        )
        graph = SimpleUndirectedGraph(vertices=vertices, edges=edges)

        with pytest.raises(OperationResourceAdmissionError) as exc_info:
            check_orientable_embedding(graph, tuple(() for _ in range(size)))
        assert exc_info.value.errors()[0]["type"] == "topology.embedding.edge_bound"

    def test_rotation_entry_bound_is_a_resource_boundary(self) -> None:
        size = 20
        vertices = tuple(str(index) for index in range(size))
        graph = SimpleUndirectedGraph(vertices=vertices, edges=())
        rotations = tuple((0,) * 64 for _ in range(size))

        with pytest.raises(OperationResourceAdmissionError) as exc_info:
            check_orientable_embedding(graph, rotations)
        assert exc_info.value.errors()[0]["type"] == "topology.embedding.rotation_bound"

    def test_empty_graph_is_a_domain_boundary(self) -> None:
        graph = SimpleUndirectedGraph(vertices=(), edges=())

        with pytest.raises(OperationDomainValidationError) as exc_info:
            check_orientable_embedding(graph, ())
        assert exc_info.value.errors()[0]["type"] == "topology.embedding.empty_graph"
