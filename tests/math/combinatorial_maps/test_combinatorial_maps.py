"""Tests for combinatorial-map operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorial_maps import FiniteCombinatorialMap
from jacobian.math.combinatorial_maps._models import (
    ConnectedComponentsRequest,
    DualRequest,
    EulerCharacteristicRequest,
    FacesRequest,
    FacesResult,
    OrientableGenusRequest,
    OrientationReverseRequest,
    OrientationReverseResult,
    VertexFaceIncidenceRequest,
)
from jacobian.math.combinatorial_maps._operations import (
    compute_connected_components,
    compute_dual,
    compute_euler_characteristic,
    compute_faces,
    compute_orientable_genus,
    compute_orientation_reverse,
    compute_vertex_face_incidence,
    verify_faces_result,
    verify_orientation_reverse_result,
)
from jacobian.math.combinatorial_maps._tools import TOOLS
from jacobian.math.combinatorial_maps.operations import (
    face_orbits,
    rotation_successor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _four_cycle() -> FiniteCombinatorialMap:
    """A 4-cycle embedded on the sphere: V=4, E=4, F=2, chi=2, g=0."""
    return FiniteCombinatorialMap(
        vertex_count=4,
        darts=(
            (0, 1, 1),
            (1, 0, 0),
            (1, 2, 3),
            (2, 1, 2),
            (2, 3, 5),
            (3, 2, 4),
            (3, 0, 7),
            (0, 3, 6),
        ),
        rotations=((0, 7), (1, 2), (3, 4), (5, 6)),
    )


def _isolated_vertex() -> FiniteCombinatorialMap:
    """A sphere vertex under the self-loop convention.

    The accepted map category requires every vertex to be incident to at
    least one dart, so an isolated sphere vertex is represented by a single
    self-loop: V=1, E=1, F=2, chi=2, g=0.
    """
    return FiniteCombinatorialMap(
        vertex_count=1,
        darts=((0, 0, 1), (0, 0, 0)),
        rotations=((0, 1),),
    )


def _theta_graph() -> FiniteCombinatorialMap:
    """Three parallel edges between two vertices: V=2, E=3, F=3, chi=2.

    The map is deliberately not reverse-symmetric: reversing the local
    rotations permutes the faces, so a face bijection matched by dart-set
    equality would fail.
    """
    return FiniteCombinatorialMap(
        vertex_count=2,
        darts=(
            (0, 1, 1),
            (1, 0, 0),
            (0, 1, 3),
            (1, 0, 2),
            (0, 1, 5),
            (1, 0, 4),
        ),
        rotations=((0, 2, 4), (1, 5, 3)),
    )


def _torus() -> FiniteCombinatorialMap:
    """Standard minimal torus cellulation: 1 vertex, 2 loops, 1 face.

    V=1, E=2, F=1, chi=0, genus=1.
    """
    return FiniteCombinatorialMap(
        vertex_count=1,
        darts=(
            (0, 0, 1),
            (0, 0, 0),
            (0, 0, 3),
            (0, 0, 2),
        ),
        rotations=((0, 2, 1, 3),),
    )


def _tree() -> FiniteCombinatorialMap:
    """A 3-vertex path (a tree): V=3, E=2, F=1, chi=2, g=0.

    Vertex 0 - vertex 1 - vertex 2.
    """
    return FiniteCombinatorialMap(
        vertex_count=3,
        darts=(
            (0, 1, 1),
            (1, 0, 0),
            (1, 2, 3),
            (2, 1, 2),
        ),
        rotations=((0,), (1, 2), (3,)),
    )


def _disconnected() -> FiniteCombinatorialMap:
    """Two disjoint copies of the isolated vertex-with-loop convention.

    Each component is a sphere (g=0), so total genus = 0 and total chi = 4.
    """
    return FiniteCombinatorialMap(
        vertex_count=2,
        darts=(
            (0, 0, 1),
            (0, 0, 0),
            (1, 1, 3),
            (1, 1, 2),
        ),
        rotations=((0, 1), (2, 3)),
    )


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_catalog_contains_only_audited_agent_outcomes() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "combinatorial_map.faces.compute",
        "combinatorial_map.euler_characteristic.compute",
        "combinatorial_map.orientable_genus.compute",
        "combinatorial_map.orientation_reverse.compute",
        "combinatorial_map.connected_components.compute",
        "combinatorial_map.dual.compute",
        "combinatorial_map.vertex_face_incidence.compute",
    }


# ---------------------------------------------------------------------------
# Faces
# ---------------------------------------------------------------------------


class TestFaces:
    def test_four_cycle_has_two_faces(self) -> None:
        result = compute_faces(FacesRequest(map=_four_cycle()))
        assert len(result.face_walks) == 2
        # Each face is a 4-walk.
        assert all(len(walk) == 4 for walk in result.face_walks)
        # Every dart appears exactly once.
        seen: set[int] = set()
        for walk in result.face_walks:
            for dart in walk:
                assert dart not in seen
                seen.add(dart)
        assert seen == set(range(8))

    def test_face_of_dart_assigns_every_dart(self) -> None:
        m = _four_cycle()
        result = compute_faces(FacesRequest(map=m))
        # face_of_dart is a tuple indexed by dart; every dart has a face.
        assert len(result.face_of_dart) == 8
        assert set(result.face_of_dart) == {0, 1}

    def test_torus_has_one_face(self) -> None:
        result = compute_faces(FacesRequest(map=_torus()))
        assert len(result.face_walks) == 1
        assert len(result.face_walks[0]) == 4

    def test_tree_has_one_face(self) -> None:
        result = compute_faces(FacesRequest(map=_tree()))
        assert len(result.face_walks) == 1

    def test_disconnected_has_four_faces(self) -> None:
        result = compute_faces(FacesRequest(map=_disconnected()))
        assert len(result.face_walks) == 4

    def test_native_kernel_validates_cross_field_inputs(self) -> None:
        with pytest.raises(ValueError, match="vertex_count"):
            FiniteCombinatorialMap(
                vertex_count=3,
                darts=((0, 1, 1), (1, 0, 0)),
                rotations=(
                    (0,),
                    (1,),
                ),
            )

    def test_external_face_claim_uses_explicit_verifier(self) -> None:
        result = compute_faces(FacesRequest(map=_four_cycle()))
        claimed = FacesResult.model_validate(
            {
                **result.model_dump(mode="json"),
                "face_of_dart": [1 - face for face in result.face_of_dart],
            }
        )
        assert not verify_faces_result(claimed)


# ---------------------------------------------------------------------------
# Euler characteristic
# ---------------------------------------------------------------------------


class TestEulerCharacteristic:
    def test_four_cycle_sphere(self) -> None:
        result = compute_euler_characteristic(
            EulerCharacteristicRequest(map=_four_cycle())
        )
        assert result.total == {"V": 4, "E": 4, "F": 2, "chi": 2}
        assert len(result.per_component) == 1
        assert result.per_component[0] == {"V": 4, "E": 4, "F": 2, "chi": 2}

    def test_torus(self) -> None:
        result = compute_euler_characteristic(EulerCharacteristicRequest(map=_torus()))
        assert result.total == {"V": 1, "E": 2, "F": 1, "chi": 0}

    def test_tree(self) -> None:
        result = compute_euler_characteristic(EulerCharacteristicRequest(map=_tree()))
        assert result.total == {"V": 3, "E": 2, "F": 1, "chi": 2}

    def test_disconnected_sums(self) -> None:
        result = compute_euler_characteristic(
            EulerCharacteristicRequest(map=_disconnected())
        )
        assert len(result.per_component) == 2
        # Each component: V=1, E=1, F=2, chi=2. Total chi = 4.
        assert result.total == {"V": 2, "E": 2, "F": 4, "chi": 4}

    def test_isolated_vertex_sphere(self) -> None:
        result = compute_euler_characteristic(
            EulerCharacteristicRequest(map=_isolated_vertex())
        )
        assert result.total == {"V": 1, "E": 1, "F": 2, "chi": 2}


# ---------------------------------------------------------------------------
# Orientable genus
# ---------------------------------------------------------------------------


class TestOrientableGenus:
    def test_four_cycle_genus_zero(self) -> None:
        result = compute_orientable_genus(OrientableGenusRequest(map=_four_cycle()))
        assert result.per_component == (0,)
        assert result.total == 0

    def test_torus_genus_one(self) -> None:
        result = compute_orientable_genus(OrientableGenusRequest(map=_torus()))
        assert result.per_component == (1,)
        assert result.total == 1

    def test_tree_genus_zero(self) -> None:
        result = compute_orientable_genus(OrientableGenusRequest(map=_tree()))
        assert result.per_component == (0,)
        assert result.total == 0

    def test_disconnected_sums(self) -> None:
        result = compute_orientable_genus(OrientableGenusRequest(map=_disconnected()))
        assert result.per_component == (0, 0)
        assert result.total == 0


# ---------------------------------------------------------------------------
# Orientation reversal
# ---------------------------------------------------------------------------


class TestOrientationReverse:
    def test_four_cycle_double_reverse_is_identity(self) -> None:
        m = _four_cycle()
        result = compute_orientation_reverse(OrientationReverseRequest(map=m))
        reversed_map = result.reversed_map
        assert reversed_map.vertex_count == m.vertex_count
        # Apply again.
        inner = compute_orientation_reverse(OrientationReverseRequest(map=reversed_map))
        assert inner.reversed_map == m

    def test_orientation_preserves_euler(self) -> None:
        m = _four_cycle()
        reversed_map = compute_orientation_reverse(
            OrientationReverseRequest(map=m)
        ).reversed_map
        original = compute_euler_characteristic(EulerCharacteristicRequest(map=m))
        after = compute_euler_characteristic(
            EulerCharacteristicRequest(map=reversed_map)
        )
        assert original.total == after.total

    def test_orientation_preserves_genus(self) -> None:
        m = _torus()
        reversed_map = compute_orientation_reverse(
            OrientationReverseRequest(map=m)
        ).reversed_map
        original = compute_orientable_genus(OrientableGenusRequest(map=m))
        after = compute_orientable_genus(OrientableGenusRequest(map=reversed_map))
        assert original.total == after.total

    def test_face_bijection_size(self) -> None:
        m = _four_cycle()
        result = compute_orientation_reverse(OrientationReverseRequest(map=m))
        assert len(result.face_bijection) == 2
        assert set(result.face_bijection.values()) == {0, 1}

    def test_theta_graph_reversal_is_face_bijection(self) -> None:
        m = _theta_graph()
        result = compute_orientation_reverse(OrientationReverseRequest(map=m))
        assert set(result.face_bijection) == {0, 1, 2}
        assert set(result.face_bijection.values()) == {0, 1, 2}
        # Old face O corresponds to the new face containing the reversed
        # darts of O (the reversal image under the face-permutation
        # conjugation), not to a new orbit with the same dart set.
        old_walks = compute_faces(FacesRequest(map=m)).face_walks
        new_walks = compute_faces(FacesRequest(map=result.reversed_map)).face_walks
        for old_face, new_face in result.face_bijection.items():
            reversal_image = frozenset(m.darts[d][2] for d in old_walks[old_face])
            assert reversal_image == frozenset(new_walks[new_face])
            assert frozenset(old_walks[old_face]) != frozenset(new_walks[new_face])

    def test_theta_graph_double_reverse_is_identity(self) -> None:
        m = _theta_graph()
        result = compute_orientation_reverse(OrientationReverseRequest(map=m))
        inner = compute_orientation_reverse(
            OrientationReverseRequest(map=result.reversed_map)
        )
        assert inner.reversed_map == m

    def test_external_orientation_claim_uses_explicit_verifier(self) -> None:
        result = compute_orientation_reverse(
            OrientationReverseRequest(map=_four_cycle())
        )
        claimed = OrientationReverseResult.model_validate(
            {
                **result.model_dump(mode="json"),
                "face_bijection": {"0": 0, "1": 1},
            }
        )
        assert not verify_orientation_reverse_result(claimed)


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------


class TestConnectedComponents:
    def test_four_cycle_single_component(self) -> None:
        result = compute_connected_components(
            ConnectedComponentsRequest(map=_four_cycle())
        )
        assert set(result.vertex_component) == {0}
        assert set(result.dart_component) == {0}
        assert set(result.face_component) == {0}

    def test_disconnected_two_components(self) -> None:
        result = compute_connected_components(
            ConnectedComponentsRequest(map=_disconnected())
        )
        assert set(result.vertex_component) == {0, 1}
        assert result.vertex_component[0] != result.vertex_component[1]


# ---------------------------------------------------------------------------
# Dual
# ---------------------------------------------------------------------------


class TestDual:
    def test_dual_vertex_count_is_face_count(self) -> None:
        m = _four_cycle()
        result = compute_dual(DualRequest(map=m))
        assert result.dual.vertex_count == 2

    def test_dual_of_dual_recovers_primal_vertex_count(self) -> None:
        m = _four_cycle()
        dual = compute_dual(DualRequest(map=m)).dual
        dual_of_dual = compute_dual(DualRequest(map=dual)).dual
        assert dual_of_dual.vertex_count == m.vertex_count

    def test_primal_to_dual_bijection_is_identity(self) -> None:
        m = _four_cycle()
        result = compute_dual(DualRequest(map=m))
        assert result.primal_to_dual == {i: i for i in range(len(m.darts))}

    def test_torus_dual_is_one_vertex(self) -> None:
        m = _torus()
        result = compute_dual(DualRequest(map=m))
        assert result.dual.vertex_count == 1

    def test_theta_graph_dual_faces_match_primal_vertices(self) -> None:
        m = _theta_graph()
        result = compute_dual(DualRequest(map=m))
        # The dual has one vertex per primal face and one face per primal
        # vertex (V=2, E=3, F=3 in the primal).
        assert result.dual.vertex_count == 3
        assert len(compute_faces(FacesRequest(map=result.dual)).face_walks) == 2
        assert len(result.dual.darts) == len(m.darts)

    def test_dual_preserves_euler_characteristic(self) -> None:
        for m in (_four_cycle(), _torus(), _tree(), _theta_graph(), _disconnected()):
            primal = compute_euler_characteristic(EulerCharacteristicRequest(map=m))
            dual = compute_dual(DualRequest(map=m)).dual
            dual_euler = compute_euler_characteristic(
                EulerCharacteristicRequest(map=dual)
            )
            assert dual_euler.total["chi"] == primal.total["chi"]
            # Duality swaps vertices and faces and preserves edges.
            assert dual_euler.total["V"] == primal.total["F"]
            assert dual_euler.total["E"] == primal.total["E"]
            assert dual_euler.total["F"] == primal.total["V"]

    def test_dual_of_dual_recovers_primal_structure(self) -> None:
        m = _theta_graph()
        dual = compute_dual(DualRequest(map=m)).dual
        dual_of_dual = compute_dual(DualRequest(map=dual)).dual
        assert dual_of_dual.vertex_count == m.vertex_count
        primal_faces = compute_faces(FacesRequest(map=m)).face_walks
        transported_faces = compute_faces(FacesRequest(map=dual_of_dual)).face_walks
        assert sorted(map(len, transported_faces)) == sorted(map(len, primal_faces))


# ---------------------------------------------------------------------------
# Vertex-face incidence
# ---------------------------------------------------------------------------


class TestVertexFaceIncidence:
    def test_four_cycle_incidence(self) -> None:
        m = _four_cycle()
        result = compute_vertex_face_incidence(VertexFaceIncidenceRequest(map=m))
        # Each of 4 vertices is incident to both faces.
        for vertex in range(4):
            assert set(result.boolean_incidence[vertex]) == {0, 1}
        # Each vertex appears once per face -> 8 multiplicity entries.
        assert sum(len(row) for row in result.multiplicity.values()) == 8
        for row in result.multiplicity.values():
            for value in row.values():
                assert value == 1

    def test_torus_incidence(self) -> None:
        m = _torus()
        result = compute_vertex_face_incidence(VertexFaceIncidenceRequest(map=m))
        assert set(result.boolean_incidence[0]) == {0}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_wrong_rotation_length_rejected(self) -> None:
        # Vertex 0 has one outgoing dart (dart 0) but its rotation lists two.
        with pytest.raises(ValidationError) as exc_info:
            FiniteCombinatorialMap(
                vertex_count=2,
                darts=((0, 1, 1), (1, 0, 0)),
                rotations=((0, 1), (1,)),
            )
        assert exc_info.value.errors()[0]["type"] == "combinatorial_map.rotation_length"

    def test_reverse_not_involution_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FiniteCombinatorialMap(
                vertex_count=2,
                darts=((0, 1, 1), (1, 0, 1)),
                rotations=((0,), (1,)),
            )
        assert (
            exc_info.value.errors()[0]["type"]
            == "combinatorial_map.reverse_not_involution"
        )

    def test_fixed_point_reverse_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FiniteCombinatorialMap(
                vertex_count=1,
                darts=((0, 0, 0),),
                rotations=((0,),),
            )
        assert (
            exc_info.value.errors()[0]["type"]
            == "combinatorial_map.reverse_fixed_point"
        )

    def test_foreign_dart_in_rotation_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FiniteCombinatorialMap(
                vertex_count=2,
                darts=((0, 1, 1), (1, 0, 0)),
                rotations=((1,), (0,)),
            )
        assert (
            exc_info.value.errors()[0]["type"]
            == "combinatorial_map.foreign_rotation_dart"
        )

    def test_vertex_count_too_large_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FiniteCombinatorialMap(
                vertex_count=10000,
                darts=((0, 1, 1), (1, 0, 0)),
                rotations=((0,), (1,)),
            )

    def test_negative_dart_index_in_rotation_rejected(self) -> None:
        # Dart -1 aliases the last dart through Python negative indexing; it
        # must be rejected instead of silently accepted.
        with pytest.raises(ValidationError) as exc_info:
            FiniteCombinatorialMap(
                vertex_count=1,
                darts=((0, 0, 1), (0, 0, 0)),
                rotations=((-1, 0),),
            )
        assert (
            exc_info.value.errors()[0]["type"]
            == "combinatorial_map.rotation_dart_out_of_range"
        )

    def test_out_of_range_dart_index_rejected_without_index_error(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FiniteCombinatorialMap(
                vertex_count=1,
                darts=((0, 0, 1), (0, 0, 0)),
                rotations=((0, 5),),
            )
        assert (
            exc_info.value.errors()[0]["type"]
            == "combinatorial_map.rotation_dart_out_of_range"
        )

    def test_isolated_vertex_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FiniteCombinatorialMap(
                vertex_count=2,
                darts=((0, 0, 1), (0, 0, 0)),
                rotations=((0, 1), ()),
            )
        assert (
            exc_info.value.errors()[0]["type"]
            == "combinatorial_map.vertex_without_dart"
        )

    def test_long_facial_walk_rejected_to_keep_dual_bounded(self) -> None:
        # A 65-vertex cycle has two facial walks of length 65, which would
        # overflow the dual rotation budget; the map is rejected up front.
        size = 65
        darts: list[tuple[int, int, int]] = []
        rotations: list[tuple[int, ...]] = []
        for i in range(size):
            j = (i + 1) % size
            darts.append((i, j, 2 * i + 1))
            darts.append((j, i, 2 * i))
            rotations.append((2 * i, 2 * ((i - 1) % size) + 1))
        with pytest.raises(ValidationError) as exc_info:
            FiniteCombinatorialMap(
                vertex_count=size,
                darts=tuple(darts),
                rotations=tuple(rotations),
            )
        assert (
            exc_info.value.errors()[0]["type"]
            == "combinatorial_map.facial_walk_too_long"
        )


# ---------------------------------------------------------------------------
# Rotation success and face orbits
# ---------------------------------------------------------------------------


def test_rotation_successor_cyclic() -> None:
    m = _four_cycle()
    # At vertex 0, rotation is (0, 7). Successor of 0 is 7; of 7 is 0.
    assert rotation_successor(m, 0) == 7
    assert rotation_successor(m, 7) == 0


def test_face_orbits_covers_all_darts() -> None:
    m = _four_cycle()
    walks, face_of_dart, _, _ = face_orbits(m)
    all_darts: list[int] = []
    for walk in walks:
        all_darts.extend(walk)
    assert sorted(all_darts) == list(range(len(m.darts)))
    assert set(face_of_dart.keys()) == set(range(len(m.darts)))
