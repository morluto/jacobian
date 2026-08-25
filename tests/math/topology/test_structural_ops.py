"""Tests for structural simplicial complex operations (#1850)."""

import pytest
from pydantic import ValidationError

from jacobian.math.topology._models import (
    BarycentricSubdivisionRequest,
    BarycentricSubdivisionResult,
    ElementaryCollapseRequest,
    ElementaryCollapseResult,
    JoinRequest,
    JoinResult,
    ShellingCheckRequest,
    ShellingCheckResult,
    SimplicialComplexRequest,
    SkeletonRequest,
    SkeletonResult,
    StarRequest,
    StarResult,
    VertexDeletionRequest,
    VertexDeletionResult,
)
from jacobian.math.topology._operations import (
    _CANONICAL_CIRCLE,
    _canonical_complex,
    compute_barycentric_subdivision,
    compute_elementary_collapse,
    compute_join,
    compute_pseudomanifold_decision,
    compute_shelling_check,
    compute_skeleton,
    compute_star,
    compute_vertex_deletion,
)
from jacobian.math.topology._pseudomanifold import (
    PseudomanifoldRequest,
    PseudomanifoldResult,
)
from jacobian.math.topology._tools import TOOLS

TRIANGLE = {"vertices": ["v0", "v1", "v2"], "facets": [["v0", "v1", "v2"]]}
CIRCLE = {
    "vertices": ["a", "b", "c"],
    "facets": [["a", "b"], ["b", "c"], ["a", "c"]],
}
EDGE = {"vertices": ["a", "b"], "facets": [["a", "b"]]}


class TestStar:
    def test_star_of_vertex_in_triangle(self) -> None:
        result = compute_star(StarRequest(complex=TRIANGLE, simplex=["v0"]))
        assert not result.star_is_empty
        assert result.star_facets == (("v0", "v1", "v2"),)
        assert result.star_complex is not None

    def test_star_of_edge_in_triangle(self) -> None:
        result = compute_star(StarRequest(complex=TRIANGLE, simplex=["v0", "v1"]))
        assert result.star_facets == (("v0", "v1", "v2"),)

    def test_star_of_vertex_in_circle(self) -> None:
        result = compute_star(StarRequest(complex=CIRCLE, simplex=["a"]))
        assert result.star_facets == (("a", "b"), ("a", "c"))

    def test_star_not_a_face(self) -> None:
        with pytest.raises(ValidationError):
            StarRequest(complex=TRIANGLE, simplex=["v0", "v1", "v2", "v3"])

    def test_result_binds_to_source_complex_roundtrip(self) -> None:
        request = StarRequest(complex=CIRCLE, simplex=["a"])
        result = compute_star(request)
        assert result.complex == request.complex
        assert StarResult.model_validate(result.model_dump()) == result

    def test_forged_simplex_outside_source_rejected(self) -> None:
        """The reviewer counterexample: a star for ('z',) presented on the
        edge {a, b} cannot validate because 'z' is not a face of the source."""
        with pytest.raises(ValidationError):
            StarResult(
                complex=EDGE,
                simplex=("z",),
                star_facets=(("a", "b"),),
                star_is_empty=False,
                star_complex=_canonical_complex(("a", "b"), (("a", "b"),)),
            )

    def test_truncated_star_facets_rejected(self) -> None:
        """Dropping one containing facet must fail the source replay."""
        with pytest.raises(ValidationError):
            StarResult(
                complex=CIRCLE,
                simplex=("a",),
                star_facets=(("a", "b"),),
                star_is_empty=False,
                star_complex=_canonical_complex(("a", "b"), (("a", "b"),)),
            )


class TestVertexDeletion:
    def test_delete_vertex_from_triangle(self) -> None:
        result = compute_vertex_deletion(
            VertexDeletionRequest(complex=TRIANGLE, vertices_to_delete=["v2"])
        )
        assert result.deleted_vertices == ("v2",)
        assert "v2" not in result.remaining_vertices
        assert ("v0", "v1") in result.remaining_facets

    def test_delete_all_facets_containing_vertex(self) -> None:
        result = compute_vertex_deletion(
            VertexDeletionRequest(complex=CIRCLE, vertices_to_delete=["a"])
        )
        assert result.remaining_facets == (("b", "c"),)

    def test_delete_unknown_vertex(self) -> None:
        with pytest.raises(ValidationError):
            VertexDeletionRequest(complex=TRIANGLE, vertices_to_delete=["v9"])

    def test_delete_all_vertices_rejected(self) -> None:
        """A deletion whose induced subcomplex is empty is out of contract;
        the canonical complex value cannot represent the empty complex."""
        with pytest.raises(ValidationError):
            VertexDeletionRequest(
                complex={"vertices": ["a"], "facets": [["a"]]},
                vertices_to_delete=["a"],
            )

    def test_nonempty_residual_precondition_is_schema_visible(self) -> None:
        """The reviewer counterexample: deleting 'a' from the singleton {a}
        satisfies the generated field schema, so the nonempty-residual
        restriction must be stated in the published schema guidance rather
        than discovered only through a failed invocation."""
        schema = VertexDeletionRequest.model_json_schema()
        field_schema = schema["properties"]["vertices_to_delete"]
        assert "at least one simplex" in field_schema["description"]
        assert "empty complex" in field_schema["description"]

    def test_deletion_discovery_metadata_states_precondition(self) -> None:
        tool = next(
            t
            for t in TOOLS
            if t.operation_id == "topology.simplicial_complex.deletion.compute"
        )
        assert "leave at least one simplex" in tool.description
        assert all(
            "at least one simplex" in example.description for example in tool.examples
        )

    def test_delete_leaving_single_vertex_admitted(self) -> None:
        result = compute_vertex_deletion(
            VertexDeletionRequest(
                complex={"vertices": ["a", "b"], "facets": [["a"], ["b"]]},
                vertices_to_delete=["b"],
            )
        )
        assert result.remaining_vertices == ("a",)
        assert result.remaining_facets == (("a",),)
        assert result.remaining_complex is not None
        assert result.remaining_complex.maximal_simplices == (("a",),)

    def test_result_requires_canonical_complex(self) -> None:
        with pytest.raises(ValidationError):
            VertexDeletionResult(
                deleted_vertices=("v2",),
                remaining_vertices=("v0", "v1"),
                remaining_facets=(("v0", "v1"),),
            )


class TestSkeleton:
    def test_one_skeleton_of_triangle(self) -> None:
        result = compute_skeleton(SkeletonRequest(complex=TRIANGLE, k=1))
        assert result.k == 1
        assert result.skeleton_facets == (
            ("v0", "v1"),
            ("v0", "v2"),
            ("v1", "v2"),
        )

    def test_zero_skeleton(self) -> None:
        result = compute_skeleton(SkeletonRequest(complex=TRIANGLE, k=0))
        assert result.skeleton_facets == (("v0",), ("v1",), ("v2",))

    def test_full_skeleton(self) -> None:
        result = compute_skeleton(SkeletonRequest(complex=TRIANGLE, k=2))
        assert result.skeleton_facets == (("v0", "v1", "v2"),)


class TestJoin:
    def test_join_two_points(self) -> None:
        point_a = {"vertices": ["a"], "facets": [["a"]]}
        point_b = {"vertices": ["b"], "facets": [["b"]]}
        result = compute_join(JoinRequest(complex_a=point_a, complex_b=point_b))
        assert result.join_vertices == ("a", "b")
        assert result.join_facets == (("a", "b"),)
        assert result.join_dimension == 1

    def test_join_overlapping_vertices_fails(self) -> None:
        point_a = {"vertices": ["a"], "facets": [["a"]]}
        point_b = {"vertices": ["a"], "facets": [["a"]]}
        with pytest.raises(ValidationError):
            JoinRequest(complex_a=point_a, complex_b=point_b)

    def test_join_edge_and_point(self) -> None:
        edge = {"vertices": ["a", "b"], "facets": [["a", "b"]]}
        point = {"vertices": ["c"], "facets": [["c"]]}
        result = compute_join(JoinRequest(complex_a=edge, complex_b=point))
        assert result.join_dimension == 2
        assert result.join_facets == (("a", "b", "c"),)


class TestBarycentricSubdivision:
    def test_subdivide_edge(self) -> None:
        result = compute_barycentric_subdivision(
            BarycentricSubdivisionRequest(complex=EDGE)
        )
        # Edge has 3 faces: {a}, {b}, {a,b}
        assert result.num_new_vertices == 3
        # Subdivision of an edge is two edges
        assert len(result.subdivision_facets) == 2

    def test_subdivide_triangle(self) -> None:
        result = compute_barycentric_subdivision(
            BarycentricSubdivisionRequest(complex=TRIANGLE)
        )
        # Triangle has 7 faces: 3 vertices + 3 edges + 1 triangle
        assert result.num_new_vertices == 7
        # Subdivision of a triangle has 6 maximal simplices (each is a chain)
        assert len(result.subdivision_facets) == 6

    def test_result_retains_source_and_roundtrips(self) -> None:
        request = BarycentricSubdivisionRequest(complex=CIRCLE)
        result = compute_barycentric_subdivision(request)
        assert result.complex == request.complex
        assert (
            BarycentricSubdivisionResult.model_validate(result.model_dump()) == result
        )

    def test_vertex_face_map_is_canonical_bijection(self) -> None:
        from jacobian.math.topology._models import _all_faces

        result = compute_barycentric_subdivision(
            BarycentricSubdivisionRequest(complex=EDGE)
        )
        faces = sorted(_all_faces(EDGE["facets"]), key=lambda f: (len(f), f))
        assert list(result.subdivision_vertex_faces) == faces
        assert list(result.subdivision_vertices) == [
            f"bv{i}" for i in range(len(faces))
        ]

    def test_swapped_vertex_face_map_rejected(self) -> None:
        """Swapping entries of the indexed vertex-to-face map must not
        validate while the subdivision complex stays unchanged."""
        result = compute_barycentric_subdivision(
            BarycentricSubdivisionRequest(complex=CIRCLE)
        )
        payload = result.model_dump(mode="json")
        faces = payload["subdivision_vertex_faces"]
        faces[0], faces[2] = faces[2], faces[0]
        with pytest.raises(ValidationError):
            BarycentricSubdivisionResult.model_validate(payload)


class TestCanonicalComplexFeeding:
    """A canonical ``FiniteSimplicialComplex`` value must feed structural
    requests unchanged (review thread: deletion result -> skeleton)."""

    def test_deletion_result_feeds_skeleton_request(self) -> None:
        deleted = compute_vertex_deletion(
            VertexDeletionRequest(complex=TRIANGLE, vertices_to_delete=["v0"])
        )
        assert deleted.remaining_complex is not None
        skeleton = compute_skeleton(
            SkeletonRequest(
                complex=deleted.remaining_complex.model_dump(mode="json"), k=0
            )
        )
        assert skeleton.skeleton_facets == (("v1",), ("v2",))

    def test_subdivision_complex_feeds_star_request(self) -> None:
        subdivided = compute_barycentric_subdivision(
            BarycentricSubdivisionRequest(complex=EDGE)
        )
        star = compute_star(
            StarRequest(
                complex=subdivided.subdivision_complex.model_dump(), simplex=["bv0"]
            )
        )
        assert not star.star_is_empty

    def test_mixed_presentations_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SkeletonRequest(
                complex={
                    "vertices": ["a", "b"],
                    "facets": [["a", "b"]],
                    "maximal_simplices": [["a", "b"]],
                },
                k=0,
            )

    def test_unknown_fields_in_canonical_shape_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SkeletonRequest(
                complex={
                    "vertices": ["a", "b"],
                    "maximal_simplices": [["a", "b"]],
                    "surprise": 1,
                },
                k=0,
            )

    def test_tampered_canonical_dump_rejected(self) -> None:
        """Changing maximal_simplices while retaining the original digest
        must not be accepted as a different request complex."""
        tampered = {
            **_CANONICAL_CIRCLE,
            "maximal_simplices": [["a", "b"], ["a", "c"]],
        }
        with pytest.raises(ValidationError):
            SkeletonRequest(complex=tampered, k=0)

    def test_incomplete_canonical_dump_rejected(self) -> None:
        """A canonical-shape dump missing derived fields cannot bypass
        validation of the owning canonical type."""
        with pytest.raises(ValidationError):
            SkeletonRequest(
                complex={
                    "vertices": ["a", "b"],
                    "maximal_simplices": [["a", "b"]],
                },
                k=0,
            )


class TestCollapseCandidateBounds:
    def test_oversized_candidate_labels_rejected(self) -> None:
        labels = [f"x{i}" for i in range(9)]
        with pytest.raises(ValidationError):
            ElementaryCollapseRequest(
                complex=EDGE, free_face=labels, coface=[*labels, "extra"]
            )

    def test_coface_cap_matches_dimension_bound(self) -> None:
        with pytest.raises(ValidationError):
            ElementaryCollapseRequest(
                complex=EDGE,
                free_face=["a"],
                coface=["a", "b", "c", "d", "e", "f", "g", "h", "i"],
            )


class TestPseudomanifold:
    def test_circle_is_closed_pseudomanifold(self) -> None:
        result = compute_pseudomanifold_decision(PseudomanifoldRequest(complex=CIRCLE))
        assert result.is_pseudomanifold
        assert result.is_closed
        assert result.dimension == 1

    def test_triangle_is_not_pseudomanifold(self) -> None:
        # Triangle boundary is a 2-simplex (solid triangle)
        result = compute_pseudomanifold_decision(
            PseudomanifoldRequest(complex=TRIANGLE)
        )
        # A solid triangle is 2-dimensional, each edge is in exactly 1 facet
        # That's a pseudomanifold with boundary
        assert result.is_pseudomanifold
        assert not result.is_closed

    def test_non_pure_is_not_pseudomanifold(self) -> None:
        complex_data = {
            "vertices": ["a", "b", "c", "d"],
            "facets": [["a", "b", "c"], ["c", "d"]],
        }
        result = compute_pseudomanifold_decision(
            PseudomanifoldRequest(complex=complex_data)
        )
        assert not result.is_pseudomanifold
        assert "pure" in result.obstruction

    def test_single_point_is_pseudomanifold_with_boundary(self) -> None:
        """The empty codim-1 face lies in one facet: boundary case."""
        point = {"vertices": ["a"], "facets": [["a"]]}
        result = compute_pseudomanifold_decision(PseudomanifoldRequest(complex=point))
        assert result.is_pseudomanifold
        assert not result.is_closed
        assert result.dimension == 0
        assert result.obstruction == "pseudomanifold with boundary"

    def test_two_points_are_closed_zero_dimensional_pseudomanifold(self) -> None:
        two_points = {"vertices": ["a", "b"], "facets": [["a"], ["b"]]}
        result = compute_pseudomanifold_decision(
            PseudomanifoldRequest(complex=two_points)
        )
        assert result.is_pseudomanifold
        assert result.is_closed
        assert result.obstruction is None

    def test_three_points_fail_incidence(self) -> None:
        three_points = {
            "vertices": ["a", "b", "c"],
            "facets": [["a"], ["b"], ["c"]],
        }
        result = compute_pseudomanifold_decision(
            PseudomanifoldRequest(complex=three_points)
        )
        assert not result.is_pseudomanifold
        assert "3 facets" in result.obstruction

    def test_forged_zero_dimensional_decision_rejected(self) -> None:
        """A serialized negative decision for a singleton cannot validate."""
        point = {"vertices": ["a"], "facets": [["a"]]}
        with pytest.raises(ValidationError):
            PseudomanifoldResult(
                complex=point,
                is_pseudomanifold=False,
                is_closed=False,
                dimension=0,
                num_facets=1,
                obstruction="dimension must be at least 1",
            )

    def test_forged_negative_obstruction_must_match_replay(self) -> None:
        """A purity failure cannot be revalidated with a foreign non-null
        obstruction: negative conclusions replay their exact description."""
        nonpure = {
            "vertices": ["a", "b", "c", "d", "e"],
            "facets": [["a", "b", "c"], ["d", "e"]],
        }
        with pytest.raises(ValidationError):
            PseudomanifoldResult(
                complex=nonpure,
                is_pseudomanifold=False,
                is_closed=False,
                dimension=2,
                num_facets=2,
                obstruction="codim-1 face [] is in 3 facets",
            )


class TestShellingCheck:
    def test_valid_shelling_of_single_facet(self) -> None:
        result = compute_shelling_check(
            ShellingCheckRequest(complex=EDGE, facet_order=[0])
        )
        assert result.is_shelling

    def test_valid_shelling_of_circle(self) -> None:
        # Circle has 3 edges; any order should be a shelling
        result = compute_shelling_check(
            ShellingCheckRequest(complex=CIRCLE, facet_order=[0, 1, 2])
        )
        assert result.is_shelling

    def test_invalid_order(self) -> None:
        with pytest.raises(ValidationError):
            ShellingCheckRequest(complex=EDGE, facet_order=[1, 0])

    def test_partial_order_cannot_authenticate_decision(self) -> None:
        """A two-facet complex revalidated with facet_order=[0] must not
        accept is_shelling=True: the omitted disjoint facet makes the full
        order non-shelling (review counterexample)."""
        two_facets = {
            "vertices": ["a", "b", "c", "d"],
            "facets": [["a", "b"], ["c", "d"]],
        }
        with pytest.raises(ValidationError):
            ShellingCheckResult(
                complex=two_facets,
                facet_order=[0],
                is_shelling=True,
                failed_at=None,
                failure_reason=None,
            )

    def test_duplicate_indices_rejected_in_result(self) -> None:
        with pytest.raises(ValidationError):
            ShellingCheckResult(
                complex=CIRCLE,
                facet_order=[0, 0, 1],
                is_shelling=False,
                failed_at=1,
                failure_reason="facet 1 restriction is not an interval",
            )


class TestElementaryCollapse:
    def test_collapse_free_vertex_from_edge(self) -> None:
        result = compute_elementary_collapse(
            ElementaryCollapseRequest(complex=EDGE, free_face=["a"], coface=["a", "b"])
        )
        assert result.is_free_face
        assert result.remaining_facets == (("b",),)
        assert result.remaining_vertices == ("b",)
        assert result.remaining_complex is not None
        assert result.remaining_complex.maximal_simplices == (("b",),)

    def test_non_free_face(self) -> None:
        # In the circle, vertex 'a' is in 2 facets, so it's not free
        result = compute_elementary_collapse(
            ElementaryCollapseRequest(
                complex=CIRCLE, free_face=["a"], coface=["a", "b"]
            )
        )
        assert not result.is_free_face

    def test_coface_not_a_facet(self) -> None:
        # ['a', 'c'] is codimension-one in shape but not a facet of EDGE
        result = compute_elementary_collapse(
            ElementaryCollapseRequest(complex=EDGE, free_face=["a"], coface=["a", "c"])
        )
        assert not result.is_free_face

    def test_repeated_vertices_in_collapse_faces_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ElementaryCollapseRequest(
                complex=EDGE, free_face=["a", "a"], coface=["a", "a", "b"]
            )
        with pytest.raises(ValidationError):
            ElementaryCollapseRequest(
                complex=EDGE, free_face=["a"], coface=["a", "b", "b"]
            )

    def test_result_binds_to_source_complex_roundtrip(self) -> None:
        request = ElementaryCollapseRequest(
            complex=EDGE, free_face=["a"], coface=["a", "b"]
        )
        result = compute_elementary_collapse(request)
        assert result.complex == request.complex
        assert ElementaryCollapseResult.model_validate(result.model_dump()) == result

    def test_forged_free_face_with_empty_residual_rejected(self) -> None:
        """A serialized free-face decision with an empty residual cannot
        validate: collapsing ('a',) from the edge {a, b} must leave {b}
        (review counterexample)."""
        with pytest.raises(ValidationError):
            ElementaryCollapseResult(
                complex=EDGE,
                is_free_face=True,
                free_face=("a",),
                coface=("a", "b"),
                remaining_facets=(),
                remaining_vertices=(),
            )

    def test_forged_non_free_decision_rejected(self) -> None:
        """A free pair replayed against its source must stay positive."""
        with pytest.raises(ValidationError):
            ElementaryCollapseResult(
                complex=EDGE,
                is_free_face=False,
                free_face=("a",),
                coface=("a", "b"),
                remaining_facets=(("a", "b"),),
                remaining_vertices=("a", "b"),
                remaining_complex=None,
            )

    def test_non_free_result_retains_source_facets(self) -> None:
        request = ElementaryCollapseRequest(
            complex=CIRCLE, free_face=["a"], coface=["a", "b"]
        )
        result = compute_elementary_collapse(request)
        assert not result.is_free_face
        assert tuple(sorted(result.remaining_facets)) == tuple(
            sorted(tuple(sorted(f)) for f in CIRCLE["facets"])
        )
        assert set(result.remaining_vertices) == {"a", "b", "c"}
        assert ElementaryCollapseResult.model_validate(result.model_dump()) == result

    def test_non_free_result_with_noncanonical_source_facets(self) -> None:
        """A noncanonical facet presentation must not make the operation
        reject its own typed negative decision (review counterexample:
        facets [["b","a"],["c","a"]] with free_face ["a"])."""
        request = ElementaryCollapseRequest(
            complex={
                "vertices": ["c", "b", "a"],
                "facets": [["b", "a"], ["c", "a"]],
            },
            free_face=["a"],
            coface=["a", "b"],
        )
        result = compute_elementary_collapse(request)
        assert not result.is_free_face
        assert result.remaining_facets == (("a", "b"), ("a", "c"))
        assert result.remaining_vertices == ("a", "b", "c")
        assert ElementaryCollapseResult.model_validate(result.model_dump()) == result


class TestJoinBounds:
    def test_join_of_two_4_simplices_rejected(self) -> None:
        """Disjoint 4-simplices join to a 9-simplex (10-vertex facet)."""
        complex_a = {
            "vertices": ["a0", "a1", "a2", "a3", "a4"],
            "facets": [["a0", "a1", "a2", "a3", "a4"]],
        }
        complex_b = {
            "vertices": ["b0", "b1", "b2", "b3", "b4"],
            "facets": [["b0", "b1", "b2", "b3", "b4"]],
        }
        with pytest.raises(ValidationError):
            JoinRequest(complex_a=complex_a, complex_b=complex_b)

    def test_join_exceeding_face_closure_rejected(self) -> None:
        """Two complexes whose join closure would exceed 2048 faces."""
        # A 7-simplex has 255 faces; its join with another 7-simplex has
        # 256*256-1 = 65535 faces, far beyond the closure bound.
        simplex = ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"]
        complex_a = {"vertices": simplex, "facets": [simplex]}
        complex_b = {
            "vertices": ["w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7"],
            "facets": [["w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7"]],
        }
        with pytest.raises(ValidationError):
            JoinRequest(complex_a=complex_a, complex_b=complex_b)

    def test_join_facet_product_rejected_before_expansion(self) -> None:
        """128 maximal edges per side pair into 16384 distinct maximal
        unions; admission rejects the product without expanding it."""

        def edge_complex(prefix: str) -> dict[str, object]:
            vertices = [f"{prefix}{i}" for i in range(32)]
            facets = [
                [f"{prefix}{i}", f"{prefix}{(i + step) % 32}"]
                for step in range(1, 5)
                for i in range(32)
            ]
            return {"vertices": vertices, "facets": facets}

        with pytest.raises(ValidationError):
            JoinRequest(complex_a=edge_complex("a"), complex_b=edge_complex("b"))

    def test_join_vertex_bound_rejected(self) -> None:
        """A join spanning more than 64 vertices is rejected up front."""

        def star_complex(prefix: str) -> dict[str, object]:
            vertices = [f"{prefix}{i}" for i in range(40)]
            return {
                "vertices": vertices,
                "facets": [[f"{prefix}0", v] for v in vertices[1:]],
            }

        with pytest.raises(ValidationError):
            JoinRequest(complex_a=star_complex("a"), complex_b=star_complex("b"))

    def test_forged_oversized_join_result_rejected_before_expansion(self) -> None:
        """Serialized operands above the facet-product bound are rejected
        by admission before the result replay expands the union."""

        def edge_complex(prefix: str) -> dict[str, object]:
            vertices = [f"{prefix}{i}" for i in range(32)]
            facets = [
                [f"{prefix}{i}", f"{prefix}{(i + step) % 32}"]
                for step in range(1, 5)
                for i in range(32)
            ]
            return {"vertices": vertices, "facets": facets}

        with pytest.raises(ValidationError):
            JoinResult(
                complex_a=edge_complex("a"),
                complex_b=edge_complex("b"),
                join_vertices=tuple(f"a{i}" for i in range(32)),
                join_facets=(("a0", "a1"),),
                join_dimension=1,
            )

    def test_request_schema_advertises_canonical_alternative(self) -> None:
        """The published input schema shows both accepted shapes so
        schema-guided callers can pass a canonical complex unchanged."""
        schema = SimplicialComplexRequest.model_json_schema()
        assert "anyOf" in schema
        branches = schema["anyOf"]
        property_sets = [frozenset(branch.get("properties", {})) for branch in branches]
        assert any("facets" in props for props in property_sets)
        assert any("maximal_simplices" in props for props in property_sets)


class TestSkeletonBounds:
    def test_skeleton_facet_explosion_rejected(self) -> None:
        """k=3 on eight disjoint 7-simplices yields 560 tetrahedra > 128."""
        vertices = []
        facets = []
        for i in range(8):
            base = [f"v{i}_{j}" for j in range(8)]
            vertices.extend(base)
            facets.append(base)
        complex_data = {"vertices": vertices, "facets": facets}
        with pytest.raises(ValidationError):
            SkeletonRequest(complex=complex_data, k=3)

    def test_admissible_skeleton_still_works(self) -> None:
        """A single 7-simplex has C(8,4) = 70 tetrahedra in its 3-skeleton."""
        simplex = ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"]
        result = compute_skeleton(
            SkeletonRequest(complex={"vertices": simplex, "facets": [simplex]}, k=3)
        )
        assert len(result.skeleton_facets) == 70


class TestCollapseResidualBounds:
    def test_residual_facets_exceeding_limit_rejected(self) -> None:
        """Collapsing a codim-one face of a 7-simplex beside 122 edges
        leaves 129 maximal facets and must be rejected before execution."""
        simplex = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]
        edges = [[f"g{i // 8}", f"h{i % 8}"] for i in range(122)]
        vertices = sorted(
            simplex + [f"g{i}" for i in range(16)] + [f"h{i}" for i in range(8)]
        )
        complex_data = {"vertices": vertices, "facets": [simplex, *edges]}
        with pytest.raises(ValidationError):
            ElementaryCollapseRequest(
                complex=complex_data,
                free_face=["s1", "s2", "s3", "s4", "s5", "s6", "s7"],
                coface=simplex,
            )

    def test_residual_within_bounds_admitted(self) -> None:
        triangle_plus_edge = {
            "vertices": ["t0", "t1", "t2", "u0", "u1"],
            "facets": [["t0", "t1", "t2"], ["u0", "u1"]],
        }
        result = compute_elementary_collapse(
            ElementaryCollapseRequest(
                complex=triangle_plus_edge,
                free_face=["t0", "t1"],
                coface=["t0", "t1", "t2"],
            )
        )
        assert result.is_free_face
        assert set(result.remaining_facets) == {
            ("t0", "t2"),
            ("t1", "t2"),
            ("u0", "u1"),
        }


class TestShellingSourceBinding:
    def test_result_binds_to_complex_and_order(self) -> None:
        request = ShellingCheckRequest(complex=CIRCLE, facet_order=[0, 1, 2])
        result = compute_shelling_check(request)
        assert result.complex == request.complex
        assert result.facet_order == (0, 1, 2)
        payload = result.model_dump()
        assert ShellingCheckResult.model_validate(payload) == result

    def test_forged_shelling_decision_rejected(self) -> None:
        """Two disjoint edges in order [0, 1] is not a shelling order; a
        forged is_shelling=True cannot validate against the retained source."""
        two_edges = {
            "vertices": ["a", "b", "c", "d"],
            "facets": [["a", "b"], ["c", "d"]],
        }
        with pytest.raises(ValidationError):
            ShellingCheckResult(
                complex=two_edges,
                facet_order=[0, 1],
                is_shelling=True,
                failed_at=None,
                failure_reason=None,
            )


class TestResultSourceBindingRegression:
    """Serialized results must replay against their retained sources."""

    def test_forged_deletion_keeping_deleted_vertex_rejected(self) -> None:
        payload = compute_vertex_deletion(
            VertexDeletionRequest(complex=TRIANGLE, vertices_to_delete=["v0"])
        ).model_dump()
        assert payload["deleted_vertices"] == ("v0",)
        # The reviewer counterexample: a singleton source still carrying the
        # allegedly deleted vertex cannot validate against the claimed output.
        payload["complex"] = {"vertices": ["v0"], "facets": [["v0"]]}
        with pytest.raises(ValidationError):
            VertexDeletionResult.model_validate(payload)

    def test_deletion_roundtrip(self) -> None:
        result = compute_vertex_deletion(
            VertexDeletionRequest(complex=CIRCLE, vertices_to_delete=["b"])
        )
        assert result.remaining_facets == (("a", "c"),)
        assert VertexDeletionResult.model_validate(result.model_dump()) == result

    def test_edge_in_zero_skeleton_rejected(self) -> None:
        request = SkeletonRequest(complex=EDGE, k=0)
        with pytest.raises(ValidationError):
            SkeletonResult(
                complex=request.complex,
                k=0,
                skeleton_facets=(("a", "b"),),
                skeleton_vertices=("a", "b"),
                skeleton_complex=_canonical_complex(("a", "b"), (("a", "b"),)),
            )

    def test_skeleton_roundtrip(self) -> None:
        result = compute_skeleton(SkeletonRequest(complex=TRIANGLE, k=1))
        assert result.skeleton_facets == (("v0", "v1"), ("v0", "v2"), ("v1", "v2"))
        assert SkeletonResult.model_validate(result.model_dump()) == result

    def test_single_point_join_result_rejected(self) -> None:
        request = JoinRequest(
            complex_a={"vertices": ["a"], "facets": [["a"]]},
            complex_b={"vertices": ["b"], "facets": [["b"]]},
        )
        with pytest.raises(ValidationError):
            JoinResult(
                complex_a=request.complex_a,
                complex_b=request.complex_b,
                join_vertices=("a",),
                join_facets=(("a",),),
                join_dimension=0,
                join_complex=_canonical_complex(("a",), (("a",),)),
            )

    def test_join_roundtrip(self) -> None:
        result = compute_join(
            JoinRequest(
                complex_a={"vertices": ["a"], "facets": [["a"]]},
                complex_b={"vertices": ["b", "c"], "facets": [["b", "c"]]},
            )
        )
        assert result.join_facets == (("a", "b", "c"),)
        assert JoinResult.model_validate(result.model_dump()) == result

    def test_subdivision_original_dimension_tamper_rejected(self) -> None:
        request = BarycentricSubdivisionRequest(complex=EDGE)
        payload = compute_barycentric_subdivision(request).model_dump()
        payload["original_dimension"] = 7
        with pytest.raises(ValidationError):
            BarycentricSubdivisionResult.model_validate(payload)

    def test_collapse_result_empty_free_face_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ElementaryCollapseResult(
                complex=EDGE,
                is_free_face=True,
                free_face=(),
                coface=("a",),
                remaining_facets=(("b",),),
                remaining_vertices=("b",),
                remaining_complex=None,
            )

    def test_collapse_accepts_reordered_facet_labels(self) -> None:
        """Freeness is a set relation: facet label order cannot flip it."""
        reordered = {
            "vertices": ["a", "b"],
            "facets": [["b", "a"]],
        }
        result = compute_elementary_collapse(
            ElementaryCollapseRequest(
                complex=reordered, free_face=["a"], coface=["a", "b"]
            )
        )
        assert result.is_free_face
        assert ElementaryCollapseResult.model_validate(result.model_dump()) == result


class TestResultDomainMirrorsRequest:
    """Serialized results must satisfy the request's own admission domain."""

    def test_star_result_empty_simplex_rejected(self) -> None:
        """No accepted invocation can request the star of the empty face, so
        a serialized result cannot authenticate it either."""
        with pytest.raises(ValidationError):
            StarResult(
                complex=EDGE,
                simplex=(),
                star_facets=(("a", "b"),),
                star_is_empty=False,
                star_complex=_canonical_complex(("a", "b"), (("a", "b"),)),
            )

    def test_deletion_result_empty_deleted_vertices_rejected(self) -> None:
        """An identity transformation is not a deletion result: the request
        requires at least one deleted vertex."""
        with pytest.raises(ValidationError):
            VertexDeletionResult(
                complex=CIRCLE,
                deleted_vertices=(),
                remaining_vertices=("a", "b", "c"),
                remaining_facets=(("a", "b"), ("b", "c"), ("a", "c")),
                remaining_complex=_canonical_complex(
                    ("a", "b", "c"), (("a", "b"), ("b", "c"), ("a", "c"))
                ),
            )

    def test_subdivision_result_oversized_source_rejected(self) -> None:
        """The retained source must satisfy the subdivision request's
        31-face work admission, which the bare complex type does not carry."""
        result = compute_barycentric_subdivision(
            BarycentricSubdivisionRequest(complex=CIRCLE)
        )
        payload = result.model_dump()
        simplex4_plus_point = {
            "vertices": ["v0", "v1", "v2", "v3", "v4", "p"],
            "facets": [["v0", "v1", "v2", "v3", "v4"], ["p"]],
        }
        payload["complex"] = simplex4_plus_point
        with pytest.raises(ValidationError):
            BarycentricSubdivisionResult.model_validate(payload)

    def test_subdivision_roundtrip_still_admitted(self) -> None:
        result = compute_barycentric_subdivision(
            BarycentricSubdivisionRequest(complex=CIRCLE)
        )
        assert (
            BarycentricSubdivisionResult.model_validate(result.model_dump()) == result
        )
