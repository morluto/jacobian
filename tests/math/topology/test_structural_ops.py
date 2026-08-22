"""Tests for structural simplicial complex operations (#1850)."""

import pytest
from pydantic import ValidationError

from jacobian.math.topology._models import (
    BarycentricSubdivisionRequest,
    ElementaryCollapseRequest,
    JoinRequest,
    PseudomanifoldRequest,
    ShellingCheckRequest,
    SkeletonRequest,
    StarRequest,
    VertexDeletionRequest,
)
from jacobian.math.topology._operations import (
    compute_barycentric_subdivision,
    compute_elementary_collapse,
    compute_join,
    compute_pseudomanifold_decision,
    compute_shelling_check,
    compute_skeleton,
    compute_star,
    compute_vertex_deletion,
)

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

    def test_star_of_edge_in_triangle(self) -> None:
        result = compute_star(StarRequest(complex=TRIANGLE, simplex=["v0", "v1"]))
        assert result.star_facets == (("v0", "v1", "v2"),)

    def test_star_of_vertex_in_circle(self) -> None:
        result = compute_star(StarRequest(complex=CIRCLE, simplex=["a"]))
        assert result.star_facets == (("a", "b"), ("a", "c"))

    def test_star_not_a_face(self) -> None:
        with pytest.raises(ValidationError, match="must be in the complex"):
            StarRequest(complex=TRIANGLE, simplex=["v0", "v1", "v2", "v3"])


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
        with pytest.raises(ValidationError, match="must be in the complex"):
            VertexDeletionRequest(complex=TRIANGLE, vertices_to_delete=["v9"])


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
        with pytest.raises(ValidationError, match="disjoint"):
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


class TestPseudomanifold:
    def test_circle_is_closed_pseudomanifold(self) -> None:
        result = compute_pseudomanifold_decision(
            PseudomanifoldRequest(complex=CIRCLE)
        )
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
        with pytest.raises(ValidationError, match="permutation"):
            ShellingCheckRequest(complex=EDGE, facet_order=[1, 0])


class TestElementaryCollapse:
    def test_collapse_free_vertex_from_edge(self) -> None:
        result = compute_elementary_collapse(
            ElementaryCollapseRequest(
                complex=EDGE, free_face=["a"], coface=["a", "b"]
            )
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
        result = compute_elementary_collapse(
            ElementaryCollapseRequest(
                complex=EDGE, free_face=["a"], coface=["a", "b", "c"]
            )
        )
        assert not result.is_free_face
