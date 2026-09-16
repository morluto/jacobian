"""Tests for the simplicial cone operation (#1798)."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology._models import SimplicialComplexRequest
from jacobian.math.topology._structural import (
    ConeRequest,
    ConeResult,
    VertexDeletionRequest,
    compute_cone,
    compute_vertex_deletion,
)
from jacobian.math.topology._tools import compute_cone_entry


def _request(
    vertices: list[str], facets: list[list[str]], apex: str = "cone_apex"
) -> ConeRequest:
    return ConeRequest(
        complex=SimplicialComplexRequest.model_validate(
            {"vertices": vertices, "facets": facets}
        ),
        apex=apex,
    )


class TestKnownAnswer:
    def test_cone_over_an_edge_is_a_triangle(self) -> None:
        result = compute_cone(_request(["a", "b"], [["a", "b"]]))
        assert isinstance(result, ConeResult)
        assert result.apex == "cone_apex"
        assert result.cone_facets == (("a", "b", "cone_apex"),)
        assert result.cone_dimension == 2
        assert result.cone_complex.dimension == 2

    def test_cone_over_a_point_is_an_edge(self) -> None:
        result = compute_cone(_request(["a"], [["a"]], apex="c"))
        assert result.cone_facets == (("a", "c"),)
        assert result.cone_dimension == 1

    def test_cone_facets_add_apex_to_every_source_facet(self) -> None:
        request = _request(["a", "b", "c"], [["a", "b"], ["b", "c"]])
        result = compute_cone(request)
        assert result.cone_facets == (("a", "b", "cone_apex"), ("b", "c", "cone_apex"))


class TestBoundaryDegenerate:
    def test_cone_over_empty_looking_singleton(self) -> None:
        result = compute_cone(_request(["x"], [["x"]]))
        assert len(result.source_face_transport) == 1
        row = result.source_face_transport[0]
        assert row.source_face == ("x",)
        assert row.cone_face == ("cone_apex", "x")

    def test_transport_covers_every_source_face(self) -> None:
        request = _request(["a", "b"], [["a", "b"]])
        result = compute_cone(request)
        # Faces of an edge: {a}, {b}, {a,b}.
        assert len(result.source_face_transport) == 3
        for row in result.source_face_transport:
            assert set(row.cone_face) == set(row.source_face) | {"cone_apex"}


class TestAdversarial:
    def test_colliding_apex_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            compute_cone(_request(["a", "b"], [["a", "b"]], apex="a"))

    def test_catalog_entry_rejects_collision(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            compute_cone_entry(_request(["a", "b"], [["a", "b"]], apex="b"))


class TestDefiningInvariant:
    def test_facets_are_source_unions_with_apex(self) -> None:
        request = _request(["a", "b", "c"], [["a", "b"], ["b", "c"]])
        result = compute_cone(request)
        expected = {
            tuple(sorted(set(facet) | {request.apex}))
            for facet in request.complex.facets
        }
        assert {tuple(sorted(f)) for f in result.cone_facets} == expected

    def test_dimension_increases_by_one(self) -> None:
        for vertices, facets, source_dim in [
            (["a"], [["a"]], 0),
            (["a", "b"], [["a", "b"]], 1),
            (["a", "b", "c"], [["a", "b", "c"]], 2),
        ]:
            result = compute_cone(_request(vertices, facets))
            assert result.cone_dimension == source_dim + 1

    def test_deleting_apex_returns_source(self) -> None:
        request = _request(["a", "b", "c"], [["a", "b"], ["b", "c"]])
        result = compute_cone(request)
        deletion = compute_vertex_deletion(
            VertexDeletionRequest(
                complex=SimplicialComplexRequest(
                    vertices=result.cone_vertices, facets=result.cone_facets
                ),
                vertices_to_delete=(request.apex,),
            )
        )
        assert sorted(map(sorted, deletion.remaining_facets)) == sorted(
            map(sorted, request.complex.facets)
        )


class TestNativeCatalogParity:
    def test_entry_matches_kernel(self) -> None:
        request = _request(["a", "b"], [["a", "b"]])
        assert compute_cone_entry(request) == compute_cone(request)
