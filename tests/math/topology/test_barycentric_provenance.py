from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology._models import (
    BarycentricSubdivisionRequest,
    BarycentricSubdivisionResult,
    SimplicialComplexRequest,
)
from jacobian.math.topology._tools import compute_barycentric_subdivision


def test_triangle_subdivision_facets_bind_exactly_to_source_chains() -> None:
    request = BarycentricSubdivisionRequest(
        complex=SimplicialComplexRequest(
            vertices=("a", "b", "c"), facets=(("a", "b", "c"),)
        )
    )
    result = compute_barycentric_subdivision(request)
    expected_chains = {
        (("a",), ("a", "b"), ("a", "b", "c")),
        (("a",), ("a", "c"), ("a", "b", "c")),
        (("b",), ("a", "b"), ("a", "b", "c")),
        (("b",), ("b", "c"), ("a", "b", "c")),
        (("c",), ("a", "c"), ("a", "b", "c")),
        (("c",), ("b", "c"), ("a", "b", "c")),
    }

    vertex_to_face = dict(
        zip(result.subdivision_vertices, result.subdivision_vertex_faces, strict=True)
    )
    actual_chains = set(result.subdivision_facet_face_chains)
    assert actual_chains == expected_chains
    assert len(result.subdivision_facets) == len(expected_chains)
    for facet, chain in zip(
        result.subdivision_facets,
        result.subdivision_facet_face_chains,
        strict=True,
    ):
        assert tuple(sorted(vertex_to_face[vertex] for vertex in facet)) == tuple(
            sorted(chain)
        )
    assert BarycentricSubdivisionResult.model_validate(result.model_dump()) == result


def test_subdivision_result_rejects_facet_chain_mismatch() -> None:
    result = compute_barycentric_subdivision(
        BarycentricSubdivisionRequest(
            complex=SimplicialComplexRequest(vertices=("a", "b"), facets=(("a", "b"),))
        )
    )
    payload = result.model_dump()
    payload["subdivision_facet_face_chains"] = (
        ("a",),
        *payload["subdivision_facet_face_chains"][1:],
    )
    with pytest.raises(ValidationError):
        BarycentricSubdivisionResult.model_validate(payload)


def _four_simplex_with_isolated_points(
    point_count: int,
) -> BarycentricSubdivisionRequest:
    points = tuple(f"p{i:02}" for i in range(point_count))
    return BarycentricSubdivisionRequest(
        complex=SimplicialComplexRequest(
            vertices=("a", "b", "c", "d", "e", *points),
            facets=(("a", "b", "c", "d", "e"), *((point,) for point in points)),
        )
    )


def test_maximal_chain_output_bound_accepts_128_and_rejects_129() -> None:
    accepted = compute_barycentric_subdivision(_four_simplex_with_isolated_points(8))
    assert len(accepted.subdivision_facets) == 128
    assert accepted.subdivision_complex is not None
    assert accepted.subdivision_complex.closure_size <= 2048

    with pytest.raises(OperationResourceAdmissionError, match="more than 128"):
        compute_barycentric_subdivision(_four_simplex_with_isolated_points(9))
