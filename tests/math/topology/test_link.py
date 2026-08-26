"""Tests for simplicial complex link operation."""

from jacobian.math.topology._structural import (
    FVectorRequest,
    LinkRequest,
    compute_link,
)


def test_link_of_vertex_in_triangle() -> None:
    result = compute_link(
        LinkRequest(
            complex={"vertices": ["v0", "v1", "v2"], "facets": [["v0", "v1", "v2"]]},
            simplex=("v0",),
        )
    )
    assert result.link_facets == (("v1", "v2"),)
    assert result.link_is_empty is False


def test_link_of_edge_in_triangle() -> None:
    result = compute_link(
        LinkRequest(
            complex={"vertices": ["v0", "v1", "v2"], "facets": [["v0", "v1", "v2"]]},
            simplex=("v0", "v1"),
        )
    )
    assert result.link_facets == (("v2",),)


def test_link_of_vertex_in_discrete_complex() -> None:
    result = compute_link(
        LinkRequest(
            complex={"vertices": ["v0", "v1"], "facets": [["v0"], ["v1"]]},
            simplex=("v0",),
        )
    )
    assert result.link_is_empty is True


def test_link_of_face_in_boundary() -> None:
    result = compute_link(
        LinkRequest(
            complex={
                "vertices": ["v0", "v1", "v2"],
                "facets": [["v0", "v1"], ["v1", "v2"], ["v0", "v2"]],
            },
            simplex=("v0",),
        )
    )
    assert ("v1",) in result.link_facets
    assert ("v2",) in result.link_facets


def test_f_vector_filled_triangle() -> None:
    """Filled triangle has f=(3,3,1) and h=(1,0,0,0)."""
    from jacobian.math.topology._structural import compute_f_vector

    request = FVectorRequest(
        complex={"vertices": ["v0", "v1", "v2"], "facets": [["v0", "v1", "v2"]]}
    )
    result = compute_f_vector(request)
    assert result.f_vector == (3, 3, 1)
    assert result.h_vector == (1, 0, 0, 0)


def test_f_vector_single_edge() -> None:
    """Single edge has f=(2,1) and h=(1,0,0)."""
    from jacobian.math.topology._structural import compute_f_vector

    request = FVectorRequest(
        complex={"vertices": ["v0", "v1"], "facets": [["v0", "v1"]]}
    )
    result = compute_f_vector(request)
    assert result.f_vector == (2, 1)
    assert result.h_vector == (1, 0, 0)


def test_f_vector_h_vector_invariant() -> None:
    """h_0 must always be 1 for a non-empty complex."""
    from jacobian.math.topology._structural import compute_f_vector

    request = FVectorRequest(
        complex={"vertices": ["v0", "v1", "v2"], "facets": [["v0", "v1", "v2"]]}
    )
    result = compute_f_vector(request)
    assert result.h_vector[0] == 1


def test_link_rejects_non_face() -> None:
    """Non-face simplex should be rejected, not silently returned as empty link."""
    from pydantic import ValidationError

    try:
        compute_link(
            LinkRequest(
                complex={
                    "vertices": ["v0", "v1", "v2", "v3"],
                    "facets": [["v0", "v1"], ["v2", "v3"]],
                },
                simplex=("v0", "v2"),
            )
        )
        raise AssertionError("Should have raised ValidationError")
    except ValidationError:
        pass
