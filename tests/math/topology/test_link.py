"""Tests for simplicial complex link operation."""

from jacobian.math.topology._models import LinkRequest
from jacobian.math.topology._operations import compute_link


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
