"""Tests for simplicial complex f-vector operation."""

from jacobian.math.topology._structural import FVectorRequest, compute_f_vector


def test_triangle() -> None:
    result = compute_f_vector(
        FVectorRequest(
            complex={"vertices": ["v0", "v1", "v2"], "facets": [["v0", "v1", "v2"]]}
        )
    )
    assert result.f_vector == (3, 3, 1)
    assert result.euler_characteristic == 1
    assert result.dimension == 2


def test_edge() -> None:
    result = compute_f_vector(
        FVectorRequest(complex={"vertices": ["v0", "v1"], "facets": [["v0", "v1"]]})
    )
    assert result.f_vector == (2, 1)
    assert result.euler_characteristic == 1
    assert result.dimension == 1


def test_single_vertex() -> None:
    result = compute_f_vector(
        FVectorRequest(complex={"vertices": ["v0"], "facets": [["v0"]]})
    )
    assert result.f_vector == (1,)
    assert result.euler_characteristic == 1
    assert result.dimension == 0
