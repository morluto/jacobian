"""Tests for exact graph characteristic-polynomial operations."""

from fractions import Fraction

from jacobian.math.graphs.spectral._models import (
    GraphEdgeList,
    GraphSpectrumRequest,
)
from jacobian.math.graphs.spectral._operations import (
    compute_adjacency_characteristic_polynomial,
    compute_laplacian_characteristic_polynomial,
)


def _request(edges, vc):
    return GraphSpectrumRequest(graph=GraphEdgeList(vertex_count=vc, edges=tuple(tuple(e) for e in edges)))


class TestAdjacencyCharacteristicPolynomial:
    def test_path_p3(self):
        # P3 adjacency eigenvalues: 0, sqrt(2), -sqrt(2) -> charpoly x(x^2-2) = x^3 - 2x.
        result = compute_adjacency_characteristic_polynomial(_request([[0, 1], [1, 2]], 3))
        coeffs = [c.as_fraction() for c in result.coefficients]
        assert coeffs == [Fraction(0), Fraction(-2), Fraction(0), Fraction(1)]

    def test_edge_k2(self):
        # K2 adjacency: [[0,1],[1,0]] eigenvalues 1,-1 -> charpoly x^2 - 1.
        result = compute_adjacency_characteristic_polynomial(_request([[0, 1]], 2))
        coeffs = [c.as_fraction() for c in result.coefficients]
        assert coeffs == [Fraction(-1), Fraction(0), Fraction(1)]

    def test_isolated_vertex(self):
        # One isolated vertex: adjacency matrix [0] -> charpoly x.
        result = compute_adjacency_characteristic_polynomial(_request([], 1))
        assert [c.as_fraction() for c in result.coefficients] == [Fraction(0), Fraction(1)]

    def test_result_is_monic(self):
        result = compute_adjacency_characteristic_polynomial(_request([[0, 1], [1, 2], [0, 2]], 3))
        assert result.coefficients[-1].as_fraction() == 1


class TestLaplacianCharacteristicPolynomial:
    def test_path_p3(self):
        # P3 Laplacian eigenvalues 0,1,3 -> charpoly x(x-1)(x-3) = x^3 - 4x^2 + 3x.
        result = compute_laplacian_characteristic_polynomial(_request([[0, 1], [1, 2]], 3))
        coeffs = [c.as_fraction() for c in result.coefficients]
        assert coeffs == [Fraction(0), Fraction(3), Fraction(-4), Fraction(1)]

    def test_laplacian_has_zero_root(self):
        # The Laplacian of any graph has a zero eigenvalue, so the constant term is 0.
        result = compute_laplacian_characteristic_polynomial(_request([[0, 1], [1, 2], [0, 2]], 3))
        assert result.coefficients[0].as_fraction() == 0

    def test_result_is_monic(self):
        result = compute_laplacian_characteristic_polynomial(_request([[0, 1]], 2))
        assert result.coefficients[-1].as_fraction() == 1
