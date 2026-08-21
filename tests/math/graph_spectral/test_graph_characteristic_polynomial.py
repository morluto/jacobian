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
    return GraphSpectrumRequest(
        graph=GraphEdgeList(vertex_count=vc, edges=tuple(tuple(e) for e in edges))
    )


def _coeffs(result):
    """Return dense increasing-degree coefficients including implicit zeros."""
    degree = 0
    out: list[Fraction] = []
    terms = {
        term.exponents[0]: term.coefficient.as_fraction()
        for term in result.polynomial.polynomial.terms
    }
    top = max(terms) if terms else 0
    while degree <= top:
        out.append(terms.get(degree, Fraction(0)))
        degree += 1
    return out


class TestAdjacencyCharacteristicPolynomial:
    def test_path_p3(self):
        # P3 adjacency eigenvalues: 0, sqrt(2), -sqrt(2) -> charpoly x(x^2-2) = x^3 - 2x.
        result = compute_adjacency_characteristic_polynomial(
            _request([[0, 1], [1, 2]], 3)
        )
        assert _coeffs(result) == [Fraction(0), Fraction(-2), Fraction(0), Fraction(1)]
        assert result.convention == "ADJACENCY"

    def test_edge_k2(self):
        # K2 adjacency: [[0,1],[1,0]] eigenvalues 1,-1 -> charpoly x^2 - 1.
        result = compute_adjacency_characteristic_polynomial(_request([[0, 1]], 2))
        assert _coeffs(result) == [Fraction(-1), Fraction(0), Fraction(1)]

    def test_isolated_vertex(self):
        # One isolated vertex: adjacency matrix [0] -> charpoly x.
        result = compute_adjacency_characteristic_polynomial(_request([], 1))
        assert _coeffs(result) == [Fraction(0), Fraction(1)]

    def test_result_is_monic(self):
        result = compute_adjacency_characteristic_polynomial(
            _request([[0, 1], [1, 2], [0, 2]], 3)
        )
        top = max(term.exponents[0] for term in result.polynomial.polynomial.terms)
        leading = next(
            term
            for term in result.polynomial.polynomial.terms
            if term.exponents[0] == top
        )
        assert leading.coefficient.as_fraction() == 1

    def test_result_binds_source_graph(self):
        request = _request([[0, 1], [1, 2]], 3)
        result = compute_adjacency_characteristic_polynomial(request)
        assert result.graph == request.graph


class TestLaplacianCharacteristicPolynomial:
    def test_path_p3(self):
        # P3 Laplacian eigenvalues 0,1,3 -> charpoly x(x-1)(x-3) = x^3 - 4x^2 + 3x.
        result = compute_laplacian_characteristic_polynomial(
            _request([[0, 1], [1, 2]], 3)
        )
        assert _coeffs(result) == [Fraction(0), Fraction(3), Fraction(-4), Fraction(1)]
        assert result.convention == "LAPLACIAN"

    def test_laplacian_has_zero_root(self):
        # The Laplacian of any graph has a zero eigenvalue, so the constant term is 0.
        result = compute_laplacian_characteristic_polynomial(
            _request([[0, 1], [1, 2], [0, 2]], 3)
        )
        constant = next(
            (
                term
                for term in result.polynomial.polynomial.terms
                if term.exponents[0] == 0
            ),
            None,
        )
        assert constant is None

    def test_result_is_monic(self):
        result = compute_laplacian_characteristic_polynomial(_request([[0, 1]], 2))
        top = max(term.exponents[0] for term in result.polynomial.polynomial.terms)
        leading = next(
            term
            for term in result.polynomial.polynomial.terms
            if term.exponents[0] == top
        )
        assert leading.coefficient.as_fraction() == 1


def test_forged_result_rejected():
    import pytest
    from pydantic import ValidationError

    from jacobian.math.graphs.spectral._models import (
        GraphCharacteristicPolynomialResult,
    )

    graph = GraphEdgeList(vertex_count=3, edges=((0, 1), (1, 2)))
    with pytest.raises(ValidationError):
        GraphCharacteristicPolynomialResult.model_validate(
            {
                "graph": graph.model_dump(),
                "convention": "ADJACENCY",
                "polynomial": {
                    "variables": ["x"],
                    "polynomial": {
                        "terms": [
                            {"coefficient": {"num": "1", "den": "1"}, "exponents": [0]}
                        ]
                    },
                },
            }
        )
