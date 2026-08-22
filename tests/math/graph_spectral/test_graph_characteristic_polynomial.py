"""Tests for exact graph characteristic-polynomial operations."""

from fractions import Fraction

from jacobian.math.graphs.spectral import (
    adjacency_characteristic_polynomial,
    laplacian_characteristic_polynomial,
)
from jacobian.math.graphs.spectral._models import (
    GraphCharacteristicPolynomialResult,
    GraphEdgeList,
    GraphSpectrumRequest,
)
from jacobian.math.graphs.spectral._operations import (
    compute_adjacency_characteristic_polynomial,
    compute_laplacian_characteristic_polynomial,
)
from jacobian.math.graphs.spectral._tools import TOOLS
from jacobian.math.polynomials._elementary_operations import (
    rational_polynomial_derivative,
)
from jacobian.math.polynomials._models import RationalPolynomialRequest
from jacobian.math.polynomials.values import RationalPolynomial


def _graph(edges, vc):
    return GraphEdgeList(vertex_count=vc, edges=tuple(tuple(e) for e in edges))


def _request(edges, vc):
    return GraphSpectrumRequest(graph=_graph(edges, vc))


def _coeffs(polynomial: RationalPolynomial):
    """Return dense increasing-degree coefficients including implicit zeros."""
    terms = {
        term.exponents[0]: term.coefficient.as_fraction()
        for term in polynomial.polynomial.terms
    }
    top = max(terms) if terms else 0
    return [terms.get(degree, Fraction(0)) for degree in range(top + 1)]


def _exponents(polynomial: RationalPolynomial) -> tuple[int, ...]:
    return tuple(term.exponents[0] for term in polynomial.polynomial.terms)


class TestAdjacencyCharacteristicPolynomial:
    def test_path_p3(self):
        # P3 adjacency eigenvalues: 0, sqrt(2), -sqrt(2) -> charpoly x(x^2-2) = x^3 - 2x.
        result = compute_adjacency_characteristic_polynomial(
            _request([[0, 1], [1, 2]], 3)
        )
        assert _coeffs(result.polynomial) == [
            Fraction(0),
            Fraction(-2),
            Fraction(0),
            Fraction(1),
        ]
        assert result.convention == "ADJACENCY"

    def test_edge_k2(self):
        # K2 adjacency: [[0,1],[1,0]] eigenvalues 1,-1 -> charpoly x^2 - 1.
        result = compute_adjacency_characteristic_polynomial(_request([[0, 1]], 2))
        assert _coeffs(result.polynomial) == [Fraction(-1), Fraction(0), Fraction(1)]

    def test_isolated_vertex(self):
        # One isolated vertex: adjacency matrix [0] -> charpoly x.
        result = compute_adjacency_characteristic_polynomial(_request([], 1))
        assert _coeffs(result.polynomial) == [Fraction(0), Fraction(1)]

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
        assert _coeffs(result.polynomial) == [
            Fraction(0),
            Fraction(3),
            Fraction(-4),
            Fraction(1),
        ]
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


def test_native_adjacency_returns_canonical_polynomial():
    polynomial = adjacency_characteristic_polynomial(_graph([[0, 1], [1, 2]], 3))
    assert type(polynomial) is RationalPolynomial
    assert polynomial.variables == ("x",)
    assert _exponents(polynomial) == (3, 1)
    assert _coeffs(polynomial) == [Fraction(0), Fraction(-2), Fraction(0), Fraction(1)]


def test_native_laplacian_returns_canonical_polynomial():
    polynomial = laplacian_characteristic_polynomial(_graph([[0, 1], [1, 2]], 3))
    assert type(polynomial) is RationalPolynomial
    assert _exponents(polynomial) == (3, 2, 1)
    assert _coeffs(polynomial) == [Fraction(0), Fraction(3), Fraction(-4), Fraction(1)]


def test_native_polynomial_is_accepted_unchanged_by_a_polynomial_consumer():
    polynomial = adjacency_characteristic_polynomial(_graph([[0, 1], [1, 2]], 3))
    request = RationalPolynomialRequest(polynomial=polynomial)
    assert request.polynomial is polynomial

    serialized = RationalPolynomialRequest.model_validate(
        {"polynomial": polynomial.model_dump(mode="json")}
    )
    derivative = rational_polynomial_derivative(request).derivative
    assert rational_polynomial_derivative(serialized).derivative == derivative
    assert _coeffs(derivative) == [Fraction(-2), Fraction(0), Fraction(3)]


def test_native_isolated_vertex_composes_without_reshaping():
    polynomial = adjacency_characteristic_polynomial(_graph([], 1))
    request = RationalPolynomialRequest(polynomial=polynomial)
    assert request.polynomial is polynomial
    assert _coeffs(rational_polynomial_derivative(request).derivative) == [Fraction(1)]


def test_catalog_result_round_trips_source_binding():
    request = _request([[0, 1], [1, 2]], 3)
    result = compute_adjacency_characteristic_polynomial(request)
    restored = GraphCharacteristicPolynomialResult.model_validate(
        result.model_dump(mode="json")
    )
    assert restored == result
    assert restored.graph == request.graph
    assert restored.convention == "ADJACENCY"
    assert restored.polynomial == adjacency_characteristic_polynomial(request.graph)


def test_serialized_native_terms_use_descending_exponent_order():
    polynomial = laplacian_characteristic_polynomial(_graph([[0, 1], [1, 2]], 3))
    payload = polynomial.model_dump(mode="json")
    exponents = [tuple(term["exponents"]) for term in payload["polynomial"]["terms"]]
    assert exponents == [(3,), (2,), (1,)]
    assert exponents == sorted(exponents, reverse=True)


def test_discovery_names_canonical_sparse_polynomial():
    for operation_id in (
        "graph.spectrum.adjacency.characteristic_polynomial.compute",
        "graph.spectrum.laplacian.characteristic_polynomial.compute",
    ):
        description = next(
            tool.description for tool in TOOLS if tool.operation_id == operation_id
        )
        assert "increasing-degree" not in description
        assert "canonical sparse RationalPolynomial" in description
        assert "descending exponent order" in description
