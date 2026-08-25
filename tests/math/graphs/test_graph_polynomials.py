"""Tests for graph polynomial operations."""

from jacobian.math.graphs.polynomials._models import (
    GraphEdge,
    GraphPolynomialRequest,
    GraphSpec,
    MatchingPolynomialRequest,
)
from jacobian.math.graphs.polynomials._operations import (
    compute_chromatic_polynomial,
    compute_flow_polynomial,
    compute_matching_polynomial,
    compute_tutte_polynomial,
)


def _cycle_graph(n: int) -> GraphSpec:
    edges = [GraphEdge(u=i, v=(i + 1) % n) for i in range(n)]
    return GraphSpec(vertex_count=n, edges=tuple(edges))


def _path_graph(n: int) -> GraphSpec:
    edges = [GraphEdge(u=i, v=i + 1) for i in range(n - 1)]
    return GraphSpec(vertex_count=n, edges=tuple(edges))


def _terms_to_dict(result):
    return {t.degree: t.coefficient for t in result.terms}


class TestTuttePolynomial:
    def test_cycle_c4(self):
        req = GraphPolynomialRequest(graph=_cycle_graph(4))
        result = compute_tutte_polynomial(req)
        # T(C4, x, y) = x^3 + x^2 + x + y
        d = {term.exponents: term.coefficient for term in result.terms}
        assert result.variables == ("x", "y")
        assert d == {(0, 1): 1, (1, 0): 1, (2, 0): 1, (3, 0): 1}

    def test_single_edge(self):
        req = GraphPolynomialRequest(
            graph=GraphSpec(vertex_count=2, edges=(GraphEdge(u=0, v=1),))
        )
        result = compute_tutte_polynomial(req)
        # T(K2) = x
        d = {term.exponents: term.coefficient for term in result.terms}
        assert d == {(1, 0): 1}


class TestChromaticPolynomial:
    def test_cycle_c3(self):
        req = GraphPolynomialRequest(graph=_cycle_graph(3))
        result = compute_chromatic_polynomial(req)
        # chi(C3) = x(x-1)(x-2) = x^3 - 3x^2 + 2x
        d = _terms_to_dict(result)
        assert d.get(3) == 1
        assert d.get(2) == -3
        assert d.get(1) == 2

    def test_path_p3(self):
        req = GraphPolynomialRequest(graph=_path_graph(3))
        result = compute_chromatic_polynomial(req)
        # chi(P3) = x(x-1)^2 = x^3 - 2x^2 + x
        d = _terms_to_dict(result)
        assert d.get(3) == 1
        assert d.get(2) == -2
        assert d.get(1) == 1


class TestFlowPolynomial:
    def test_cycle_c4(self):
        req = GraphPolynomialRequest(graph=_cycle_graph(4))
        result = compute_flow_polynomial(req)
        # F(C4) = x - 1, from (-1)^{|E|-|V|+k} T(0, 1-x).
        d = _terms_to_dict(result)
        assert d.get(1) == 1
        assert d.get(0) == -1

    def test_rejects_graph_beyond_deletion_budget(self):
        import pytest
        from pydantic import ValidationError

        edges = tuple(GraphEdge(u=i, v=j) for i in range(8) for j in range(i))
        with pytest.raises(ValidationError):
            GraphPolynomialRequest(
                graph=GraphSpec(vertex_count=8, edges=edges),
            )

    def test_bridge_is_zero_polynomial(self):
        req = GraphPolynomialRequest(
            graph=GraphSpec(vertex_count=2, edges=(GraphEdge(u=0, v=1),))
        )
        result = compute_flow_polynomial(req)
        assert result.terms == ()


class TestMatchingPolynomial:
    def test_single_edge(self):
        req = MatchingPolynomialRequest(
            graph=GraphSpec(vertex_count=2, edges=(GraphEdge(u=0, v=1),))
        )
        result = compute_matching_polynomial(req)
        # M(K2) = x^2 - 1
        d = _terms_to_dict(result)
        assert d.get(2) == 1
        assert d.get(0) == -1

    def test_path_p3(self):
        req = MatchingPolynomialRequest(graph=_path_graph(3))
        result = compute_matching_polynomial(req)
        # P3 has edges (0,1) and (1,2)
        # 0-matchings: 1, 1-matching: 2, no 2-matchings
        # M = x^3 - 2x
        d = _terms_to_dict(result)
        assert d.get(3) == 1
        assert d.get(1) == -2
