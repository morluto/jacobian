"""Tests for graph polynomial operations."""

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.polynomials._models import (
    GraphPolynomialRequest,
    GraphPolynomialResult,
    MatchingPolynomialRequest,
    SparseMultivariatePolynomial,
)
from jacobian.math.graphs.polynomials.operations import (
    chromatic_polynomial,
    flow_polynomial,
    matching_polynomial,
    tutte_polynomial,
)
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph


def _cycle_graph(n: int) -> IndexedSimpleUndirectedGraph:
    edges = tuple(sorted((min(i, (i + 1) % n), max(i, (i + 1) % n)) for i in range(n)))
    return IndexedSimpleUndirectedGraph(vertex_count=n, edges=edges)


def _path_graph(n: int) -> IndexedSimpleUndirectedGraph:
    edges = tuple((i, i + 1) for i in range(n - 1))
    return IndexedSimpleUndirectedGraph(vertex_count=n, edges=edges)


def _terms_to_dict(result: GraphPolynomialResult) -> dict[int, int]:
    return {t.degree: t.coefficient for t in result.terms}


def _run_tutte(request: GraphPolynomialRequest) -> SparseMultivariatePolynomial:
    return SparseMultivariatePolynomial(
        variables=("x", "y"), terms=tutte_polynomial(request.graph)
    )


def _run_chromatic(request: GraphPolynomialRequest) -> GraphPolynomialResult:
    return GraphPolynomialResult(terms=chromatic_polynomial(request.graph))


def _run_flow(request: GraphPolynomialRequest) -> GraphPolynomialResult:
    return GraphPolynomialResult(terms=flow_polynomial(request.graph))


def _run_matching(request: MatchingPolynomialRequest) -> GraphPolynomialResult:
    return GraphPolynomialResult(terms=matching_polynomial(request.graph))


class TestTuttePolynomial:
    def test_cycle_c4(self) -> None:
        req = GraphPolynomialRequest(graph=_cycle_graph(4))
        result = _run_tutte(req)
        # T(C4, x, y) = x^3 + x^2 + x + y
        d = {term.exponents: term.coefficient for term in result.terms}
        assert result.variables == ("x", "y")
        assert d == {(0, 1): 1, (1, 0): 1, (2, 0): 1, (3, 0): 1}

    def test_single_edge(self) -> None:
        req = GraphPolynomialRequest(
            graph=IndexedSimpleUndirectedGraph(vertex_count=2, edges=((0, 1),))
        )
        result = _run_tutte(req)
        # T(K2) = x
        d = {term.exponents: term.coefficient for term in result.terms}
        assert d == {(1, 0): 1}


class TestChromaticPolynomial:
    def test_cycle_c3(self) -> None:
        req = GraphPolynomialRequest(graph=_cycle_graph(3))
        result = _run_chromatic(req)
        # chi(C3) = x(x-1)(x-2) = x^3 - 3x^2 + 2x
        d = _terms_to_dict(result)
        assert d.get(3) == 1
        assert d.get(2) == -3
        assert d.get(1) == 2

    def test_path_p3(self) -> None:
        req = GraphPolynomialRequest(graph=_path_graph(3))
        result = _run_chromatic(req)
        # chi(P3) = x(x-1)^2 = x^3 - 2x^2 + x
        d = _terms_to_dict(result)
        assert d.get(3) == 1
        assert d.get(2) == -2
        assert d.get(1) == 1


class TestFlowPolynomial:
    def test_cycle_c4(self) -> None:
        req = GraphPolynomialRequest(graph=_cycle_graph(4))
        result = _run_flow(req)
        # F(C4) = x - 1, from (-1)^{|E|-|V|+k} T(0, 1-x).
        d = _terms_to_dict(result)
        assert d.get(1) == 1
        assert d.get(0) == -1

    def test_rejects_graph_beyond_deletion_budget(self) -> None:
        edges = tuple((j, i) for i in range(8) for j in range(i))
        request = GraphPolynomialRequest(
            graph=IndexedSimpleUndirectedGraph(vertex_count=8, edges=edges),
        )
        with pytest.raises(
            OperationDomainValidationError, match="exact computation envelope"
        ):
            _run_flow(request)

    def test_bridge_is_zero_polynomial(self) -> None:
        req = GraphPolynomialRequest(
            graph=IndexedSimpleUndirectedGraph(vertex_count=2, edges=((0, 1),))
        )
        result = _run_flow(req)
        assert result.terms == ()


class TestMatchingPolynomial:
    def test_single_edge(self) -> None:
        req = MatchingPolynomialRequest(
            graph=IndexedSimpleUndirectedGraph(vertex_count=2, edges=((0, 1),))
        )
        result = _run_matching(req)
        # M(K2) = x^2 - 1
        d = _terms_to_dict(result)
        assert d.get(2) == 1
        assert d.get(0) == -1

    def test_path_p3(self) -> None:
        req = MatchingPolynomialRequest(graph=_path_graph(3))
        result = _run_matching(req)
        # P3 has edges (0,1) and (1,2)
        # 0-matchings: 1, 1-matching: 2, no 2-matchings
        # M = x^3 - 2x
        d = _terms_to_dict(result)
        assert d.get(3) == 1
        assert d.get(1) == -2
