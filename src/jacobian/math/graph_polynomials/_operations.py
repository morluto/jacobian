"""Domain-owned graph polynomial operations backed by NetworkX and SymPy."""

from __future__ import annotations

from functools import cache

import networkx as nx
import sympy
from sympy import Poly, Symbol, expand

from jacobian.math.graph_polynomials._models import (
    GraphPolynomialRequest,
    GraphPolynomialResult,
    MatchingPolynomialRequest,
    PolynomialTerm,
)


def _build_graph(
    request: GraphPolynomialRequest | MatchingPolynomialRequest,
) -> nx.Graph[int]:
    g = nx.Graph()  # type: ignore[var-annotated]
    g.add_nodes_from(range(request.graph.vertex_count))
    for edge in request.graph.edges:
        g.add_edge(edge.u, edge.v)
    return g


def _poly_to_terms(poly_expr: object, var: sympy.Symbol) -> tuple[PolynomialTerm, ...]:
    """Convert a sympy polynomial expression to sorted nonzero PolynomialTerm tuples."""
    poly = Poly(poly_expr, var)
    terms: list[PolynomialTerm] = []
    for monom, coeff in poly.terms():
        if coeff == 0:
            continue
        terms.append(PolynomialTerm(coefficient=int(coeff), degree=monom[0]))
    return tuple(sorted(terms, key=lambda term: term.degree))


def compute_tutte_polynomial(request: GraphPolynomialRequest) -> GraphPolynomialResult:
    """Compute the exact Tutte polynomial T_G(x, y).

    Terms are encoded as (coefficient, degree) where degree = x_deg * 100 + y_deg
    to fit the single-int constraint of PolynomialTerm.
    """
    x, y = sympy.symbols("x y")
    g = _build_graph(request)
    result = nx.tutte_polynomial(g)
    poly = Poly(result, x, y)
    terms: list[PolynomialTerm] = []
    for monom, coeff in poly.terms():
        if coeff == 0:
            continue
        x_deg, y_deg = monom
        terms.append(PolynomialTerm(coefficient=int(coeff), degree=x_deg * 100 + y_deg))
    return GraphPolynomialResult(
        terms=tuple(sorted(terms, key=lambda term: term.degree))
    )


def compute_chromatic_polynomial(
    request: GraphPolynomialRequest,
) -> GraphPolynomialResult:
    """Compute the exact chromatic polynomial chi_G(x)."""
    x = Symbol("x")
    g = _build_graph(request)
    result = nx.chromatic_polynomial(g)
    return GraphPolynomialResult(terms=_poly_to_terms(result, x))


def compute_flow_polynomial(request: GraphPolynomialRequest) -> GraphPolynomialResult:
    """Compute the exact nowhere-zero flow polynomial F_G(x).

    The identity is F_G(x) = (-1)^{|E|-|V|+k(G)} T_G(0, 1-x).
    """
    g = _build_graph(request)
    tutte = nx.tutte_polynomial(g)
    components = nx.number_connected_components(g)
    sign = (-1) ** (g.number_of_edges() - g.number_of_nodes() + components)
    flow_x = sympy.Symbol("flow_x")
    flow_expr = tutte.subs({sympy.Symbol("x"): 0, sympy.Symbol("y"): 1 - flow_x})
    flow_expr = sign * expand(flow_expr)
    return GraphPolynomialResult(terms=_poly_to_terms(flow_expr, flow_x))


def compute_matching_polynomial(
    request: MatchingPolynomialRequest,
) -> GraphPolynomialResult:
    """Compute the exact matching polynomial M_G(x).

    M_G(x) = sum_{k} (-1)^k m_k x^{n-2k}, computed by the deletion recurrence
    on induced subgraphs of at most 16 vertices.
    """
    g = _build_graph(request)
    n = g.number_of_nodes()
    if n == 0:
        return GraphPolynomialResult(terms=(PolynomialTerm(coefficient=1, degree=0),))

    adjacency = [0] * n
    for u, v in g.edges():
        adjacency[u] |= 1 << v
        adjacency[v] |= 1 << u

    @cache
    def coefficients(mask: int) -> tuple[int, ...]:
        bits = mask.bit_count()
        if bits == 0:
            return (1,)
        vertex = (mask & -mask).bit_length() - 1
        rest = mask ^ (1 << vertex)
        without = coefficients(rest)
        result = [0] * (bits + 1)
        for degree, coeff in enumerate(without):
            result[degree + 1] += coeff
        neighbors = adjacency[vertex] & rest
        while neighbors:
            bit = neighbors & -neighbors
            neighbor = bit.bit_length() - 1
            deleted = coefficients(rest ^ (1 << neighbor))
            for degree, coeff in enumerate(deleted):
                result[degree] -= coeff
            neighbors ^= bit
        return tuple(result)

    terms = tuple(
        PolynomialTerm(coefficient=coeff, degree=degree)
        for degree, coeff in enumerate(coefficients((1 << n) - 1))
        if coeff
    )
    return GraphPolynomialResult(terms=terms)


__all__ = [
    "compute_chromatic_polynomial",
    "compute_flow_polynomial",
    "compute_matching_polynomial",
    "compute_tutte_polynomial",
]
