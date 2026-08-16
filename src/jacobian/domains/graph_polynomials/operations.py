"""Domain adapter for graph polynomial operations backed by NetworkX and SymPy."""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import sympy
from sympy import Poly, Symbol, expand

from jacobian.contracts.graph_polynomials import (
    GraphPolynomialRequest,
    GraphPolynomialResult,
    PolynomialTerm,
)


def _build_graph(request: GraphPolynomialRequest) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(request.graph.vertex_count))
    for edge in request.graph.edges:
        g.add_edge(edge.u, edge.v)
    return g


def _poly_to_terms(poly_expr, var: sympy.Symbol) -> tuple[PolynomialTerm, ...]:
    """Convert a sympy polynomial expression to sorted PolynomialTerm tuples."""
    poly = Poly(poly_expr, var)
    terms: list[PolynomialTerm] = []
    for monom, coeff in poly.terms():
        if coeff == 0:
            continue
        terms.append(PolynomialTerm(coefficient=int(coeff), degree=monom[0]))
    if not terms:
        terms.append(PolynomialTerm(coefficient=0, degree=0))
    return tuple(sorted(terms, key=lambda t: t.degree))


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
        terms.append(
            PolynomialTerm(coefficient=int(coeff), degree=x_deg * 100 + y_deg)
        )
    if not terms:
        terms.append(PolynomialTerm(coefficient=0, degree=0))
    return GraphPolynomialResult(terms=tuple(sorted(terms, key=lambda t: t.degree)))


def compute_chromatic_polynomial(
    request: GraphPolynomialRequest,
) -> GraphPolynomialResult:
    """Compute the exact chromatic polynomial chi_G(x)."""
    x = Symbol("x")
    g = _build_graph(request)
    result = nx.chromatic_polynomial(g)
    terms = _poly_to_terms(result, x)
    return GraphPolynomialResult(terms=terms)


def compute_flow_polynomial(request: GraphPolynomialRequest) -> GraphPolynomialResult:
    """Compute the exact nowhere-zero flow polynomial F_G(x).

    Derived from the Tutte polynomial: F_G(x) = (-1)^n * T_G(0, 1-x).
    """
    g = _build_graph(request)
    tutte = nx.tutte_polynomial(g)
    n = g.number_of_nodes()
    # The Tutte polynomial uses symbols x and y. Extract them.
    sym_x = sympy.Symbol("x")
    sym_y = sympy.Symbol("y")
    flow_x = sympy.Symbol("flow_x")
    # Substitute: Tutte's x -> 0, Tutte's y -> 1 - flow_x
    flow_expr = tutte.subs({sym_x: 0, sym_y: 1 - flow_x})
    flow_expr = (-1) ** n * expand(flow_expr)
    terms = _poly_to_terms(flow_expr, flow_x)
    return GraphPolynomialResult(terms=terms)


def compute_matching_polynomial(
    request: GraphPolynomialRequest,
) -> GraphPolynomialResult:
    """Compute the exact matching polynomial M_G(x).

    M_G(x) = sum_{k=0}^{n/2} (-1)^k * m_k * x^{n - 2k}
    where m_k is the number of k-matchings (sets of k independent edges).
    """
    x = Symbol("x")  # noqa: F841
    g = _build_graph(request)
    n = g.number_of_nodes()
    if n == 0:
        return GraphPolynomialResult(
            terms=(PolynomialTerm(coefficient=1, degree=0),)
        )

    edges = list(g.edges())
    matching_counts = [0] * (n + 1)
    matching_counts[0] = 1

    max_k = min(n // 2, len(edges))
    for k in range(1, max_k + 1):
        for edge_set in combinations(edges, k):
            used: set[int] = set()
            for u, v in edge_set:
                if u in used or v in used:
                    break
                used.add(u)
                used.add(v)
            else:
                matching_counts[k] += 1

    terms: list[PolynomialTerm] = []
    for k in range(n // 2 + 1):
        coeff = ((-1) ** k) * matching_counts[k]
        if coeff != 0:
            degree = n - 2 * k
            terms.append(PolynomialTerm(coefficient=coeff, degree=degree))
    if not terms:
        terms.append(PolynomialTerm(coefficient=0, degree=0))
    return GraphPolynomialResult(terms=tuple(sorted(terms, key=lambda t: t.degree)))


__all__ = [
    "compute_tutte_polynomial",
    "compute_chromatic_polynomial",
    "compute_flow_polynomial",
    "compute_matching_polynomial",
]
