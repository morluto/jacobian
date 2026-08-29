"""Exact native graph-polynomial operations."""

from __future__ import annotations

from functools import cache

import networkx as nx
import sympy
from sympy import Poly, Symbol, expand

from jacobian._exact import CanonicalRational
from jacobian.canonical import CanonicalLimits, format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.polynomials._models import (
    MAX_GRAPH_POLYNOMIAL_EDGES,
    MAX_GRAPH_POLYNOMIAL_VERTICES,
    MAX_MATCHING_EDGES,
    MAX_MATCHING_VERTICES,
    MultivariatePolynomialTerm,
    PolynomialTerm,
    TreeIndependencePolynomialAdmissionError,
    _admitted_tree_profile,
    _maximum_independence_result_bytes,
    _TreeProfile,
)
from jacobian.math.graphs.values import (
    IndexedSimpleUndirectedGraph,
    SimpleUndirectedGraph,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _add_coefficients(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    size = max(len(left), len(right))
    return tuple(
        (left[index] if index < len(left) else 0)
        + (right[index] if index < len(right) else 0)
        for index in range(size)
    )


def _convolve_coefficients(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...]:
    result = [0] * (len(left) + len(right) - 1)
    for left_degree, left_coefficient in enumerate(left):
        for right_degree, right_coefficient in enumerate(right):
            result[left_degree + right_degree] += left_coefficient * right_coefficient
    return tuple(result)


def _compute_independence_coefficients(
    profile: _TreeProfile,
) -> tuple[int, ...]:
    """Run the tree dynamic program using an already-admitted profile."""
    states: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for vertex in profile.postorder:
        excluded: tuple[int, ...] = (1,)
        included: tuple[int, ...] = (0, 1)
        for child in profile.children[vertex]:
            child_excluded, child_included = states.pop(child)
            excluded = _convolve_coefficients(
                excluded,
                _add_coefficients(child_excluded, child_included),
            )
            included = _convolve_coefficients(included, child_excluded)
        states[vertex] = (excluded, included)

    root_excluded, root_included = states[profile.root]
    coefficients = _add_coefficients(root_excluded, root_included)
    if len(coefficients) != profile.independence_degree + 1:
        raise ValueError("independence polynomial degree invariant failed")
    return coefficients


def _admit_tree(graph: SimpleUndirectedGraph) -> _TreeProfile:
    profile = _admitted_tree_profile(graph)
    output_limit = CanonicalLimits().max_output_bytes
    if (
        _maximum_independence_result_bytes(graph, profile.independence_degree)
        > output_limit
    ):
        raise TreeIndependencePolynomialAdmissionError(
            "tree independence polynomial would exceed the canonical output "
            "limit after retaining its source; shorten vertex labels"
        )
    return profile


def independence_polynomial_coefficients(
    graph: SimpleUndirectedGraph,
) -> tuple[int, ...]:
    """Return ``i_0, ..., i_alpha`` for one admitted finite tree.

    This native projection matches the dense coefficients returned alongside
    the canonical sparse ``RationalPolynomial`` by the catalog operation.
    """

    return _compute_independence_coefficients(_admit_tree(graph))


def _polynomial_from_coefficients(
    coefficients: tuple[int, ...],
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(
                        num=format_canonical_integer(coefficient),
                        den="1",
                    ),
                    exponents=(degree,),
                )
                for degree, coefficient in reversed(list(enumerate(coefficients)))
                if coefficient != 0
            )
        ),
    )


def independence_polynomial(graph: SimpleUndirectedGraph) -> RationalPolynomial:
    """Return the exact independence polynomial of one admitted finite tree."""

    return _polynomial_from_coefficients(
        _compute_independence_coefficients(_admit_tree(graph))
    )


def _build_graph(
    graph: IndexedSimpleUndirectedGraph,
    *,
    matching: bool = False,
) -> nx.Graph[int]:
    if matching:
        max_vertices, max_edges = MAX_MATCHING_VERTICES, MAX_MATCHING_EDGES
        label = "matching polynomial"
        code = "graph.matching_polynomial.exact_computation_envelope"
    else:
        max_vertices, max_edges = (
            MAX_GRAPH_POLYNOMIAL_VERTICES,
            MAX_GRAPH_POLYNOMIAL_EDGES,
        )
        label = "graph polynomial"
        code = "graph.polynomial.exact_computation_envelope"
    if graph.vertex_count > max_vertices or len(graph.edges) > max_edges:
        raise OperationDomainValidationError(
            location=("graph",),
            code=code,
            message=f"{label} graph exceeds its exact computation envelope",
        )
    g: nx.Graph[int] = nx.Graph()
    g.add_nodes_from(range(graph.vertex_count))
    g.add_edges_from(graph.edges)
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


def tutte_polynomial(
    graph: IndexedSimpleUndirectedGraph,
) -> tuple[MultivariatePolynomialTerm, ...]:
    """Compute the exact Tutte polynomial T_G(x, y).

    Monomials retain their bivariate exponent tuples.
    """
    x, y = sympy.symbols("x y")
    g = _build_graph(graph)
    result = nx.tutte_polynomial(g)
    poly = Poly(result, x, y)
    terms: list[MultivariatePolynomialTerm] = []
    for monom, coeff in poly.terms():
        if coeff == 0:
            continue
        terms.append(
            MultivariatePolynomialTerm(coefficient=int(coeff), exponents=tuple(monom))
        )
    return tuple(sorted(terms, key=lambda term: term.exponents))


def chromatic_polynomial(
    graph: IndexedSimpleUndirectedGraph,
) -> tuple[PolynomialTerm, ...]:
    """Compute the exact chromatic polynomial chi_G(x)."""
    x = Symbol("x")
    g = _build_graph(graph)
    result = nx.chromatic_polynomial(g)
    return _poly_to_terms(result, x)


def flow_polynomial(graph: IndexedSimpleUndirectedGraph) -> tuple[PolynomialTerm, ...]:
    """Compute the exact nowhere-zero flow polynomial F_G(x).

    The identity is F_G(x) = (-1)^{|E|-|V|+k(G)} T_G(0, 1-x).
    """
    g = _build_graph(graph)
    tutte = nx.tutte_polynomial(g)
    components = nx.number_connected_components(g)
    sign = (-1) ** (g.number_of_edges() - g.number_of_nodes() + components)
    flow_x = sympy.Symbol("flow_x")
    flow_expr = tutte.subs({sympy.Symbol("x"): 0, sympy.Symbol("y"): 1 - flow_x})
    flow_expr = sign * expand(flow_expr)
    return _poly_to_terms(flow_expr, flow_x)


def matching_polynomial(
    graph: IndexedSimpleUndirectedGraph,
) -> tuple[PolynomialTerm, ...]:
    """Compute the exact matching polynomial M_G(x).

    M_G(x) = sum_{k} (-1)^k m_k x^{n-2k}, computed by the deletion recurrence
    on induced subgraphs of at most 16 vertices.
    """
    g = _build_graph(graph, matching=True)
    n = g.number_of_nodes()
    if n == 0:
        return (PolynomialTerm(coefficient=1, degree=0),)

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
    return terms


__all__ = [
    "chromatic_polynomial",
    "flow_polynomial",
    "independence_polynomial",
    "independence_polynomial_coefficients",
    "matching_polynomial",
    "tutte_polynomial",
]
