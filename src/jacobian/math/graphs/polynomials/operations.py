"""Exact native graph-polynomial operations."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

MAX_INDEPENDENCE_POLYNOMIAL_VERTICES = 256
# The canonical graph representation accepts at most 256 vertices, and every
# budget below derives from that input envelope rather than from any other
# operation's consumer limit:
#   - a tree on n vertices has independence number at most n - 1, so its
#     dense coefficient profile carries at most n terms;
#   - each coefficient counts independent k-element vertex sets, hence is at
#     most 2^n, and the total independent-set count sums those coefficients;
#   - the kernel performs exactly two dense convolutions per rooted edge and
#     each factor carries at most one coefficient per term.
MAX_INDEPENDENCE_POLYNOMIAL_TERMS = MAX_INDEPENDENCE_POLYNOMIAL_VERTICES
MAX_INDEPENDENCE_POLYNOMIAL_EXPONENT = MAX_INDEPENDENCE_POLYNOMIAL_TERMS - 1
MAX_INDEPENDENCE_POLYNOMIAL_COEFFICIENT_DIGITS = len(
    format_canonical_integer(1 << MAX_INDEPENDENCE_POLYNOMIAL_VERTICES)
)
MAX_INDEPENDENCE_CONVOLUTION_PRODUCTS_PER_PASS = (
    2
    * (MAX_INDEPENDENCE_POLYNOMIAL_VERTICES - 1)
    * MAX_INDEPENDENCE_POLYNOMIAL_TERMS**2
)


@dataclass(frozen=True, slots=True)
class _TreeProfile:
    root: str
    children: dict[str, tuple[str, ...]]
    postorder: tuple[str, ...]
    independence_degree: int
    convolution_products: int


def _admitted_tree_profile(graph: SimpleUndirectedGraph) -> _TreeProfile:
    """Validate the native tree/work domain before expanding coefficients."""

    import networkx as nx

    from jacobian.math.graphs._networkx import graph_from_value

    vertex_count = len(graph.vertices)
    if vertex_count == 0:
        raise ValueError("independence polynomial requires a nonempty tree")
    if vertex_count > MAX_INDEPENDENCE_POLYNOMIAL_VERTICES:
        raise ValueError(
            "independence polynomial supports at most "
            f"{MAX_INDEPENDENCE_POLYNOMIAL_VERTICES} vertices"
        )

    transient = graph_from_value(graph)
    if not nx.is_tree(transient):
        raise ValueError("independence polynomial requires a connected acyclic graph")

    root = graph.vertices[0]
    child_lists: dict[str, list[str]] = {vertex: [] for vertex in graph.vertices}
    order = [root]
    for parent, child in nx.bfs_edges(
        transient,
        source=root,
        sort_neighbors=sorted,
    ):
        child_lists[parent].append(child)
        order.append(child)
    children = {vertex: tuple(child_lists[vertex]) for vertex in graph.vertices}
    postorder = tuple(reversed(order))

    excluded_degree: dict[str, int] = {}
    included_degree: dict[str, int] = {}
    convolution_products = 0
    for vertex in postorder:
        excluded = 0
        included = 1
        for child in children[vertex]:
            child_total = max(excluded_degree[child], included_degree[child])
            convolution_products += (excluded + 1) * (child_total + 1)
            convolution_products += (included + 1) * (excluded_degree[child] + 1)
            excluded += child_total
            included += excluded_degree[child]
        excluded_degree[vertex] = excluded
        included_degree[vertex] = included

    independence_degree = max(excluded_degree[root], included_degree[root])
    if convolution_products > MAX_INDEPENDENCE_CONVOLUTION_PRODUCTS_PER_PASS:
        raise ValueError(
            "tree independence polynomial exceeds the "
            f"{MAX_INDEPENDENCE_CONVOLUTION_PRODUCTS_PER_PASS}-product "
            "coefficient-convolution budget"
        )

    return _TreeProfile(
        root=root,
        children=children,
        postorder=postorder,
        independence_degree=independence_degree,
        convolution_products=convolution_products,
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


def independence_polynomial_coefficients(
    graph: SimpleUndirectedGraph,
) -> tuple[int, ...]:
    """Return ``i_0, ..., i_alpha`` for one admitted finite tree.

    This native projection matches the dense coefficients returned alongside
    the canonical sparse ``RationalPolynomial`` by the catalog operation.
    """

    profile = _admitted_tree_profile(graph)
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

    return _polynomial_from_coefficients(independence_polynomial_coefficients(graph))


__all__ = [
    "independence_polynomial",
    "independence_polynomial_coefficients",
]
