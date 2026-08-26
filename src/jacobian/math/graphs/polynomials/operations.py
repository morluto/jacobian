"""Exact native graph-polynomial operations."""

from __future__ import annotations

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.graphs.polynomials._models import _admitted_tree_profile
from jacobian.math.graphs.values import SimpleUndirectedGraph
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
