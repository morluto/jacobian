"""Typed wire contracts for graph spectral operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
    require_polynomial_budget,
)

_VARIABLE = "x"
_MAX_CHARPOLY_TERMS = 33
_MAX_CHARPOLY_EXPONENT = 32
_MAX_CHARPOLY_COEFFICIENT_DIGITS = 128


class GraphEdgeList(StrictModel):
    """A simple undirected graph given by an edge list."""

    vertex_count: int = Field(ge=1, le=32)
    edges: tuple[tuple[int, int], ...] = Field(max_length=512)

    @model_validator(mode="after")
    def require_simple_graph(self) -> Self:
        seen: set[tuple[int, int]] = set()
        for u, v in self.edges:
            if not (0 <= u < self.vertex_count and 0 <= v < self.vertex_count):
                raise PydanticCustomError(
                    "graph.edge_vertices_must_be_in_0_vertex_count_1",
                    "edge vertices must be in 0..vertex_count-1",
                )
            if u == v:
                raise PydanticCustomError(
                    "graph.a_simple_graph_cannot_contain_self_loops",
                    "a simple graph cannot contain self-loops",
                )
            edge = (min(u, v), max(u, v))
            if edge in seen:
                raise PydanticCustomError(
                    "graph.a_simple_graph_cannot_contain_duplicate_edges",
                    "a simple graph cannot contain duplicate edges",
                )
            seen.add(edge)
        return self


class GraphSpectrumRequest(StrictModel):
    graph: GraphEdgeList


class GraphSpectrumResult(StrictModel):
    """The exact eigenvalues with algebraic multiplicities of a graph matrix.

    Retains the source graph and matrix convention so validation replays
    the exact spectrum of the same matrix.  Each ``eigenvalues`` entry is
    the canonical exact SymPy rendering (over the algebraic closure of QQ,
    never a float) of one distinct eigenvalue; multiplicities are positive
    and sum to the graph order, so the claimed spectrum reconstructs
    ``det(xI - M)`` exactly, where ``M`` is the adjacency matrix A for
    ``matrix_convention="ADJACENCY"`` and the Laplacian matrix L for
    ``matrix_convention="LAPLACIAN"``.  Pair order carries no mathematical
    meaning: validation compares ``(eigenvalue, multiplicity)`` pairs as a
    multiset rather than by backend iteration order.
    """

    graph: GraphEdgeList
    matrix_convention: Literal["ADJACENCY", "LAPLACIAN"]
    eigenvalues: tuple[str, ...]
    multiplicities: tuple[int, ...]
    convention: Literal["SYMPY_EIGENVALS"] = "SYMPY_EIGENVALS"

    @model_validator(mode="after")
    def require_source_bound(self) -> Self:
        from jacobian.math.graphs.spectral.operations import (
            adjacency_spectrum,
            laplacian_spectrum,
        )

        order = self.graph.vertex_count
        if len(self.eigenvalues) != len(self.multiplicities):
            raise PydanticCustomError(
                "graph.eigenvalue_multiplicity_tuples_have_equal_length",
                "eigenvalue and multiplicity tuples must have equal length",
            )
        if len(set(self.eigenvalues)) != len(self.eigenvalues):
            raise PydanticCustomError(
                "graph.eigenvalues_must_be_distinct", "eigenvalues must be distinct"
            )
        if any(multiplicity < 1 for multiplicity in self.multiplicities):
            raise PydanticCustomError(
                "graph.algebraic_multiplicities_must_be_positive",
                "algebraic multiplicities must be positive",
            )
        if sum(self.multiplicities) != order:
            raise PydanticCustomError(
                "graph.multiplicities_must_sum_to_the_graph_order",
                "multiplicities must sum to the graph order",
            )
        replayed = (
            adjacency_spectrum(self.graph)
            if self.matrix_convention == "ADJACENCY"
            else laplacian_spectrum(self.graph)
        )
        claimed = dict(zip(self.eigenvalues, self.multiplicities, strict=True))
        if dict(replayed) != claimed:
            raise PydanticCustomError(
                "graph.spectrum_must_be_the_exact_spectrum_of_the_sourc",
                "spectrum must be the exact spectrum of the source graph",
            )
        return self


def _dense_to_canonical_polynomial(
    coefficients: tuple[CanonicalRational, ...],
) -> RationalPolynomial:
    """Convert increasing-degree dense coefficients to the canonical value."""
    terms = tuple(
        RationalPolynomialTerm(coefficient=coefficient, exponents=(degree,))
        for degree, coefficient in reversed(list(enumerate(coefficients)))
        if coefficient.as_fraction() != 0
    )
    return RationalPolynomial(
        variables=(_VARIABLE,),
        polynomial=SparseRationalPolynomial(terms=terms),
    )


class GraphCharacteristicPolynomialResult(StrictModel):
    """The exact monic characteristic polynomial of a graph matrix over QQ.

    ``polynomial`` is the domain-owned canonical sparse value so downstream
    polynomial operations compose without translation.  The result retains
    its source graph and convention so validation can replay the defining
    determinant relation.
    """

    graph: GraphEdgeList
    convention: Literal["ADJACENCY", "LAPLACIAN"]
    polynomial: RationalPolynomial

    @model_validator(mode="after")
    def require_bound_to_source(self) -> Self:
        from jacobian.math.graphs.spectral.operations import (
            _adjacency_matrix,
            _laplacian_matrix,
        )
        from jacobian.math.polynomials._conversions import (
            rational_polynomial_to_sympy,
        )

        if self.polynomial.variables != (_VARIABLE,):
            raise PydanticCustomError(
                "graph.characteristic_polynomial_must_be_univariate_in_",
                "characteristic polynomial must be univariate in x",
            )
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=_MAX_CHARPOLY_TERMS,
            maximum_exponent=_MAX_CHARPOLY_EXPONENT,
            maximum_coefficient_digits=_MAX_CHARPOLY_COEFFICIENT_DIGITS,
            label="characteristic polynomial",
        )
        matrix = (
            _adjacency_matrix(self.graph)
            if self.convention == "ADJACENCY"
            else _laplacian_matrix(self.graph)
        )
        poly_sym = rational_polynomial_to_sympy(self.polynomial)
        actual = poly_sym.as_expr()
        charpoly = matrix.charpoly()
        expected = charpoly.as_expr().subs(charpoly.gen, poly_sym.gen)
        if expected != actual:
            raise PydanticCustomError(
                "graph.characteristic_polynomial_does_match_source",
                "characteristic polynomial does not match the source graph",
            )
        return self


__all__ = [
    "GraphCharacteristicPolynomialResult",
    "GraphEdgeList",
    "GraphSpectrumRequest",
    "GraphSpectrumResult",
]
