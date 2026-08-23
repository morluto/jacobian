"""Typed wire contracts for graph spectral operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

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
                raise ValueError("edge vertices must be in 0..vertex_count-1")
            if u == v:
                raise ValueError("a simple graph cannot contain self-loops")
            edge = (min(u, v), max(u, v))
            if edge in seen:
                raise ValueError("a simple graph cannot contain duplicate edges")
            seen.add(edge)
        return self


class GraphSpectrumRequest(StrictModel):
    graph: GraphEdgeList


class GraphSpectrumResult(StrictModel):
    """The exact eigenvalues with algebraic multiplicities of a graph matrix."""

    eigenvalues: tuple[str, ...]
    multiplicities: tuple[int, ...]
    convention: Literal["SYMPY_EIGENVALS"] = "SYMPY_EIGENVALS"


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
            raise ValueError("characteristic polynomial must be univariate in x")
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
            raise ValueError(
                "characteristic polynomial does not match the source graph"
            )
        return self


__all__ = [
    "GraphCharacteristicPolynomialResult",
    "GraphEdgeList",
    "GraphSpectrumRequest",
    "GraphSpectrumResult",
]
