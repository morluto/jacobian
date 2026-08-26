"""Typed wire contracts for graph spectral operations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import WithJsonSchema, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph
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


_MAX_SPECTRAL_VERTICES = 32


def _require_spectral_graph(graph: IndexedSimpleUndirectedGraph) -> None:
    if graph.vertex_count > _MAX_SPECTRAL_VERTICES:
        raise PydanticCustomError(
            "graph.spectral_vertex_count_exceeds_operation_bound",
            "spectral operations support at most 32 vertices",
        )


def _spectral_graph_schema() -> JsonSchemaValue:
    """Project the spectral-operation envelope onto the shared graph value."""

    schema = IndexedSimpleUndirectedGraph.model_json_schema()
    schema["description"] = (
        "An integer-indexed simple undirected graph accepted by the spectral "
        "operations: at most 32 vertices and at most 512 edges."
    )
    schema["properties"]["vertex_count"].update(maximum=_MAX_SPECTRAL_VERTICES)
    schema["properties"]["edges"].update(maxItems=512)
    return schema


SpectralGraph = Annotated[
    IndexedSimpleUndirectedGraph,
    WithJsonSchema(_spectral_graph_schema()),
]


class GraphSpectrumRequest(StrictModel):
    graph: SpectralGraph

    @model_validator(mode="after")
    def require_admitted_graph(self) -> Self:
        _require_spectral_graph(self.graph)
        return self


class GraphSpectrumResult(StrictModel):
    """The exact eigenvalues with algebraic multiplicities of a graph matrix.

    Retains the source graph and matrix convention.  Each ``eigenvalues`` entry is
    the canonical exact SymPy rendering (over the algebraic closure of QQ,
    never a float) of one distinct eigenvalue; multiplicities are positive
    and sum to the graph order, so the claimed spectrum reconstructs
    ``det(xI - M)`` exactly, where ``M`` is the adjacency matrix A for
    ``matrix_convention="ADJACENCY"`` and the Laplacian matrix L for
    ``matrix_convention="LAPLACIAN"``.  Pair order carries no mathematical
    meaning.  The explicit owner verifier compares ``(eigenvalue,
    multiplicity)`` pairs as a multiset when an independently supplied claim
    needs checking.
    """

    graph: SpectralGraph
    matrix_convention: Literal["ADJACENCY", "LAPLACIAN"]
    eigenvalues: tuple[str, ...]
    multiplicities: tuple[int, ...]
    convention: Literal["SYMPY_EIGENVALS"] = "SYMPY_EIGENVALS"

    @model_validator(mode="after")
    def require_structural_shape(self) -> Self:
        _require_spectral_graph(self.graph)
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
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        graph: IndexedSimpleUndirectedGraph,
        matrix_convention: Literal["ADJACENCY", "LAPLACIAN"],
        eigenvalues: tuple[str, ...],
        multiplicities: tuple[int, ...],
    ) -> Self:
        """Construct a spectrum emitted by the trusted exact kernel."""

        return cls.model_construct(
            graph=graph,
            matrix_convention=matrix_convention,
            eigenvalues=eigenvalues,
            multiplicities=multiplicities,
            convention="SYMPY_EIGENVALS",
        )


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
    its source graph and convention; the explicit owner verifier checks the
    defining determinant relation for independently supplied claims.
    """

    graph: SpectralGraph
    convention: Literal["ADJACENCY", "LAPLACIAN"]
    polynomial: RationalPolynomial

    @model_validator(mode="after")
    def require_structural_shape(self) -> Self:
        _require_spectral_graph(self.graph)
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
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        graph: IndexedSimpleUndirectedGraph,
        convention: Literal["ADJACENCY", "LAPLACIAN"],
        polynomial: RationalPolynomial,
    ) -> Self:
        """Construct a polynomial emitted by the trusted exact kernel."""

        return cls.model_construct(
            graph=graph, convention=convention, polynomial=polynomial
        )


__all__ = [
    "GraphCharacteristicPolynomialResult",
    "GraphSpectrumRequest",
    "GraphSpectrumResult",
]
