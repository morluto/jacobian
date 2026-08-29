"""Typed wire contracts for graph spectral operations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import WithJsonSchema, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.graphs.values import (
    IndexedSimpleUndirectedGraph,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
    require_polynomial_budget,
)

_VARIABLE = "x"
# Every order-k Laplacian principal minor is at most n**k by Hadamard, and
# each coefficient sums at most binomial(n, k) such minors. The uniform
# (2n)**n bound therefore covers every coefficient for n <= 256.
_MAX_CHARPOLY_VERTICES = 256
_MAX_CHARPOLY_MATRIX_CELLS = _MAX_CHARPOLY_VERTICES**2
_MAX_CHARPOLY_WORK = _MAX_CHARPOLY_VERTICES**4
_MAX_CHARPOLY_EDGES = _MAX_CHARPOLY_VERTICES * (_MAX_CHARPOLY_VERTICES - 1) // 2


_MAX_SPECTRAL_VERTICES = 32


def _charpoly_coefficient_digit_bound(order: int) -> int:
    return 1 if order == 0 else len(str((2 * order) ** order))


def _require_characteristic_polynomial_graph(
    graph: IndexedSimpleUndirectedGraph,
) -> None:
    order = graph.vertex_count
    if order > _MAX_CHARPOLY_VERTICES:
        raise PydanticCustomError(
            "graph.characteristic_polynomial_vertex_count_exceeds_operation_bound",
            "characteristic-polynomial operations support at most 256 vertices",
        )
    if order**2 > _MAX_CHARPOLY_MATRIX_CELLS or order**4 > _MAX_CHARPOLY_WORK:
        raise PydanticCustomError(
            "graph.characteristic_polynomial_work_exceeds_operation_bound",
            "characteristic-polynomial matrix work exceeds the operation bound",
        )


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


def _characteristic_polynomial_graph_schema() -> JsonSchemaValue:
    schema = IndexedSimpleUndirectedGraph.model_json_schema()
    schema["description"] = (
        "An integer-indexed simple undirected graph accepted by exact "
        "characteristic-polynomial operations: at most 256 vertices and "
        "at most 32640 canonical edges."
    )
    schema["properties"]["vertex_count"].update(maximum=_MAX_CHARPOLY_VERTICES)
    schema["properties"]["edges"].update(maxItems=_MAX_CHARPOLY_EDGES)
    return schema


CharacteristicPolynomialGraph = Annotated[
    IndexedSimpleUndirectedGraph,
    WithJsonSchema(_characteristic_polynomial_graph_schema()),
]


class GraphSpectrumRequest(StrictModel):
    graph: SpectralGraph


class GraphCharacteristicPolynomialRequest(StrictModel):
    """A graph whose dense integer matrix and exact polynomial are bounded."""

    graph: CharacteristicPolynomialGraph


class GraphSpectrumResult(StrictModel):
    """The exact eigenvalues with algebraic multiplicities of a graph matrix.

    Retains the source graph and matrix convention.  Each ``eigenvalues`` entry is
    the canonical exact SymPy rendering (over the algebraic closure of QQ,
    never a float) of one distinct eigenvalue; multiplicities are positive
    and sum to the graph order, so the claimed spectrum reconstructs
    ``det(xI - M)`` exactly, where ``M`` is the adjacency matrix A for
    ``matrix_convention="ADJACENCY"`` and the Laplacian matrix L for
    ``matrix_convention="LAPLACIAN"``. Pair order carries no mathematical
    meaning.
    """

    graph: SpectralGraph
    matrix_convention: Literal["ADJACENCY", "LAPLACIAN"]
    eigenvalues: tuple[str, ...]
    multiplicities: tuple[int, ...]

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
    polynomial operations compose without translation. The result retains its
    source graph and convention.
    """

    graph: CharacteristicPolynomialGraph
    convention: Literal["ADJACENCY", "LAPLACIAN"]
    polynomial: RationalPolynomial

    @model_validator(mode="after")
    def require_structural_shape(self) -> Self:
        _require_characteristic_polynomial_graph(self.graph)
        if self.polynomial.variables != (_VARIABLE,):
            raise PydanticCustomError(
                "graph.characteristic_polynomial_must_be_univariate_in_",
                "characteristic polynomial must be univariate in x",
            )
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=self.graph.vertex_count + 1,
            maximum_exponent=self.graph.vertex_count,
            maximum_coefficient_digits=_charpoly_coefficient_digit_bound(
                self.graph.vertex_count
            ),
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
    "GraphCharacteristicPolynomialRequest",
    "GraphCharacteristicPolynomialResult",
    "GraphSpectrumRequest",
    "GraphSpectrumResult",
]
