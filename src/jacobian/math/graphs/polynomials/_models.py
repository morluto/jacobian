"""Typed wire contracts for exact graph polynomial operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, ExactInteger
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math.graphs.values import (
    IndexedSimpleUndirectedGraph,
    SimpleUndirectedGraph,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
    require_polynomial_budget,
)

MAX_INDEPENDENCE_POLYNOMIAL_VERTICES = 256
# The canonical graph representation accepts at most 256 vertices, and every
# budget below derives from that input envelope rather than from any other
# operation's consumer limit. A tree on n vertices has at most n independence
# polynomial terms, each coefficient is at most 2^n, and the dynamic program
# performs two dense convolutions for every rooted edge.
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


class TreeIndependencePolynomialAdmissionError(ValueError):
    """Native admission failure for a tree independence polynomial."""


@dataclass(frozen=True, slots=True)
class _TreeProfile:
    """Pure structural admission data shared by the tree kernel."""

    root: str
    children: dict[str, tuple[str, ...]]
    postorder: tuple[str, ...]
    independence_degree: int
    convolution_products: int


def _admitted_tree_profile(graph: SimpleUndirectedGraph) -> _TreeProfile:
    """Validate one tree and bound its coefficient-convolution work.

    This preflight deliberately uses only canonical graph values.  NetworkX is
    a private kernel dependency and is not needed to establish connectedness,
    acyclicity, or the fixed dynamic-program work envelope.
    """

    vertex_count = len(graph.vertices)
    if vertex_count == 0:
        raise TreeIndependencePolynomialAdmissionError(
            "independence polynomial requires a nonempty tree"
        )
    if vertex_count > MAX_INDEPENDENCE_POLYNOMIAL_VERTICES:
        raise TreeIndependencePolynomialAdmissionError(
            "independence polynomial supports at most "
            f"{MAX_INDEPENDENCE_POLYNOMIAL_VERTICES} vertices"
        )
    if len(graph.edges) != vertex_count - 1:
        raise TreeIndependencePolynomialAdmissionError(
            "independence polynomial requires a connected acyclic graph"
        )

    adjacency: dict[str, list[str]] = {vertex: [] for vertex in graph.vertices}
    for left, right in graph.edges:
        adjacency[left].append(right)
        adjacency[right].append(left)

    root = graph.vertices[0]
    parents: dict[str, str | None] = {root: None}
    order = [root]
    for parent in order:
        for child in sorted(adjacency[parent]):
            if child == parents[parent]:
                continue
            if child in parents:
                raise TreeIndependencePolynomialAdmissionError(
                    "independence polynomial requires a connected acyclic graph"
                )
            parents[child] = parent
            order.append(child)
    if len(order) != vertex_count:
        raise TreeIndependencePolynomialAdmissionError(
            "independence polynomial requires a connected acyclic graph"
        )

    child_lists: dict[str, list[str]] = {vertex: [] for vertex in graph.vertices}
    for child, predecessor in parents.items():
        if predecessor is not None:
            child_lists[predecessor].append(child)
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
        raise TreeIndependencePolynomialAdmissionError(
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


MAX_GRAPH_POLYNOMIAL_VERTICES = 12
MAX_GRAPH_POLYNOMIAL_EDGES = 24


class GraphPolynomialRequest(StrictModel):
    """Request a Tutte, chromatic, or flow polynomial on a tractable graph."""

    graph: IndexedSimpleUndirectedGraph


MAX_MATCHING_VERTICES = 16
MAX_MATCHING_EDGES = 48


class MatchingPolynomialRequest(StrictModel):
    """Request a matching polynomial on a graph this recurrence can exhaust."""

    graph: IndexedSimpleUndirectedGraph


class TreeIndependencePolynomialRequest(StrictModel):
    """Request the exact independence polynomial of one admitted tree.

    The materialized canonical graph must be nonempty, connected, and acyclic.
    Admission derives every budget from this operation's own work and result
    envelope: at most 256 vertices and a scalar preflight whose exact
    coefficient-convolution count stays inside the kernel pass budget.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Compute the exact independence polynomial of one materialized "
                "canonical tree. The graph must be nonempty, connected, and "
                "acyclic. Admission bounds rooted convolution work by an exact "
                "scalar preflight; coefficient count and digit length are "
                "bounded by the admitted graph order."
            )
        }
    )

    graph: SimpleUndirectedGraph = Field(
        description=(
            "Canonical finite simple undirected graph. It may contain at most "
            f"{MAX_INDEPENDENCE_POLYNOMIAL_VERTICES} vertices and must be a "
            "nonempty tree whose independence polynomial fits this "
            "operation's convolution-work and coefficient-digit bounds."
        )
    )


class TreeIndependencePolynomialResult(StrictModel):
    """One source-bound exact independence polynomial and its defining values.

    The trusted tree kernel constructs this result without making Pydantic
    validation execute the coefficient dynamic program again.
    """

    graph: SimpleUndirectedGraph
    polynomial: RationalPolynomial
    independence_number: StrictInt = Field(
        ge=1,
        le=MAX_INDEPENDENCE_POLYNOMIAL_EXPONENT,
        description="The tree independence number alpha(T).",
    )
    independent_set_count: Annotated[ExactInteger, Field(ge=0)] = Field(
        description="The total I_T(1) as a canonical decimal integer."
    )

    @property
    def coefficients(self) -> tuple[int, ...]:
        """Native dense integer projection; no duplicate polynomial is serialized."""
        terms = self.polynomial.polynomial.terms
        if any(term.coefficient.den != 1 for term in terms):
            raise ValueError(
                "integer coefficient projection requires integral coefficients"
            )
        degree = max((term.exponents[0] for term in terms), default=0)
        values = {
            term.exponents[0]: term.coefficient.as_fraction().numerator
            for term in terms
        }
        return tuple(values.get(i, 0) for i in range(degree + 1))

    @model_validator(mode="after")
    def require_structural_shape(self) -> Self:
        if self.polynomial.variables != ("x",):
            raise PydanticCustomError(
                "graph.independence_polynomial_must_belong_to_qq_x",
                "independence polynomial must belong to QQ[x]",
            )
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=MAX_INDEPENDENCE_POLYNOMIAL_TERMS,
            maximum_exponent=MAX_INDEPENDENCE_POLYNOMIAL_EXPONENT,
            maximum_coefficient_digits=MAX_INDEPENDENCE_POLYNOMIAL_COEFFICIENT_DIGITS,
            label="independence polynomial",
        )
        if (
            len(format_canonical_integer(self.independent_set_count))
            > MAX_INDEPENDENCE_POLYNOMIAL_COEFFICIENT_DIGITS
        ):
            raise PydanticCustomError(
                "graph.independent_set_count_exceeds_max_independence_polynomial",
                "independent-set count exceeds the "
                f"{MAX_INDEPENDENCE_POLYNOMIAL_COEFFICIENT_DIGITS}-digit bound",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        graph: SimpleUndirectedGraph,
        coefficients: tuple[int, ...],
    ) -> Self:
        """Construct a result emitted by the trusted tree polynomial kernel."""

        return cls.model_construct(
            graph=graph,
            polynomial=_polynomial_from_dense_coefficients(coefficients),
            independence_number=len(coefficients) - 1,
            independent_set_count=sum(coefficients),
        )


def _polynomial_from_dense_coefficients(
    coefficients: tuple[int, ...],
) -> RationalPolynomial:
    """Construct the canonical QQ[x] value carried by a dense coefficient list."""

    return RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(
                        num=coefficient,
                        den=1,
                    ),
                    exponents=(degree,),
                )
                for degree, coefficient in reversed(list(enumerate(coefficients)))
                if coefficient != 0
            )
        ),
    )


class PolynomialTerm(StrictModel):
    """One monomial term: coefficient times x^degree."""

    coefficient: int
    degree: int = Field(ge=0)


class GraphPolynomialResult(StrictModel):
    """A graph polynomial over QQ with its source and variable convention."""

    graph: IndexedSimpleUndirectedGraph
    kind: Literal["CHROMATIC", "FLOW", "MATCHING", "TUTTE"]
    polynomial: RationalPolynomial

    @model_validator(mode="after")
    def require_variable_convention(self) -> Self:
        variables = (
            ("x", "y")
            if self.kind == "TUTTE"
            else ("flow_x",)
            if self.kind == "FLOW"
            else ("x",)
        )
        if self.polynomial.variables != variables:
            raise PydanticCustomError(
                "graph.polynomial_variable_convention",
                "polynomial axes must match the graph-polynomial convention",
            )
        return self


class MultivariatePolynomialTerm(StrictModel):
    coefficient: int
    exponents: tuple[int, ...]

    @model_validator(mode="after")
    def require_nonzero_nonnegative_term(self) -> Self:
        if self.coefficient == 0 or any(exponent < 0 for exponent in self.exponents):
            raise PydanticCustomError(
                "graph.multivariate_terms_require_nonzero_coefficients_nonnegative_exponents",
                "multivariate terms require nonzero coefficients and nonnegative exponents",
            )
        return self


class SparseMultivariatePolynomial(StrictModel):
    variables: tuple[str, ...] = Field(min_length=1)
    terms: tuple[MultivariatePolynomialTerm, ...]

    @model_validator(mode="after")
    def require_canonical_terms(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise PydanticCustomError(
                "graph.polynomial_variables_must_be_unique",
                "polynomial variables must be unique",
            )
        exponents = [term.exponents for term in self.terms]
        if any(len(item) != len(self.variables) for item in exponents):
            raise PydanticCustomError(
                "graph.every_exponent_tuple_must_match_the_variable_axi",
                "every exponent tuple must match the variable axis",
            )
        if exponents != sorted(exponents) or len(set(exponents)) != len(exponents):
            raise PydanticCustomError(
                "graph.multivariate_terms_have_unique_sorted_exponent_tuples",
                "multivariate terms must have unique sorted exponent tuples",
            )
        return self


__all__ = [
    "GraphPolynomialRequest",
    "GraphPolynomialResult",
    "MatchingPolynomialRequest",
    "MultivariatePolynomialTerm",
    "PolynomialTerm",
    "SparseMultivariatePolynomial",
    "TreeIndependencePolynomialRequest",
    "TreeIndependencePolynomialResult",
]
