"""Typed wire contracts for exact graph polynomial operations."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import ConfigDict, Field, StrictInt, StringConstraints, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
)
from jacobian.math.graphs.polynomials.operations import (
    MAX_INDEPENDENCE_POLYNOMIAL_COEFFICIENT_DIGITS,
    MAX_INDEPENDENCE_POLYNOMIAL_EXPONENT,
    MAX_INDEPENDENCE_POLYNOMIAL_TERMS,
    MAX_INDEPENDENCE_POLYNOMIAL_VERTICES,
    _admitted_tree_profile,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    require_polynomial_budget,
)

_NonnegativeCanonicalInteger = Annotated[
    str,
    StringConstraints(pattern=r"^(?:0|[1-9][0-9]*)$", strict=True),
]


class GraphEdge(StrictModel):
    """One undirected edge between two non-negative vertex indices."""

    u: int = Field(ge=0)
    v: int = Field(ge=0)

    @model_validator(mode="after")
    def require_no_loop(self) -> Self:
        if self.u == self.v:
            raise ValueError("graph edges must not be loops")
        return self


class GraphSpec(StrictModel):
    """A finite simple undirected graph as vertex count + edge list."""

    vertex_count: int = Field(ge=0, le=64)
    edges: tuple[GraphEdge, ...] = Field(default=(), max_length=512)

    @model_validator(mode="after")
    def require_valid_edges(self) -> Self:
        for edge in self.edges:
            if edge.u >= self.vertex_count or edge.v >= self.vertex_count:
                raise ValueError("edge endpoints must be < vertex_count")
        seen: set[tuple[int, int]] = set()
        for edge in self.edges:
            key = (min(edge.u, edge.v), max(edge.u, edge.v))
            if key in seen:
                raise ValueError("duplicate edges are not allowed")
            seen.add(key)
        return self


MAX_GRAPH_POLYNOMIAL_VERTICES = 12
MAX_GRAPH_POLYNOMIAL_EDGES = 24


class GraphPolynomialRequest(StrictModel):
    """Request a Tutte, chromatic, or flow polynomial on a tractable graph."""

    graph: GraphSpec

    @model_validator(mode="after")
    def require_deletion_contraction_budget(self) -> Self:
        if self.graph.vertex_count > MAX_GRAPH_POLYNOMIAL_VERTICES:
            raise ValueError(
                "Tutte, chromatic, and flow polynomials may have at most "
                f"{MAX_GRAPH_POLYNOMIAL_VERTICES} vertices"
            )
        if len(self.graph.edges) > MAX_GRAPH_POLYNOMIAL_EDGES:
            raise ValueError(
                "Tutte, chromatic, and flow polynomials may have at most "
                f"{MAX_GRAPH_POLYNOMIAL_EDGES} edges"
            )
        return self


MAX_MATCHING_VERTICES = 16
MAX_MATCHING_EDGES = 48


class MatchingPolynomialRequest(StrictModel):
    """Request a matching polynomial on a graph this recurrence can exhaust."""

    graph: GraphSpec

    @model_validator(mode="after")
    def require_matching_budget(self) -> Self:
        if self.graph.vertex_count > MAX_MATCHING_VERTICES:
            raise ValueError(
                f"matching polynomial graphs may have at most {MAX_MATCHING_VERTICES} vertices"
            )
        if len(self.graph.edges) > MAX_MATCHING_EDGES:
            raise ValueError(
                f"matching polynomial graphs may have at most {MAX_MATCHING_EDGES} edges"
            )
        return self


def _maximum_independence_result_bytes(
    graph: SimpleUndirectedGraph,
    degree: int,
) -> int:
    """Bound the complete source-bound result at one admitted degree."""

    maximum_coefficient = "9" * MAX_INDEPENDENCE_POLYNOMIAL_COEFFICIENT_DIGITS
    payload = {
        "graph": graph.model_dump(mode="json"),
        "coefficients": [maximum_coefficient] * (degree + 1),
        "polynomial": {
            "polynomial_schema_version": "1",
            "domain": "QQ",
            "variables": ["x"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {
                            "num": maximum_coefficient,
                            "den": "1",
                        },
                        "exponents": [exponent],
                    }
                    for exponent in range(degree, -1, -1)
                ]
            },
        },
        "independence_number": degree,
        "independent_set_count": maximum_coefficient,
    }
    output_limit = CanonicalLimits().max_output_bytes
    measurement_limits = CanonicalLimits(max_output_bytes=2 * output_limit)
    return len(encode_strict_json(payload, limits=measurement_limits))


class TreeIndependencePolynomialRequest(StrictModel):
    """Request the exact independence polynomial of one admitted tree.

    The materialized canonical graph must be nonempty, connected, and acyclic.
    Admission derives every budget from this operation's own work and result
    envelope: at most 256 vertices, a scalar preflight whose exact
    coefficient-convolution count stays inside the kernel pass budget, and a
    serialized-result reservation that keeps the echoed source plus dense
    coefficients inside the canonical output limit.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Compute the exact independence polynomial of one materialized "
                "canonical tree. The graph must be nonempty, connected, and "
                "acyclic. Admission bounds rooted convolution work by an exact "
                "scalar preflight and reserves canonical-output space for the "
                "echoed source and dense coefficients."
            )
        }
    )

    graph: SimpleUndirectedGraph = Field(
        description=(
            "Canonical finite simple undirected graph. It may contain at most "
            f"{MAX_INDEPENDENCE_POLYNOMIAL_VERTICES} vertices and must be a "
            "nonempty tree whose independence polynomial fits this "
            "operation's convolution-work and serialized-output budgets."
        )
    )

    @model_validator(mode="after")
    def require_admitted_tree_and_output(self) -> Self:
        profile = _admitted_tree_profile(self.graph)
        output_limit = CanonicalLimits().max_output_bytes
        if (
            _maximum_independence_result_bytes(
                self.graph,
                profile.independence_degree,
            )
            > output_limit
        ):
            raise ValueError(
                "tree independence polynomial would exceed the "
                f"{output_limit}-byte canonical output limit after retaining "
                "its source; shorten vertex labels"
            )
        return self


class TreeIndependencePolynomialResult(StrictModel):
    """One source-bound exact independence polynomial and its defining values."""

    graph: SimpleUndirectedGraph
    coefficients: tuple[_NonnegativeCanonicalInteger, ...] = Field(
        min_length=2,
        max_length=MAX_INDEPENDENCE_POLYNOMIAL_TERMS,
        description=(
            "Dense coefficients i_0, ..., i_alpha as canonical decimal "
            "nonnegative integers."
        ),
    )
    polynomial: RationalPolynomial
    independence_number: StrictInt = Field(
        ge=1,
        le=MAX_INDEPENDENCE_POLYNOMIAL_EXPONENT,
        description="The tree independence number alpha(T).",
    )
    independent_set_count: _NonnegativeCanonicalInteger = Field(
        description="The total I_T(1) as a canonical decimal integer."
    )

    @model_validator(mode="after")
    def require_values_bound_to_source(self) -> Self:
        if self.polynomial.variables != ("x",):
            raise ValueError("independence polynomial must belong to QQ[x]")
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=MAX_INDEPENDENCE_POLYNOMIAL_TERMS,
            maximum_exponent=MAX_INDEPENDENCE_POLYNOMIAL_EXPONENT,
            maximum_coefficient_digits=MAX_INDEPENDENCE_POLYNOMIAL_COEFFICIENT_DIGITS,
            label="independence polynomial",
        )
        if any(
            len(coefficient.lstrip("-"))
            > MAX_INDEPENDENCE_POLYNOMIAL_COEFFICIENT_DIGITS
            for coefficient in self.coefficients
        ):
            raise ValueError(
                "independence coefficient exceeds the "
                f"{MAX_INDEPENDENCE_POLYNOMIAL_COEFFICIENT_DIGITS}-digit bound"
            )
        if (
            len(self.independent_set_count.lstrip("-"))
            > MAX_INDEPENDENCE_POLYNOMIAL_COEFFICIENT_DIGITS
        ):
            raise ValueError(
                "independent-set count exceeds the "
                f"{MAX_INDEPENDENCE_POLYNOMIAL_COEFFICIENT_DIGITS}-digit bound"
            )
        TreeIndependencePolynomialRequest(graph=self.graph)

        from jacobian.math.graphs.polynomials.operations import (
            _polynomial_from_coefficients,
            independence_polynomial_coefficients,
        )

        expected_coefficients = independence_polynomial_coefficients(self.graph)
        expected_wire_coefficients = tuple(
            format_canonical_integer(coefficient)
            for coefficient in expected_coefficients
        )
        if self.polynomial != _polynomial_from_coefficients(expected_coefficients):
            raise ValueError("independence polynomial does not match the source tree")
        if self.coefficients != expected_wire_coefficients:
            raise ValueError("independence coefficients do not match the source tree")
        if self.independence_number != len(expected_coefficients) - 1:
            raise ValueError("independence number does not match the source tree")
        if self.independent_set_count != format_canonical_integer(
            sum(expected_coefficients)
        ):
            raise ValueError("independent-set count does not match the source tree")
        return self


class PolynomialTerm(StrictModel):
    """One monomial term: coefficient times x^degree."""

    coefficient: int
    degree: int = Field(ge=0)


class GraphPolynomialResult(StrictModel):
    """A sparse polynomial represented as a list of (coefficient, degree) terms."""

    terms: tuple[PolynomialTerm, ...]

    @model_validator(mode="after")
    def require_canonical(self) -> Self:
        degrees = [term.degree for term in self.terms]
        if degrees != sorted(degrees):
            raise ValueError("polynomial terms must be sorted by degree")
        if len(set(degrees)) != len(degrees):
            raise ValueError("polynomial degrees must be unique")
        if any(term.coefficient == 0 for term in self.terms):
            raise ValueError("polynomial terms must have nonzero coefficients")
        return self


class MultivariatePolynomialTerm(StrictModel):
    coefficient: int
    exponents: tuple[int, ...]

    @model_validator(mode="after")
    def require_nonzero_nonnegative_term(self) -> Self:
        if self.coefficient == 0 or any(exponent < 0 for exponent in self.exponents):
            raise ValueError(
                "multivariate terms require nonzero coefficients and nonnegative exponents"
            )
        return self


class SparseMultivariatePolynomial(StrictModel):
    variables: tuple[str, ...] = Field(min_length=1)
    terms: tuple[MultivariatePolynomialTerm, ...]

    @model_validator(mode="after")
    def require_canonical_terms(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("polynomial variables must be unique")
        exponents = [term.exponents for term in self.terms]
        if any(len(item) != len(self.variables) for item in exponents):
            raise ValueError("every exponent tuple must match the variable axis")
        if exponents != sorted(exponents) or len(set(exponents)) != len(exponents):
            raise ValueError(
                "multivariate terms must have unique sorted exponent tuples"
            )
        return self


__all__ = [
    "GraphEdge",
    "GraphPolynomialRequest",
    "GraphPolynomialResult",
    "GraphSpec",
    "MatchingPolynomialRequest",
    "MultivariatePolynomialTerm",
    "PolynomialTerm",
    "SparseMultivariatePolynomial",
    "TreeIndependencePolynomialRequest",
    "TreeIndependencePolynomialResult",
]
