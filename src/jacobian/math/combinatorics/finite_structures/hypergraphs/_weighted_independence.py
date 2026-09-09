"""Exact rational-weighted independent vertex selection."""

from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

MAX_WEIGHTED_INDEPENDENCE_VERTICES = 40
MAX_WEIGHTED_INDEPENDENCE_NODES = 1_000_000


class WeightedIndependentSelectionRequest(StrictModel):
    hypergraph: FiniteHypergraph
    weights: tuple[CanonicalRational, ...] = Field(
        max_length=MAX_WEIGHTED_INDEPENDENCE_VERTICES
    )
    search_node_limit: StrictInt = Field(
        default=1_000_000, ge=1, le=MAX_WEIGHTED_INDEPENDENCE_NODES
    )

    @model_validator(mode="after")
    def bind_weights(self) -> Self:
        if len(self.weights) != len(self.hypergraph.vertices):
            raise ValueError("weights must align with the complete vertex axis")
        return self


class WeightedIndependentSelectionResult(StrictModel):
    source: WeightedIndependentSelectionRequest
    status: Literal["EXACT", "BOUNDED"]
    incumbent_vertices: tuple[str, ...]
    incumbent_value: CanonicalRational
    lower_bound: CanonicalRational
    upper_bound: CanonicalRational
    searched_node_count: StrictInt = Field(ge=1, le=MAX_WEIGHTED_INDEPENDENCE_NODES)

    @model_validator(mode="after")
    def bind_incumbent_and_bounds(self) -> Self:
        vertices = self.source.hypergraph.vertices
        selected = set(self.incumbent_vertices)
        if self.incumbent_vertices != tuple(
            vertex for vertex in vertices if vertex in selected
        ):
            raise PydanticCustomError(
                "hypergraph.weighted_independence.incumbent_axis",
                "incumbent vertices must be a unique subsequence of the source axis",
            )
        if any(set(members) <= selected for _, members in self.source.hypergraph.edges):
            raise PydanticCustomError(
                "hypergraph.weighted_independence.incumbent_not_independent",
                "the attaining incumbent must be hypergraph-independent",
            )
        value = sum(
            (
                weight.as_fraction()
                for vertex, weight in zip(vertices, self.source.weights, strict=True)
                if vertex in selected
            ),
            Fraction(),
        )
        if (
            self.incumbent_value.as_fraction() != value
            or self.lower_bound.as_fraction() != value
        ):
            raise PydanticCustomError(
                "hypergraph.weighted_independence.incumbent_value",
                "incumbent value and lower bound must equal the source-weighted witness",
            )
        upper = self.upper_bound.as_fraction()
        if upper < value or (self.status == "EXACT" and upper != value):
            raise PydanticCustomError(
                "hypergraph.weighted_independence.bounds",
                "bounds must contain the incumbent and coincide for an exact result",
            )
        return self


def maximum_weight_independent_selection(
    request: WeightedIndependentSelectionRequest,
) -> WeightedIndependentSelectionResult:
    hypergraph = request.hypergraph
    if len(hypergraph.vertices) > MAX_WEIGHTED_INDEPENDENCE_VERTICES:
        raise OperationResourceAdmissionError(
            location=("hypergraph", "vertices"),
            code="hypergraph.weighted_independence.vertex_bound",
            message="weighted independent selection admits at most 40 vertices",
        )
    if any(not members for _, members in hypergraph.edges):
        raise OperationDomainValidationError(
            location=("hypergraph", "edges"),
            code="hypergraph.weighted_independence.empty_edge",
            message="weighted independent selection requires nonempty hyperedges",
        )
    weights = tuple(weight.as_fraction() for weight in request.weights)
    index = {vertex: i for i, vertex in enumerate(hypergraph.vertices)}
    edge_masks = tuple(
        sum(1 << index[vertex] for vertex in members) for _, members in hypergraph.edges
    )
    root_upper = sum((max(weight, Fraction()) for weight in weights), Fraction())
    incumbent_mask = 0
    incumbent_value = Fraction()
    stack: list[tuple[int, int, Fraction]] = [(0, 0, Fraction())]
    visited = 0
    while stack and visited < request.search_node_limit:
        position, chosen, value = stack.pop()
        visited += 1
        if position == len(weights):
            if value > incumbent_value:
                incumbent_mask, incumbent_value = chosen, value
            continue
        remaining_upper = value + sum(
            (max(weight, Fraction()) for weight in weights[position:]), Fraction()
        )
        if remaining_upper <= incumbent_value:
            continue
        stack.append((position + 1, chosen, value))
        included = chosen | (1 << position)
        if not any(included & edge == edge for edge in edge_masks):
            stack.append((position + 1, included, value + weights[position]))
    exact = not stack
    incumbent_vertices = tuple(
        vertex
        for i, vertex in enumerate(hypergraph.vertices)
        if incumbent_mask & (1 << i)
    )
    lower = CanonicalRational.from_fraction(incumbent_value)
    upper = CanonicalRational.from_fraction(incumbent_value if exact else root_upper)
    return WeightedIndependentSelectionResult(
        source=request,
        status="EXACT" if exact else "BOUNDED",
        incumbent_vertices=incumbent_vertices,
        incumbent_value=lower,
        lower_bound=lower,
        upper_bound=upper,
        searched_node_count=visited,
    )


__all__ = ["maximum_weight_independent_selection"]
