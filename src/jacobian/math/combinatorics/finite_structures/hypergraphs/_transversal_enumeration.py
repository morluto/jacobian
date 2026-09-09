"""Bounded-cardinality minimal transversal enumeration."""

from itertools import combinations
from math import comb

from pydantic import Field, StrictInt

from jacobian._models import StrictModel
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_VERTICES,
    FiniteHypergraph,
)

MAX_TRANSVERSAL_ENUMERATION_CANDIDATES = 1_000_000
MAX_ENUMERATED_TRANSVERSALS = 100_000


class MinimalTransversalEnumerationRequest(StrictModel):
    hypergraph: FiniteHypergraph
    maximum_cardinality: StrictInt = Field(ge=0, le=MAX_VERTICES)


class MinimalTransversalEnumerationResult(StrictModel):
    hypergraph: FiniteHypergraph
    maximum_cardinality: StrictInt = Field(ge=0, le=MAX_VERTICES)
    transversals: tuple[tuple[str, ...], ...] = Field(
        max_length=MAX_ENUMERATED_TRANSVERSALS
    )


def enumerate_minimal_transversals(
    request: MinimalTransversalEnumerationRequest,
) -> MinimalTransversalEnumerationResult:
    vertices = request.hypergraph.vertices
    maximum = min(request.maximum_cardinality, len(vertices))
    candidate_bound = sum(comb(len(vertices), size) for size in range(maximum + 1))
    if candidate_bound > MAX_TRANSVERSAL_ENUMERATION_CANDIDATES:
        raise OperationResourceAdmissionError(
            location=("maximum_cardinality",),
            code="hypergraph.minimal_transversal.candidate_bound",
            message=(
                f"cardinality-bounded enumeration has {candidate_bound} candidate "
                f"subsets; maximum is {MAX_TRANSVERSAL_ENUMERATION_CANDIDATES}"
            ),
        )
    edges = tuple(frozenset(members) for _, members in request.hypergraph.edges)
    results: list[tuple[str, ...]] = []
    if not edges:
        results.append(())
    elif not any(not edge for edge in edges):
        for size in range(1, maximum + 1):
            for candidate in combinations(vertices, size):
                selected = frozenset(candidate)
                if not all(selected & edge for edge in edges):
                    continue
                if any(
                    all((selected - {vertex}) & edge for edge in edges)
                    for vertex in selected
                ):
                    continue
                results.append(candidate)
    return MinimalTransversalEnumerationResult(
        hypergraph=request.hypergraph,
        maximum_cardinality=request.maximum_cardinality,
        transversals=tuple(results),
    )
