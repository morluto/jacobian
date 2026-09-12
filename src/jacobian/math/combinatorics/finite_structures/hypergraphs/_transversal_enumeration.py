"""Bounded-cardinality minimal transversal enumeration."""

from itertools import combinations
from math import comb

from pydantic import Field, StrictInt

from jacobian._models import StrictModel
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_VERTICES,
    FiniteHypergraph,
)

MAX_TRANSVERSAL_ENUMERATION_CANDIDATES = 1_000_000
MAX_ENUMERATED_TRANSVERSALS = 100_000
MAX_TRANSVERSAL_ENUMERATION_RESULT_BYTES = 10 * 1024 * 1024
_TRANSVERSAL_ROW_OVERHEAD_BYTES = 128


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
    edges = tuple(frozenset(members) for _, members in request.hypergraph.edges)
    # These degenerate families have a closed result and need no candidate
    # enumeration, regardless of the caller's cardinality slice.
    if not edges:
        return MinimalTransversalEnumerationResult(
            hypergraph=request.hypergraph,
            maximum_cardinality=request.maximum_cardinality,
            transversals=((),),
        )
    if any(not edge for edge in edges):
        return MinimalTransversalEnumerationResult(
            hypergraph=request.hypergraph,
            maximum_cardinality=request.maximum_cardinality,
            transversals=(),
        )
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
    # The result carrier is a materialized complete family.  A candidate
    # subset is a sound upper bound on the number of retained rows, so reject
    # a profile that could overflow either the row or canonical transport
    # envelope before starting candidate generation.
    if candidate_bound > MAX_ENUMERATED_TRANSVERSALS:
        raise OperationResourceAdmissionError(
            location=("maximum_cardinality",),
            code="hypergraph.minimal_transversal.output_bound",
            message=(
                f"the admitted candidate slice permits {candidate_bound} result rows; "
                f"maximum is {MAX_ENUMERATED_TRANSVERSALS}"
            ),
        )
    source_bytes = len(encode_strict_json(request.hypergraph.model_dump(mode="json")))
    max_vertex_bytes = max(
        (len(encode_strict_json(vertex)) for vertex in vertices), default=2
    )
    predicted_result_bytes = (
        source_bytes
        + candidate_bound
        * (_TRANSVERSAL_ROW_OVERHEAD_BYTES + maximum * max_vertex_bytes)
        + 256
    )
    if predicted_result_bytes > MAX_TRANSVERSAL_ENUMERATION_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("maximum_cardinality",),
            code="hypergraph.minimal_transversal.result_bytes_bound",
            message=(
                "the bounded minimal-transversal result is predicted to occupy "
                f"{predicted_result_bytes} bytes; maximum is "
                f"{MAX_TRANSVERSAL_ENUMERATION_RESULT_BYTES}"
            ),
        )
    results: list[tuple[str, ...]] = []
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
