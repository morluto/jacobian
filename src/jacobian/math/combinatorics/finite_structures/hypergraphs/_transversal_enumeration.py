"""Bounded-cardinality minimal transversal enumeration."""

import time
from itertools import combinations
from math import comb
from typing import Annotated, Self

from pydantic import Field, StrictInt, ValidationError, model_validator

from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_VERTICES,
    FiniteHypergraph,
)

MAX_TRANSVERSAL_ENUMERATION_CANDIDATES = 1_000_000
MAX_TRANSVERSAL_ENUMERATION_WORK = 50_000_000
MAX_ENUMERATED_TRANSVERSALS = 100_000
MAX_TRANSVERSAL_OUTPUT_INCIDENCES = 1_000_000
_OWNER_DEADLINE_SECONDS = 3600.0
_MinimalTransversalRow = Annotated[tuple[str, ...], Field(max_length=MAX_VERTICES)]


class MinimalTransversalEnumerationRequest(StrictModel):
    """One canonical source hypergraph and a bounded rank slice."""

    hypergraph: FiniteHypergraph
    maximum_cardinality: StrictInt = Field(
        ge=0,
        le=MAX_VERTICES,
        description=(
            "Largest allowed transversal cardinality. The owner admits all "
            "candidate-edge and minimality checks before exhaustive search."
        ),
    )


class MinimalTransversalCardinalityCount(StrictModel):
    """Count of returned minimal transversals of one cardinality."""

    cardinality: StrictInt = Field(ge=0, le=MAX_VERTICES)
    count: StrictInt = Field(ge=0, le=MAX_ENUMERATED_TRANSVERSALS)


class MinimalTransversalEnumerationResult(StrictModel):
    """A complete, canonically ordered minimal-transversal rank slice.

    Validation binds rows to the source vertex axis and to the requested rank
    slice, but deliberately does not replay the hitting or inclusion-minimality
    predicates established by the enumeration kernel.
    """

    hypergraph: FiniteHypergraph
    maximum_cardinality: StrictInt = Field(ge=0, le=MAX_VERTICES)
    transversals: tuple[_MinimalTransversalRow, ...] = Field(
        max_length=MAX_ENUMERATED_TRANSVERSALS
    )
    cardinality_profile: tuple[MinimalTransversalCardinalityCount, ...] = Field(
        max_length=MAX_VERTICES + 1
    )

    @model_validator(mode="after")
    def bind_structural_profile(self) -> Self:
        """Enforce source-axis, uniqueness, ordering, and profile invariants."""

        vertices = self.hypergraph.vertices
        vertex_set = set(vertices)
        vertex_position = {vertex: index for index, vertex in enumerate(vertices)}
        effective_maximum = min(self.maximum_cardinality, len(vertices))
        previous: tuple[str, ...] | None = None
        counts = [0] * (self.maximum_cardinality + 1)
        for transversal in self.transversals:
            if len(transversal) > effective_maximum:
                raise ValueError(
                    "minimal transversal exceeds the requested cardinality"
                )
            transversal_set = set(transversal)
            if len(transversal_set) != len(transversal):
                raise ValueError("minimal transversal vertices must be distinct")
            if not transversal_set <= vertex_set:
                raise ValueError(
                    "minimal transversal vertices must be declared source vertices"
                )
            expected_order = tuple(
                vertex for vertex in vertices if vertex in transversal_set
            )
            if transversal != expected_order:
                raise ValueError(
                    "minimal transversal vertices must be unique and in declared vertex order"
                )
            if previous is not None and (
                len(transversal),
                tuple(vertex_position[v] for v in transversal),
            ) <= (
                len(previous),
                tuple(vertex_position[v] for v in previous),
            ):
                raise ValueError(
                    "minimal transversals must be in canonical cardinality order"
                )
            counts[len(transversal)] += 1
            previous = transversal
        expected_profile = tuple(
            MinimalTransversalCardinalityCount(cardinality=index, count=count)
            for index, count in enumerate(counts)
        )
        if self.cardinality_profile != expected_profile:
            raise ValueError("cardinality profile must match the returned transversals")
        return self


def _validated_request(
    request: MinimalTransversalEnumerationRequest,
) -> MinimalTransversalEnumerationRequest:
    """Re-establish the typed request boundary for direct native callers."""

    if not isinstance(request, MinimalTransversalEnumerationRequest):
        raise TypeError(
            "minimal transversal enumeration requires its typed request model"
        )
    try:
        if not isinstance(request.hypergraph, FiniteHypergraph):
            raise TypeError("request hypergraph is outside its typed model")
        hypergraph = FiniteHypergraph.model_validate(
            request.hypergraph.model_dump(mode="python"), strict=True
        )
        return MinimalTransversalEnumerationRequest.model_validate(
            {
                "hypergraph": hypergraph,
                "maximum_cardinality": request.maximum_cardinality,
            },
            strict=True,
        )
    except (AttributeError, TypeError, ValueError, ValidationError) as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="hypergraph.minimal_transversal.malformed_request",
            message="minimal transversal enumeration received a malformed typed request",
        ) from exc


def _admit_enumeration(
    request: MinimalTransversalEnumerationRequest,
) -> tuple[int, tuple[frozenset[str], ...]]:
    """Admit exhaustive work and every materialized result envelope."""

    vertices = request.hypergraph.vertices
    maximum = min(request.maximum_cardinality, len(vertices))
    edges = tuple(frozenset(members) for _, members in request.hypergraph.edges)

    # These degenerate families have closed results and need no candidate
    # enumeration, regardless of the caller's cardinality slice.
    if not edges or any(not edge for edge in edges):
        return maximum, edges

    candidate_count = sum(comb(len(vertices), size) for size in range(1, maximum + 1))
    weighted_candidate_count = sum(
        size * comb(len(vertices), size) for size in range(1, maximum + 1)
    )
    edge_count = len(edges)
    candidate_edge_work = candidate_count * edge_count
    minimality_work = weighted_candidate_count * edge_count
    total_work = candidate_edge_work + minimality_work

    # Minimal transversals form an antichain. By the LYM inequality, the
    # number of rows in ranks 1..k is at most the largest binomial coefficient
    # among those ranks.
    possible_rows = max(
        (comb(len(vertices), size) for size in range(1, maximum + 1)),
        default=0,
    )
    if possible_rows > MAX_ENUMERATED_TRANSVERSALS:
        raise OperationResourceAdmissionError(
            location=("maximum_cardinality",),
            code="hypergraph.minimal_transversal.output_bound",
            message=(
                "the possible minimal-transversal antichain has "
                f"{possible_rows} result rows; maximum is "
                f"{MAX_ENUMERATED_TRANSVERSALS}"
            ),
        )
    if candidate_count > MAX_TRANSVERSAL_ENUMERATION_CANDIDATES:
        raise OperationResourceAdmissionError(
            location=("maximum_cardinality",),
            code="hypergraph.minimal_transversal.candidate_bound",
            message=(
                f"cardinality-bounded enumeration has {candidate_count} candidate "
                f"subsets; maximum is {MAX_TRANSVERSAL_ENUMERATION_CANDIDATES}"
            ),
        )
    if total_work > MAX_TRANSVERSAL_ENUMERATION_WORK:
        raise OperationResourceAdmissionError(
            location=("maximum_cardinality",),
            code="hypergraph.minimal_transversal.work_bound",
            message=(
                "candidate-edge membership and minimality work has "
                f"{total_work} checks; maximum is "
                f"{MAX_TRANSVERSAL_ENUMERATION_WORK}"
            ),
        )

    output_incidences = possible_rows * maximum
    if output_incidences > MAX_TRANSVERSAL_OUTPUT_INCIDENCES:
        raise OperationResourceAdmissionError(
            location=("maximum_cardinality",),
            code="hypergraph.minimal_transversal.incidences_bound",
            message=(
                "the possible minimal-transversal output has "
                f"{output_incidences} vertex incidences; maximum is "
                f"{MAX_TRANSVERSAL_OUTPUT_INCIDENCES}"
            ),
        )

    return maximum, edges


def enumerate_minimal_transversals(
    request: MinimalTransversalEnumerationRequest,
) -> MinimalTransversalEnumerationResult:
    """Enumerate every inclusion-minimal transversal through the requested rank."""

    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return enumerate_minimal_transversals(request)
    if execution.deadline is None:
        bind_request_deadline(execution.started_at + _OWNER_DEADLINE_SECONDS)
    request_checkpoint("before minimal transversal admission")
    request = _validated_request(request)
    request_checkpoint("after minimal transversal request validation")
    vertices = request.hypergraph.vertices
    maximum, edges = _admit_enumeration(request)
    request_checkpoint("after minimal transversal admission")
    if not edges:
        results: tuple[tuple[str, ...], ...] = ((),)
    elif any(not edge for edge in edges) or maximum == 0:
        results = ()
    else:
        materialized: list[tuple[str, ...]] = []
        for size in range(1, maximum + 1):
            for candidate in combinations(vertices, size):
                request_checkpoint("during minimal transversal enumeration")
                selected = frozenset(candidate)
                if not all(selected & edge for edge in edges):
                    continue
                if any(
                    all((selected - {vertex}) & edge for edge in edges)
                    for vertex in selected
                ):
                    continue
                materialized.append(candidate)
        results = tuple(materialized)

    request_checkpoint("before minimal transversal result construction")
    counts = [0] * (request.maximum_cardinality + 1)
    for transversal in results:
        counts[len(transversal)] += 1
    profile = tuple(
        MinimalTransversalCardinalityCount(cardinality=cardinality, count=count)
        for cardinality, count in enumerate(counts)
    )
    return MinimalTransversalEnumerationResult(
        hypergraph=request.hypergraph,
        maximum_cardinality=request.maximum_cardinality,
        transversals=results,
        cardinality_profile=profile,
    )
