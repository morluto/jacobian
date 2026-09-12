"""Bounded-cardinality minimal transversal enumeration."""

import time
from itertools import combinations
from math import comb
from typing import Annotated, Self

from pydantic import Field, StrictInt, ValidationError, model_validator

from jacobian._execution import (
    current_request_execution,
    execution_deadline,
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


def _candidate_hits_all_edges(
    selected: frozenset[str], edges: tuple[frozenset[str], ...]
) -> tuple[bool, int]:
    """Check candidate-edge membership and return the number of checks made."""

    checks = 0
    for edge in edges:
        checks += 1
        if not selected & edge:
            return False, checks
    return True, checks


def _candidate_has_redundant_vertex(
    selected: frozenset[str], edges: tuple[frozenset[str], ...]
) -> tuple[bool, int]:
    """Check minimality and return the number of candidate-edge checks made."""

    checks = 0
    for vertex in selected:
        reduced = selected - {vertex}
        for edge in edges:
            checks += 1
            if not reduced & edge:
                break
        else:
            return True, checks
    return False, checks


def _unique_edges(
    request: MinimalTransversalEnumerationRequest,
) -> tuple[frozenset[str], ...]:
    seen: set[frozenset[str]] = set()
    edges: list[frozenset[str]] = []
    for index, (_, members) in enumerate(request.hypergraph.edges):
        if index % 256 == 0:
            request_checkpoint("during minimal transversal edge dedup")
        edge = frozenset(members)
        if edge in seen:
            continue
        seen.add(edge)
        edges.append(edge)
    return tuple(edges)


def _forced_vertices(edges: tuple[frozenset[str], ...]) -> frozenset[str]:
    forced: set[str] = set()
    for edge in edges:
        if len(edge) == 1:
            forced.update(edge)
    return frozenset(forced)


def _domination_comparison_count(edges: tuple[frozenset[str], ...]) -> int:
    """Count strict-subset tests against strictly smaller edges only."""

    counts: dict[int, int] = {}
    for edge in edges:
        size = len(edge)
        counts[size] = counts.get(size, 0) + 1
    smaller = 0
    total = 0
    for size in sorted(counts):
        total += counts[size] * smaller
        smaller += counts[size]
    return total


def _minimal_edges(
    edges: tuple[frozenset[str], ...],
) -> tuple[frozenset[str], ...]:
    """Drop dominated edges after a size-ordered subset scan."""

    comparison_count = _domination_comparison_count(edges)
    if comparison_count > MAX_TRANSVERSAL_ENUMERATION_WORK:
        raise OperationResourceAdmissionError(
            location=("hypergraph", "edges"),
            code="hypergraph.minimal_transversal.antichain_work_bound",
            message=(
                "edge-domination presolve has "
                f"{comparison_count} subset comparisons; maximum is "
                f"{MAX_TRANSVERSAL_ENUMERATION_WORK}"
            ),
        )
    buckets: dict[int, list[frozenset[str]]] = {}
    for edge in edges:
        buckets.setdefault(len(edge), []).append(edge)
    smaller: tuple[frozenset[str], ...] = ()
    kept: list[frozenset[str]] = []
    comparisons = 0
    for size in sorted(buckets):
        size_kept: list[frozenset[str]] = []
        for edge in buckets[size]:
            dominated = False
            for other in smaller:
                comparisons += 1
                if comparisons % 256 == 0:
                    request_checkpoint("during minimal transversal domination")
                if other < edge:
                    dominated = True
                    break
            if not dominated:
                size_kept.append(edge)
        kept.extend(size_kept)
        smaller = tuple(kept)
    return tuple(kept)


def _admit_enumeration(
    request: MinimalTransversalEnumerationRequest,
) -> tuple[int, tuple[frozenset[str], ...], frozenset[str], bool, bool]:
    """Admit exhaustive work after edge dedup and singleton presolve."""

    vertices = request.hypergraph.vertices
    maximum = min(request.maximum_cardinality, len(vertices))
    unique_edges = _unique_edges(request)
    if not unique_edges:
        return maximum, unique_edges, frozenset(), True, False
    if any(not edge for edge in unique_edges):
        return maximum, unique_edges, frozenset(), False, True
    forced = _forced_vertices(unique_edges)
    remaining_unique = tuple(edge for edge in unique_edges if not edge & forced)
    if len(forced) > maximum:
        return maximum, remaining_unique, forced, False, False
    if not remaining_unique:
        return maximum, remaining_unique, forced, False, False
    # Cardinality-1 search never needs domination. Equal-cardinality distinct
    # remaining edges are already an antichain, so skip the pairwise subset scan.
    edge_sizes = {len(edge) for edge in remaining_unique}
    need_domination = maximum > 1 and len(edge_sizes) > 1
    domination_work = (
        _domination_comparison_count(remaining_unique) if need_domination else 0
    )
    edges = _minimal_edges(remaining_unique) if need_domination else remaining_unique
    remaining_edges = edges
    if len(forced) >= maximum:
        return maximum, remaining_edges, forced, False, False
    if len(remaining_edges) == 1:
        return maximum, remaining_edges, forced, False, False

    occupied = frozenset().union(*remaining_edges)
    free_vertices = tuple(
        vertex for vertex in vertices if vertex not in forced and vertex in occupied
    )
    free_maximum = maximum - len(forced)
    candidate_count = sum(
        comb(len(free_vertices), size)
        for size in range(0 if forced else 1, free_maximum + 1)
    )
    constraint_count = len(remaining_edges) + len(forced)
    candidate_edge_work = candidate_count * len(remaining_edges)
    minimality_work = sum(
        (len(forced) + size) * constraint_count * comb(len(free_vertices), size)
        for size in range(0 if forced else 1, free_maximum + 1)
    )
    total_work = candidate_edge_work + minimality_work + domination_work

    possible_rows = max(
        (
            comb(len(free_vertices), size)
            for size in range(0 if forced else 1, free_maximum + 1)
        ),
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

    return maximum, remaining_edges, forced, False, False


def enumerate_minimal_transversals(
    request: MinimalTransversalEnumerationRequest,
) -> MinimalTransversalEnumerationResult:
    """Enumerate every inclusion-minimal transversal through the requested rank."""

    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return enumerate_minimal_transversals(request)
    execution_deadline(_OWNER_DEADLINE_SECONDS)
    request_checkpoint("before minimal transversal admission")
    request = _validated_request(request)
    request_checkpoint("after minimal transversal request validation")
    vertices = request.hypergraph.vertices
    maximum, remaining_edges, forced, source_empty, has_empty_edge = _admit_enumeration(
        request
    )
    request_checkpoint("after minimal transversal admission")
    forced_row = tuple(vertex for vertex in vertices if vertex in forced)
    if source_empty:
        results: tuple[tuple[str, ...], ...] = ((),)
    elif has_empty_edge or len(forced) > maximum:
        results = ()
    elif not remaining_edges:
        results = (forced_row,)
    elif len(forced) >= maximum:
        results = ()
    elif len(remaining_edges) == 1:
        results = tuple(
            tuple(vertex for vertex in vertices if vertex in forced or vertex == extra)
            for extra in vertices
            if extra in remaining_edges[0]
        )
    else:
        materialized: list[tuple[str, ...]] = []
        free_vertices = tuple(
            vertex
            for vertex in vertices
            if vertex not in forced and any(vertex in edge for edge in remaining_edges)
        )
        free_maximum = maximum - len(forced)
        for size in range(0 if forced else 1, free_maximum + 1):
            for extra in combinations(free_vertices, size):
                request_checkpoint("during minimal transversal enumeration")
                selected = forced | frozenset(extra)
                hits_all_edges, _ = _candidate_hits_all_edges(selected, remaining_edges)
                if not hits_all_edges:
                    continue
                has_redundant_vertex, _ = _candidate_has_redundant_vertex(
                    selected,
                    remaining_edges + tuple(frozenset({vertex}) for vertex in forced),
                )
                if has_redundant_vertex:
                    continue
                materialized.append(
                    tuple(vertex for vertex in vertices if vertex in selected)
                )
        results = tuple(materialized)

    request_checkpoint("before minimal transversal result construction")
    counts = [0] * (request.maximum_cardinality + 1)
    for transversal in results:
        counts[len(transversal)] += 1
    profile = tuple(
        MinimalTransversalCardinalityCount(cardinality=cardinality, count=count)
        for cardinality, count in enumerate(counts)
    )
    result = MinimalTransversalEnumerationResult(
        hypergraph=request.hypergraph,
        maximum_cardinality=request.maximum_cardinality,
        transversals=results,
        cardinality_profile=profile,
    )
    request_checkpoint("after minimal transversal result validation")
    return result
