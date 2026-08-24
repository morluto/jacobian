"""Provider-independent values for bounded independence-number search."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.math.graphs.values import SimpleUndirectedGraph

IndependenceSearchStatus = Literal["EXACT", "UNKNOWN"]
IndependenceTermination = Literal[
    "OPTIMUM_ESTABLISHED",
    "WALL_TIME",
    "SOLVER_UNKNOWN",
    "SOLVER_UNSAT",
    "SPECIAL_CASE",
    "REPLAY_INCOMPLETE",
]


class IndependenceNumberBudget(StrictModel):
    """Explicit public limits for one bounded independence-number search."""

    wall_seconds: StrictInt = Field(default=5, ge=1, le=120)
    max_solver_calls: StrictInt = Field(
        default=1,
        ge=1,
        le=33,
        description=(
            "Compatibility budget retained from the threshold-search contract; "
            "the version-2 optimizer uses one solver call."
        ),
    )
    max_order: StrictInt = Field(default=128, ge=0, le=128)


# The result retains its source graph and echoes every witness identifier,
# so a request near the canonical input limit can serialize a response past
# the identical output limit.  Admission reserves this much for the fixed
# scalar fields, the bounded detail string, and the result envelope beyond
# the echoed graph and worst-case witness labels.
_RESULT_ENVELOPE_RESERVE_BYTES = 2_048


def _graph_wire_bytes(graph: SimpleUndirectedGraph) -> int:
    return len(encode_strict_json(graph.model_dump(mode="json")))


def _label_wire_bytes(graph: SimpleUndirectedGraph) -> int:
    return sum(len(encode_strict_json(label) + b",") for label in graph.vertices)


def _require_output_headroom(source_bytes: int, witness_label_bytes: int) -> None:
    estimated_result_bytes = (
        source_bytes + witness_label_bytes + _RESULT_ENVELOPE_RESERVE_BYTES
    )
    output_limit = CanonicalLimits().max_output_bytes
    if estimated_result_bytes > output_limit:
        raise ValueError(
            "the independence-number result retains its source graph and "
            "witness labels and would exceed the "
            f"{output_limit}-byte canonical output limit; "
            "shorten vertex labels or shrink the graph"
        )


class IndependenceNumberRequest(StrictModel):
    """One finite simple graph and its operation-owned search budget."""

    graph: SimpleUndirectedGraph
    resource_budget: IndependenceNumberBudget = Field(
        default_factory=IndependenceNumberBudget
    )

    @model_validator(mode="after")
    def require_supported_order(self) -> Self:
        order = len(self.graph.vertices)
        if order > self.resource_budget.max_order:
            raise ValueError("graph order exceeds the declared max_order budget")
        if order > 128:
            raise ValueError("independence-number search supports order at most 128")
        return self

    @model_validator(mode="after")
    def require_transportable_result(self) -> Self:
        # The result echoes the retained source graph and repeats up to
        # every vertex identifier as the canonically sorted witness, so
        # admission bounds that predicted serialization before any solve.
        _require_output_headroom(
            _graph_wire_bytes(self.graph),
            _label_wire_bytes(self.graph),
        )
        return self


_EXACT_REPLAY_SEARCH_NODES = 200_000


def _replay_deadline_elapsed(deadline: float | None) -> bool:
    """Report whether the shared request wall-clock envelope has elapsed."""

    return deadline is not None and time.monotonic() >= deadline


def _require_replay_budget(node_expansions: int, deadline: float | None) -> None:
    """Reject the claim once the replay exhausts its node or wall budget."""

    if node_expansions > _EXACT_REPLAY_SEARCH_NODES:
        raise ValueError(
            "claimed exact optimum was not reproduced by the bounded "
            "source-graph replay"
        )
    if _replay_deadline_elapsed(deadline):
        raise ValueError(
            "claimed exact optimum replay exceeded the request wall-clock deadline"
        )


def _component_masks(
    neighbours: list[int],
    unvisited: int,
) -> Iterator[int]:
    """Yield each connected component of ``unvisited`` as a vertex bitmask."""

    while unvisited:
        component = unvisited & -unvisited
        frontier = component
        while frontier:
            bit = frontier & -frontier
            frontier ^= bit
            fresh = neighbours[bit.bit_length() - 1] & ~component
            component |= fresh
            frontier |= fresh
        unvisited &= ~component
        yield component


def _greedy_clique_cover_size(candidates: int, neighbours: list[int]) -> int:
    """Count clique classes in one greedy cover of ``candidates``.

    Vertices enter in descending candidate-degree order and join the first
    class whose members they are all adjacent to, so every class stays a
    clique and an independent set meets it at most once.
    """

    weighted = []
    rest = candidates
    while rest:
        bit = rest & -rest
        rest ^= bit
        position = bit.bit_length() - 1
        weighted.append((-(neighbours[position] & candidates).bit_count(), position))
    weighted.sort()
    classes: list[int] = []
    for _, position in weighted:
        adjacent = neighbours[position]
        for slot, members in enumerate(classes):
            if members & ~adjacent == 0:
                classes[slot] = members | (1 << position)
                break
        else:
            classes.append(1 << position)
    return len(classes)


def _replay_exact_optimum(
    graph: SimpleUndirectedGraph,
    claimed_optimum: int,
    deadline: float | None = None,
) -> None:
    """Replay the claimed optimum as an exact search over the source graph.

    The producing solve establishes optimality with its own budgeted Z3 call;
    independently supplied results must reproduce the claim through this
    deterministic branch-and-bound before an ``EXACT`` conclusion validates.
    The replay decomposes the source graph into connected components and sums
    their exact maxima, which is sound because the independence number is
    additive over components.  Inside one component it forces every
    candidate-isolated vertex without branching (each such vertex belongs to
    every maximum independent set of the remaining candidates), prunes
    through a greedy clique cover of the candidates (an independent set
    meets each clique at most once, so the class count bounds any
    completion), and otherwise branches on a maximum-degree candidate, so
    structured sparse graphs such as matchings or disjoint gadget unions
    stay linear while dense graphs prune through the cover and
    candidate-popcount bounds.  Each component seeds its incumbent with one
    deterministic lowest-index greedy independent set before branching, so
    pruning compares completions against an achieved feasible size.  Each
    expanded node performs bounded bitset work over at most 128 vertices,
    and the whole replay is bounded by ``_EXACT_REPLAY_SEARCH_NODES`` node
    expansions charged across all components plus one linear greedy pass per
    component; exhausting that budget rejects the claim fail-closed.  When
    ``deadline`` carries a monotonic timestamp, each expansion also charges
    the replay to that shared request envelope and rejects the claim once it
    elapses, so a producing solve never spends time beyond its own budget.
    """

    vertices = tuple(sorted(graph.vertices))
    index = {vertex: position for position, vertex in enumerate(vertices)}
    order = len(vertices)
    neighbours = [0] * order
    for left, right in graph.edges:
        mask = (1 << index[left]) | (1 << index[right])
        neighbours[index[left]] |= mask
        neighbours[index[right]] |= mask
    state_nodes = 0

    def component_optimum(component: int) -> int:
        nonlocal state_nodes
        greedy = 0
        rest = component
        while rest:
            bit = rest & -rest
            rest &= ~(bit | neighbours[bit.bit_length() - 1])
            greedy += 1
        best_size = greedy

        def search(candidates: int, chosen: int) -> None:
            nonlocal state_nodes, best_size
            state_nodes += 1
            _require_replay_budget(state_nodes, deadline)
            neighbourhood = 0
            rest = candidates
            while rest:
                bit = rest & -rest
                rest ^= bit
                neighbourhood |= neighbours[bit.bit_length() - 1]
            forced = candidates & ~neighbourhood
            if forced:
                chosen += forced.bit_count()
                candidates ^= forced
            if chosen + candidates.bit_count() <= best_size:
                return
            if not candidates:
                best_size = chosen
                return
            if chosen + _greedy_clique_cover_size(candidates, neighbours) <= best_size:
                return
            pivot_degree = -1
            rest = candidates
            while rest:
                bit = rest & -rest
                rest ^= bit
                degree = (neighbours[bit.bit_length() - 1] & candidates).bit_count()
                if degree > pivot_degree:
                    pivot_degree = degree
                    pivot = bit.bit_length() - 1
            search(candidates & ~((1 << pivot) | neighbours[pivot]), chosen + 1)
            search(candidates & ~(1 << pivot), chosen)

        search(component, 0)
        return best_size

    unvisited = (1 << order) - 1
    total_optimum = 0
    for component in _component_masks(neighbours, unvisited):
        total_optimum += component_optimum(component)
    if total_optimum != claimed_optimum:
        raise ValueError(
            "claimed exact optimum contradicts the independent source-graph replay"
        )


class IndependenceNumberResult(StrictModel):
    """Exact optimum or bounded incumbent and bounds for one supplied graph.

    Retains the canonical source graph so validation replays the defining
    incumbent invariant: every witness identifier belongs to the source,
    no source edge has both endpoints in the witness, the incumbent equals
    the witness cardinality, and the reported order matches the source.
    An ``EXACT`` conclusion additionally replays a bounded maximum
    independent-set search on the retained source graph — the same replay
    the producing solve runs before claiming optimality, so every produced
    result validates — and a feasible but non-maximum witness cannot
    validate a forged optimum or upper bound.
    Operational ``UNKNOWN`` stays distinct from a mathematical optimum;
    ``REPLAY_INCOMPLETE`` marks a solver optimum that the bounded
    source-graph replay could not certify, so no optimum is claimed.  An
    incomplete outcome reports the graph order as its independently safe
    upper bound, so no unauthenticated incumbent gap survives validation.
    """

    result_schema_version: Literal["2"] = "2"
    graph: SimpleUndirectedGraph
    status: IndependenceSearchStatus
    order: StrictInt = Field(ge=0, le=128)
    optimum_value: StrictInt | None = Field(default=None, ge=0, le=128)
    incumbent_value: StrictInt = Field(ge=0, le=128)
    lower_bound: StrictInt = Field(ge=0, le=128)
    upper_bound: StrictInt = Field(ge=0, le=128)
    witness_vertices: tuple[str, ...] = Field(max_length=128)
    termination_reason: IndependenceTermination
    detail: str = Field(min_length=1, max_length=1024)
    convention: Literal["MAXIMUM_EDGE_FREE_VERTEX_SUBSET"] = (
        "MAXIMUM_EDGE_FREE_VERTEX_SUBSET"
    )

    @model_validator(mode="after")
    def bind_result(self) -> Self:
        if self.order != len(self.graph.vertices):
            raise ValueError("reported order must match the retained source graph")
        vertices = set(self.graph.vertices)
        if any(vertex not in vertices for vertex in self.witness_vertices):
            raise ValueError("every witness vertex must belong to the source graph")
        witness = set(self.witness_vertices)
        if any(
            left in witness and right in witness for left, right in self.graph.edges
        ):
            raise ValueError("witness must not contain both endpoints of a source edge")
        if self.witness_vertices != tuple(sorted(self.witness_vertices)) or len(
            set(self.witness_vertices)
        ) != len(self.witness_vertices):
            raise ValueError("witness vertices must be unique and canonically sorted")
        if self.incumbent_value != len(self.witness_vertices):
            raise ValueError("witness cardinality must match the incumbent")
        if self.lower_bound != self.incumbent_value:
            raise ValueError("a maximum-search incumbent is the lower bound")
        if not self.lower_bound <= self.upper_bound <= self.order:
            raise ValueError("independence-number bounds must lie inside graph order")
        if self.status == "EXACT":
            if (
                self.optimum_value is None
                or self.optimum_value != self.incumbent_value
                or self.optimum_value != self.upper_bound
                or self.termination_reason
                not in {"OPTIMUM_ESTABLISHED", "SPECIAL_CASE"}
            ):
                raise ValueError("exact result must bind one coincident optimum")
            _replay_exact_optimum(self.graph, self.optimum_value)
        elif self.optimum_value is not None:
            raise ValueError("incomplete search cannot claim an optimum")
        elif self.upper_bound != self.order:
            raise ValueError(
                "an incomplete result must report the graph order as its "
                "independently safe upper bound"
            )
        return self


def independence_number(request: IndependenceNumberRequest) -> IndependenceNumberResult:
    """Return an exact optimum when bounded Z3 optimization establishes it."""

    from jacobian.math.graphs import _independence_z3

    return _independence_z3.solve_independence_number(request)


__all__ = [
    "IndependenceNumberBudget",
    "IndependenceNumberRequest",
    "IndependenceNumberResult",
    "IndependenceSearchStatus",
    "IndependenceTermination",
    "independence_number",
]
