"""Private Z3/NetworkX backend for bounded independence-number search."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Literal

import z3  # type: ignore[import-untyped]

from jacobian.math.graphs.independence import (
    IndependenceNumberBudget,
    IndependenceNumberRequest,
    IndependenceNumberResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

# Independent result verification is deliberately separate from structural
# Pydantic parsing. Its branch-and-bound search has both a finite node ledger
# and a cooperatively enforced monotonic deadline; callers cannot silently trigger it by
# deserializing an otherwise well-formed result.
_EXACT_REPLAY_SEARCH_NODES = 200_000
_INDEPENDENT_VERIFICATION_WALL_SECONDS = 5
_MAX_INDEPENDENT_VERIFICATION_WALL_SECONDS = 120


def _integer_bound(value: z3.ArithRef, fallback: int) -> int:
    return value.as_long() if z3.is_int_value(value) else fallback


def _replay_deadline_elapsed(deadline: float) -> bool:
    return time.monotonic() >= deadline


def _require_replay_budget(node_expansions: int, deadline: float) -> None:
    """Reject an exact claim once its finite replay envelope is exhausted."""

    if node_expansions > _EXACT_REPLAY_SEARCH_NODES:
        raise ValueError(
            "claimed exact optimum was not reproduced by the bounded "
            "source-graph replay"
        )
    if _replay_deadline_elapsed(deadline):
        raise ValueError(
            "claimed exact optimum replay exceeded its wall-clock deadline"
        )


def _component_masks(neighbours: list[int], unvisited: int) -> Iterator[int]:
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
    """Return a greedy clique-cover upper bound on an independent-set size."""

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
    *,
    deadline: float,
) -> None:
    """Reproduce an exact independence claim within one finite envelope.

    The replay decomposes the source graph into connected components and sums
    their maxima. Each component uses deterministic branch-and-bound with
    forced isolated vertices and a greedy clique-cover bound. Every expanded
    node performs bitset work over at most 128 vertices. Across all components
    it is charged to ``_EXACT_REPLAY_SEARCH_NODES`` and to ``deadline``.
    Exhaustion rejects the claim fail-closed.
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

    total_optimum = sum(
        component_optimum(component)
        for component in _component_masks(neighbours, (1 << order) - 1)
    )
    if total_optimum != claimed_optimum:
        raise ValueError(
            "claimed exact optimum contradicts the independent source-graph replay"
        )


def verify_independence_result(
    result: IndependenceNumberResult,
    *,
    wall_seconds: int = _INDEPENDENT_VERIFICATION_WALL_SECONDS,
) -> bool:
    """Verify an independently supplied exact outcome in a bounded envelope.

    Structural Pydantic validation has already bound the source graph and
    witness. Only an ``EXACT`` payload asserts a nonlocal mathematical fact;
    that assertion is replayed under this verifier's explicit 1--120 second
    wall envelope and fixed node ledger. ``UNKNOWN`` does not assert an
    optimum, so its structural invariants are sufficient.
    """

    if not 1 <= wall_seconds <= _MAX_INDEPENDENT_VERIFICATION_WALL_SECONDS:
        raise ValueError("verification wall_seconds must lie between 1 and 120")
    if result.status != "EXACT":
        return True
    if result.optimum_value is None:  # Structural validation prevents this.
        return False
    try:
        _replay_exact_optimum(
            result.graph,
            result.optimum_value,
            deadline=time.monotonic() + wall_seconds,
        )
    except ValueError:
        return False
    return True


def solve_independence_number(
    request: IndependenceNumberRequest,
) -> IndependenceNumberResult:
    """Run the retained catalog/MCP request adapter."""

    return solve_independence_number_values(request.graph, request.resource_budget)


def solve_independence_number_values(
    graph: SimpleUndirectedGraph,
    resource_budget: IndependenceNumberBudget,
) -> IndependenceNumberResult:
    """Run one wall-clock-bounded exact maximum independent-set optimization.

    The trusted factory still performs every structural source and witness
    check. An ``EXACT`` conclusion additionally reproduces its optimum under
    the request deadline; a replay that cannot certify the solver optimum
    demotes to the typed ``UNKNOWN`` outcome. Every incomplete outcome,
    including a ``sat`` optimize whose objective bounds stay open, reports
    the graph order as its independently safe upper bound.
    """

    started = time.monotonic()
    vertices = graph.vertices
    order = len(vertices)
    if not vertices:
        return IndependenceNumberResult._from_kernel(
            graph=graph,
            status="EXACT",
            optimum_value=0,
            upper_bound=0,
            incumbent_vertices=(),
            termination_reason="SPECIAL_CASE",
            detail="the empty graph has independence number zero",
        )

    incumbent: tuple[str, ...] = (min(vertices),)
    remaining_ms = int(
        (resource_budget.wall_seconds - (time.monotonic() - started)) * 1000
    )
    if remaining_ms <= 0:
        return IndependenceNumberResult._from_kernel(
            graph=graph,
            status="UNKNOWN",
            optimum_value=None,
            upper_bound=len(vertices),
            incumbent_vertices=incumbent,
            termination_reason="WALL_TIME",
            detail="the wall-clock budget expired after the initial feasible witness",
        )

    optimizer = z3.Optimize()
    optimizer.set(timeout=max(1, remaining_ms))
    selected = {
        vertex: z3.Bool(f"selected_{index}") for index, vertex in enumerate(vertices)
    }
    for left, right in graph.edges:
        optimizer.add(z3.Or(z3.Not(selected[left]), z3.Not(selected[right])))
    objective = optimizer.maximize(
        z3.Sum([z3.If(selected[vertex], 1, 0) for vertex in vertices])
    )

    status = optimizer.check()
    if status == z3.sat:
        model = optimizer.model()
        optimized = tuple(
            sorted(
                vertex
                for vertex, variable in selected.items()
                if z3.is_true(model.eval(variable, model_completion=True))
            )
        )
        if len(optimized) > len(incumbent):
            incumbent = optimized
        lower = objective.lower()
        upper = objective.upper()
        lower_bound = max(len(incumbent), _integer_bound(lower, len(incumbent)))
        upper_bound = max(lower_bound, min(order, _integer_bound(upper, order)))
        if lower_bound == upper_bound == len(incumbent):
            try:
                _replay_exact_optimum(
                    graph,
                    len(incumbent),
                    deadline=started + resource_budget.wall_seconds,
                )
            except ValueError:
                return IndependenceNumberResult._from_kernel(
                    graph=graph,
                    status="UNKNOWN",
                    optimum_value=None,
                    upper_bound=len(vertices),
                    incumbent_vertices=incumbent,
                    termination_reason="REPLAY_INCOMPLETE",
                    detail=(
                        "bounded source-graph replay could not certify the "
                        "solver optimum, so no exact optimum is claimed"
                    ),
                )
            return IndependenceNumberResult._from_kernel(
                graph=graph,
                status="EXACT",
                optimum_value=len(incumbent),
                upper_bound=len(incumbent),
                incumbent_vertices=incumbent,
                termination_reason="OPTIMUM_ESTABLISHED",
                detail="bounded Z3 optimization seeded by a NetworkX feasible witness",
            )
    elif status == z3.unsat:
        return IndependenceNumberResult._from_kernel(
            graph=graph,
            status="UNKNOWN",
            optimum_value=None,
            upper_bound=len(vertices),
            incumbent_vertices=incumbent,
            termination_reason="SOLVER_UNSAT",
            detail="bounded Z3 optimization returned unsat, which is unexpected "
            "for an independence-number problem that always has a feasible witness",
        )
    termination: Literal["WALL_TIME", "SOLVER_UNKNOWN"] = (
        "WALL_TIME"
        if time.monotonic() - started >= resource_budget.wall_seconds
        else "SOLVER_UNKNOWN"
    )
    return IndependenceNumberResult._from_kernel(
        graph=graph,
        status="UNKNOWN",
        optimum_value=None,
        upper_bound=len(vertices),
        incumbent_vertices=incumbent,
        termination_reason=termination,
        detail="bounded Z3 optimization did not establish an exact optimum",
    )
