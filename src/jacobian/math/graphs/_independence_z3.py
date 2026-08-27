"""Private Z3/NetworkX backend for bounded independence-number search."""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from jacobian.canonical import CanonicalLimits
from jacobian.math.graphs.independence import (
    IndependenceNumberBudget,
    IndependenceNumberRequest,
    IndependenceNumberResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

# Independent result verification is deliberately separate from structural
# Pydantic parsing. Its branch-and-bound search has both a finite node ledger
# and a cooperatively enforced monotonic deadline; callers cannot silently trigger it by
# deserializing an otherwise well-formed result.
_EXACT_REPLAY_SEARCH_NODES = 200_000
_INDEPENDENT_VERIFICATION_WALL_SECONDS = 5
_MAX_INDEPENDENT_VERIFICATION_WALL_SECONDS = 120
_INDEPENDENCE_WORKER = Path(__file__).with_name("_independence_z3_worker.py")
_WORKER_OUTPUT_BYTES = CanonicalLimits().max_output_bytes
_WORKER_ERROR_BYTES = 16_384
_WORKER_ADDRESS_SPACE_BYTES = 1_536 * 1024 * 1024
_WORKER_FILE_SIZE_BYTES = 1_024 * 1_024


def _integer_bound(value: Any, fallback: int) -> int:
    import z3  # type: ignore[import-untyped]

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


def _solve_independence_number_values_kernel(
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

    import z3

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


def solve_independence_number_values(
    graph: SimpleUndirectedGraph,
    resource_budget: IndependenceNumberBudget,
) -> IndependenceNumberResult:
    """Run all Z3 optimization and exact replay in one bounded owner worker."""

    incumbent = () if not graph.vertices else (min(graph.vertices),)

    def fallback(detail: str) -> IndependenceNumberResult:
        return IndependenceNumberResult._from_kernel(
            graph=graph,
            status="UNKNOWN",
            optimum_value=None,
            upper_bound=len(graph.vertices),
            incumbent_vertices=incumbent,
            termination_reason="SOLVER_UNKNOWN",
            detail=detail,
        )

    deadline = time.monotonic() + resource_budget.wall_seconds
    try:
        with TemporaryDirectory(prefix="jacobian-graph-independence-") as directory:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                return fallback(
                    "the graph independence request expired before worker startup"
                )
            completed = run_bounded_process(
                [sys.executable, str(_INDEPENDENCE_WORKER)],
                input_bytes=json.dumps(
                    {
                        "graph": graph.model_dump(mode="json"),
                        "resource_budget": resource_budget.model_dump(mode="json"),
                    },
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8"),
                timeout_seconds=remaining_seconds,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_WORKER_OUTPUT_BYTES,
                stderr_limit=_WORKER_ERROR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(resource_budget.wall_seconds)),
                    address_space_bytes=_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=directory,
            )
    except OSError:
        return fallback("the bounded graph independence worker could not be started")
    if (
        completed.timed_out
        or completed.cancelled
        or completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        return fallback(
            "the bounded graph independence worker did not establish an outcome"
        )
    if time.monotonic() >= deadline:
        return fallback(
            "the graph independence request expired before response validation"
        )
    try:
        result = IndependenceNumberResult.model_validate(
            {
                **json.loads(completed.stdout.decode("utf-8")),
                "graph": graph.model_dump(mode="json"),
            }
        )
        return (
            result
            if time.monotonic() < deadline
            else fallback(
                "the graph independence request expired during response validation"
            )
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return fallback(
            "the bounded graph independence worker returned malformed output"
        )
