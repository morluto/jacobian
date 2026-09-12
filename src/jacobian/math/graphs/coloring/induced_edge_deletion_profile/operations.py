"""Induced edge deletion profile kernel with bounded exact colourability."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import suppress
from itertools import combinations
from threading import Event, Thread
from typing import Any

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_cancellation,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.coloring.induced_edge_deletion_profile._models import (
    DEFAULT_INDUCED_SOLVER_CONFLICTS,
    MAX_INDUCED_DELETION_R,
    MAX_INDUCED_DELETION_VERTICES,
    MAX_INDUCED_EDGE_MATERIALIZATION,
    MAX_INDUCED_LEDGER_CONFLICTS,
    MAX_INDUCED_RETAINED_LABEL_CHARACTERS,
    MAX_INDUCED_SOLVER_CALLS,
    InducedDeletionRow,
    InducedEdgeDeletionProfileResult,
    PerSizeMaximum,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

_OWNER_DEADLINE_SECONDS = 3600.0


def _require_execution_active(stage: str) -> None:
    request_checkpoint(stage)
    execution = current_request_execution()
    if (
        execution is not None
        and execution.deadline is not None
        and time.monotonic() >= execution.deadline
    ):
        raise OperationExecutionTimeoutError(f"request deadline expired {stage}")


def _set_remaining_z3_timeout(solver: Any) -> None:
    """Give each in-process Z3 call the request's remaining wall-time budget."""
    execution = current_request_execution()
    if execution is None or execution.deadline is None:
        return
    remaining_milliseconds = int((execution.deadline - time.monotonic()) * 1000)
    if remaining_milliseconds <= 0:
        raise OperationExecutionTimeoutError("request deadline expired before Z3 call")
    solver.set("timeout", remaining_milliseconds)


def _run_z3_check(solver: Any, stage: str) -> Any:
    """Run one solver check with cooperative cancellation and deadline interrupts.

    Z3's native timeout is necessary but cancellation signals are external to the
    solver.  A small request-scoped watchdog interrupts the solver object itself,
    so an in-process backend cannot outlive its request envelope.
    """
    execution = current_request_execution()
    cancellation = current_request_cancellation() or (
        execution.cancellation_signal if execution is not None else None
    )
    deadline = execution.deadline if execution is not None else None
    stop = Event()
    interrupted_for: list[str | None] = [None]

    def watch() -> None:
        while not stop.wait(0.005):
            if cancellation is not None and cancellation.is_set():
                interrupted_for[0] = "cancelled"
                with suppress(Exception):
                    solver.interrupt()
                return
            if deadline is not None and time.monotonic() >= deadline:
                interrupted_for[0] = "deadline"
                with suppress(Exception):
                    solver.interrupt()
                return

    watcher = Thread(target=watch, name="jacobian-z3-watchdog", daemon=True)
    watcher.start()
    try:
        outcome = solver.check()
    finally:
        stop.set()
        watcher.join(timeout=0.1)

    if cancellation is not None and cancellation.is_set():
        raise OperationExecutionCancelledError(f"request cancelled {stage}")
    if interrupted_for[0] == "cancelled":
        raise OperationExecutionCancelledError(f"request cancelled {stage}")
    _require_execution_active(f"after {stage}")
    if interrupted_for[0] == "deadline":
        raise OperationExecutionTimeoutError(f"request deadline expired {stage}")
    return outcome


def _is_bipartite_vertices_edges(
    vertices: list[str], edges: list[tuple[str, str]]
) -> bool:
    adj: dict[str, set[str]] = {v: set() for v in vertices}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    color: dict[str, bool] = {}
    for start in vertices:
        if start in color:
            continue
        color[start] = False
        stack = [start]
        while stack:
            v = stack.pop()
            for nb in adj[v]:
                if nb not in color:
                    color[nb] = not color[v]
                    stack.append(nb)
                elif color[nb] == color[v]:
                    return False
    return True


def _is_r_colorable_without_deletion(
    vertices: list[str],
    edges: list[tuple[str, str]],
    r: int,
    solver_conflicts: int,
) -> bool:
    """Check if graph (vertices, edges) is r-colourable using bounded Z3 or trivial cases."""
    if not edges:
        return True
    if r == 1:
        return False
    if r >= len(vertices):
        return True
    if r == 2:
        return _is_bipartite_vertices_edges(vertices, edges)
    # r >=3 use Z3
    return _z3_is_r_colorable(vertices, edges, r, solver_conflicts)


def _z3_is_r_colorable(
    vertices: list[str],
    edges: list[tuple[str, str]],
    r: int,
    solver_conflicts: int,
) -> bool:
    import z3

    _require_execution_active("during r-colourability check")
    solver = z3.Solver()
    solver.set("max_conflicts", solver_conflicts)
    _set_remaining_z3_timeout(solver)
    # map vertex -> int var
    var_map = {vertex: z3.Int(f"c_{index}") for index, vertex in enumerate(vertices)}
    for variable in var_map.values():
        solver.add(variable >= 0, variable < r)
    for a, b in edges:
        solver.add(var_map[a] != var_map[b])
    outcome = _run_z3_check(solver, "r-colourability check")
    if outcome == z3.sat:
        return True
    if outcome == z3.unsat:
        return False
    if solver.reason_unknown() in {"timeout", "max-conflicts-reached"}:
        raise OperationExecutionTimeoutError(
            "induced deletion profile r-colourability check exhausted its conflict budget"
        )
    raise RuntimeError("bounded r-colourability worker did not establish an outcome")


def _z3_exists_deletion_leq_k(
    vertices: list[str],
    edges: list[tuple[str, str]],
    r: int,
    k: int,
    solver_conflicts: int,
    partial: dict[int, bool] | None = None,
    exact: bool = False,
) -> bool:
    """Check existence of deletion set with |F| <=k (or ==k if exact) making graph r-colourable."""
    import z3

    _require_execution_active("during deletion feasibility check")
    if k < 0:
        return False
    m = len(edges)
    if k >= m and partial is None:
        # deleting all edges yields edgeless, which is r-colourable for r>=1
        return True
    if r == 1 and partial is None:
        # r=1 feasible iff we can delete all edges that remain (i.e., make edgeless)
        if exact:
            return k == m
        return k >= m
    solver = z3.Solver()
    solver.set("max_conflicts", solver_conflicts)
    _set_remaining_z3_timeout(solver)
    color_vars = {vertex: z3.Int(f"c_{index}") for index, vertex in enumerate(vertices)}
    for variable in color_vars.values():
        solver.add(variable >= 0, variable < r)
    deletion_vars = [z3.Bool(f"d_{idx}") for idx in range(m)]
    for idx, (a, b) in enumerate(edges):
        # if not deleted then colours differ
        solver.add(
            z3.Implies(z3.Not(deletion_vars[idx]), color_vars[a] != color_vars[b])
        )
    # cardinality
    total = z3.Sum([z3.If(v, 1, 0) for v in deletion_vars])
    if exact:
        solver.add(total == k)
    else:
        solver.add(total <= k)
    if partial is not None:
        for idx, val in partial.items():
            if val:
                solver.add(deletion_vars[idx])
            else:
                solver.add(z3.Not(deletion_vars[idx]))
    outcome = _run_z3_check(solver, "deletion feasibility check")
    if outcome == z3.sat:
        return True
    if outcome == z3.unsat:
        return False
    if solver.reason_unknown() in {"timeout", "max-conflicts-reached"}:
        raise OperationExecutionTimeoutError(
            "induced deletion feasibility check exhausted its conflict budget"
        )
    raise RuntimeError(
        "bounded deletion feasibility check did not establish an outcome"
    )


def _admit_induced_edge_deletion_profile(
    graph: SimpleUndirectedGraph,
    r: int,
    solver_conflicts: int,
) -> None:
    if type(r) is not int or not 1 <= r <= MAX_INDUCED_DELETION_R:
        raise OperationDomainValidationError(
            location=("r",),
            code="graph.induced_edge_deletion.r_out_of_range",
            message="r must be a positive integer",
        )
    if type(solver_conflicts) is not int or not 1 <= solver_conflicts <= 1_000_000:
        raise OperationDomainValidationError(
            location=("solver_conflicts",),
            code="graph.induced_edge_deletion.solver_conflicts_out_of_range",
            message="solver_conflicts must be an integer between 1 and 1000000",
        )
    n = len(graph.vertices)
    if n > MAX_INDUCED_DELETION_VERTICES:
        raise OperationDomainValidationError(
            location=("graph", "vertices"),
            code="graph.induced_edge_deletion.vertex_count_exceeds_bound",
            message=(
                f"induced-edge deletion profile supports at most {MAX_INDUCED_DELETION_VERTICES} "
                f"vertices (got {n})"
            ),
        )
    rows = 1 << n
    # rows bound already via n, but check
    if rows > (1 << MAX_INDUCED_DELETION_VERTICES):
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.induced_edge_deletion.rows_exceed_bound",
            message="induced-edge deletion profile subset rows exceed the admitted bound",
        )
    # retained label characters
    source_chars = sum(len(v) for v in graph.vertices) + sum(
        len(u) + len(v) for u, v in graph.edges
    )
    vertex_lengths = sorted((len(v) for v in graph.vertices), reverse=True)
    edge_lengths = sorted((len(u) + len(v) for u, v in graph.edges), reverse=True)
    # worst witness per row: all vertices + all edges (complete case)
    max_m = n * (n - 1) // 2
    largest_vertex_witness = sum(vertex_lengths)  # all vertices in subset
    largest_edge_witness = (
        sum(edge_lengths[: min(len(edge_lengths), max_m)]) if edge_lengths else 0
    )
    largest_witness = largest_vertex_witness + largest_edge_witness
    # per row we also store vertex_subset tuple (vertex labels) and deleted edges
    if source_chars + rows * largest_witness > MAX_INDUCED_RETAINED_LABEL_CHARACTERS:
        raise OperationDomainValidationError(
            location=("graph", "vertices"),
            code="graph.induced_edge_deletion.retained_labels_exceed_bound",
            message="induced-edge deletion profile exceeds the retained label-character bound",
        )
    m = len(graph.edges)
    edge_materialization_work = rows * m
    if edge_materialization_work > MAX_INDUCED_EDGE_MATERIALIZATION:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.induced_edge_deletion.materialization_exceeds_bound",
            message="induced-edge deletion profile induced-edge materialization exceeds its bound",
        )
    # Predicted solver calls: quick enumeration of induced edge counts per subset.
    # r=2 is solved exactly by finite cut enumeration below, so it must not be
    # charged against the Z3 ledger at all.
    sorted_vertices = sorted(graph.vertices)
    sorted_edges = tuple(sorted(graph.edges))
    # quick map for counting
    predicted_calls = 0
    for size in range(n + 1):
        for subset in combinations(sorted_vertices, size):
            _require_execution_active("during admission subset scan")
            subset_set = set(subset)
            m_s = 0
            for a, b in sorted_edges:
                if a in subset_set and b in subset_set:
                    m_s += 1
            if m_s == 0 or r in (1, 2) or r >= len(subset):
                continue
            # Initial feasibility + up to m_s optimum checks + up to m_s
            # canonical tie-break checks + one reconstruction check.
            predicted_calls += 2 * m_s + 2
            if predicted_calls > MAX_INDUCED_SOLVER_CALLS:
                raise OperationDomainValidationError(
                    location=("graph",),
                    code="graph.induced_edge_deletion.solver_calls_exceed_bound",
                    message="induced-edge deletion profile solver-call bound exceeded",
                )
    ledger = predicted_calls * solver_conflicts
    if ledger > MAX_INDUCED_LEDGER_CONFLICTS:
        raise OperationDomainValidationError(
            location=("solver_conflicts",),
            code="graph.induced_edge_deletion.ledger_exceeds_bound",
            message="induced-edge deletion profile ledger exceeds its conflict budget",
        )


def _min_bipartite_deletions(
    subset_vertices: tuple[str, ...], induced_edges: list[tuple[str, str]]
) -> tuple[int, tuple[tuple[str, str], ...]]:
    """Return the exact canonical edge-bipartization witness for one subset."""
    if _is_bipartite_vertices_edges(list(subset_vertices), induced_edges):
        return 0, ()
    vertex_index = {vertex: index for index, vertex in enumerate(subset_vertices)}
    best: tuple[int, tuple[tuple[str, str], ...]] | None = None
    for mask in range(1 << len(subset_vertices)):
        _require_execution_active("during bipartite cut enumeration")
        deleted = tuple(
            edge
            for edge in induced_edges
            if not (
                ((mask >> vertex_index[edge[0]]) ^ (mask >> vertex_index[edge[1]])) & 1
            )
        )
        candidate = (len(deleted), deleted)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best


def _compute_min_deletions_for_subset(  # noqa: C901
    subset_vertices: tuple[str, ...],
    induced_edges: list[tuple[str, str]],
    r: int,
    solver_conflicts: int,
) -> tuple[int, tuple[tuple[str, str], ...]]:
    """Return (min_deletions, canonical deleted_edges) for one subset S."""
    m = len(induced_edges)
    n_s = len(subset_vertices)
    if m == 0:
        return 0, ()
    if r >= n_s:
        return 0, ()
    if r == 1:
        # need to delete all edges to become edgeless
        return m, tuple(sorted(induced_edges))

    if r == 2:
        # Edge deletion to bipartiteness is the complement of a maximum cut.
        # The admitted n<=8 envelope makes exhaustive Boolean cut enumeration
        # exact, deterministic, and independent of the unrelated Z3 coloring
        # backend.  Choosing by (deletion count, deleted-edge tuple) gives the
        # required lexicographically smallest attaining source-edge set.
        return _min_bipartite_deletions(subset_vertices, induced_edges)

    # quick check if already r-colourable with 0 deletions
    _require_execution_active("during subset optimisation")
    if _is_r_colorable_without_deletion(
        list(subset_vertices), induced_edges, r, solver_conflicts
    ):
        return 0, ()

    # linear scan for minimal k
    min_k: int | None = None
    for k in range(1, m + 1):
        _require_execution_active("during deletion optimum search")
        if _z3_exists_deletion_leq_k(
            list(subset_vertices),
            induced_edges,
            r,
            k,
            solver_conflicts,
            partial=None,
            exact=False,
        ):
            min_k = k
            break
    if min_k is None:
        # Should not happen because deleting all edges yields edgeless which is r-colourable
        # For r>=1, edgeless is 1-colourable so feasible
        min_k = m
    k = min_k

    # Find canonical attaining set via greedy lexicographic fixing
    # induced_edges are already sorted lexicographically
    partial: dict[int, bool] = {}
    # we know total exactly k
    for idx in range(m):
        _require_execution_active("during canonical tie-break")
        already_chosen = sum(1 for v in partial.values() if v)
        remaining_needed = k - already_chosen
        remaining_positions = m - idx
        # mandatory cases without solver
        if remaining_needed == 0:
            partial[idx] = False
            continue
        if remaining_needed == remaining_positions:
            # must delete all remaining
            for j in range(idx, m):
                partial[j] = True
            break
        if remaining_needed < 0 or remaining_needed > remaining_positions:
            raise AssertionError("lexicographic remaining count infeasible")
        # try include idx
        test_partial = dict(partial)
        test_partial[idx] = True
        feasible_include = _z3_exists_deletion_leq_k(
            list(subset_vertices),
            induced_edges,
            r,
            k,
            solver_conflicts,
            partial=test_partial,
            exact=True,
        )
        if feasible_include:
            partial[idx] = True
        else:
            partial[idx] = False
            # must be feasible with False, else no solution exists which contradicts existence of k
            # we could assert
            # check quickly that False branch is feasible (should be)
            # Not needed to solver check again; greedy ensures alternative works
            # But we could verify for debugging
    deleted = tuple(induced_edges[i] for i, val in sorted(partial.items()) if val)
    # sanity: should be sorted already because induced_edges sorted and we iterate in order
    assert deleted == tuple(sorted(deleted))
    assert len(deleted) == k
    # verify that G[S]-F is indeed r-colourable (reconstruction)
    remaining_edges = [e for e in induced_edges if e not in set(deleted)]
    if not _is_r_colorable_without_deletion(
        list(subset_vertices), remaining_edges, r, solver_conflicts
    ):
        raise AssertionError("canonical deletion set does not yield r-colourable graph")
    return k, deleted


def compute_induced_edge_deletion_profile(
    graph: SimpleUndirectedGraph,
    r: int,
    solver_conflicts: int = DEFAULT_INDUCED_SOLVER_CONFLICTS,
) -> InducedEdgeDeletionProfileResult:
    """Return D_{G,r}(S) for every vertex subset S with canonical attaining F and per-size maxima.

    For each subset S of V(G), D(S) is the minimum number of edges to delete from
    the induced subgraph G[S] to obtain an r-colourable graph, and the returned
    F is the lexicographically smallest source-edge subset attaining that minimum.
    The per-size maxima are derived as max_{|S|=m} D(S).
    """
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return compute_induced_edge_deletion_profile(graph, r, solver_conflicts)
    if execution.deadline is None:
        bind_request_deadline(execution.started_at + _OWNER_DEADLINE_SECONDS)
    _require_execution_active("before admission")
    _admit_induced_edge_deletion_profile(graph, r, solver_conflicts)

    sorted_vertices = sorted(graph.vertices)
    sorted_edges = tuple(sorted(graph.edges))
    n = len(sorted_vertices)

    rows: list[InducedDeletionRow] = []
    # enumerate subsets by increasing size then lexicographic
    for size in range(n + 1):
        _require_execution_active("during profile enumeration")
        for subset in combinations(sorted_vertices, size):
            _require_execution_active("during subset processing")
            subset_tuple = tuple(subset)  # already sorted
            subset_set = set(subset_tuple)
            induced_edges = [
                e for e in sorted_edges if e[0] in subset_set and e[1] in subset_set
            ]
            induced_edge_count = len(induced_edges)
            min_k, deleted = _compute_min_deletions_for_subset(
                subset_tuple, induced_edges, r, solver_conflicts
            )
            rows.append(
                InducedDeletionRow(
                    vertex_subset=subset_tuple,
                    min_deletions=min_k,
                    deleted_edges=deleted,
                    induced_edge_count=induced_edge_count,
                )
            )

    # rows already in canonical order due to enumeration order
    # derive per-size maxima
    grouped: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        grouped[len(row.vertex_subset)].append(row.min_deletions)
    per_size: list[PerSizeMaximum] = []
    for size in range(n + 1):
        vals = grouped.get(size, [])
        if not vals:
            # Should not happen because there is at least one subset per size
            # For completeness, zero
            per_size.append(
                PerSizeMaximum(
                    subset_size=size, maximum_min_deletions=0, attaining_subset_count=0
                )
            )
        else:
            max_val = max(vals)
            count = sum(1 for v in vals if v == max_val)
            per_size.append(
                PerSizeMaximum(
                    subset_size=size,
                    maximum_min_deletions=max_val,
                    attaining_subset_count=count,
                )
            )

    return InducedEdgeDeletionProfileResult._from_kernel(
        graph=graph,
        r=r,
        rows=tuple(rows),
        max_deletions_by_size=tuple(per_size),
    )


__all__ = ["compute_induced_edge_deletion_profile"]
