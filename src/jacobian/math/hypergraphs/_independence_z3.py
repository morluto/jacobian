"""Private Z3 backend for bounded hypergraph independence-number search."""

from __future__ import annotations

import time
from typing import Any

import z3  # type: ignore[import-untyped]

from jacobian.math.hypergraphs._models import (
    FiniteHypergraph,
    HypergraphIndependenceRequest,
    HypergraphIndependenceResult,
    HypergraphIndependenceStatus,
    HypergraphIndependenceTermination,
    _greedy_independent_vertices,
    _independence_upper_bound,
)


def _remaining_ms(started: float, wall_seconds: int) -> int:
    return int((wall_seconds - (time.monotonic() - started)) * 1000)


def _build_solver(
    source: FiniteHypergraph,
) -> tuple[Any, dict[str, Any], Any]:
    solver = z3.Solver()
    selected = {
        vertex: z3.Bool(f"hypergraph_selected_{index}")
        for index, vertex in enumerate(source.vertices)
    }
    for _, members in source.edges:
        solver.add(z3.Or(*(z3.Not(selected[vertex]) for vertex in members)))
    cardinality = z3.Sum([z3.If(selected[vertex], 1, 0) for vertex in source.vertices])
    return solver, selected, cardinality


def verify_upper_bound(
    source: FiniteHypergraph,
    upper_bound: int,
    wall_seconds: int,
) -> bool:
    """Replay that no independent set exceeds the reported upper bound."""

    try:
        solver, _, cardinality = _build_solver(source)
        solver.set(timeout=wall_seconds * 1000)
        solver.add(cardinality >= upper_bound + 1)
        return bool(solver.check() == z3.unsat)
    except z3.Z3Exception:
        return False


def _check_threshold(
    solver: Any,
    selected: dict[str, Any],
    cardinality: Any,
    threshold: int,
    timeout_ms: int,
    vertex_order: tuple[str, ...],
) -> tuple[object, tuple[str, ...], str]:
    solver.push()
    try:
        solver.set(timeout=max(1, timeout_ms))
        solver.add(cardinality >= threshold)
        status = solver.check()
        if status != z3.sat:
            reason = solver.reason_unknown() if status == z3.unknown else ""
            return status, (), reason
        model = solver.model()
        witness = tuple(
            vertex
            for vertex in vertex_order
            if z3.is_true(model.eval(selected[vertex], model_completion=True))
        )
        return status, witness, ""
    finally:
        solver.pop()


def _result(
    request: HypergraphIndependenceRequest,
    *,
    status: HypergraphIndependenceStatus,
    independence_number: int | None,
    incumbent: tuple[str, ...],
    upper_bound: int,
    solver_calls: int,
    wall_budget_exhausted: bool,
    termination_reason: HypergraphIndependenceTermination,
    detail: str,
) -> HypergraphIndependenceResult:
    return HypergraphIndependenceResult._from_kernel(
        hypergraph=request.hypergraph,
        resource_budget=request.resource_budget,
        status=status,
        independence_number=independence_number,
        incumbent_vertices=incumbent,
        upper_bound=upper_bound,
        solver_calls=solver_calls,
        wall_budget_exhausted=wall_budget_exhausted,
        termination_reason=termination_reason,
        detail=detail,
    )


def verify_independence_result(result: HypergraphIndependenceResult) -> bool:
    """Independently replay a claimed strict upper bound within its budget."""

    source_upper_bound = _independence_upper_bound(result.hypergraph)
    if result.upper_bound == source_upper_bound:
        return True
    return verify_upper_bound(
        result.hypergraph,
        result.upper_bound,
        result.resource_budget.wall_seconds,
    )


def solve_independence_number(
    request: HypergraphIndependenceRequest,
) -> HypergraphIndependenceResult:
    """Search cardinality thresholds from a sound source-derived upper bound."""

    started = time.monotonic()
    source = request.hypergraph
    vertices = source.vertices
    incumbent = _greedy_independent_vertices(source)
    source_upper_bound = _independence_upper_bound(source)
    upper_bound = source_upper_bound
    if len(incumbent) == upper_bound:
        return _result(
            request,
            status="EXACT",
            independence_number=len(incumbent),
            incumbent=incumbent,
            upper_bound=upper_bound,
            solver_calls=0,
            wall_budget_exhausted=False,
            termination_reason="SPECIAL_CASE",
            detail=(
                "the deterministic feasible witness meets the source-derived "
                "singleton-edge upper bound"
            ),
        )

    try:
        solver, selected, cardinality = _build_solver(source)
    except z3.Z3Exception as exc:
        return _result(
            request,
            status="UNKNOWN",
            independence_number=None,
            incumbent=incumbent,
            upper_bound=source_upper_bound,
            solver_calls=0,
            wall_budget_exhausted=False,
            termination_reason="SOLVER_ERROR",
            detail=f"the exact backend rejected the admitted encoding: {str(exc)[:800]}",
        )

    solver_calls = 0
    for threshold in range(upper_bound, len(incumbent), -1):
        if solver_calls >= request.resource_budget.max_solver_calls:
            return _result(
                request,
                status="UNKNOWN",
                independence_number=None,
                incumbent=incumbent,
                upper_bound=upper_bound,
                solver_calls=solver_calls,
                wall_budget_exhausted=False,
                termination_reason="SOLVER_CALL_LIMIT",
                detail="the descending threshold search exhausted its solver-call budget",
            )
        remaining_ms = _remaining_ms(started, request.resource_budget.wall_seconds)
        if remaining_ms <= 0:
            return _result(
                request,
                status="UNKNOWN",
                independence_number=None,
                incumbent=incumbent,
                upper_bound=upper_bound,
                solver_calls=solver_calls,
                wall_budget_exhausted=True,
                termination_reason="WALL_TIME",
                detail="the wall-clock budget expired before the next threshold query",
            )

        try:
            solver_calls += 1
            solver_status, candidate, reason = _check_threshold(
                solver,
                selected,
                cardinality,
                threshold,
                remaining_ms,
                vertices,
            )
        except z3.Z3Exception as exc:
            return _result(
                request,
                status="UNKNOWN",
                independence_number=None,
                incumbent=incumbent,
                upper_bound=source_upper_bound,
                solver_calls=solver_calls,
                wall_budget_exhausted=False,
                termination_reason="SOLVER_ERROR",
                detail=f"the exact backend failed during a threshold query: {str(exc)[:800]}",
            )
        if solver_status == z3.unsat:
            upper_bound = threshold - 1
            continue
        if solver_status == z3.sat:
            if len(candidate) < threshold:
                return _result(
                    request,
                    status="UNKNOWN",
                    independence_number=None,
                    incumbent=incumbent,
                    upper_bound=source_upper_bound,
                    solver_calls=solver_calls,
                    wall_budget_exhausted=False,
                    termination_reason="SOLVER_ERROR",
                    detail=(
                        "the exact backend returned a satisfying witness below "
                        f"the submitted threshold {threshold}"
                    ),
                )
            return _result(
                request,
                status="EXACT",
                independence_number=len(candidate),
                incumbent=candidate,
                upper_bound=len(candidate),
                solver_calls=solver_calls,
                wall_budget_exhausted=False,
                termination_reason="OPTIMUM_ESTABLISHED",
                detail="the exact backend established the first feasible descending threshold",
            )

        wall_expired = (
            _remaining_ms(started, request.resource_budget.wall_seconds) <= 0
            or "timeout" in reason.lower()
        )
        return _result(
            request,
            status="UNKNOWN",
            independence_number=None,
            incumbent=incumbent,
            upper_bound=upper_bound,
            solver_calls=solver_calls,
            wall_budget_exhausted=wall_expired,
            termination_reason="WALL_TIME" if wall_expired else "SOLVER_UNKNOWN",
            detail=(
                "the exact backend returned unknown for the next threshold: "
                f"{reason or 'no reason'}"
            ),
        )

    return _result(
        request,
        status="EXACT",
        independence_number=len(incumbent),
        incumbent=incumbent,
        upper_bound=len(incumbent),
        solver_calls=solver_calls,
        wall_budget_exhausted=False,
        termination_reason="OPTIMUM_ESTABLISHED",
        detail="all larger cardinality thresholds were proved unsatisfiable",
    )


__all__: list[str] = []
