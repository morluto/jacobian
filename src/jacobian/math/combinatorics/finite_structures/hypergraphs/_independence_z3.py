"""Private Z3 backend for bounded hypergraph independence-number search."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
    HypergraphIndependenceRequest,
    HypergraphIndependenceResult,
    HypergraphIndependenceStatus,
    HypergraphIndependenceTermination,
    _greedy_independent_vertices,
    _hypergraph_digest,
    _independence_upper_bound,
)
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_INDEPENDENCE_WORKER = Path(__file__).with_name("_independence_z3_worker.py")
_WORKER_OUTPUT_BYTES = 64 * 1024
_WORKER_ERROR_BYTES = 16_384
_WORKER_ADDRESS_SPACE_BYTES = 1_536 * 1024 * 1024
_WORKER_FILE_SIZE_BYTES = 1_024 * 1_024


def _remaining_ms(started: float, wall_seconds: int) -> int:
    return int((wall_seconds - (time.monotonic() - started)) * 1000)


def _build_solver(
    source: FiniteHypergraph,
) -> tuple[Any, dict[str, Any], Any]:
    import z3

    solver = z3.Solver()
    selected = {
        vertex: z3.Bool(f"hypergraph_selected_{index}")
        for index, vertex in enumerate(source.vertices)
    }
    for _, members in source.edges:
        solver.add(z3.Or(*(z3.Not(selected[vertex]) for vertex in members)))
    cardinality = z3.Sum([z3.If(selected[vertex], 1, 0) for vertex in source.vertices])
    return solver, selected, cardinality


def _check_threshold(
    solver: Any,
    selected: dict[str, Any],
    cardinality: Any,
    threshold: int,
    started: float,
    wall_seconds: int,
    vertex_order: tuple[str, ...],
) -> tuple[object, tuple[str, ...], str]:
    import z3

    solver.push()
    try:
        solver.add(cardinality >= threshold)
        remaining_ms = _remaining_ms(started, wall_seconds)
        if remaining_ms <= 0:
            return z3.unknown, (), "the wall-clock budget expired during encoding"
        solver.set(timeout=max(1, remaining_ms))
        status = solver.check()
        if status != z3.sat:
            reason = solver.reason_unknown() if status == z3.unknown else ""
            return status, (), reason
        if _remaining_ms(started, wall_seconds) <= 0:
            return (
                z3.unknown,
                (),
                "the wall-clock budget expired before model extraction",
            )
        model = solver.model()
        witness = tuple(
            vertex
            for vertex in vertex_order
            if z3.is_true(model.eval(selected[vertex], model_completion=True))
        )
        if _remaining_ms(started, wall_seconds) <= 0:
            return (
                z3.unknown,
                (),
                "the wall-clock budget expired during model extraction",
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


def _solver_witness_is_canonical_and_independent(
    source: FiniteHypergraph,
    candidate: tuple[str, ...],
) -> bool:
    """Check backend witness shape and the defining independent-set invariant."""

    candidate_set = set(candidate)
    if len(candidate_set) != len(candidate) or any(
        vertex not in source.vertices for vertex in candidate
    ):
        return False
    if (
        tuple(vertex for vertex in source.vertices if vertex in candidate_set)
        != candidate
    ):
        return False
    return not any(set(members) <= candidate_set for _, members in source.edges)


def _solve_independence_number_kernel(
    request: HypergraphIndependenceRequest,
) -> HypergraphIndependenceResult:
    """Search cardinality thresholds from a sound source-derived upper bound."""

    import z3

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
        if _remaining_ms(started, request.resource_budget.wall_seconds) <= 0:
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
                started,
                request.resource_budget.wall_seconds,
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
            if len(
                candidate
            ) < threshold or not _solver_witness_is_canonical_and_independent(
                source, candidate
            ):
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


def _run_independence_worker(
    payload: dict[str, object], *, timeout_seconds: float
) -> object | None:
    """Run one complete Z3 kernel in an isolated bounded owner process."""

    try:
        with TemporaryDirectory(
            prefix="jacobian-hypergraph-independence-"
        ) as directory:
            completed = run_bounded_process(
                [sys.executable, str(_INDEPENDENCE_WORKER)],
                input_bytes=json.dumps(
                    payload, separators=(",", ":"), ensure_ascii=False
                ).encode("utf-8"),
                timeout_seconds=timeout_seconds,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_WORKER_OUTPUT_BYTES,
                stderr_limit=_WORKER_ERROR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(timeout_seconds)),
                    address_space_bytes=_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=directory,
            )
    except OSError:
        return None
    if (
        completed.timed_out
        or completed.cancelled
        or completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        return None
    try:
        return cast(object, json.loads(completed.stdout.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def solve_independence_number(
    request: HypergraphIndependenceRequest,
) -> HypergraphIndependenceResult:
    """Run every Z3 phase under one process and resource envelope."""

    source_upper_bound = _independence_upper_bound(request.hypergraph)
    incumbent = _greedy_independent_vertices(request.hypergraph)
    started = time.monotonic()
    remaining_seconds = (
        _remaining_ms(started, request.resource_budget.wall_seconds) / 1_000
    )
    if remaining_seconds <= 0:
        return _result(
            request,
            status="UNKNOWN",
            independence_number=None,
            incumbent=incumbent,
            upper_bound=source_upper_bound,
            solver_calls=0,
            wall_budget_exhausted=True,
            termination_reason="WALL_TIME",
            detail="the hypergraph independence request expired before worker startup",
        )
    response = _run_independence_worker(
        {"kind": "solve", "request": request.model_dump(mode="json")},
        timeout_seconds=remaining_seconds,
    )
    if not isinstance(response, dict):
        return _result(
            request,
            status="UNKNOWN",
            independence_number=None,
            incumbent=incumbent,
            upper_bound=source_upper_bound,
            solver_calls=0,
            wall_budget_exhausted=False,
            termination_reason="SOLVER_ERROR",
            detail="the bounded hypergraph independence worker did not establish an outcome",
        )
    if _remaining_ms(started, request.resource_budget.wall_seconds) <= 0:
        return _result(
            request,
            status="UNKNOWN",
            independence_number=None,
            incumbent=incumbent,
            upper_bound=source_upper_bound,
            solver_calls=0,
            wall_budget_exhausted=True,
            termination_reason="WALL_TIME",
            detail="the hypergraph independence request expired before response validation",
        )
    try:
        # The worker returns only its bounded outcome projection.  Retained
        # source data belongs to this parent request and must not consume the
        # worker channel or be trusted from child output.
        result = HypergraphIndependenceResult.model_validate(
            {
                **response,
                "hypergraph": request.hypergraph.model_dump(mode="json"),
                "hypergraph_digest": _hypergraph_digest(request.hypergraph),
                "resource_budget": request.resource_budget.model_dump(mode="json"),
            }
        )
        if _remaining_ms(started, request.resource_budget.wall_seconds) > 0:
            return result
        raise ValueError("request expired during response validation")
    except (TypeError, ValueError):
        return _result(
            request,
            status="UNKNOWN",
            independence_number=None,
            incumbent=incumbent,
            upper_bound=source_upper_bound,
            solver_calls=0,
            wall_budget_exhausted=False,
            termination_reason="SOLVER_ERROR",
            detail="the bounded hypergraph independence worker returned malformed output",
        )


__all__: list[str] = []
