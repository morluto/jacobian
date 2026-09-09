"""Private Z3 backend for bounded hypergraph independence-number search."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from jacobian._execution import (
    BackendFailureReason,
    OperationBackendError,
    lease_operation_phases,
    remaining_timeout_ms,
    require_execution_deadline,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
    HypergraphIndependenceBudget,
    HypergraphIndependenceResult,
    HypergraphIndependenceStatus,
    HypergraphIndependenceTermination,
    _greedy_independent_vertices,
    _independence_upper_bound,
)
from jacobian.process import (
    ProcessResourceLimits,
    check_bounded_process_result,
    decode_checked_worker_output,
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
        solver.set(timeout=remaining_timeout_ms(max(1, remaining_ms)))
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
    source: FiniteHypergraph,
    resource_budget: HypergraphIndependenceBudget,
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
        hypergraph=source,
        resource_budget=resource_budget,
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
    source: FiniteHypergraph,
    resource_budget: HypergraphIndependenceBudget,
) -> HypergraphIndependenceResult:
    """Refine cardinality thresholds by binary search on a monotone predicate.

    ``exists independent set of size >= k`` is monotone decreasing in ``k``
    (subsets of independent sets are independent), so one UNSAT at ``k``
    proves every larger threshold infeasible while one SAT witness of size
    ``m`` proves every threshold ``<= m`` feasible. The loop keeps a feasible
    lower bound ``lo`` with its incumbent witness and a proved upper bound
    ``hi``, querying ``mid = (lo + hi + 1) // 2`` until they coincide.
    """

    import z3

    started = time.monotonic()
    vertices = source.vertices
    incumbent = _greedy_independent_vertices(source)
    source_upper_bound = _independence_upper_bound(source)
    upper_bound = source_upper_bound
    if len(incumbent) == upper_bound:
        return _result(
            source,
            resource_budget,
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
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT) from exc

    solver_calls = 0
    lower_bound = len(incumbent)
    while lower_bound < upper_bound:
        if solver_calls >= resource_budget.max_solver_calls:
            return _result(
                source,
                resource_budget,
                status="UNKNOWN",
                independence_number=None,
                incumbent=incumbent,
                upper_bound=upper_bound,
                solver_calls=solver_calls,
                wall_budget_exhausted=False,
                termination_reason="SOLVER_CALL_LIMIT",
                detail="the binary threshold search exhausted its solver-call budget",
            )
        if _remaining_ms(started, resource_budget.wall_seconds) <= 0:
            return _result(
                source,
                resource_budget,
                status="UNKNOWN",
                independence_number=None,
                incumbent=incumbent,
                upper_bound=upper_bound,
                solver_calls=solver_calls,
                wall_budget_exhausted=True,
                termination_reason="WALL_TIME",
                detail="the wall-clock budget expired before the next threshold query",
            )

        threshold = (lower_bound + upper_bound + 1) // 2
        try:
            solver_calls += 1
            solver_status, candidate, reason = _check_threshold(
                solver,
                selected,
                cardinality,
                threshold,
                started,
                resource_budget.wall_seconds,
                vertices,
            )
        except z3.Z3Exception as exc:
            raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT) from exc
        if solver_status == z3.unsat:
            upper_bound = threshold - 1
            continue
        if solver_status == z3.sat:
            if len(
                candidate
            ) < threshold or not _solver_witness_is_canonical_and_independent(
                source, candidate
            ):
                raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
            if len(candidate) > len(incumbent):
                incumbent = candidate
                lower_bound = max(lower_bound, len(candidate))
            else:
                lower_bound = max(lower_bound, threshold)
            continue

        wall_expired = (
            _remaining_ms(started, resource_budget.wall_seconds) <= 0
            or "timeout" in reason.lower()
        )
        return _result(
            source,
            resource_budget,
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
        source,
        resource_budget,
        status="EXACT",
        independence_number=len(incumbent),
        incumbent=incumbent,
        upper_bound=len(incumbent),
        solver_calls=solver_calls,
        wall_budget_exhausted=False,
        termination_reason="OPTIMUM_ESTABLISHED",
        detail="the binary threshold search established coincident feasible and infeasible bounds",
    )


def _run_independence_worker(payload: dict[str, object], *, deadline: float) -> object:
    """Run one complete Z3 kernel in an isolated bounded owner process."""

    require_execution_deadline(deadline)
    timeout_seconds = deadline - time.monotonic()
    try:
        with TemporaryDirectory(
            prefix="jacobian-hypergraph-independence-"
        ) as directory:
            completed = run_bounded_process(
                [sys.executable, str(_INDEPENDENCE_WORKER)],
                input_bytes=json.dumps(
                    {"_deadline": deadline, **payload},
                    separators=(",", ":"),
                    ensure_ascii=False,
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
    except OSError as exc:
        require_execution_deadline(deadline)
        raise OperationBackendError(BackendFailureReason.STARTUP) from exc
    check_bounded_process_result(completed)
    require_execution_deadline(deadline)
    response = decode_checked_worker_output(
        completed.stdout,
        decode_result=lambda value: cast(object, value),
        checkpoint=lambda: require_execution_deadline(deadline),
    )
    require_execution_deadline(deadline)
    return response


def solve_independence_number(
    source: FiniteHypergraph,
    resource_budget: HypergraphIndependenceBudget,
) -> HypergraphIndependenceResult:
    """Run every Z3 phase under one process and resource envelope."""

    lease = lease_operation_phases(
        resource_budget.wall_seconds,
        admitted_response_bytes=_WORKER_OUTPUT_BYTES,
        validation_work=len(source.vertices) + len(source.edges),
    )
    response = _run_independence_worker(
        {
            "kind": "solve",
            "hypergraph": source.model_dump(mode="json"),
            "resource_budget": resource_budget.model_dump(mode="json"),
        },
        deadline=lease.backend_deadline,
    )
    require_execution_deadline(lease.operation_deadline)
    if (
        not isinstance(response, dict)
        or "hypergraph" in response
        or "resource_budget" in response
    ):
        raise OperationBackendError(BackendFailureReason.MALFORMED_RESPONSE)
    try:
        result = HypergraphIndependenceResult.model_validate(
            {
                **response,
                "hypergraph": source.model_dump(mode="json"),
                "resource_budget": resource_budget.model_dump(mode="json"),
            }
        )
    except (TypeError, ValueError) as exc:
        require_execution_deadline(lease.operation_deadline)
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT) from exc
    require_execution_deadline(lease.operation_deadline)
    return result


__all__: list[str] = []
