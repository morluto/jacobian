"""Private Z3/NetworkX backend for bounded independence-number search."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from jacobian._execution import (
    BackendFailureReason,
    OperationBackendError,
    OperationExecutionTimeoutError,
    lease_operation_phases,
    remaining_timeout_ms,
    request_checkpoint,
    require_execution_deadline,
)
from jacobian._worker_protocol import (
    encode_worker_result_frame,
)
from jacobian.math.graphs.independence import (
    IndependenceNumberBudget,
    IndependenceNumberResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.process import (
    ProcessResourceLimits,
    check_bounded_process_result,
    decode_checked_worker_output,
    run_bounded_process,
    worker_environment,
)

_INDEPENDENCE_WORKER = Path(__file__).with_name("_independence_z3_worker.py")
_WORKER_ERROR_BYTES = 16_384
_WORKER_ADDRESS_SPACE_BYTES = 1_536 * 1024 * 1024
_WORKER_FILE_SIZE_BYTES = 1_024 * 1_024


def _independence_worker_stdout_limit(graph: SimpleUndirectedGraph) -> int:
    """Measure a complete feasible upper envelope for the private projection."""

    projection = {
        "status": "UNKNOWN",
        "order": 128,
        "optimum_value": None,
        "incumbent_value": 128,
        "lower_bound": 128,
        "upper_bound": 128,
        "witness_vertices": list(graph.vertices),
        "termination_reason": "OPTIMUM_ESTABLISHED",
        "detail": "x" * 1_024,
        "convention": "MAXIMUM_EDGE_FREE_VERTEX_SUBSET",
    }
    return max(
        8192,
        len(encode_worker_result_frame(projection)),
    )


def _integer_bound(value: Any, fallback: int) -> int:
    import z3

    return value.as_long() if z3.is_int_value(value) else fallback


def _closed_objective_value(objective: Any) -> int | None:
    """Return an objective value only when both Optimize bounds are closed."""

    import z3

    lower = objective.lower()
    upper = objective.upper()
    if not (z3.is_int_value(lower) and z3.is_int_value(upper)):
        return None
    lower_value = lower.as_long()
    upper_value = upper.as_long()
    return lower_value if lower_value == upper_value else None


def _solve_independence_number_values_kernel(
    graph: SimpleUndirectedGraph,
    resource_budget: IndependenceNumberBudget,
    *,
    canonicalize_witness: bool = False,
) -> IndependenceNumberResult:
    """Run one wall-clock-bounded exact maximum independent-set optimization.

    The trusted factory performs the structural source and witness checks.
    Every incomplete outcome, including a ``sat`` optimize whose objective
    bounds stay open, reports the graph order as its independently safe upper
    bound.
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
    optimizer.set(priority="lex")
    optimizer.set(timeout=remaining_timeout_ms(max(1, remaining_ms)))
    selected = {
        vertex: z3.Bool(f"selected_{index}") for index, vertex in enumerate(vertices)
    }
    for left, right in graph.edges:
        optimizer.add(z3.Or(z3.Not(selected[left]), z3.Not(selected[right])))
    cardinality = z3.Sum([z3.If(selected[vertex], 1, 0) for vertex in vertices])
    objective = optimizer.maximize(cardinality)

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
        if lower_bound == upper_bound == len(incumbent) and (
            _closed_objective_value(objective) == len(incumbent)
        ):
            if canonicalize_witness:
                remaining_ms = int(
                    (resource_budget.wall_seconds - (time.monotonic() - started))
                    * 1000
                )
                if remaining_ms > 0:
                    lex_optimizer = z3.Optimize()
                    lex_optimizer.set(priority="lex")
                    lex_optimizer.set(timeout=remaining_timeout_ms(max(1, remaining_ms)))
                    lex_selected = {
                        vertex: z3.Bool(f"lex_{index}")
                        for index, vertex in enumerate(vertices)
                    }
                    for left, right in graph.edges:
                        lex_optimizer.add(
                            z3.Or(
                                z3.Not(lex_selected[left]),
                                z3.Not(lex_selected[right]),
                            )
                        )
                    lex_optimizer.add(
                        z3.Sum(
                            [
                                z3.If(lex_selected[vertex], 1, 0)
                                for vertex in vertices
                            ]
                        )
                        == len(incumbent)
                    )
                    for vertex in vertices:
                        lex_optimizer.maximize(z3.If(lex_selected[vertex], 1, 0))
                    if lex_optimizer.check() == z3.sat:
                        lex_model = lex_optimizer.model()
                        incumbent = tuple(
                            sorted(
                                vertex
                                for vertex, variable in lex_selected.items()
                                if z3.is_true(
                                    lex_model.eval(variable, model_completion=True)
                                )
                            )
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
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
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
    *,
    canonicalize_witness: bool = False,
) -> IndependenceNumberResult:
    """Run Z3 optimization in one bounded owner worker and decode its result."""

    stdout_limit = _independence_worker_stdout_limit(graph)
    lease = lease_operation_phases(
        resource_budget.wall_seconds,
        admitted_response_bytes=stdout_limit,
        validation_work=len(graph.vertices) + len(graph.edges),
    )
    deadline = lease.operation_deadline
    try:
        with TemporaryDirectory(prefix="jacobian-graph-independence-") as directory:
            remaining_seconds = lease.backend_deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise OperationExecutionTimeoutError("operation deadline expired")
            completed = run_bounded_process(
                [sys.executable, str(_INDEPENDENCE_WORKER)],
                input_bytes=json.dumps(
                    {
                        "_deadline": lease.backend_deadline,
                        "graph": graph.model_dump(mode="json"),
                        "resource_budget": resource_budget.model_dump(mode="json"),
                        "canonicalize_witness": canonicalize_witness,
                    },
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8"),
                timeout_seconds=remaining_seconds,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=stdout_limit,
                stderr_limit=_WORKER_ERROR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(resource_budget.wall_seconds)),
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
    try:

        def decode(response: Any) -> IndependenceNumberResult:
            if not isinstance(response, dict) or "graph" in response:
                raise ValueError
            return IndependenceNumberResult.model_validate(
                {**response, "graph": graph.model_dump(mode="json")}
            )

        result = decode_checked_worker_output(
            completed.stdout,
            decode_result=decode,
            checkpoint=lambda: require_execution_deadline(deadline),
        )
        request_checkpoint("after graph independence response validation")
        require_execution_deadline(deadline)
        return result
    except (TypeError, ValueError) as exc:
        request_checkpoint("during graph independence response validation")
        require_execution_deadline(deadline)
        raise OperationBackendError(BackendFailureReason.MALFORMED_RESPONSE) from exc
