"""Private Z3/NetworkX backend for bounded independence-number search."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from jacobian.canonical import CanonicalLimits
from jacobian.math.graphs.independence import (
    IndependenceNumberBudget,
    IndependenceNumberResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_INDEPENDENCE_WORKER = Path(__file__).with_name("_independence_z3_worker.py")
_WORKER_OUTPUT_BYTES = CanonicalLimits().max_output_bytes
_WORKER_ERROR_BYTES = 16_384
_WORKER_ADDRESS_SPACE_BYTES = 1_536 * 1024 * 1024
_WORKER_FILE_SIZE_BYTES = 1_024 * 1_024


def _integer_bound(value: Any, fallback: int) -> int:
    import z3

    return value.as_long() if z3.is_int_value(value) else fallback


def _solve_independence_number_values_kernel(
    graph: SimpleUndirectedGraph,
    resource_budget: IndependenceNumberBudget,
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
    """Run Z3 optimization in one bounded owner worker and decode its result."""

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
