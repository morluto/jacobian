"""Private Z3/NetworkX backend for bounded independence-number search."""

from __future__ import annotations

import time
from typing import Literal

import z3  # type: ignore[import-untyped]

from jacobian.math.graphs.independence import (
    IndependenceNumberRequest,
    IndependenceNumberResult,
)


def _integer_bound(value: z3.ArithRef, fallback: int) -> int:
    return value.as_long() if z3.is_int_value(value) else fallback


def solve_independence_number(
    request: IndependenceNumberRequest,
) -> IndependenceNumberResult:
    """Run one wall-clock-bounded exact maximum independent-set optimization."""

    started = time.monotonic()
    vertices = request.graph.vertices
    order = len(vertices)
    if not vertices:
        return IndependenceNumberResult(
            status="EXACT",
            order=0,
            optimum_value=0,
            incumbent_value=0,
            lower_bound=0,
            upper_bound=0,
            witness_vertices=(),
            termination_reason="SPECIAL_CASE",
            detail="the empty graph has independence number zero",
        )

    incumbent: tuple[str, ...] = (min(vertices),)
    remaining_ms = int(
        (request.resource_budget.wall_seconds - (time.monotonic() - started)) * 1000
    )
    if remaining_ms <= 0:
        return IndependenceNumberResult(
            status="UNKNOWN",
            order=order,
            optimum_value=None,
            incumbent_value=len(incumbent),
            lower_bound=len(incumbent),
            upper_bound=order,
            witness_vertices=incumbent,
            termination_reason="WALL_TIME",
            detail="the wall-clock budget expired after the initial feasible witness",
        )

    optimizer = z3.Optimize()
    optimizer.set(timeout=max(1, remaining_ms))
    selected = {
        vertex: z3.Bool(f"selected_{index}") for index, vertex in enumerate(vertices)
    }
    for left, right in request.graph.edges:
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
            return IndependenceNumberResult(
                status="EXACT",
                order=order,
                optimum_value=len(incumbent),
                incumbent_value=len(incumbent),
                lower_bound=len(incumbent),
                upper_bound=len(incumbent),
                witness_vertices=incumbent,
                termination_reason="OPTIMUM_ESTABLISHED",
                detail="bounded Z3 optimization seeded by a NetworkX feasible witness",
            )
    elif status == z3.unsat:
        return IndependenceNumberResult(
            status="UNKNOWN",
            order=order,
            optimum_value=None,
            incumbent_value=len(incumbent),
            lower_bound=len(incumbent),
            upper_bound=order,
            witness_vertices=incumbent,
            termination_reason="SOLVER_UNSAT",
            detail="bounded Z3 optimization returned unsat, which is unexpected "
            "for an independence-number problem that always has a feasible witness",
        )
    else:
        upper_bound = order

    termination: Literal["WALL_TIME", "SOLVER_UNKNOWN"] = (
        "WALL_TIME"
        if time.monotonic() - started >= request.resource_budget.wall_seconds
        else "SOLVER_UNKNOWN"
    )
    return IndependenceNumberResult(
        status="UNKNOWN",
        order=order,
        optimum_value=None,
        incumbent_value=len(incumbent),
        lower_bound=len(incumbent),
        upper_bound=upper_bound,
        witness_vertices=incumbent,
        termination_reason=termination,
        detail="bounded Z3 optimization did not establish an exact optimum",
    )
