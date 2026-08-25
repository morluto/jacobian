"""Private Z3/NetworkX backend for bounded independence-number search."""

from __future__ import annotations

import time
from typing import Literal

import z3  # type: ignore[import-untyped]

from jacobian.math.graphs.independence import (
    IndependenceNumberRequest,
    IndependenceNumberResult,
    _replay_exact_optimum,
)


def _integer_bound(value: z3.ArithRef, fallback: int) -> int:
    return value.as_long() if z3.is_int_value(value) else fallback


def solve_independence_number(
    request: IndependenceNumberRequest,
) -> IndependenceNumberResult:
    """Run one wall-clock-bounded exact maximum independent-set optimization.

    Results are built through ``model_construct`` after the producing solve
    established every field invariant under its own declared budget.  An
    ``EXACT`` conclusion additionally reproduces its optimum through the
    same bounded source-graph replay that independent validation runs,
    charged against the same request deadline, so every returned result
    validates; a replay that cannot certify the solver optimum demotes to
    the typed ``UNKNOWN`` outcome instead of an ``EXACT`` payload that
    would fail revalidation.  Every incomplete outcome, including a
    ``sat`` optimize whose objective bounds stay open, reports the graph
    order as its upper bound, matching the validating source-binding
    contract.
    """

    started = time.monotonic()
    vertices = request.graph.vertices
    order = len(vertices)
    if not vertices:
        return IndependenceNumberResult.model_construct(
            graph=request.graph,
            status="EXACT",
            order=0,
            optimum_value=0,
            incumbent_value=0,
            lower_bound=0,
            upper_bound=0,
            witness_vertices=(),
            termination_reason="SPECIAL_CASE",
            detail="the empty graph has independence number zero",
            convention="MAXIMUM_EDGE_FREE_VERTEX_SUBSET",
        )

    incumbent: tuple[str, ...] = (min(vertices),)
    remaining_ms = int(
        (request.resource_budget.wall_seconds - (time.monotonic() - started)) * 1000
    )
    if remaining_ms <= 0:
        return IndependenceNumberResult.model_construct(
            graph=request.graph,
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
            try:
                _replay_exact_optimum(
                    request.graph,
                    len(incumbent),
                    deadline=started + request.resource_budget.wall_seconds,
                )
            except ValueError:
                return IndependenceNumberResult.model_construct(
                    graph=request.graph,
                    status="UNKNOWN",
                    order=order,
                    optimum_value=None,
                    incumbent_value=len(incumbent),
                    lower_bound=len(incumbent),
                    upper_bound=order,
                    witness_vertices=incumbent,
                    termination_reason="REPLAY_INCOMPLETE",
                    detail=(
                        "bounded source-graph replay could not certify the "
                        "solver optimum, so no exact optimum is claimed"
                    ),
                )
            return IndependenceNumberResult.model_construct(
                graph=request.graph,
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
        return IndependenceNumberResult.model_construct(
            graph=request.graph,
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
    termination: Literal["WALL_TIME", "SOLVER_UNKNOWN"] = (
        "WALL_TIME"
        if time.monotonic() - started >= request.resource_budget.wall_seconds
        else "SOLVER_UNKNOWN"
    )
    return IndependenceNumberResult.model_construct(
        graph=request.graph,
        status="UNKNOWN",
        order=order,
        optimum_value=None,
        incumbent_value=len(incumbent),
        lower_bound=len(incumbent),
        upper_bound=order,
        witness_vertices=incumbent,
        termination_reason=termination,
        detail="bounded Z3 optimization did not establish an exact optimum",
    )
