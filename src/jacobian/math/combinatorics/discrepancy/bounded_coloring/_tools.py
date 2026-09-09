"""Finite per-set discrepancy feasibility declaration."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.discrepancy.bounded_coloring._models import (
    BoundedColoringRequest,
    BoundedColoringResult,
)
from jacobian.math.combinatorics.discrepancy.bounded_coloring.operations import decide


def _decide(request: BoundedColoringRequest) -> BoundedColoringResult:
    return decide(request.set_system, request.absolute_bounds, request.resource_budget)


TOOLS: MathTools = (
    MathTool(
        operation_id="discrepancy.theory.bounded_coloring.decide",
        title="Decide signed colorings with per-set imbalance bounds",
        description=(
            "For a finite indexed set system and one nonnegative integer bound "
            "per set, decide whether a complete +/-1 coloring keeps each absolute "
            "signed sum within its own bound. SATISFIABLE returns the complete "
            "coloring and exact signed-sum ledger; UNSATISFIABLE is an exact finite "
            "decision. Operational timeout, resource exhaustion, and backend failure "
            "are errors rather than mathematical outcomes. Repeated and empty sets retain their source "
            "positions. Admits the full 64-element, 1000-set carrier, at most 64000 "
            "incidences and 128000 PB literal occurrences, with a request-visible "
            "solver work cap and wall budget measured from request start. Caller deadlines "
            "and cancellation remain operational errors. This decides supplied "
            "bounds without optimizing a global discrepancy or an asymptotic parameter."
        ),
        request_type=BoundedColoringRequest,
        result_type=BoundedColoringResult,
        run=_decide,
        tags=("discrepancy", "set-system", "coloring", "feasibility", "exact"),
        examples=(
            OperationExample(
                name="different_bounds",
                description=(
                    "Decide whether a +/-1 coloring of {0,1} keeps |sum| of the "
                    "pair at most 0 and the singleton at most 1; each bound must "
                    "not exceed its set's size."
                ),
                input={
                    "set_system": {"ground_set_size": 2, "sets": [[0, 1], [0]]},
                    "absolute_bounds": [0, 1],
                },
            ),
            OperationExample(
                name="impossible_singleton",
                description=(
                    "Decide whether a singleton can have signed sum 0; the bound "
                    "must be at most the set size, so 0 is a valid but unsatisfiable "
                    "request."
                ),
                input={
                    "set_system": {"ground_set_size": 1, "sets": [[0]]},
                    "absolute_bounds": [0],
                },
            ),
        ),
    ),
)
