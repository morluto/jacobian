"""Petri net operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.petri_nets._models import (
    FireTransitionRequest,
    FireTransitionResult,
    ReachabilityRequest,
    ReachabilityResult,
)
from jacobian.math.petri_nets._operations import (
    compute_fire_transition,
    compute_reachability,
)


def _op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


# Simple net: 2 places, 2 transitions
# t0: p0 -> p1 (pre=[[1,0],[0,0]], post=[[0,0],[0,1]])
# t1: p1 -> p0 (pre=[[0,0],[0,1]], post=[[1,0],[0,0]])
_NET = {
    "net": {
        "place_count": 2,
        "transition_count": 2,
        "pre": [[1, 0], [0, 0]],
        "post": [[0, 0], [0, 1]],
    },
}

PETRI_NET_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "petri_net.fire_transition.compute",
        "Fire one transition in a Petri net",
        "Fire a single transition at the given marking and return whether it "
        "succeeded and the resulting marking. If the transition is not "
        "enabled, it does not fire.",
        FireTransitionRequest,
        FireTransitionResult,
        compute_fire_transition,
        "petri-net",
        "firing",
        "exact",
        examples=(
            example(
                "fire_t0",
                "Fire transition 0 from marking [2, 0].",
                {
                    "net": _NET["net"],
                    "marking": {"tokens": [2, 0]},
                    "transition": 0,
                },
            ),
        ),
    ),
    _op(
        "petri_net.reachability_graph.compute",
        "Compute the bounded reachability graph of a Petri net",
        "Return an exact BFS reachability graph when exploration closes within "
        "max_states. If the bound is exhausted, return TRUNCATED with every "
        "enabled omitted firing in an explicit replayable frontier.",
        ReachabilityRequest,
        ReachabilityResult,
        compute_reachability,
        "petri-net",
        "reachability",
        "exact",
        examples=(
            example(
                "simple_net_reachability",
                "Reachability graph of a simple 2-place net.",
                {
                    "net": _NET["net"],
                    "initial_marking": {"tokens": [1, 0]},
                    "max_states": 100,
                },
            ),
        ),
    ),
)

TOOLS = PETRI_NET_OPERATIONS

__all__ = ["TOOLS"]
