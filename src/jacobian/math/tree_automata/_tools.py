"""Tree automaton operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.tree_automata._models import (
    AcceptedTreeCountRequest,
    AcceptedTreeCountResult,
    TreeAutomatonReachabilityRequest,
    TreeRunRequest,
    TreeRunResult,
)
from jacobian.math.tree_automata._operations import (
    compute_accepted_tree_count,
    compute_tree_automaton_reachability,
    compute_tree_run,
)
from jacobian.math.tree_automata.values import ReachableStateProfile


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
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


# Automaton: states {0, 1}, symbols {a (arity 0), f (arity 2)}
# Transitions: a -> 0, f(0, 0) -> 0, f(1, 0) -> 1, f(0, 1) -> 1, f(1, 1) -> 1
# Final states: {0}
_RUN_EXAMPLE = {
    "automaton": {
        "state_count": 2,
        "arity": [0, 2],
        "transitions": [
            {"symbol": 0, "child_states": [], "target_state": 0},
            {"symbol": 1, "child_states": [0, 0], "target_state": 0},
            {"symbol": 1, "child_states": [0, 1], "target_state": 1},
            {"symbol": 1, "child_states": [1, 0], "target_state": 1},
            {"symbol": 1, "child_states": [1, 1], "target_state": 1},
        ],
        "final_states": [0],
    },
    "tree": {
        "symbol": 1,
        "children": [
            {"symbol": 0, "children": []},
            {"symbol": 0, "children": []},
        ],
    },
}

TREE_AUTOMATA_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "tree_automaton.states.reachable.compute",
        "Compute bottom-up tree-automaton reachable states",
        "Return the complete least-fixed-point set of states reachable by a "
        "finite ground ranked tree, its complement, and one canonical "
        "minimum-node witness tree per reachable state. A transition is "
        "enabled only when all of its ordered child states are reachable. "
        "When several derivations tie at the minimum node count, each state's "
        "witness is the unique one whose root transition (symbol, "
        "child_states, target_state) is lexicographically smallest, comparing "
        "child_states element-wise as integers, with each child witness "
        "chosen by the same rule recursively.",
        TreeAutomatonReachabilityRequest,
        ReachableStateProfile,
        compute_tree_automaton_reachability,
        "tree-automata",
        "reachability",
        "fixed-point",
        "exact",
        examples=(
            example(
                "leaf_seed_and_binary_extension",
                "Find states generated from a leaf and a binary "
                "constructor; every transition's symbol must index the "
                "ranked alphabet and its child_states count must equal "
                "arity[symbol].",
                {
                    "automaton": _RUN_EXAMPLE["automaton"],
                },
            ),
        ),
    ),
    _op(
        "tree_automaton.run.compute",
        "Run a bottom-up tree automaton on a ranked tree",
        "Execute a nondeterministic bottom-up tree automaton on a ranked "
        "tree and return the set of reachable root states and whether the "
        "tree is accepted.",
        TreeRunRequest,
        TreeRunResult,
        compute_tree_run,
        "tree-automata",
        "run",
        "exact",
        examples=(
            example(
                "simple_run",
                "Run a tree automaton on f(a, a).",
                _RUN_EXAMPLE,
            ),
        ),
    ),
    _op(
        "tree_automaton.accepted_tree_count.compute",
        "Count accepted trees of a given size",
        "Count the number of ranked trees of a given size accepted by a "
        "bottom-up nondeterministic tree automaton. On-the-fly subset-state "
        "dynamic programming counts each distinct tree once, even when it has "
        "multiple accepting runs; the validated request carries a conservative "
        "work bound.",
        AcceptedTreeCountRequest,
        AcceptedTreeCountResult,
        compute_accepted_tree_count,
        "tree-automata",
        "counting",
        "exact",
        examples=(
            example(
                "count_size_1",
                "Count accepted trees of size 1.",
                {
                    "automaton": _RUN_EXAMPLE["automaton"],
                    "tree_size": 1,
                },
            ),
        ),
    ),
)

TOOLS = TREE_AUTOMATA_OPERATIONS

__all__ = ["TOOLS"]
