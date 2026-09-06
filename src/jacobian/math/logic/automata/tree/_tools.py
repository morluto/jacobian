"""Tree automaton operation declarations."""

from typing import Any

from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.logic.automata.tree._models import (
    AcceptedTreeCountRequest,
    AcceptedTreeCountResult,
    TreeAutomatonReachabilityRequest,
    TreeRunRequest,
    TreeRunResult,
)
from jacobian.math.logic.automata.tree.operations import (
    _tree_state_chart_unchecked,
    accepted_tree_count,
    reachable_state_profile,
)
from jacobian.math.logic.automata.tree.values import (
    ReachableStateProfile,
    TreeStateChartEntry,
    accepted_tree_count_work_bound,
    validate_ranked_tree,
)


def compute_tree_run(request: TreeRunRequest) -> TreeRunResult:
    try:
        node_count = validate_ranked_tree(request.automaton, request.tree)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("tree",),
            code="tree_automata.ranked_tree_domain",
            message=str(exc),
        ) from exc
    chart = _tree_state_chart_unchecked(request.automaton, request.tree)
    states = set(chart[-1][1])
    accepting = set(states) & set(request.automaton.final_states)
    return TreeRunResult._from_kernel(
        request,
        accepted=bool(accepting),
        root_states=tuple(sorted(states)),
        state_chart=tuple(
            TreeStateChartEntry(position=position, states=states)
            for position, states in chart
        ),
        node_count=node_count,
    )


def compute_accepted_tree_count(
    request: AcceptedTreeCountRequest,
) -> AcceptedTreeCountResult:
    try:
        estimated_work_bound = accepted_tree_count_work_bound(
            request.automaton, request.tree_size
        )
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("tree_size",),
            code="tree_automata.accepted_tree_count_work_bound",
            message=str(exc),
        ) from exc
    return AcceptedTreeCountResult._from_kernel(
        request,
        count=accepted_tree_count(request.automaton, request.tree_size),
        estimated_work_bound=estimated_work_bound,
    )


def compute_tree_automaton_reachability(
    request: TreeAutomatonReachabilityRequest,
) -> ReachableStateProfile:
    """Compute the exact reachable-state profile with minimum tree witnesses."""

    return reachable_state_profile(request.automaton)


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

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="tree_automaton.states.reachable.compute",
        title="Compute bottom-up tree-automaton reachable states",
        description="Return the complete least-fixed-point set of states reachable by a "
        "finite ground ranked tree, its complement, and one canonical "
        "minimum-node witness tree per reachable state. A transition is "
        "enabled only when all of its ordered child states are reachable. "
        "When several derivations tie at the minimum node count, each state's "
        "witness is the unique one whose root transition (symbol, "
        "child_states, target_state) is lexicographically smallest, comparing "
        "child_states element-wise as integers, with each child witness "
        "chosen by the same rule recursively.",
        request_type=TreeAutomatonReachabilityRequest,
        result_type=ReachableStateProfile,
        run=compute_tree_automaton_reachability,
        tags=("tree-automata", "reachability", "fixed-point", "exact"),
        examples=(
            OperationExample(
                name="leaf_seed_and_binary_extension",
                description="Find states generated from a leaf and a binary "
                "constructor; every transition's symbol must index the "
                "ranked alphabet and its child_states count must equal "
                "arity[symbol].",
                input={
                    "automaton": _RUN_EXAMPLE["automaton"],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tree_automaton.run.compute",
        title="Run a bottom-up tree automaton on a ranked tree",
        description="Execute a nondeterministic bottom-up tree automaton on a ranked "
        "tree and return the set of reachable root states and whether the "
        "tree is accepted.",
        request_type=TreeRunRequest,
        result_type=TreeRunResult,
        run=compute_tree_run,
        tags=("tree-automata", "run", "exact"),
        examples=(
            OperationExample(
                name="simple_run",
                description="Run a tree automaton on f(a, a).",
                input=_RUN_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="tree_automaton.accepted_tree_count.compute",
        title="Count accepted trees of a given size",
        description="Count the number of ranked trees of a given size accepted by a "
        "bottom-up nondeterministic tree automaton. On-the-fly subset-state "
        "dynamic programming counts each distinct tree once, even when it has "
        "multiple accepting runs; the validated request carries a conservative "
        "work bound.",
        request_type=AcceptedTreeCountRequest,
        result_type=AcceptedTreeCountResult,
        run=compute_accepted_tree_count,
        tags=("tree-automata", "counting", "exact"),
        examples=(
            OperationExample(
                name="count_size_1",
                description="Count accepted trees of size 1.",
                input={
                    "automaton": _RUN_EXAMPLE["automaton"],
                    "tree_size": 1,
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
