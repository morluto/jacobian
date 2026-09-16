"""Tree automaton operation declarations."""

from typing import Any

from jacobian.catalog.models import (
    MathTool,
    OperationExample,
)
from jacobian.math.logic.automata.tree._models import (
    AcceptedTreeCountRequest,
    AcceptedTreeCountResult,
    TreeAutomatonReachabilityRequest,
    TreeAutomatonTrimRequest,
    TreeAutomatonTrimResult,
    TreeDeterminizeRequest,
    TreeDeterminizeResult,
    TreeRunRequest,
    TreeRunResult,
)
from jacobian.math.logic.automata.tree.operations import (
    _accepted_tree_count_admitted,
    _tree_state_chart_unchecked,
    determinize_tree_automaton,
    reachable_state_profile,
    trim_tree_automaton,
)
from jacobian.math.logic.automata.tree.values import (
    ReachableStateProfile,
    TreeStateChartEntry,
    accepted_tree_count_work_bound,
    validate_ranked_tree,
)


def compute_tree_run(request: TreeRunRequest) -> TreeRunResult:
    node_count = validate_ranked_tree(request.automaton, request.tree)
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
    estimated_work_bound = accepted_tree_count_work_bound(
        request.automaton, request.tree_size
    )
    return AcceptedTreeCountResult._from_kernel(
        request,
        count=_accepted_tree_count_admitted(request.automaton, request.tree_size),
        estimated_work_bound=estimated_work_bound,
    )


def compute_tree_automaton_reachability(
    request: TreeAutomatonReachabilityRequest,
) -> ReachableStateProfile:
    """Compute the exact reachable-state profile with minimum tree witnesses."""

    return reachable_state_profile(request.automaton)


def compute_tree_automaton_trim(
    request: TreeAutomatonTrimRequest,
) -> TreeAutomatonTrimResult:
    """Restrict an automaton to its reachable and productive states."""

    return trim_tree_automaton(request.automaton)


def compute_tree_automaton_determinize(
    request: TreeDeterminizeRequest,
) -> TreeDeterminizeResult:
    """Determinize an automaton by bounded subset construction."""

    return determinize_tree_automaton(
        request.automaton,
        request.max_subset_states,
        request.sample_max_height,
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

# Nondeterministic automaton for boolean AND-trees: states {0 (false),
# 1 (true), 2 (true alias)}. Symbols: 0=false (arity 0), 1=true (arity 0),
# 2=AND (arity 2). Subset construction reaches {{0}, {1, 2}}.
_DETERMINIZE_EXAMPLE = {
    "state_count": 3,
    "arity": [0, 0, 2],
    "transitions": [
        {"symbol": 0, "child_states": [], "target_state": 0},
        {"symbol": 1, "child_states": [], "target_state": 1},
        {"symbol": 1, "child_states": [], "target_state": 2},
        {"symbol": 2, "child_states": [0, 0], "target_state": 0},
        {"symbol": 2, "child_states": [0, 1], "target_state": 0},
        {"symbol": 2, "child_states": [0, 2], "target_state": 0},
        {"symbol": 2, "child_states": [1, 0], "target_state": 0},
        {"symbol": 2, "child_states": [2, 0], "target_state": 0},
        {"symbol": 2, "child_states": [1, 1], "target_state": 1},
        {"symbol": 2, "child_states": [1, 2], "target_state": 1},
        {"symbol": 2, "child_states": [2, 1], "target_state": 1},
        {"symbol": 2, "child_states": [2, 2], "target_state": 1},
        {"symbol": 2, "child_states": [2, 2], "target_state": 2},
    ],
    "final_states": [1],
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
        operation_id="tree_automaton.trim.compute",
        title="Trim a bottom-up tree automaton",
        description="Restrict a bottom-up nondeterministic tree automaton to the "
        "states that are both reachable by a ground tree and productive inside "
        "an accepting run, returning the trimmed automaton, old/new state "
        "maps, and one replayed minimum-node witness tree per kept state on "
        "the trimmed automaton. Restriction preserves the accepted language "
        "exactly; the empty language trims to the canonical one-state "
        "automaton with no transitions and no final states.",
        request_type=TreeAutomatonTrimRequest,
        result_type=TreeAutomatonTrimResult,
        run=compute_tree_automaton_trim,
        tags=("tree-automata", "trim", "reachability", "exact"),
        discovery_terms=("tree automaton trim", "useful states", "trimming"),
        examples=(
            OperationExample(
                name="drop_unreachable_state",
                description="Drop the unreachable state 1 from a two-state automaton.",
                input={
                    "automaton": _RUN_EXAMPLE["automaton"],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="tree_automaton.determinize.compute",
        title="Determinize a bottom-up tree automaton",
        description="Run bounded subset construction on a nondeterministic "
        "bottom-up tree automaton. On success return the complete "
        "deterministic machine with its subset map, a replayed "
        "transition-closure certificate, and acceptance agreement on every "
        "ground tree up to a bounded height. When the powerset exceeds the "
        "state-set budget, return the partial construction with TRUNCATED "
        "status and no language-equivalence claim.",
        request_type=TreeDeterminizeRequest,
        result_type=TreeDeterminizeResult,
        run=compute_tree_automaton_determinize,
        tags=("tree-automata", "determinization", "subset-construction", "exact"),
        discovery_terms=("determinize", "subset construction", "powerset"),
        examples=(
            OperationExample(
                name="determinize_and_trees",
                description="Determinize a nondeterministic automaton for "
                "boolean AND-trees with an aliased true state.",
                input={
                    "automaton": _DETERMINIZE_EXAMPLE,
                    "max_subset_states": 64,
                    "sample_max_height": 3,
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
