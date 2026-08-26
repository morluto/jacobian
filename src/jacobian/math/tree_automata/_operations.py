"""Domain adapter for tree automaton operations."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.math.tree_automata._models import (
    AcceptedTreeCountRequest,
    AcceptedTreeCountResult,
    TreeAutomatonReachabilityRequest,
    TreeRunRequest,
    TreeRunResult,
)
from jacobian.math.tree_automata.operations import (
    ReachableStateProfile,
    accepted_tree_count,
    reachable_state_profile,
    run_tree_automaton,
    tree_state_chart,
)
from jacobian.math.tree_automata.values import (
    accepted_tree_count_work_bound,
    validate_ranked_tree,
)

__all__ = [
    "compute_accepted_tree_count",
    "compute_tree_automaton_reachability",
    "compute_tree_run",
    "verify_accepted_tree_count_result",
    "verify_reachable_state_profile",
    "verify_tree_run_result",
]


def compute_tree_run(request: TreeRunRequest) -> TreeRunResult:
    states = run_tree_automaton(request.automaton, request.tree)
    accepting = set(states) & set(request.automaton.final_states)
    return TreeRunResult._from_kernel(
        request,
        accepted=bool(accepting),
        root_states=tuple(sorted(states)),
        state_chart=tree_state_chart(request.automaton, request.tree),
        node_count=validate_ranked_tree(request.automaton, request.tree),
    )


def compute_accepted_tree_count(
    request: AcceptedTreeCountRequest,
) -> AcceptedTreeCountResult:
    return AcceptedTreeCountResult._from_kernel(
        request,
        count=format_canonical_integer(
            accepted_tree_count(request.automaton, request.tree_size)
        ),
        estimated_work_bound=accepted_tree_count_work_bound(
            request.automaton, request.tree_size
        ),
    )


def compute_tree_automaton_reachability(
    request: TreeAutomatonReachabilityRequest,
) -> ReachableStateProfile:
    """Compute the exact reachable-state profile with minimum tree witnesses."""

    return reachable_state_profile(request.automaton)


def verify_tree_run_result(result: TreeRunResult) -> bool:
    """Replay a separately supplied tree-run claim inside its admitted envelope."""

    try:
        expected_chart = tree_state_chart(result.automaton, result.tree)
        expected_states = expected_chart[-1][1]
        return (
            result.state_chart == expected_chart
            and result.root_states == expected_states
            and result.node_count == validate_ranked_tree(result.automaton, result.tree)
            and result.accepted
            == bool(set(expected_states) & set(result.automaton.final_states))
        )
    except ValueError:
        return False


def verify_accepted_tree_count_result(result: AcceptedTreeCountResult) -> bool:
    """Replay a separately supplied exact counting claim within request bounds."""

    try:
        return result.estimated_work_bound == accepted_tree_count_work_bound(
            result.automaton, result.tree_size
        ) and int(result.count) == accepted_tree_count(
            result.automaton, result.tree_size
        )
    except ValueError:
        return False


def verify_reachable_state_profile(result: ReachableStateProfile) -> bool:
    """Replay a separately supplied profile using the reachability work ledger."""

    try:
        return result == reachable_state_profile(result.automaton)
    except ValueError:
        return False
