"""Domain adapter for tree automaton operations."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.automata.tree._models import (
    AcceptedTreeCountRequest,
    AcceptedTreeCountResult,
    TreeAutomatonReachabilityRequest,
    TreeRunRequest,
    TreeRunResult,
)
from jacobian.math.logic.automata.tree.operations import (
    ReachableStateProfile,
    _tree_state_chart_unchecked,
    accepted_tree_count,
    reachable_state_profile,
)
from jacobian.math.logic.automata.tree.values import (
    accepted_tree_count_work_bound,
    validate_ranked_tree,
)

__all__ = [
    "compute_accepted_tree_count",
    "compute_tree_automaton_reachability",
    "compute_tree_run",
]


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
        state_chart=chart,
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
        count=format_canonical_integer(
            accepted_tree_count(request.automaton, request.tree_size)
        ),
        estimated_work_bound=estimated_work_bound,
    )


def compute_tree_automaton_reachability(
    request: TreeAutomatonReachabilityRequest,
) -> ReachableStateProfile:
    """Compute the exact reachable-state profile with minimum tree witnesses."""

    return reachable_state_profile(request.automaton)
