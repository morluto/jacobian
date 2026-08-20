"""Domain adapter for tree automaton operations."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.math.tree_automata._models import (
    AcceptedTreeCountRequest,
    AcceptedTreeCountResult,
    TreeRunRequest,
    TreeRunResult,
)
from jacobian.math.tree_automata.operations import (
    accepted_tree_count,
    run_tree_automaton,
    tree_state_chart,
)
from jacobian.math.tree_automata.values import (
    accepted_tree_count_work_bound,
    validate_ranked_tree,
)

__all__ = ["compute_accepted_tree_count", "compute_tree_run"]


def compute_tree_run(request: TreeRunRequest) -> TreeRunResult:
    states = run_tree_automaton(request.automaton, request.tree)
    accepting = set(states) & set(request.automaton.final_states)
    return TreeRunResult(
        **request.model_dump(),
        accepted=bool(accepting),
        root_states=tuple(sorted(states)),
        state_chart=tree_state_chart(request.automaton, request.tree),
        node_count=validate_ranked_tree(request.automaton, request.tree),
    )


def compute_accepted_tree_count(
    request: AcceptedTreeCountRequest,
) -> AcceptedTreeCountResult:
    return AcceptedTreeCountResult(
        **request.model_dump(),
        count=format_canonical_integer(
            accepted_tree_count(request.automaton, request.tree_size)
        ),
        estimated_work_bound=accepted_tree_count_work_bound(
            request.automaton, request.tree_size
        ),
    )
