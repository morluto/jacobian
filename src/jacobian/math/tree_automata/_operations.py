"""Domain adapter for tree automaton operations."""

from __future__ import annotations

from jacobian.math.tree_automata._models import (
    AcceptedTreeCountRequest,
    AcceptedTreeCountResult,
    TreeRunRequest,
    TreeRunResult,
)
from jacobian.math.tree_automata.operations import (
    accepted_tree_count,
    run_tree_automaton,
)

__all__ = ["compute_accepted_tree_count", "compute_tree_run"]


def compute_tree_run(request: TreeRunRequest) -> TreeRunResult:
    states = run_tree_automaton(request.automaton, request.tree)
    accepting = set(states) & set(request.automaton.final_states)
    return TreeRunResult(
        accepted=bool(accepting),
        root_states=tuple(sorted(states)),
    )


def compute_accepted_tree_count(
    request: AcceptedTreeCountRequest,
) -> AcceptedTreeCountResult:
    return AcceptedTreeCountResult(
        count=accepted_tree_count(request.automaton, request.tree_size)
    )
