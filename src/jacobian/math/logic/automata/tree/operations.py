"""Domain-owned bottom-up tree automaton kernels."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from itertools import product
from math import prod

from jacobian.math.logic.automata.tree._models import (
    AcceptedTreeCountResult,
    TreeRunResult,
)
from jacobian.math.logic.automata.tree.values import (
    BottomUpTreeAutomaton,
    RankedTree,
    ReachableStateProfile,
    TreeAutomatonTransition,
    TreeStateChartEntry,
    _build_reachable_state_profile,
    accepted_tree_count_work_bound,
    validate_ranked_tree,
)

__all__ = [
    "ReachableStateProfile",
    "accepted_tree_count",
    "reachable_state_profile",
    "run_tree_automaton",
    "tree_state_chart",
    "verify_accepted_tree_count",
    "verify_reachable_state_profile",
    "verify_tree_run",
]


def reachable_state_profile(
    automaton: BottomUpTreeAutomaton,
) -> ReachableStateProfile:
    """Return each reachable state and its canonical minimum-node witness tree."""

    return _build_reachable_state_profile(automaton)


def run_tree_automaton(
    automaton: BottomUpTreeAutomaton,
    tree: RankedTree,
) -> set[int]:
    """Run a bottom-up tree automaton and return the reachable root states."""

    return set(tree_state_chart(automaton, tree)[-1][1])


def tree_state_chart(
    automaton: BottomUpTreeAutomaton,
    tree: RankedTree,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Return the canonical postorder position/state chart for a ranked tree."""

    validate_ranked_tree(automaton, tree)
    return _tree_state_chart_unchecked(automaton, tree)


def _tree_state_chart_unchecked(
    automaton: BottomUpTreeAutomaton,
    tree: RankedTree,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Build a chart after the owner operation has validated the tree."""

    chart: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def visit(node: RankedTree, position: tuple[int, ...]) -> set[int]:
        child_states = tuple(
            visit(child, (*position, index))
            for index, child in enumerate(node.children)
        )
        states = {
            transition.target_state
            for transition in automaton.transitions
            if transition.symbol == node.symbol
            and len(transition.child_states) == len(child_states)
            and all(
                transition.child_states[index] in states
                for index, states in enumerate(child_states)
            )
        }
        chart.append((position, tuple(sorted(states))))
        return states

    visit(tree, ())
    return tuple(chart)


def accepted_tree_count(
    automaton: BottomUpTreeAutomaton,
    tree_size: int,
) -> int:
    """Count distinct accepted ranked trees, not accepting runs."""

    if tree_size < 1:
        return 0
    accepted_tree_count_work_bound(automaton, tree_size)
    transitions_by_symbol = {
        symbol: tuple(
            transition
            for transition in automaton.transitions
            if transition.symbol == symbol
        )
        for symbol in range(len(automaton.arity))
    }
    counts_by_size: list[dict[int, int]] = [{} for _ in range(tree_size + 1)]
    for size in range(1, tree_size + 1):
        size_counts: defaultdict[int, int] = defaultdict(int)
        for symbol, arity in enumerate(automaton.arity):
            _accumulate_symbol_trees(
                arity=arity,
                size=size,
                transitions=transitions_by_symbol[symbol],
                counts_by_size=counts_by_size,
                size_counts=size_counts,
            )
        counts_by_size[size] = dict(size_counts)

    final_mask = sum(1 << state for state in automaton.final_states)
    return sum(
        count
        for state_subset, count in counts_by_size[tree_size].items()
        if state_subset & final_mask
    )


def _accumulate_symbol_trees(
    *,
    arity: int,
    size: int,
    transitions: tuple[TreeAutomatonTransition, ...],
    counts_by_size: list[dict[int, int]],
    size_counts: defaultdict[int, int],
) -> None:
    if not transitions:
        return
    if arity == 0:
        if size == 1:
            root_subset = _target_subset(transitions, ())
            if root_subset:
                size_counts[root_subset] += 1
        return
    for child_sizes in _positive_compositions(size - 1, arity):
        if any(not counts_by_size[value] for value in child_sizes):
            continue
        child_choices = [counts_by_size[value].items() for value in child_sizes]
        for child_items in product(*child_choices):
            child_subsets = tuple(item[0] for item in child_items)
            root_subset = _target_subset(transitions, child_subsets)
            if root_subset:
                size_counts[root_subset] += prod(item[1] for item in child_items)


def _target_subset(
    transitions: tuple[TreeAutomatonTransition, ...],
    child_subsets: tuple[int, ...],
) -> int:
    target_subset = 0
    for transition in transitions:
        if all(
            child_subsets[index] & (1 << child_state)
            for index, child_state in enumerate(transition.child_states)
        ):
            target_subset |= 1 << transition.target_state
    return target_subset


def _positive_compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    if parts == 1:
        if total >= 1:
            yield (total,)
        return
    for first in range(1, total - parts + 2):
        for remainder in _positive_compositions(total - first, parts - 1):
            yield (first, *remainder)


def verify_tree_run(claim: TreeRunResult) -> bool:
    """Verify a serialized run chart and root claim against its sources."""

    try:
        chart = tree_state_chart(claim.automaton, claim.tree)
        roots = chart[-1][1]
        accepted = bool(set(roots) & set(claim.automaton.final_states))
        typed_chart = tuple(
            TreeStateChartEntry(position=position, states=states)
            for position, states in chart
        )
        return (
            claim.state_chart == typed_chart
            and claim.root_states == roots
            and claim.accepted == accepted
            and claim.node_count == len(chart)
        )
    except (TypeError, ValueError):
        return False


def verify_reachable_state_profile(claim: ReachableStateProfile) -> bool:
    """Verify reachable states and each claimed witness against the automaton."""

    try:
        return reachable_state_profile(claim.automaton) == claim
    except (TypeError, ValueError):
        return False


def verify_accepted_tree_count(claim: AcceptedTreeCountResult) -> bool:
    """Verify an accepted-tree count against its bounded source automaton."""

    try:
        return (
            accepted_tree_count(claim.automaton, claim.tree_size) == claim.count
        )
    except (TypeError, ValueError):
        return False
