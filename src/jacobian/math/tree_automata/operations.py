"""Domain-owned bottom-up tree automaton kernels."""

from __future__ import annotations

from jacobian.math.tree_automata.values import (
    BottomUpTreeAutomaton,
    RankedTree,
)

__all__ = [
    "accepted_tree_count",
    "run_tree_automaton",
]


def run_tree_automaton(
    automaton: BottomUpTreeAutomaton, tree: RankedTree,
) -> set[int]:
    """Run a bottom-up tree automaton on a ranked tree.

    Returns the set of states reachable at the root.
    """
    child_states: list[set[int]] = []
    if tree.children:
        child_states = [
            run_tree_automaton(automaton, child) for child in tree.children
        ]
    matching: set[int] = set()
    for tr in automaton.transitions:
        if tr.symbol != tree.symbol:
            continue
        if len(tr.child_states) != len(child_states):
            continue
        match = True
        for i, states in enumerate(child_states):
            if tr.child_states[i] not in states:
                match = False
                break
        if match:
            matching.add(tr.target_state)
    return matching


def accepted_tree_count(  # noqa: C901
    automaton: BottomUpTreeAutomaton, tree_size: int,
) -> int:
    """Count accepted trees of a given size via dynamic programming.

    For each state, ``dp[s][k]`` = number of trees of size ``k`` evaluating
    to state ``s``. We iterate size from 1 to tree_size, and for each size
    and each transition, compute the count from child state contributions.
    """
    if tree_size < 1:
        return 0
    dp: list[dict[int, int]] = [{} for _ in range(automaton.state_count)]
    for size in range(1, tree_size + 1):
        for tr in automaton.transitions:
            if not tr.child_states:
                if size == 1:
                    dp[tr.target_state][1] = (
                        dp[tr.target_state].get(1, 0) + 1
                    )
            else:
                child_size = size - 1
                for child_sizes in _all_child_size_splits(
                    tr.child_states, child_size, dp,
                ):
                    total = sum(child_sizes)
                    if total != child_size:
                        continue
                    count = 1
                    valid = True
                    for i, cs in enumerate(child_sizes):
                        child_state = tr.child_states[i]
                        if cs not in dp[child_state]:
                            valid = False
                            break
                        count *= dp[child_state][cs]
                    if valid and count > 0:
                        dp[tr.target_state][size] = (
                            dp[tr.target_state].get(size, 0) + count
                        )
    total_count = 0
    for f in automaton.final_states:
        total_count += dp[f].get(tree_size, 0)
    return total_count


def _all_child_size_splits(
    child_states: tuple[int, ...], total: int, dp: list[dict[int, int]],
) -> list[tuple[int, ...]]:
    """Yield all ways to split ``total`` among children, using available dp sizes."""
    if not child_states:
        return [()] if total == 0 else []
    state = child_states[0]
    available = [s for s in dp[state] if s <= total]
    if len(child_states) == 1:
        return [(s,) for s in available if s == total]
    result: list[tuple[int, ...]] = []
    for first in available:
        rest = _all_child_size_splits(child_states[1:], total - first, dp)
        for r in rest:
            result.append((first, *r))
    return result