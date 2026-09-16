"""Domain-owned bottom-up tree automaton kernels."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from itertools import product
from math import prod

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.tree._models import (
    AcceptedTreeCountResult,
    TreeAutomatonTrimResult,
    TreeRunResult,
)
from jacobian.math.logic.automata.tree.values import (
    MAX_TREE_AUTOMATON_WORK,
    BottomUpTreeAutomaton,
    RankedTree,
    ReachableStateProfile,
    TreeAutomatonTransition,
    TreeStateChartEntry,
    _build_reachable_state_profile,
    _reject_tree,
    accepted_tree_count_work_bound,
    validate_ranked_tree,
)

__all__ = [
    "ReachableStateProfile",
    "accepted_tree_count",
    "reachable_state_profile",
    "run_tree_automaton",
    "tree_state_chart",
    "trim_tree_automaton",
    "verify_accepted_tree_count",
    "verify_reachable_state_profile",
    "verify_tree_run",
    "verify_trim_tree_automaton",
]


def reachable_state_profile(
    automaton: BottomUpTreeAutomaton,
) -> ReachableStateProfile:
    """Return each reachable state and its canonical minimum-node witness tree."""

    return _build_reachable_state_profile(automaton)


def _productive_states(automaton: BottomUpTreeAutomaton) -> set[int]:
    """Return every state occurring inside some accepting run.

    Backward least fixed point seeded with the final states: a transition
    whose target is productive makes all of its child states productive. The
    pass prices the same saturation it runs and shares the tree-automaton
    work envelope with the reachability admission.
    """

    maximum_arity = max(
        (len(row.child_states) for row in automaton.transitions), default=0
    )
    rounds = automaton.state_count + 1
    if rounds * len(automaton.transitions) * (maximum_arity + 1) > (
        MAX_TREE_AUTOMATON_WORK
    ):
        _reject_tree("tree automaton productivity work bound exceeded")
    useful = set(automaton.final_states)
    for _ in range(rounds):
        grown = set(useful)
        for transition in automaton.transitions:
            if transition.target_state in useful:
                grown.update(transition.child_states)
        if grown == useful:
            return useful
        useful = grown
    raise RuntimeError("tree automaton productivity did not reach a fixed point")


def trim_tree_automaton(
    automaton: BottomUpTreeAutomaton,
) -> TreeAutomatonTrimResult:
    """Restrict an automaton to its reachable and productive states.

    Every accepting run of the source uses reachable productive states only,
    so restriction preserves the accepted language exactly. The empty
    language trims to the canonical one-state automaton with no transitions
    and no final states. Witnesses are replayed on the trimmed automaton, so
    each uses kept states only.
    """

    profile = _build_reachable_state_profile(automaton)
    kept = tuple(sorted(set(profile.reachable_states) & _productive_states(automaton)))
    dropped = tuple(
        state for state in range(automaton.state_count) if state not in set(kept)
    )
    old_to_new = tuple(
        kept.index(state) if state in set(kept) else -1
        for state in range(automaton.state_count)
    )
    if not kept:
        trimmed = BottomUpTreeAutomaton(
            state_count=1,
            arity=automaton.arity,
            transitions=(),
            final_states=(),
        )
        return TreeAutomatonTrimResult._from_kernel(
            automaton=automaton,
            trimmed=trimmed,
            kept_states=(),
            dropped_states=dropped,
            old_to_new=old_to_new,
            new_to_old=(),
            empty_language=True,
            witnesses=(),
        )
    kept_set = set(kept)
    remap = {old: new for new, old in enumerate(kept)}
    trimmed = BottomUpTreeAutomaton(
        state_count=len(kept),
        arity=automaton.arity,
        transitions=tuple(
            sorted(
                (
                    TreeAutomatonTransition(
                        symbol=transition.symbol,
                        child_states=tuple(
                            remap[state] for state in transition.child_states
                        ),
                        target_state=remap[transition.target_state],
                    )
                    for transition in automaton.transitions
                    if transition.target_state in kept_set
                    and all(state in kept_set for state in transition.child_states)
                ),
                key=lambda row: (row.symbol, row.child_states, row.target_state),
            )
        ),
        final_states=tuple(
            sorted(
                remap[state] for state in automaton.final_states if state in kept_set
            )
        ),
    )
    replay = _build_reachable_state_profile(trimmed)
    if set(replay.reachable_states) != set(range(len(kept))):
        raise OperationDomainValidationError(
            location=("automaton",),
            code="tree_automata.trim_replay_mismatch",
            message="a kept state is unreachable in the trimmed automaton",
        )
    if not trimmed.final_states:
        raise OperationDomainValidationError(
            location=("automaton",),
            code="tree_automata.trim_replay_mismatch",
            message="a nonempty trim must retain a final state",
        )
    return TreeAutomatonTrimResult._from_kernel(
        automaton=automaton,
        trimmed=trimmed,
        kept_states=kept,
        dropped_states=dropped,
        old_to_new=old_to_new,
        new_to_old=kept,
        empty_language=not trimmed.final_states,
        witnesses=replay.witnesses,
    )


def verify_trim_tree_automaton(claim: TreeAutomatonTrimResult) -> bool:
    """Verify a trim against its retained source automaton."""

    try:
        return trim_tree_automaton(claim.automaton) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


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

    if type(tree_size) is not int:
        _reject_tree("tree size must be an integer", resource=False)
    if tree_size < 1:
        return 0
    accepted_tree_count_work_bound(automaton, tree_size)
    return _accepted_tree_count_admitted(automaton, tree_size)


def _accepted_tree_count_admitted(
    automaton: BottomUpTreeAutomaton, tree_size: int
) -> int:
    if not any(transition.child_states for transition in automaton.transitions):
        # Only leaves can have a run. Count symbols, not nondeterministic runs.
        if tree_size != 1:
            return 0
        finals = set(automaton.final_states)
        return len(
            {
                transition.symbol
                for transition in automaton.transitions
                if transition.target_state in finals
            }
        )
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
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_reachable_state_profile(claim: ReachableStateProfile) -> bool:
    """Verify reachable states and each claimed witness against the automaton."""

    try:
        return reachable_state_profile(claim.automaton) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_accepted_tree_count(claim: AcceptedTreeCountResult) -> bool:
    """Verify an accepted-tree count against its bounded source automaton."""

    try:
        return accepted_tree_count(claim.automaton, claim.tree_size) == claim.count
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False
