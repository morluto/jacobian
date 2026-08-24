"""Domain-owned bottom-up tree automaton kernels."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from math import prod

from jacobian.math.tree_automata.values import (
    MAX_REACHABILITY_WITNESS_NODES,
    MAX_TREE_AUTOMATON_REACHABILITY_WORK,
    BottomUpTreeAutomaton,
    RankedTree,
    ReachableStateProfile,
    TreeAutomatonTransition,
    accepted_tree_count_work_bound,
    validate_ranked_tree,
)

__all__ = [
    "ReachableStateProfile",
    "accepted_tree_count",
    "reachable_state_profile",
    "run_tree_automaton",
    "tree_state_chart",
]


@dataclass(frozen=True)
class _WitnessChoice:
    node_count: int
    transition: TreeAutomatonTransition


def reachable_state_profile(
    automaton: BottomUpTreeAutomaton,
) -> ReachableStateProfile:
    """Return each reachable state and its canonical minimum-node witness tree.

    This is least-fixed-point reachability over transition hyperedges: a row
    becomes usable only after every child state has a witness.  It is not graph
    reachability over state vertices.

    The canonical witness of one state is the unique ground tree chosen by
    taking, among every transition row targeting the state whose ordered child
    states all have witnesses, the candidates' fewest node count
    (``1 + sum(child witness node counts)``) and breaking ties by the
    lexicographically smallest ``(symbol, child_states, target_state)``
    transition, comparing ``child_states`` element-wise as integers; each
    child's witness satisfies the same rule recursively.
    """

    if _reachability_execution_work_bound(automaton) > (
        MAX_TREE_AUTOMATON_REACHABILITY_WORK
    ):
        raise ValueError("tree automaton reachability work bound exceeded")

    transitions = tuple(sorted(automaton.transitions, key=_transition_key))
    choices, _scans = _saturate_choices(transitions, automaton.state_count)

    reachable_choices = tuple(
        (state, choice) for state, choice in enumerate(choices) if choice is not None
    )
    reachable_states = tuple(state for state, _ in reachable_choices)
    total_witness_nodes = sum(choice.node_count for _, choice in reachable_choices)
    if total_witness_nodes > MAX_REACHABILITY_WITNESS_NODES:
        raise ValueError("reachable-state witness output exceeds the node bound")

    return ReachableStateProfile.model_construct(
        automaton=automaton,
        reachable_states=reachable_states,
        unreachable_states=tuple(
            state for state, choice in enumerate(choices) if choice is None
        ),
        witnesses=tuple(
            (state, _materialize_witness(state, choices)) for state in reachable_states
        ),
    )


def _saturate_choices(
    transitions: tuple[TreeAutomatonTransition, ...],
    state_count: int,
) -> tuple[list[_WitnessChoice | None], int]:
    """Run the least-fixed-point saturation and return (choices, scan count).

    This is the single shared kernel loop: ``reachable_state_profile`` uses the
    returned choices to build witnesses, and work admission uses the returned
    scan count as the exact measured convergence depth of one profile, so the
    priced quantity and the executed quantity cannot drift apart.
    """

    choices: list[_WitnessChoice | None] = [None] * state_count
    scans = 0
    for _ in range(state_count + 1):
        scans += 1
        next_choices = choices.copy()
        for transition in transitions:
            child_choices = tuple(choices[state] for state in transition.child_states)
            if any(choice is None for choice in child_choices):
                continue
            node_count = 1 + sum(
                choice.node_count for choice in child_choices if choice is not None
            )
            candidate = _WitnessChoice(node_count, transition)
            current = next_choices[transition.target_state]
            if current is None or _witness_key(candidate) < _witness_key(current):
                next_choices[transition.target_state] = candidate
        if next_choices == choices:
            return choices, scans
        choices = next_choices
    raise RuntimeError(  # pragma: no cover - finite state bound proves convergence.
        "tree automaton reachability did not reach a fixed point"
    )


def _reachability_price_components(
    automaton: BottomUpTreeAutomaton,
) -> tuple[int, int, int]:
    """Return (sort work, per-scan work, measured saturation scan rounds)."""

    transition_count = len(automaton.transitions)
    maximum_arity = max(
        (len(row.child_states) for row in automaton.transitions), default=0
    )
    sort_rounds = max(1, (transition_count - 1).bit_length())
    sort_work = transition_count * sort_rounds * (4 + maximum_arity)
    per_scan_work = 2 * automaton.state_count + sum(
        6 + 4 * len(row.child_states) for row in automaton.transitions
    )
    sorted_transitions = tuple(sorted(automaton.transitions, key=_transition_key))
    _, scan_rounds = _saturate_choices(sorted_transitions, automaton.state_count)
    return sort_work, per_scan_work, scan_rounds


def _reachability_execution_work_bound(automaton: BottomUpTreeAutomaton) -> int:
    """Conservatively bound one native reachability profile's execution work.

    Each call first evaluates this very bound, which prices one saturation
    probe: the shared kernel loop run once over sorted rows to measure the
    exact number of scan rounds the fixed point needs.  The probe is the same
    code path the profile executes, so the measured round count equals the
    profile's actual convergence depth by construction rather than by estimate;
    it replaces the former worst-case charge of two rounds per constructible
    state, which rejected automata whose nullary seeds saturate immediately.
    The probe itself always terminates inside the input schema alone: a scan
    round repeats only when some choice was defined or improved, a
    minimum-node witness never repeats a state along a root-to-leaf path
    (substituting the higher occurrence of a repeated state for the lower one
    would strictly shrink the tree while preserving the derived state), so the
    witness heights -- and hence the rounds -- stay within ``state_count + 1``
    no matter which automaton the request model admitted.  One native call
    then executes the profile once more (sorting plus the measured scans) and
    materializes then recounts at most the admitted witness nodes, so the
    bound charges two probes' worth of sorting and scanning plus witness
    work.  Every row scan visits its child-state tuple independently to
    construct the lookup tuple, test that all children are known, add their
    node counts, and compare an equal-size candidate's canonical transition
    key.

    Native callers perform exactly one profile per call.
    """

    sort_work, per_scan_work, scan_rounds = _reachability_price_components(automaton)
    witness_work = 3 * MAX_REACHABILITY_WITNESS_NODES
    return 2 * sort_work + 2 * scan_rounds * per_scan_work + witness_work


def _reachability_public_path_work_bound(automaton: BottomUpTreeAutomaton) -> int:
    """Price the public path's admission evaluation plus its three profiles.

    The MCP request boundary performs one profile for request admission, one
    for execution, and one for source-bound result replay.  Each of those
    three invocations sorts the transitions twice -- once inside its own
    execution-bound evaluation, whose saturation probe measures the
    convergence depth before any witness is materialized, and once when the
    kernel sorts rows for its actual profile -- so together with the sort
    performed while evaluating this very public bound the path performs seven
    sorts in total, not one per profile.  Each invocation also runs two
    probes' worth of scanning (its bound-evaluation probe plus its actual
    saturation), so with the public bound's own probe the path again performs
    seven probes' worth of scanning across three witness materializations,
    all charged against the shared
    ``MAX_TREE_AUTOMATON_REACHABILITY_WORK`` envelope instead of escaping the
    budget.  Native calls perform one profile and are priced by
    ``_reachability_execution_work_bound`` alone.
    """

    sort_work, per_scan_work, scan_rounds = _reachability_price_components(automaton)
    witness_work = 3 * MAX_REACHABILITY_WITNESS_NODES
    return 7 * sort_work + 7 * scan_rounds * per_scan_work + 3 * witness_work


def _transition_key(
    transition: TreeAutomatonTransition,
) -> tuple[int, tuple[int, ...], int]:
    return (transition.symbol, transition.child_states, transition.target_state)


def _witness_key(
    choice: _WitnessChoice,
) -> tuple[int, int, tuple[int, ...], int]:
    return (choice.node_count, *_transition_key(choice.transition))


def _materialize_witness(
    state: int,
    choices: list[_WitnessChoice | None],
) -> RankedTree:
    choice = choices[state]
    if choice is None:  # pragma: no cover - callers pass reachable states only.
        raise ValueError("cannot materialize an unreachable state")
    return RankedTree(
        symbol=choice.transition.symbol,
        children=tuple(
            _materialize_witness(child_state, choices)
            for child_state in choice.transition.child_states
        ),
    )


def run_tree_automaton(
    automaton: BottomUpTreeAutomaton,
    tree: RankedTree,
) -> set[int]:
    """Run a bottom-up tree automaton on a ranked tree.

    Returns the set of states reachable at the root.
    """
    return set(tree_state_chart(automaton, tree)[-1][1])


def tree_state_chart(
    automaton: BottomUpTreeAutomaton,
    tree: RankedTree,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Return the canonical postorder position/state chart for a ranked tree."""
    validate_ranked_tree(automaton, tree)
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


def _reachable_root_states(
    automaton: BottomUpTreeAutomaton,
    tree: RankedTree,
) -> set[int]:
    child_states: list[set[int]] = []
    if tree.children:
        child_states = [
            _reachable_root_states(automaton, child) for child in tree.children
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
