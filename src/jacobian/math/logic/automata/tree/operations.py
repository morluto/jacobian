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
    TreeDeterminizeResult,
    TreeRunResult,
)
from jacobian.math.logic.automata.tree.values import (
    MAX_RUN_TREE_NODES,
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
    "determinize_tree_automaton",
    "reachable_state_profile",
    "run_tree_automaton",
    "tree_state_chart",
    "trim_tree_automaton",
    "verify_accepted_tree_count",
    "verify_determinization",
    "verify_reachable_state_profile",
    "verify_tree_run",
    "verify_trim_tree_automaton",
]


MAX_DETERMINIZE_WORK = 500_000
MAX_DETERMINIZE_SAMPLE_TREES = 4096


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


class _DeterminizeBudgetError(Exception):
    """Internal signal that subset construction exceeded its budget."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _determinize_images(
    automaton: BottomUpTreeAutomaton,
) -> dict[tuple[int, tuple[int, ...]], int]:
    """Map each source transition row to its target-state bitmask."""

    images: dict[tuple[int, tuple[int, ...]], int] = {}
    for transition in automaton.transitions:
        key = (transition.symbol, transition.child_states)
        images[key] = images.get(key, 0) | (1 << transition.target_state)
    return images


def _image_of_subsets(
    images: dict[tuple[int, tuple[int, ...]], int],
    state_count: int,
    symbol: int,
    children: tuple[tuple[int, ...], ...],
    work: list[int],
) -> tuple[int, ...]:
    """Return the sorted source-state image of one subset tuple.

    ``work`` is a single-cell budget ledger: every elementary source-state
    combination consumes one unit and raises ``_DeterminizeBudgetError`` past
    the work envelope.
    """

    mask = 0
    combos: list[tuple[int, ...]] = [()]
    for subset in children:
        work[0] += max(1, len(combos)) * max(1, len(subset))
        if work[0] > MAX_DETERMINIZE_WORK:
            raise _DeterminizeBudgetError("WORK_BUDGET")
        combos = [(*prefix, state) for prefix in combos for state in subset]
        if len(combos) > MAX_DETERMINIZE_WORK:
            raise _DeterminizeBudgetError("WORK_BUDGET")
    work[0] += max(1, len(combos))
    if work[0] > MAX_DETERMINIZE_WORK:
        raise _DeterminizeBudgetError("WORK_BUDGET")
    for combo in combos:
        mask |= images.get((symbol, combo), 0)
    return tuple(state for state in range(state_count) if mask & (1 << state))


def _combos_for_expansion(subsets: int, arity: tuple[int, ...]) -> int:
    """Count fresh subset tuples when the newest subset has an index.

    Expanding subset ``subsets - 1`` evaluates exactly the tuples whose
    maximum child index equals it; count them, stopping past the budget.
    """

    total = 0
    for rank in arity:
        if rank == 0:
            continue
        fresh = 1
        for _ in range(rank):
            fresh *= subsets
            if fresh > MAX_DETERMINIZE_WORK:
                return MAX_DETERMINIZE_WORK + 1
        old = 1
        for _ in range(rank):
            old *= subsets - 1
            if old > MAX_DETERMINIZE_WORK:
                break
        total += fresh - old
        if total > MAX_DETERMINIZE_WORK:
            return MAX_DETERMINIZE_WORK + 1
    return total


def _seed_nullary_subsets(
    automaton: BottomUpTreeAutomaton,
    images: dict[tuple[int, tuple[int, ...]], int],
    subsets: list[tuple[int, ...]],
    index_of: dict[tuple[int, ...], int],
    rows: dict[tuple[int, tuple[int, ...]], int],
    combos: list[int],
    work: list[int],
    max_subset_states: int,
) -> None:
    """Intern the image of every nullary symbol before worklist expansion."""

    for symbol, rank in enumerate(automaton.arity):
        if rank != 0:
            continue
        combos[0] += 1
        work[0] += 1
        if work[0] > MAX_DETERMINIZE_WORK:
            raise _DeterminizeBudgetError("WORK_BUDGET")
        image = _image_of_subsets(images, automaton.state_count, symbol, (), work)
        target = _intern_subset(image, subsets, index_of, max_subset_states)
        if target is not None:
            rows[(symbol, ())] = target


def _intern_subset(
    image: tuple[int, ...],
    subsets: list[tuple[int, ...]],
    index_of: dict[tuple[int, ...], int],
    max_subset_states: int,
) -> int | None:
    """Intern one nonempty image, signalling the state budget when full."""

    if not image:
        return None
    if image not in index_of:
        if len(subsets) >= max_subset_states:
            raise _DeterminizeBudgetError("STATE_BUDGET")
        index_of[image] = len(subsets)
        subsets.append(image)
    return index_of[image]


def _expand_newest_subset(
    automaton: BottomUpTreeAutomaton,
    images: dict[tuple[int, tuple[int, ...]], int],
    subsets: list[tuple[int, ...]],
    index_of: dict[tuple[int, ...], int],
    rows: dict[tuple[int, tuple[int, ...]], int],
    combos: list[int],
    work: list[int],
    newest: int,
    expanded: int,
    max_subset_states: int,
) -> None:
    """Evaluate every subset tuple whose maximum child index is ``newest``."""

    if _combos_for_expansion(expanded, automaton.arity) > (
        MAX_DETERMINIZE_WORK - work[0]
    ):
        raise _DeterminizeBudgetError("WORK_BUDGET")
    for symbol, rank in enumerate(automaton.arity):
        if rank == 0:
            continue
        for children in product(range(expanded), repeat=rank):
            if max(children) != newest:
                continue
            combos[0] += 1
            image = _image_of_subsets(
                images,
                automaton.state_count,
                symbol,
                tuple(subsets[child] for child in children),
                work,
            )
            target = _intern_subset(image, subsets, index_of, max_subset_states)
            if target is not None:
                key = (symbol, children)
                if key not in rows and len(rows) == 4096:
                    raise _DeterminizeBudgetError("WORK_BUDGET")
                rows[key] = target


def _subset_construction(
    automaton: BottomUpTreeAutomaton,
    max_subset_states: int,
) -> tuple[
    list[tuple[int, ...]], dict[tuple[int, tuple[int, ...]], int], int, str | None
]:
    """Run bounded subset construction, returning subsets and DFA rows.

    Returns ``(subsets, rows, combos_evaluated, truncation_reason)`` where
    rows map ``(symbol, child DFA indices)`` to target DFA indices over the
    discovery-ordered subset list. ``truncation_reason`` is ``None`` exactly
    when every subset tuple over the final subset list was evaluated.
    """

    images = _determinize_images(automaton)
    subsets: list[tuple[int, ...]] = []
    index_of: dict[tuple[int, ...], int] = {}
    rows: dict[tuple[int, tuple[int, ...]], int] = {}
    combos = [0]
    work = [0]
    try:
        _seed_nullary_subsets(
            automaton,
            images,
            subsets,
            index_of,
            rows,
            combos,
            work,
            max_subset_states,
        )
        expanded = 0
        while expanded < len(subsets):
            newest = expanded
            expanded += 1
            _expand_newest_subset(
                automaton,
                images,
                subsets,
                index_of,
                rows,
                combos,
                work,
                newest,
                expanded,
                max_subset_states,
            )
    except _DeterminizeBudgetError as truncated:
        return subsets, rows, combos[0], truncated.reason
    return subsets, rows, combos[0], None


def _canonical_dfa(
    automaton: BottomUpTreeAutomaton,
    subsets: list[tuple[int, ...]],
    rows: dict[tuple[int, tuple[int, ...]], int],
) -> tuple[BottomUpTreeAutomaton, tuple[tuple[int, ...], ...]]:
    """Order subsets lexicographically and build the deterministic machine."""

    order = sorted(range(len(subsets)), key=lambda index: subsets[index])
    renumber = {old: new for new, old in enumerate(order)}
    canonical = tuple(subsets[old] for old in order)
    source_finals = set(automaton.final_states)
    deterministic = BottomUpTreeAutomaton(
        state_count=max(1, len(canonical)),
        arity=automaton.arity,
        transitions=tuple(
            sorted(
                (
                    TreeAutomatonTransition(
                        symbol=symbol,
                        child_states=tuple(renumber[child] for child in children),
                        target_state=renumber[target],
                    )
                    for (symbol, children), target in rows.items()
                    if all(child < len(subsets) for child in children)
                    and target < len(subsets)
                ),
                key=lambda row: (row.symbol, row.child_states, row.target_state),
            )
        ),
        final_states=tuple(
            sorted(
                index
                for index, subset in enumerate(canonical)
                if set(subset) & source_finals
            )
        ),
    )
    return deterministic, canonical


def _replay_dfa_closure(
    automaton: BottomUpTreeAutomaton,
    deterministic: BottomUpTreeAutomaton,
    subset_map: tuple[tuple[int, ...], ...],
) -> int:
    """Replay every DFA row against the subset map; return rows checked."""

    images = _determinize_images(automaton)
    work = [0]
    checked = 0
    for transition in deterministic.transitions:
        image = _image_of_subsets(
            images,
            automaton.state_count,
            transition.symbol,
            tuple(subset_map[child] for child in transition.child_states),
            work,
        )
        if image != subset_map[transition.target_state]:
            raise RuntimeError("determinized row disagrees with subset construction")
        checked += 1
    return checked


def _trees_of_height(
    automaton: BottomUpTreeAutomaton, max_height: int
) -> tuple[RankedTree, ...]:
    """Enumerate all run-admissible ground trees of height at most ``max_height``.

    Trees whose node count exceeds the run envelope are pruned during
    generation; enumeration stops with a resource error past the sample
    budget.
    """

    levels: list[list[tuple[RankedTree, int, int]]] = []
    current: list[tuple[RankedTree, int, int]] = []
    for symbol, rank in enumerate(automaton.arity):
        if rank == 0:
            current.append((RankedTree(symbol=symbol, children=()), 0, 1))
    if len(current) > MAX_DETERMINIZE_SAMPLE_TREES:
        raise OperationResourceAdmissionError(
            location=("automaton", "sample_max_height"),
            code="tree_automata.determinize_sample_bound_exceeded",
            message="the bounded-height tree sample exceeds the admitted budget; "
            "shrink sample_max_height",
        )
    levels.append(sorted(current, key=lambda entry: entry[0].symbol))
    for _ in range(max_height):
        previous = [entry for level in levels for entry in level]
        following: list[tuple[RankedTree, int, int]] = []
        for symbol, rank in enumerate(automaton.arity):
            if rank == 0:
                continue
            for children in product(previous, repeat=rank):
                if max(child[1] for child in children) != len(levels) - 1:
                    continue
                total = 1 + sum(child[2] for child in children)
                if total > MAX_RUN_TREE_NODES:
                    continue
                following.append(
                    (
                        RankedTree(
                            symbol=symbol,
                            children=tuple(child[0] for child in children),
                        ),
                        len(levels),
                        total,
                    )
                )
                if len(previous) + len(following) > MAX_DETERMINIZE_SAMPLE_TREES:
                    raise OperationResourceAdmissionError(
                        location=("automaton", "sample_max_height"),
                        code="tree_automata.determinize_sample_bound_exceeded",
                        message="the bounded-height tree sample exceeds the admitted "
                        "budget; shrink sample_max_height",
                    )
        if not following:
            break
        following.sort(
            key=lambda entry: (
                entry[0].symbol,
                tuple(child.symbol for child in entry[0].children),
                entry[2],
            )
        )
        levels.append(following)
    return tuple(tree for level in levels for tree, _, _ in level)


def determinize_tree_automaton(
    automaton: BottomUpTreeAutomaton,
    max_subset_states: int = 64,
    sample_max_height: int = 3,
) -> TreeDeterminizeResult:
    """Determinize a bottom-up tree automaton by subset construction.

    On success return the complete deterministic machine with its subset
    map, a replayed transition-closure certificate, and acceptance
    agreement on every ground tree of height at most ``sample_max_height``.
    When the powerset exceeds ``max_subset_states`` (or the shared work
    envelope), return the partial construction with ``TRUNCATED`` status
    and no language-equivalence claim.
    """

    if type(max_subset_states) is not int or not 1 <= max_subset_states <= 64:
        raise OperationDomainValidationError(
            location=("max_subset_states",),
            code="tree_automata.determinize_subset_budget",
            message="max_subset_states must be within 1..64",
        )
    if type(sample_max_height) is not int or not 0 <= sample_max_height <= 5:
        raise OperationDomainValidationError(
            location=("sample_max_height",),
            code="tree_automata.determinize_sample_height",
            message="sample_max_height must be within 0..5",
        )
    subsets, rows, combos, truncation = _subset_construction(
        automaton, max_subset_states
    )
    if truncation is not None:
        deterministic, canonical = _canonical_dfa(automaton, subsets, rows)
        return TreeDeterminizeResult._from_kernel(
            automaton=automaton,
            max_subset_states=max_subset_states,
            sample_max_height=sample_max_height,
            status="TRUNCATED",
            truncation_reason=truncation,
            deterministic=deterministic,
            subset_map=canonical,
            equivalence_claim=False,
            closure_rows_checked=0,
            combos_evaluated=combos,
            sample_trees_checked=0,
            sample_agreement=False,
        )
    if not subsets:
        deterministic = BottomUpTreeAutomaton(
            state_count=1, arity=automaton.arity, transitions=(), final_states=()
        )
        return TreeDeterminizeResult._from_kernel(
            automaton=automaton,
            max_subset_states=max_subset_states,
            sample_max_height=sample_max_height,
            status="COMPLETE",
            truncation_reason="NONE",
            deterministic=deterministic,
            subset_map=((),),
            equivalence_claim=True,
            closure_rows_checked=0,
            combos_evaluated=combos,
            sample_trees_checked=0,
            sample_agreement=True,
        )
    deterministic, canonical = _canonical_dfa(automaton, subsets, rows)
    closure_rows = _replay_dfa_closure(automaton, deterministic, canonical)
    sample = _trees_of_height(automaton, sample_max_height)
    source_finals = set(automaton.final_states)
    for tree in sample:
        source_roots = run_tree_automaton(automaton, tree)
        deterministic_roots = run_tree_automaton(deterministic, tree)
        source_accepted = bool(source_roots & source_finals)
        deterministic_accepted = bool(
            deterministic_roots & set(deterministic.final_states)
        )
        if source_accepted != deterministic_accepted:
            raise RuntimeError("determinized automaton disagrees with its source")
    return TreeDeterminizeResult._from_kernel(
        automaton=automaton,
        max_subset_states=max_subset_states,
        sample_max_height=sample_max_height,
        status="COMPLETE",
        truncation_reason="NONE",
        deterministic=deterministic,
        subset_map=canonical,
        equivalence_claim=True,
        closure_rows_checked=closure_rows,
        combos_evaluated=combos,
        sample_trees_checked=len(sample),
        sample_agreement=True,
    )


def verify_determinization(claim: TreeDeterminizeResult) -> bool:
    """Verify a determinization against its retained source automaton."""

    try:
        return (
            determinize_tree_automaton(
                claim.automaton, claim.max_subset_states, claim.sample_max_height
            )
            == claim
        )
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False
