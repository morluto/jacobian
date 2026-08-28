"""Provider-independent values for exact bottom-up tree automata."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import comb
from typing import Annotated, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_TA_STATES = 64
MAX_TA_SYMBOLS = 32
MAX_TA_TRANSITIONS = 4096
MAX_TA_ARITY = 16
MAX_RUN_TREE_NODES = 4096
MAX_RUN_TREE_DEPTH = 128
MAX_TREE_AUTOMATON_WORK = 2_000_000
MAX_REACHABILITY_WITNESS_NODES = 4096
MAX_TREE_AUTOMATON_REACHABILITY_WORK = 30_000_000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"tree_automata.{reason}", message)


Arity = Annotated[int, Field(ge=0, le=MAX_TA_ARITY)]


class TreeAutomatonTransition(StrictModel):
    """A bottom-up tree automaton transition.

    A transition ``f(q_1, ..., q_n) -> q`` says: if the children of a
    ``f``-labelled node are in states ``q_1, ..., q_n``, the node is in
    state ``q``.
    ``symbol`` is the function symbol (label of the node).
    """

    symbol: int = Field(ge=0, le=MAX_TA_SYMBOLS - 1)
    child_states: tuple[int, ...] = Field(max_length=MAX_TA_ARITY)
    target_state: int = Field(ge=0, le=MAX_TA_STATES - 1)


class RankedTree(StrictModel):
    """A ranked tree: a node labelled by a symbol with zero or more children."""

    symbol: int = Field(ge=0, le=MAX_TA_SYMBOLS - 1)
    children: tuple[RankedTree, ...] = Field(default=(), max_length=MAX_TA_ARITY)


RankedTree.model_rebuild()


class BottomUpTreeAutomaton(StrictModel):
    """A nondeterministic bottom-up tree automaton (NFTA).

    The automaton has ``state_count`` states, a ranked alphabet where
    ``arity[symbol]`` gives the arity of each symbol, a set of transitions,
    and a set of final (accepting) states.
    """

    state_count: int = Field(ge=1, le=MAX_TA_STATES)
    arity: tuple[Arity, ...] = Field(
        max_length=MAX_TA_SYMBOLS,
        description=(
            "arity of each ranked symbol; an empty tuple is the canonical "
            "empty ranked alphabet, whose ground-tree language is empty"
        ),
    )
    transitions: tuple[TreeAutomatonTransition, ...] = Field(
        min_length=0, max_length=MAX_TA_TRANSITIONS
    )
    final_states: tuple[int, ...] = Field(min_length=0, max_length=MAX_TA_STATES)

    @model_validator(mode="after")
    def require_valid_automaton(self) -> Self:
        self._require_unique_sets()
        self._require_valid_transitions()
        self._require_valid_final_states()
        return self

    def _require_unique_sets(self) -> None:
        if len(set(self.transitions)) != len(self.transitions):
            raise _validation_error(
                "transitions_not_unique", "transitions must be unique"
            )
        if len(set(self.final_states)) != len(self.final_states):
            raise _validation_error(
                "final_states_not_unique", "final states must be unique"
            )

    def _require_valid_transitions(self) -> None:
        for tr in self.transitions:
            if not 0 <= tr.target_state < self.state_count:
                raise ValueError("transition target out of range")
            if tr.symbol >= len(self.arity):
                raise ValueError("transition symbol out of range")
            if len(tr.child_states) != self.arity[tr.symbol]:
                raise ValueError("transition child count must match symbol arity")
            for s in tr.child_states:
                if not 0 <= s < self.state_count:
                    raise ValueError("transition child state out of range")

    def _require_valid_final_states(self) -> None:
        for f in self.final_states:
            if not 0 <= f < self.state_count:
                raise ValueError("final state out of range")


def validate_ranked_tree(
    automaton: BottomUpTreeAutomaton,
    tree: RankedTree,
) -> int:
    """Validate every node against the ranked alphabet and return node count."""

    node_count = 0
    stack = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        node_count += 1
        if node_count > MAX_RUN_TREE_NODES:
            raise ValueError("tree node count exceeds bound")
        if depth > MAX_RUN_TREE_DEPTH:
            raise ValueError("tree depth exceeds bound")
        if node.symbol >= len(automaton.arity):
            raise ValueError("tree symbol out of ranked alphabet")
        if len(node.children) != automaton.arity[node.symbol]:
            raise ValueError("every tree node must match its symbol arity")
        stack.extend((child, depth + 1) for child in node.children)

    arity_factor = max(1, max(automaton.arity))
    estimated_work = node_count * max(1, len(automaton.transitions)) * arity_factor
    if estimated_work > MAX_TREE_AUTOMATON_WORK:
        raise ValueError("tree run work bound exceeded")
    return node_count


def ranked_tree_node_count(tree: RankedTree) -> int:
    """Return the bounded number of nodes in a ranked tree independent of an alphabet."""

    node_count = 0
    stack = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        node_count += 1
        if node_count > MAX_RUN_TREE_NODES:
            raise ValueError("tree node count exceeds bound")
        if depth > MAX_RUN_TREE_DEPTH:
            raise ValueError("tree depth exceeds bound")
        stack.extend((child, depth + 1) for child in node.children)
    return node_count


class ReachableStateProfile(StrictModel):
    """Exact least-fixed-point reachability profile for one NFTA.

    The domain-owned canonical value returned by ``reachable_state_profile``
    and published unchanged as the reachability operation result: every
    state is listed exactly once as reachable or unreachable, and each
    reachable state carries one canonical minimum-node ground-tree witness.
    The witness is unique by construction: among every transition row
    targeting the state whose ordered child states all carry witnesses,
    candidates are ranked by fewest node count
    (``1 + sum(child witness node counts)``), then by the lexicographically
    smallest ``(symbol, child_states, target_state)`` transition with
    ``child_states`` compared element-wise as integers, and each child's
    witness is chosen by the same rule recursively.  Result validation checks
    only this value's structural invariants.
    """

    automaton: BottomUpTreeAutomaton
    reachable_states: tuple[int, ...] = Field(max_length=MAX_TA_STATES)
    unreachable_states: tuple[int, ...] = Field(max_length=MAX_TA_STATES)
    witnesses: tuple[tuple[int, RankedTree], ...] = Field(
        max_length=MAX_TA_STATES,
        description=(
            f"one canonical minimum-node (state, tree) witness per reachable "
            f"state; when several derivations tie at the minimum node count, "
            f"the witness is the unique one whose root transition "
            f"(symbol, child_states, target_state) is lexicographically "
            f"smallest, comparing child_states element-wise as integers, with "
            f"each child's witness chosen by the same rule recursively; their "
            f"node counts are bounded in aggregate by "
            f"MAX_REACHABILITY_WITNESS_NODES ({MAX_REACHABILITY_WITNESS_NODES} nodes summed over all "
            f"reachable states)"
        ),
    )

    @model_validator(mode="after")
    def require_canonical_profile_shape(self) -> Self:
        state_count = self.automaton.state_count
        for label, states in (
            ("reachable", self.reachable_states),
            ("unreachable", self.unreachable_states),
        ):
            if states != tuple(sorted(set(states))):
                raise _validation_error(
                    "states_not_canonical", f"{label} states must be unique and sorted"
                )
            if any(not 0 <= state < state_count for state in states):
                raise _validation_error(
                    "state_out_of_range", f"{label} state out of range"
                )
        if set(self.reachable_states) & set(self.unreachable_states):
            raise _validation_error(
                "states_not_disjoint",
                "reachable and unreachable states must be disjoint",
            )
        if len(self.reachable_states) + len(self.unreachable_states) != state_count:
            raise _validation_error(
                "states_do_not_partition",
                "reachable and unreachable states must partition the automaton states",
            )
        if tuple(state for state, _ in self.witnesses) != self.reachable_states:
            raise _validation_error(
                "witnesses_not_aligned",
                "witnesses must carry exactly one entry per reachable state in order",
            )
        total_nodes = 0
        for _, tree in self.witnesses:
            total_nodes = _ranked_witness_nodes(tree, self.automaton, total_nodes)
        if total_nodes > MAX_REACHABILITY_WITNESS_NODES:
            raise _validation_error(
                "witness_output_bound",
                "reachable-state witness output exceeds the node bound",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        automaton: BottomUpTreeAutomaton,
        *,
        reachable_states: tuple[int, ...],
        unreachable_states: tuple[int, ...],
        witnesses: tuple[tuple[int, RankedTree], ...],
    ) -> Self:
        """Construct the canonical profile emitted by the trusted kernel."""

        return cls(
            automaton=automaton,
            reachable_states=reachable_states,
            unreachable_states=unreachable_states,
            witnesses=witnesses,
        )


def _ranked_witness_nodes(
    tree: RankedTree,
    automaton: BottomUpTreeAutomaton,
    running_total: int,
) -> int:
    """Return the running node total after counting one alphabet-conformant tree."""

    node_count = 0
    stack = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        node_count += 1
        if running_total + node_count > MAX_REACHABILITY_WITNESS_NODES:
            raise _validation_error(
                "witness_output_bound",
                "reachable-state witness output exceeds the node bound",
            )
        if depth > MAX_RUN_TREE_DEPTH:
            raise _validation_error(
                "witness_depth_bound", "witness depth exceeds the ranked-tree bound"
            )
        if node.symbol >= len(automaton.arity):
            raise _validation_error(
                "witness_symbol_out_of_range",
                "witness symbol out of the ranked alphabet",
            )
        if len(node.children) != automaton.arity[node.symbol]:
            raise _validation_error(
                "witness_arity_mismatch",
                "every witness node must match its symbol arity",
            )
        stack.extend((child, depth + 1) for child in node.children)
    return running_total + node_count


@dataclass(frozen=True)
class _WitnessChoice:
    node_count: int
    transition: TreeAutomatonTransition


def _build_reachable_state_profile(
    automaton: BottomUpTreeAutomaton,
) -> ReachableStateProfile:
    """Build the canonical least-fixed-point profile in one priced pass."""

    _, sort_work, per_scan_work, scan_rounds, choices = _priced_saturation(automaton)
    if sort_work + scan_rounds * per_scan_work + 3 * MAX_REACHABILITY_WITNESS_NODES > (
        MAX_TREE_AUTOMATON_REACHABILITY_WORK
    ):
        raise ValueError("tree automaton reachability work bound exceeded")
    reachable_choices = tuple(
        (state, choice) for state, choice in enumerate(choices) if choice is not None
    )
    reachable_states = tuple(state for state, _ in reachable_choices)
    if sum(choice.node_count for _, choice in reachable_choices) > (
        MAX_REACHABILITY_WITNESS_NODES
    ):
        raise ValueError("reachable-state witness output exceeds the node bound")
    return ReachableStateProfile._from_kernel(
        automaton,
        reachable_states=reachable_states,
        unreachable_states=tuple(
            state for state, choice in enumerate(choices) if choice is None
        ),
        witnesses=tuple(
            (state, _materialize_witness(state, choices)) for state in reachable_states
        ),
    )


def _saturate_choices(
    transitions: tuple[TreeAutomatonTransition, ...], state_count: int
) -> tuple[list[_WitnessChoice | None], int]:
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
    raise RuntimeError("tree automaton reachability did not reach a fixed point")


def _priced_saturation(
    automaton: BottomUpTreeAutomaton,
) -> tuple[
    tuple[TreeAutomatonTransition, ...], int, int, int, list[_WitnessChoice | None]
]:
    """Run exactly one sorted least-fixed-point pass and price that same pass."""

    transition_count = len(automaton.transitions)
    maximum_arity = max(
        (len(row.child_states) for row in automaton.transitions), default=0
    )
    sort_work = (
        transition_count
        * max(1, (transition_count - 1).bit_length())
        * (4 + maximum_arity)
    )
    per_scan_work = 2 * automaton.state_count + sum(
        6 + 4 * len(row.child_states) for row in automaton.transitions
    )
    sorted_transitions = tuple(sorted(automaton.transitions, key=_transition_key))
    choices, scan_rounds = _saturate_choices(sorted_transitions, automaton.state_count)
    return sorted_transitions, sort_work, per_scan_work, scan_rounds, choices


def reachability_price_components(
    automaton: BottomUpTreeAutomaton,
) -> tuple[int, int, int]:
    """Return the exact sort, scan, and measured-round prices for one pass."""

    _, sort_work, per_scan_work, scan_rounds, _ = _priced_saturation(automaton)
    return sort_work, per_scan_work, scan_rounds


def reachability_execution_work_bound(automaton: BottomUpTreeAutomaton) -> int:
    """Price one native profile: one pass and its bounded witness output."""

    sort_work, per_scan_work, scan_rounds = reachability_price_components(automaton)
    return sort_work + scan_rounds * per_scan_work + 3 * MAX_REACHABILITY_WITNESS_NODES


def _transition_key(
    transition: TreeAutomatonTransition,
) -> tuple[int, tuple[int, ...], int]:
    return (transition.symbol, transition.child_states, transition.target_state)


def _witness_key(
    choice: _WitnessChoice,
) -> tuple[int, int, tuple[int, ...], int]:
    return (choice.node_count, *_transition_key(choice.transition))


def _materialize_witness(
    state: int, choices: list[_WitnessChoice | None]
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


def accepted_tree_count_work_bound(
    automaton: BottomUpTreeAutomaton,
    tree_size: int,
) -> int:
    """Return a conservative bound for subset-DP transition checks."""

    transition_counts = Counter(
        transition.symbol for transition in automaton.transitions
    )
    subset_count = (1 << automaton.state_count) - 1
    work = 0
    for symbol, arity in enumerate(automaton.arity):
        transition_count = max(1, transition_counts[symbol])
        if arity == 0:
            work += transition_count
        elif tree_size > arity:
            compositions = comb(tree_size - 1, arity)
            work += compositions * subset_count**arity * transition_count
        if work > MAX_TREE_AUTOMATON_WORK:
            raise ValueError("accepted-tree count work bound exceeded")
    return work


__all__ = [
    "MAX_REACHABILITY_WITNESS_NODES",
    "MAX_RUN_TREE_DEPTH",
    "MAX_RUN_TREE_NODES",
    "MAX_TA_ARITY",
    "MAX_TA_STATES",
    "MAX_TA_SYMBOLS",
    "MAX_TA_TRANSITIONS",
    "MAX_TREE_AUTOMATON_REACHABILITY_WORK",
    "MAX_TREE_AUTOMATON_WORK",
    "BottomUpTreeAutomaton",
    "RankedTree",
    "ReachableStateProfile",
    "TreeAutomatonTransition",
    "accepted_tree_count_work_bound",
    "ranked_tree_node_count",
    "validate_ranked_tree",
]
