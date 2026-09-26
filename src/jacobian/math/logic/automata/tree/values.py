"""Provider-independent values for exact bottom-up tree automata."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from itertools import pairwise
from math import comb
from typing import Annotated, NoReturn, Self

from pydantic import Field, StrictInt, field_validator, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS
from jacobian._execution import request_checkpoint
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)

MAX_TA_STATES = 64
MAX_TA_SYMBOLS = 32
MAX_TA_TRANSITIONS = 4096
MAX_TA_ARITY = 16
MAX_RUN_TREE_NODES = 4096
MAX_RUN_TREE_DEPTH = 128
MAX_TREE_AUTOMATON_WORK = 2_000_000
MAX_TREE_COUNT_HEIGHT = 100
MAX_TREE_COUNT_OUTPUT_BYTES = 4 * 1024 * 1024
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


class RegularTreeProduction(StrictModel):
    """One unit-free regular tree grammar production ``A -> f(B1,...,Bk)``."""

    nonterminal: StrictInt = Field(ge=0, le=MAX_TA_STATES - 1)
    symbol: StrictInt = Field(ge=0, le=MAX_TA_SYMBOLS - 1)
    children: tuple[StrictInt, ...] = Field(max_length=MAX_TA_ARITY)


class RegularTreeGrammar(StrictModel):
    """A finite ranked regular tree grammar over explicitly indexed symbols.

    Nonterminals are ``0..nonterminal_count-1`` and the start nonterminal is
    explicit. Productions have the unit-free ranked form
    ``A -> f(B1,...,Bk)``; their child count must equal the arity of ``f``.
    """

    nonterminal_count: StrictInt = Field(ge=1, le=MAX_TA_STATES)
    arity: tuple[Annotated[StrictInt, Field(ge=0, le=MAX_TA_ARITY)], ...] = Field(
        max_length=MAX_TA_SYMBOLS
    )
    start_nonterminal: StrictInt = Field(ge=0, le=MAX_TA_STATES - 1)
    productions: tuple[RegularTreeProduction, ...] = Field(
        max_length=MAX_TA_TRANSITIONS
    )

    @model_validator(mode="after")
    def require_canonical_productions(self) -> Self:
        if self.start_nonterminal >= self.nonterminal_count:
            raise _validation_error(
                "grammar_start_out_of_range",
                "start nonterminal must be in the declared nonterminal set",
            )
        for production in self.productions:
            if production.nonterminal >= self.nonterminal_count:
                raise _validation_error(
                    "grammar_nonterminal_out_of_range",
                    "production left side must be a declared nonterminal",
                )
            if production.symbol >= len(self.arity):
                raise _validation_error(
                    "grammar_symbol_out_of_range",
                    "production symbol must be in the declared ranked signature",
                )
            if len(production.children) != self.arity[production.symbol]:
                raise _validation_error(
                    "grammar_rank_mismatch",
                    "production child count must match its ranked symbol",
                )
            if any(child >= self.nonterminal_count for child in production.children):
                raise _validation_error(
                    "grammar_child_out_of_range",
                    "production children must be declared nonterminals",
                )
        productions = tuple(
            sorted(
                self.productions,
                key=lambda rule: (rule.nonterminal, rule.symbol, rule.children),
            )
        )
        if any(left == right for left, right in pairwise(productions)):
            raise _validation_error(
                "grammar_duplicate_production",
                "duplicate productions do not add a new derivation rule",
            )
        object.__setattr__(self, "productions", productions)
        return self


class RankedTree(StrictModel):
    """A ranked tree: a node labelled by a symbol with zero or more children."""

    symbol: int = Field(ge=0, le=MAX_TA_SYMBOLS - 1)
    children: tuple[RankedTree, ...] = Field(default=(), max_length=MAX_TA_ARITY)


RankedTree.model_rebuild()


class TreeStateChartEntry(StrictModel):
    """One postorder chart row with an explicit node position and states."""

    position: tuple[int, ...]
    states: tuple[int, ...]

    @model_validator(mode="after")
    def require_canonical_states(self) -> Self:
        if any(state < 0 or state >= MAX_TA_STATES for state in self.states):
            raise _validation_error(
                "chart_state_out_of_range", "chart state out of range"
            )
        if self.states != tuple(sorted(set(self.states))):
            raise _validation_error(
                "chart_states_not_canonical", "chart states must be sorted and unique"
            )
        if any(position < 0 for position in self.position):
            raise _validation_error(
                "chart_position_negative", "chart positions must be nonnegative"
            )
        return self


class TreeStateWitness(StrictModel):
    """A state-indexed canonical ground-tree witness."""

    state: int = Field(ge=0, lt=MAX_TA_STATES)
    tree: RankedTree


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


class DeterministicBottomUpTreeAutomaton(BottomUpTreeAutomaton):
    """A partial or complete deterministic bottom-up tree automaton."""

    @model_validator(mode="after")
    def require_deterministic_transitions(self) -> Self:
        keys = tuple((row.symbol, row.child_states) for row in self.transitions)
        if len(keys) != len(set(keys)):
            raise _validation_error(
                "transitions_not_deterministic",
                "deterministic automata have at most one target for each symbol and child-state tuple",
            )
        return self


class CompleteDeterministicBottomUpTreeAutomaton(DeterministicBottomUpTreeAutomaton):
    """A deterministic bottom-up tree automaton with a total transition table."""

    @model_validator(mode="after")
    def require_complete_transition_table(self) -> Self:
        required = sum(self.state_count**rank for rank in self.arity)
        if len(self.transitions) != required:
            raise _validation_error(
                "transition_table_incomplete",
                "complete automata need exactly one transition for every symbol and child-state tuple",
            )
        return self


def _reject_tree(message: str, *, resource: bool = True) -> NoReturn:
    error_type = (
        OperationResourceAdmissionError if resource else OperationDomainValidationError
    )
    raise error_type(
        location=("automaton", "tree"), code="tree_automata.admission", message=message
    )


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
            _reject_tree("tree node count exceeds bound")
        if depth > MAX_RUN_TREE_DEPTH:
            _reject_tree("tree depth exceeds bound")
        if node.symbol >= len(automaton.arity):
            _reject_tree("tree symbol out of ranked alphabet", resource=False)
        if len(node.children) != automaton.arity[node.symbol]:
            _reject_tree("every tree node must match its symbol arity", resource=False)
        stack.extend((child, depth + 1) for child in node.children)

    arity_factor = max(1, max(automaton.arity))
    estimated_work = node_count * max(1, len(automaton.transitions)) * arity_factor
    if estimated_work > MAX_TREE_AUTOMATON_WORK:
        _reject_tree("tree run work bound exceeded")
    return node_count


def ranked_tree_node_count(tree: RankedTree) -> int:
    """Return the bounded number of nodes in a ranked tree independent of an alphabet."""

    node_count = 0
    stack = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        node_count += 1
        if node_count > MAX_RUN_TREE_NODES:
            _reject_tree("tree node count exceeds bound")
        if depth > MAX_RUN_TREE_DEPTH:
            _reject_tree("tree depth exceeds bound")
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
    witnesses: tuple[TreeStateWitness, ...] = Field(
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

    @field_validator("witnesses", mode="before")
    @classmethod
    def decode_legacy_witness_pairs(cls, value: object) -> object:
        """Accept tuple pairs at the structural JSON boundary."""
        if not isinstance(value, (list, tuple)):
            return value
        converted: list[object] = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                converted.append({"state": item[0], "tree": item[1]})
            else:
                converted.append(item)
        return converted

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
        if tuple(witness.state for witness in self.witnesses) != self.reachable_states:
            raise _validation_error(
                "witnesses_not_aligned",
                "witnesses must carry exactly one entry per reachable state in order",
            )
        total_nodes = 0
        for witness in self.witnesses:
            total_nodes = _ranked_witness_nodes(
                witness.tree, self.automaton, total_nodes
            )
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
        witnesses: tuple[TreeStateWitness, ...],
    ) -> Self:
        """Construct the canonical profile emitted by the trusted kernel."""

        return cls.model_construct(
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
        _reject_tree("tree automaton reachability work bound exceeded")
    reachable_choices = tuple(
        (state, choice) for state, choice in enumerate(choices) if choice is not None
    )
    reachable_states = tuple(state for state, _ in reachable_choices)
    if sum(choice.node_count for _, choice in reachable_choices) > (
        MAX_REACHABILITY_WITNESS_NODES
    ):
        _reject_tree("reachable-state witness output exceeds the node bound")
    return ReachableStateProfile._from_kernel(
        automaton,
        reachable_states=reachable_states,
        unreachable_states=tuple(
            state for state, choice in enumerate(choices) if choice is None
        ),
        witnesses=tuple(
            TreeStateWitness(state=state, tree=_materialize_witness(state, choices))
            for state in reachable_states
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

    if type(tree_size) is not int:
        _reject_tree("tree size must be an integer", resource=False)
    if not any(transition.child_states for transition in automaton.transitions):
        return len(automaton.transitions)
    if not 0 <= tree_size <= 100:
        _reject_tree("tree size exceeds the supported 0..100 bound")
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
            _reject_tree("accepted-tree count work bound exceeded")
    return work


def _ground_reachable_states(automaton: BottomUpTreeAutomaton) -> frozenset[int]:
    """Saturate ground reachability once, in linear transition-incidence work."""
    waiting: list[list[int]] = [[] for _ in range(automaton.state_count)]
    missing: list[int] = []
    reached: set[int] = set()
    queue: deque[int] = deque()
    for index, transition in enumerate(automaton.transitions):
        missing.append(len(transition.child_states))
        if not transition.child_states and transition.target_state not in reached:
            reached.add(transition.target_state)
            queue.append(transition.target_state)
        for child in transition.child_states:
            waiting[child].append(index)
        if index % 256 == 0:
            request_checkpoint("during tree-automaton reachability indexing")
    visited = 0
    while queue:
        request_checkpoint("during tree-automaton reachability saturation")
        for index in waiting[queue.popleft()]:
            missing[index] -= 1
            if missing[index] == 0:
                target = automaton.transitions[index].target_state
                if target not in reached:
                    reached.add(target)
                    queue.append(target)
            visited += 1
            if visited % 8192 == 0:
                request_checkpoint("during tree-automaton reachability saturation")
    return frozenset(reached)


@dataclass(frozen=True)
class _RunCountAdmission:
    work: int
    reachable: frozenset[int]
    zero: bool


def _admit_nondeterministic_run_counts(
    automaton: BottomUpTreeAutomaton, max_size: int
) -> _RunCountAdmission:
    """Admit the grouped polynomial DP for exact accepting-run counts.

    Each run assigns one state to every node of one ranked tree. Counting
    these assignments is a sum/product recurrence over transitions, distinct
    from the subset recurrence that counts each accepted tree once.
    """

    if type(max_size) is not int or not 1 <= max_size <= 100:
        _reject_tree("run-count maximum size must be in 1..100", resource=False)

    # No finite ground tree exists without a nullary symbol. Likewise, an
    # empty final-state set makes every accepting-run count zero. Avoid charging
    # polynomial work for these constant-answer profiles.
    if not automaton.final_states or not any(
        not transition.child_states for transition in automaton.transitions
    ):
        return _RunCountAdmission(0, frozenset(), True)

    # Both indexing and saturation visit at most one incident edge per child
    # occurrence, plus one queue step per state. Price them before running.
    reachability_work = (
        4 * automaton.state_count
        + 4 * len(automaton.transitions)
        + 3 * sum(len(row.child_states) for row in automaton.transitions)
    )
    if reachability_work > MAX_TREE_AUTOMATON_WORK:
        _reject_tree("nondeterministic run-count reachability work bound exceeded")
    reachable = _ground_reachable_states(automaton)
    if not any(state in reachable for state in automaton.final_states):
        return _RunCountAdmission(reachability_work, reachable, True)

    # An ordered tree shape has at most 4**n possibilities, each node has at
    # most 32 symbols and 64 assigned states: at most 8192**n runs. This also
    # bounds every nonnegative intermediate coefficient. Reserve one digit
    # for the strict inequality and log rounding.
    max_coefficient_digits = 4 * max_size + 1
    # Add JSON string quotes, separators, and result-field overhead.
    profile_digits = 4 * sum(range(1, max_size + 1)) + 4 * max_size + 64
    if max_coefficient_digits > MAX_CANONICAL_INTEGER_DIGITS:
        _reject_tree("run-count coefficients exceed the exact integer digit bound")
    source_chars = (
        256
        + 4 * len(automaton.arity)
        + 4 * len(automaton.final_states)
        + sum(64 + 3 * len(row.child_states) for row in automaton.transitions)
    )
    if source_chars + profile_digits > 1_000_000:
        _reject_tree("run-count profile exceeds the exact output bound")

    groups: dict[tuple[int, tuple[int, ...]], int] = {}
    for transition in automaton.transitions:
        if any(child not in reachable for child in transition.child_states):
            continue
        if transition.target_state not in reachable:
            continue
        key = (transition.symbol, transition.child_states)
        groups[key] = groups.get(key, 0) + 1

    reachable_transition_count = sum(groups.values())
    width = max_size + 1
    work = (
        sum(
            width
            + (width * width if len(child_states) == 2 else 0)
            + (2 * width * width * width * max(0, len(child_states) - 2))
            for _, child_states in groups
        )
        + reachable_transition_count * width
        + automaton.state_count * width
    )
    total_work = work + reachability_work
    if total_work > MAX_TREE_AUTOMATON_WORK:
        _reject_tree("nondeterministic run-count work bound exceeded")
    return _RunCountAdmission(total_work, reachable, False)


def nondeterministic_run_counts_work_bound(
    automaton: BottomUpTreeAutomaton, max_size: int
) -> int:
    """Return the admitted combined reachability and DP work estimate."""
    return _admit_nondeterministic_run_counts(automaton, max_size).work


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
    "MAX_TREE_COUNT_HEIGHT",
    "MAX_TREE_COUNT_OUTPUT_BYTES",
    "BottomUpTreeAutomaton",
    "CompleteDeterministicBottomUpTreeAutomaton",
    "DeterministicBottomUpTreeAutomaton",
    "RankedTree",
    "ReachableStateProfile",
    "RegularTreeGrammar",
    "RegularTreeProduction",
    "TreeAutomatonTransition",
    "TreeStateChartEntry",
    "TreeStateWitness",
    "accepted_tree_count_work_bound",
    "nondeterministic_run_counts_work_bound",
    "ranked_tree_node_count",
    "validate_ranked_tree",
]
