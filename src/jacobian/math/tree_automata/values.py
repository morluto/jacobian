"""Provider-independent values for exact bottom-up tree automata."""

from __future__ import annotations

from collections import Counter
from math import comb
from typing import Annotated, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_TA_STATES = 64
MAX_TA_SYMBOLS = 32
MAX_TA_TRANSITIONS = 4096
MAX_TA_ARITY = 16
MAX_RUN_TREE_NODES = 4096
MAX_RUN_TREE_DEPTH = 128
MAX_TREE_AUTOMATON_WORK = 2_000_000

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
    arity: tuple[Arity, ...] = Field(min_length=1, max_length=MAX_TA_SYMBOLS)
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
            raise ValueError("transitions must be unique")
        if len(set(self.final_states)) != len(self.final_states):
            raise ValueError("final states must be unique")

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
    "MAX_RUN_TREE_DEPTH",
    "MAX_RUN_TREE_NODES",
    "MAX_TA_ARITY",
    "MAX_TA_STATES",
    "MAX_TA_SYMBOLS",
    "MAX_TA_TRANSITIONS",
    "MAX_TREE_AUTOMATON_WORK",
    "BottomUpTreeAutomaton",
    "RankedTree",
    "TreeAutomatonTransition",
    "accepted_tree_count_work_bound",
    "validate_ranked_tree",
]
