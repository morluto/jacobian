"""Provider-independent values for exact bottom-up tree automata."""

from __future__ import annotations

from collections import Counter
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
    witness is chosen by the same rule recursively.  Source binding replays
    exactly this rule, so only the published witness for each state
    validates.
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

    @model_validator(mode="after")
    def require_source_bound_profile(self) -> Self:
        self.require_source_binding()
        return self

    def require_source_binding(self) -> None:
        """Replay the exact least fixed point against the retained automaton.

        Model validation invokes this replay automatically, so any
        deserialized or independently supplied profile proves itself against
        its automaton before being accepted as the declared exact result
        type.  The producing kernel instead constructs via ``model_construct``
        so one admitted execution performs exactly one profile; the public
        path invokes this method once as its source-bound result replay.
        """
        from jacobian.math.tree_automata.operations import reachable_state_profile

        if self != reachable_state_profile(self.automaton):
            raise _validation_error(
                "profile_not_bound",
                "reachability profile is not bound to its automaton",
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
