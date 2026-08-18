"""Provider-independent values for exact bottom-up tree automata."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_TA_STATES = 64
MAX_TA_SYMBOLS = 32
MAX_TA_TRANSITIONS = 4096
MAX_TA_ARITY = 16


class TreeAutomatonTransition(StrictModel):
    """A bottom-up tree automaton transition.

    A transition ``f(q_1, ..., q_n) -> q`` says: if the children of a
    ``f``-labelled node are in states ``q_1, ..., q_n``, the node is in
    state ``q``.
    ``symbol`` is the function symbol (label of the node).
    """

    symbol: int = Field(ge=0)
    child_states: tuple[int, ...] = Field(max_length=MAX_TA_ARITY)
    target_state: int = Field(ge=0)


class RankedTree(StrictModel):
    """A ranked tree: a node labelled by a symbol with zero or more children."""

    symbol: int = Field(ge=0)
    children: tuple[RankedTree, ...] = Field(default=())

    @model_validator(mode="after")
    def require_valid_tree(self) -> Self:
        if len(self.children) > MAX_TA_ARITY:
            raise ValueError("arity exceeds bound")
        return self


RankedTree.model_rebuild()


class BottomUpTreeAutomaton(StrictModel):
    """A nondeterministic bottom-up tree automaton (NFTA).

    The automaton has ``state_count`` states, a ranked alphabet where
    ``arity[symbol]`` gives the arity of each symbol, a set of transitions,
    and a set of final (accepting) states.
    """

    state_count: int = Field(ge=1, le=MAX_TA_STATES)
    arity: tuple[int, ...] = Field(min_length=1)
    transitions: tuple[TreeAutomatonTransition, ...] = Field(
        min_length=0, max_length=MAX_TA_TRANSITIONS
    )
    final_states: tuple[int, ...] = Field(min_length=0)

    @model_validator(mode="after")
    def require_valid_automaton(self) -> Self:
        for tr in self.transitions:
            if not 0 <= tr.target_state < self.state_count:
                raise ValueError("transition target out of range")
            if tr.symbol >= len(self.arity):
                raise ValueError("transition symbol out of range")
            if len(tr.child_states) != self.arity[tr.symbol]:
                raise ValueError(
                    "transition child count must match symbol arity"
                )
            for s in tr.child_states:
                if not 0 <= s < self.state_count:
                    raise ValueError("transition child state out of range")
        for f in self.final_states:
            if not 0 <= f < self.state_count:
                raise ValueError("final state out of range")
        return self


__all__ = [
    "MAX_TA_ARITY",
    "MAX_TA_STATES",
    "MAX_TA_SYMBOLS",
    "MAX_TA_TRANSITIONS",
    "BottomUpTreeAutomaton",
    "RankedTree",
    "TreeAutomatonTransition",
]
