"""Typed wire contracts for tree automaton operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.tree_automata.values import (
    BottomUpTreeAutomaton,
    RankedTree,
)


class TreeRunRequest(StrictModel):
    """Run a bottom-up tree automaton on a ranked tree.

    Returns the set of states reachable at the root.
    """

    automaton: BottomUpTreeAutomaton
    tree: RankedTree

    @model_validator(mode="after")
    def require_valid_tree_arity(self) -> Self:
        if self.tree.symbol >= len(self.automaton.arity):
            raise ValueError("tree symbol out of range")
        expected_arity = self.automaton.arity[self.tree.symbol]
        if len(self.tree.children) != expected_arity:
            raise ValueError("tree node arity must match automaton arity")
        return self


class TreeRunResult(StrictModel):
    """Result of a tree automaton run."""

    accepted: bool
    root_states: tuple[int, ...]


class AcceptedTreeCountRequest(StrictModel):
    """Count accepted trees of a given size."""

    automaton: BottomUpTreeAutomaton
    tree_size: int = Field(ge=1, le=100)


class AcceptedTreeCountResult(StrictModel):
    """Exact count of accepted trees."""

    count: int


__all__ = [
    "AcceptedTreeCountRequest",
    "AcceptedTreeCountResult",
    "TreeRunRequest",
    "TreeRunResult",
]
