"""Typed wire contracts for tree automaton operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.math.tree_automata.values import (
    BottomUpTreeAutomaton,
    RankedTree,
    accepted_tree_count_work_bound,
    validate_ranked_tree,
)


class TreeRunRequest(StrictModel):
    """Run a bottom-up tree automaton on a ranked tree.

    Returns the set of states reachable at the root.
    """

    automaton: BottomUpTreeAutomaton
    tree: RankedTree

    @model_validator(mode="after")
    def require_valid_tree_arity(self) -> Self:
        validate_ranked_tree(self.automaton, self.tree)
        return self


class TreeRunResult(TreeRunRequest):
    """Result of a tree automaton run."""

    accepted: bool
    root_states: tuple[int, ...] = Field(max_length=64)
    state_chart: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    node_count: int = Field(ge=1, le=4096)
    complete: Literal[True] = True
    method: Literal["BOTTOM_UP_REACHABLE_STATE_SETS"] = "BOTTOM_UP_REACHABLE_STATE_SETS"

    @model_validator(mode="after")
    def require_canonical_root_states(self) -> Self:
        if self.root_states != tuple(sorted(set(self.root_states))):
            raise ValueError("root states must be unique and sorted")
        from jacobian.math.tree_automata.operations import tree_state_chart

        expected_chart = tree_state_chart(self.automaton, self.tree)
        expected = expected_chart[-1][1]
        if self.state_chart != expected_chart or self.root_states != expected:
            raise ValueError("root states are not bound to the automaton and tree")
        if self.accepted != bool(set(expected) & set(self.automaton.final_states)):
            raise ValueError(
                "tree acceptance must agree with the reachable root states"
            )
        return self


class AcceptedTreeCountRequest(StrictModel):
    """Count accepted trees of a given size."""

    automaton: BottomUpTreeAutomaton
    tree_size: int = Field(ge=1, le=100)

    @model_validator(mode="after")
    def require_bounded_exact_count(self) -> Self:
        accepted_tree_count_work_bound(self.automaton, self.tree_size)
        return self


class AcceptedTreeCountResult(AcceptedTreeCountRequest):
    """Exact count of accepted trees."""

    tree_size: int = Field(ge=1, le=100)
    count: CanonicalInteger
    complete: Literal[True] = True
    method: Literal["ON_THE_FLY_SUBSET_DYNAMIC_PROGRAMMING"] = (
        "ON_THE_FLY_SUBSET_DYNAMIC_PROGRAMMING"
    )
    estimated_work_bound: int = Field(ge=0, le=2_000_000)

    @model_validator(mode="after")
    def bind_count(self) -> Self:
        from jacobian.math.tree_automata.operations import accepted_tree_count

        if int(self.count) != accepted_tree_count(self.automaton, self.tree_size):
            raise ValueError("tree count is not bound to its automaton")
        return self


__all__ = [
    "AcceptedTreeCountRequest",
    "AcceptedTreeCountResult",
    "TreeRunRequest",
    "TreeRunResult",
]
