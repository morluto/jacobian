"""Typed wire contracts for tree automaton operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.math.tree_automata.values import (
    MAX_REACHABILITY_WITNESS_NODES,
    BottomUpTreeAutomaton,
    RankedTree,
    accepted_tree_count_work_bound,
    ranked_tree_node_count,
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


class ReachableStateWitness(StrictModel):
    """One canonical minimum-node tree witnessing a reachable automaton state."""

    state: int = Field(ge=0, le=63)
    tree: RankedTree
    node_count: int = Field(ge=1, le=MAX_REACHABILITY_WITNESS_NODES)

    @model_validator(mode="after")
    def require_exact_node_count(self) -> Self:
        if self.node_count != ranked_tree_node_count(self.tree):
            raise ValueError("witness node_count must equal its tree node count")
        return self


class TreeAutomatonReachabilityRequest(StrictModel):
    """Compute ground-tree reachable states through bottom-up hyperedges.

    A schema-valid automaton can still exceed two coupled work envelopes that
    validation enforces before execution:

    - ``MAX_TREE_AUTOMATON_REACHABILITY_WORK`` (30,000,000 units) prices
      transition sorting, one constructible-state closure prepass plus the
      saturation scans over every transition row, and witness materialization
      and recount, multiplied across request admission, execution, and
      source-bound result replay.
    - ``MAX_REACHABILITY_WITNESS_NODES`` (4096 nodes) bounds the total node
      count summed over the minimum witnesses of all reachable states: it is
      an aggregate output limit across states, not a per-witness limit.

    Adjust either quantity by shrinking the automaton (fewer or cheaper
    transition rows, fewer constructible states, smaller witnesses).
    """

    automaton: BottomUpTreeAutomaton = Field(
        description=(
            "nondeterministic bottom-up tree automaton with at most 64 "
            "states, 32 ranked symbols, and 4096 unique transitions. "
            "Requests are additionally rejected when the coupled "
            "reachability work envelope (MAX_TREE_AUTOMATON_REACHABILITY_"
            "WORK = 30,000,000 units) or the aggregate witness output "
            "envelope (MAX_REACHABILITY_WITNESS_NODES = 4096 nodes summed "
            "across every reachable state's minimum witness) is exceeded"
        ),
    )

    @model_validator(mode="after")
    def require_bounded_witness_profile(self) -> Self:
        from jacobian.math.tree_automata.operations import reachable_state_profile

        reachable_state_profile(self.automaton)
        return self


class TreeAutomatonReachabilityResult(StrictModel):
    """Exact source-bound ground-tree reachability profile."""

    automaton: BottomUpTreeAutomaton
    reachable_states: tuple[int, ...] = Field(max_length=64)
    unreachable_states: tuple[int, ...] = Field(max_length=64)
    witnesses: tuple[ReachableStateWitness, ...] = Field(
        max_length=64,
        description=(
            "one canonical minimum-node witness per reachable state; their "
            "node counts are bounded in aggregate by "
            "MAX_REACHABILITY_WITNESS_NODES (4096 nodes summed over all "
            "reachable states)"
        ),
    )

    @model_validator(mode="after")
    def require_source_bound_profile(self) -> Self:
        from jacobian.math.tree_automata.operations import reachable_state_profile

        expected = reachable_state_profile(self.automaton)
        expected_witnesses = tuple(
            ReachableStateWitness(
                state=state,
                tree=tree,
                node_count=ranked_tree_node_count(tree),
            )
            for state, tree in expected.witnesses
        )
        if sum(witness.node_count for witness in self.witnesses) > (
            MAX_REACHABILITY_WITNESS_NODES
        ):
            raise ValueError("reachable-state witness output exceeds the node bound")
        if (
            self.reachable_states != expected.reachable_states
            or self.unreachable_states != expected.unreachable_states
            or self.witnesses != expected_witnesses
        ):
            raise ValueError("reachability profile is not bound to its automaton")
        return self


__all__ = [
    "AcceptedTreeCountRequest",
    "AcceptedTreeCountResult",
    "ReachableStateWitness",
    "TreeAutomatonReachabilityRequest",
    "TreeAutomatonReachabilityResult",
    "TreeRunRequest",
    "TreeRunResult",
]
