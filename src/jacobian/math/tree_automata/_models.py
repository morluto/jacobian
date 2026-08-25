"""Typed wire contracts for tree automaton operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.math.tree_automata.values import (
    MAX_TA_TRANSITIONS,
    MAX_TREE_AUTOMATON_REACHABILITY_WORK,
    BottomUpTreeAutomaton,
    RankedTree,
    accepted_tree_count_work_bound,
    validate_ranked_tree,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"tree_automata.{reason}", message)


class TreeRunRequest(StrictModel):
    """Run a bottom-up tree automaton on a ranked tree.

    Returns the set of states reachable at the root.
    """

    automaton: BottomUpTreeAutomaton
    tree: RankedTree

    @model_validator(mode="after")
    def require_valid_tree_arity(self) -> Self:
        try:
            validate_ranked_tree(self.automaton, self.tree)
        except ValueError as exc:
            raise _validation_error("invalid_tree", str(exc)) from exc
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
            raise _validation_error(
                "root_states_not_canonical", "root states must be unique and sorted"
            )
        from jacobian.math.tree_automata.operations import tree_state_chart

        expected_chart = tree_state_chart(self.automaton, self.tree)
        expected = expected_chart[-1][1]
        if self.state_chart != expected_chart or self.root_states != expected:
            raise _validation_error(
                "root_states_not_bound",
                "root states are not bound to the automaton and tree",
            )
        if self.accepted != bool(set(expected) & set(self.automaton.final_states)):
            raise _validation_error(
                "acceptance_not_bound",
                "tree acceptance must agree with the reachable root states",
            )
        return self


class AcceptedTreeCountRequest(StrictModel):
    """Count accepted trees of a given size."""

    automaton: BottomUpTreeAutomaton
    tree_size: int = Field(ge=1, le=100)

    @model_validator(mode="after")
    def require_bounded_exact_count(self) -> Self:
        try:
            accepted_tree_count_work_bound(self.automaton, self.tree_size)
        except ValueError as exc:
            raise _validation_error("count_work_bound", str(exc)) from exc
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
            raise _validation_error(
                "count_not_bound", "tree count is not bound to its automaton"
            )
        return self


class TreeAutomatonReachabilityRequest(StrictModel):
    """Compute ground-tree reachable states through bottom-up hyperedges.

    A schema-valid automaton can still exceed two coupled work envelopes that
    validation enforces before execution:

    - ``MAX_TREE_AUTOMATON_REACHABILITY_WORK`` (30,000,000 units) prices one
      profile's transition sorting, saturation scans measured to their exact
      convergence depth by a shared-code-path pass, and witness
      materialization and recount, charged across the four priced passes the
      public path performs: work admission's own convergence-measuring pass
      plus one pass consumed by each of the three profile invocations
      (request admission, execution, and source-bound result replay).  Each
      pass performs exactly one sort and one saturation, and each profile
      reuses its own pass's result, so no unpriced sort or probe executes.
      The pass always terminates within ``state_count + 1`` rounds for any
      schema-valid automaton.
    - ``MAX_REACHABILITY_WITNESS_NODES`` (4096 nodes) bounds the total node
      count summed over the minimum witnesses of all reachable states: it is
      an aggregate output limit across states, not a per-witness limit.

    Adjust either quantity by shrinking the automaton (fewer or cheaper
    transition rows, shallower witness-dependency chains, smaller witnesses).
    """

    automaton: BottomUpTreeAutomaton = Field(
        description=(
            f"nondeterministic bottom-up tree automaton with at most 64 "
            f"states, 32 ranked symbols, and {MAX_TA_TRANSITIONS} unique transitions. "
            "Requests are additionally rejected when the coupled "
            "reachability work envelope (MAX_TREE_AUTOMATON_REACHABILITY_"
            "WORK = 30,000,000 units, priced across the four passes behind "
            "request admission, execution, and source-bound result replay "
            "together with request admission's own saturation convergence "
            "pass) or the "
            "aggregate witness output envelope (MAX_REACHABILITY_WITNESS_"
            "NODES = 4096 nodes summed across every reachable state's "
            "minimum witness) is exceeded"
        ),
    )

    @model_validator(mode="after")
    def require_bounded_witness_profile(self) -> Self:
        from jacobian.math.tree_automata.operations import (
            _reachability_public_path_work_bound,
            reachable_state_profile,
        )

        if _reachability_public_path_work_bound(self.automaton) > (
            MAX_TREE_AUTOMATON_REACHABILITY_WORK
        ):
            raise _validation_error(
                "reachability_work_bound",
                "tree automaton reachability work bound exceeded",
            )
        try:
            reachable_state_profile(self.automaton)
        except ValueError as exc:
            if "witness output" in str(exc):
                raise _validation_error("witness_output_bound", str(exc)) from exc
            raise
        return self


__all__ = [
    "AcceptedTreeCountRequest",
    "AcceptedTreeCountResult",
    "TreeAutomatonReachabilityRequest",
    "TreeRunRequest",
    "TreeRunResult",
]
