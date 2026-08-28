"""Typed wire contracts for tree automaton operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.math.logic.automata.tree.values import (
    MAX_REACHABILITY_WITNESS_NODES,
    MAX_TA_STATES,
    MAX_TA_SYMBOLS,
    MAX_TA_TRANSITIONS,
    MAX_TREE_AUTOMATON_REACHABILITY_WORK,
    BottomUpTreeAutomaton,
    RankedTree,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"tree_automata.{reason}", message)


class TreeRunRequest(StrictModel):
    """Run a bottom-up tree automaton on a ranked tree.

    Returns the set of states reachable at the root.
    """

    automaton: BottomUpTreeAutomaton
    tree: RankedTree


class TreeRunResult(TreeRunRequest):
    """Result of a tree automaton run."""

    accepted: bool
    root_states: tuple[int, ...] = Field(max_length=64)
    state_chart: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    node_count: int = Field(ge=1, le=4096)

    @model_validator(mode="after")
    def require_canonical_root_states(self) -> Self:
        if self.root_states != tuple(sorted(set(self.root_states))):
            raise _validation_error(
                "root_states_not_canonical", "root states must be unique and sorted"
            )
        if any(
            not 0 <= state < self.automaton.state_count for state in self.root_states
        ):
            raise _validation_error(
                "root_state_out_of_range", "root state out of range"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: TreeRunRequest,
        *,
        accepted: bool,
        root_states: tuple[int, ...],
        state_chart: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
        node_count: int,
    ) -> Self:
        """Construct a result emitted by the trusted tree-run kernel."""

        return cls(
            **request.model_dump(),
            accepted=accepted,
            root_states=root_states,
            state_chart=state_chart,
            node_count=node_count,
        )


class AcceptedTreeCountRequest(StrictModel):
    """Count accepted trees of a given size."""

    automaton: BottomUpTreeAutomaton
    tree_size: int = Field(ge=1, le=100)


class AcceptedTreeCountResult(AcceptedTreeCountRequest):
    """Exact count of accepted trees."""

    tree_size: int = Field(ge=1, le=100)
    count: CanonicalInteger
    estimated_work_bound: int = Field(ge=0, le=2_000_000)

    @model_validator(mode="after")
    def bind_count(self) -> Self:
        if int(self.count) < 0:
            raise _validation_error("count_negative", "tree count must be nonnegative")
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: AcceptedTreeCountRequest,
        *,
        count: CanonicalInteger,
        estimated_work_bound: int,
    ) -> Self:
        """Construct a result emitted by the trusted subset-DP kernel."""

        return cls(
            **request.model_dump(),
            count=count,
            estimated_work_bound=estimated_work_bound,
        )


class TreeAutomatonReachabilityRequest(StrictModel):
    __doc__ = f"""Compute ground-tree reachable states through bottom-up hyperedges.

    A schema-valid automaton can still exceed two coupled work envelopes that
    validation enforces before execution:

    - ``MAX_TREE_AUTOMATON_REACHABILITY_WORK`` ({MAX_TREE_AUTOMATON_REACHABILITY_WORK:,} units) prices one
      profile's transition sorting, saturation scans measured to their exact
      convergence depth by a shared-code-path pass, and witness
      materialization and recount. The owner-local operation performs exactly
      one sort and one saturation, reusing that pass for admission and result
      construction. The pass always terminates within ``state_count + 1``
      rounds for any schema-valid automaton.
    - ``MAX_REACHABILITY_WITNESS_NODES`` ({MAX_REACHABILITY_WITNESS_NODES} nodes) bounds the total node
      count summed over the minimum witnesses of all reachable states: it is
      an aggregate output limit across states, not a per-witness limit.

    Adjust either quantity by shrinking the automaton (fewer or cheaper
    transition rows, shallower witness-dependency chains, smaller witnesses).
    """

    automaton: BottomUpTreeAutomaton = Field(
        description=(
            f"nondeterministic bottom-up tree automaton with at most "
            f"{MAX_TA_STATES} states, {MAX_TA_SYMBOLS} ranked symbols, and "
            f"{MAX_TA_TRANSITIONS} unique transitions. "
            "Execution is bounded by the coupled "
            "reachability work envelope (MAX_TREE_AUTOMATON_REACHABILITY_"
            f"WORK = {MAX_TREE_AUTOMATON_REACHABILITY_WORK:,} units for one owner-local saturation pass) or the "
            "aggregate witness output envelope (MAX_REACHABILITY_WITNESS_"
            f"NODES = {MAX_REACHABILITY_WITNESS_NODES} nodes summed across every reachable state's "
            "minimum witness) is exceeded"
        ),
    )


__all__ = [
    "AcceptedTreeCountRequest",
    "AcceptedTreeCountResult",
    "TreeAutomatonReachabilityRequest",
    "TreeRunRequest",
    "TreeRunResult",
]
