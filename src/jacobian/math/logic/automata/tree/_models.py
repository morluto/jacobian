"""Typed wire contracts for tree automaton operations."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import Field, StrictInt, field_validator, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.math.logic.automata.tree.values import (
    MAX_REACHABILITY_WITNESS_NODES,
    MAX_RUN_TREE_DEPTH,
    MAX_RUN_TREE_NODES,
    MAX_TA_ARITY,
    MAX_TA_STATES,
    MAX_TA_SYMBOLS,
    MAX_TA_TRANSITIONS,
    MAX_TREE_AUTOMATON_REACHABILITY_WORK,
    BottomUpTreeAutomaton,
    CompleteDeterministicBottomUpTreeAutomaton,
    DeterministicBottomUpTreeAutomaton,
    RankedTree,
    RegularTreeGrammar,
    TreeStateChartEntry,
    TreeStateWitness,
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
    state_chart: tuple[TreeStateChartEntry, ...]
    node_count: int = Field(ge=1, le=4096)

    @field_validator("state_chart", mode="before")
    @classmethod
    def decode_legacy_chart_rows(cls, value: object) -> object:
        """Accept tuple rows while keeping the canonical Python value typed."""
        if not isinstance(value, (list, tuple)):
            return value
        converted: list[object] = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                converted.append({"position": item[0], "states": item[1]})
            else:
                converted.append(item)
        return converted

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
        state_chart: tuple[TreeStateChartEntry, ...],
        node_count: int,
    ) -> Self:
        """Construct a result emitted by the trusted tree-run kernel."""

        return cls.model_construct(
            automaton=request.automaton,
            tree=request.tree,
            accepted=accepted,
            root_states=root_states,
            state_chart=state_chart,
            node_count=node_count,
        )


class RankedTreePositionsRequest(StrictModel):
    """Return every node address in a finite ranked tree."""

    tree: RankedTree


class RankedTreePositionsResult(RankedTreePositionsRequest):
    """Zero-based child-index paths in root-first preorder."""

    positions: tuple[
        Annotated[
            tuple[Annotated[StrictInt, Field(ge=0, lt=MAX_TA_ARITY)], ...],
            Field(max_length=MAX_RUN_TREE_DEPTH),
        ],
        ...,
    ] = Field(max_length=MAX_RUN_TREE_NODES)

    @model_validator(mode="after")
    def require_position_shape(self) -> Self:
        if not self.positions or self.positions[0] != ():
            raise _validation_error(
                "positions_root", "positions must start with the empty root path"
            )
        if len(set(self.positions)) != len(self.positions):
            raise _validation_error(
                "positions_unique", "tree node positions must be unique"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        tree: RankedTree,
        positions: tuple[tuple[int, ...], ...],
    ) -> Self:
        """Construct the complete position list emitted by the admitted kernel."""

        return cls.model_construct(tree=tree, positions=positions)


class RankedTreeSubtreeRequest(StrictModel):
    """Select the rooted subtree at a zero-based child-index position."""

    tree: RankedTree
    position: tuple[Annotated[StrictInt, Field(ge=0, lt=MAX_TA_ARITY)], ...] = Field(
        max_length=MAX_RUN_TREE_DEPTH
    )


class RankedTreeSubtreeResult(RankedTreeSubtreeRequest):
    """A subtree together with its exact source tree and structural address."""

    subtree: RankedTree

    @classmethod
    def _from_kernel(
        cls,
        *,
        tree: RankedTree,
        position: tuple[int, ...],
        subtree: RankedTree,
    ) -> Self:
        return cls.model_construct(tree=tree, position=position, subtree=subtree)


class AcceptedTreeCountRequest(StrictModel):
    """Count accepted trees of a given size."""

    automaton: BottomUpTreeAutomaton
    tree_size: int = Field(ge=1, le=100)


class AcceptedTreeCountResult(AcceptedTreeCountRequest):
    """Exact count of accepted trees."""

    tree_size: int = Field(ge=1, le=100)
    count: ExactInteger
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
        count: ExactInteger,
        estimated_work_bound: int,
    ) -> Self:
        """Construct a result emitted by the trusted subset-DP kernel."""

        return cls.model_construct(
            automaton=request.automaton,
            tree_size=request.tree_size,
            count=count,
            estimated_work_bound=estimated_work_bound,
        )


class TreeAutomatonTrimRequest(StrictModel):
    """Restrict an automaton to its reachable and productive states."""

    automaton: BottomUpTreeAutomaton


class TreeAutomatonTrimResult(StrictModel):
    """Trimmed automaton with old/new transport and trimmed-state witnesses."""

    automaton: BottomUpTreeAutomaton
    trimmed: BottomUpTreeAutomaton
    kept_states: tuple[int, ...]
    dropped_states: tuple[int, ...]
    old_to_new: tuple[int, ...]
    new_to_old: tuple[int, ...]
    empty_language: bool
    witnesses: tuple[TreeStateWitness, ...] = Field(
        description=(
            "One canonical minimum-node witness tree per kept state, replayed "
            "on the trimmed automaton so every witness uses kept states only."
        )
    )

    @model_validator(mode="after")
    def require_canonical_trim(self) -> Self:
        state_count = self.automaton.state_count
        if self.kept_states != tuple(sorted(set(self.kept_states))) or any(
            not 0 <= state < state_count for state in self.kept_states
        ):
            raise _validation_error(
                "kept_states_not_canonical", "kept states must be unique and sorted"
            )
        if set(self.kept_states) | set(self.dropped_states) != set(
            range(state_count)
        ) or set(self.kept_states) & set(self.dropped_states):
            raise _validation_error(
                "kept_dropped_not_partition",
                "kept and dropped states must partition the automaton states",
            )
        if len(self.old_to_new) != state_count:
            raise _validation_error(
                "old_to_new_axis", "old-to-new map must cover every source state"
            )
        for old, new in enumerate(self.old_to_new):
            expected = self.kept_states.index(old) if old in self.kept_states else -1
            if new != expected:
                raise _validation_error(
                    "old_to_new_mismatch", "old-to-new map must index the kept states"
                )
        if self.new_to_old != self.kept_states:
            raise _validation_error(
                "new_to_old_mismatch", "new-to-old map must list the kept states"
            )
        if self.empty_language != (not self.kept_states):
            raise _validation_error(
                "empty_language_mismatch",
                "empty language must agree with keeping no state",
            )
        if self.trimmed.arity != self.automaton.arity:
            raise _validation_error(
                "trimmed_alphabet", "trimming preserves the ranked alphabet"
            )
        if tuple(witness.state for witness in self.witnesses) != tuple(
            range(self.trimmed.state_count)
        ):
            raise _validation_error(
                "witnesses_not_aligned",
                "witnesses must carry exactly one entry per trimmed state in order",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class RegularTreeGrammarToAutomatonRequest(StrictModel):
    """Convert a unit-free regular tree grammar to its bottom-up automaton."""

    grammar: RegularTreeGrammar


class RegularTreeGrammarToAutomatonResult(StrictModel):
    """The source grammar and the equivalent bottom-up tree automaton."""

    grammar: RegularTreeGrammar
    automaton: BottomUpTreeAutomaton

    @classmethod
    def _from_kernel(
        cls, *, grammar: RegularTreeGrammar, automaton: BottomUpTreeAutomaton
    ) -> Self:
        """Construct the source-bound result emitted by the trusted converter."""
        return cls.model_construct(grammar=grammar, automaton=automaton)


class TreeAutomatonComplementRequest(StrictModel):
    """Complement a complete deterministic automaton over its ranked alphabet."""

    automaton: CompleteDeterministicBottomUpTreeAutomaton = Field(
        description=(
            "A deterministic complete bottom-up tree automaton. For every "
            "ranked symbol of arity k, exactly one transition must exist for "
            "each ordered k-tuple of source states."
        )
    )


class TreeAutomatonBooleanProductRequest(StrictModel):
    """Product two complete deterministic automata over one ranked alphabet."""

    left: CompleteDeterministicBottomUpTreeAutomaton
    right: CompleteDeterministicBottomUpTreeAutomaton
    connective: Literal["intersection", "union", "difference", "symmetric_difference"]


class TreeAutomatonBooleanProductResult(TreeAutomatonBooleanProductRequest):
    """Direct product machine, whose states are ordered source-state pairs."""

    product: CompleteDeterministicBottomUpTreeAutomaton
    state_pairs: tuple[tuple[int, int], ...] = Field(max_length=MAX_TA_STATES)

    @model_validator(mode="after")
    def require_product_axes(self) -> Self:
        pair_count = self.left.state_count * self.right.state_count
        if pair_count > MAX_TA_STATES:
            raise _validation_error(
                "product_state_bound",
                "the complete Cartesian product exceeds the supported state axis",
            )
        expected = tuple(
            (left, right)
            for left in range(self.left.state_count)
            for right in range(self.right.state_count)
        )
        if self.state_pairs != expected or self.product.state_count != len(expected):
            raise _validation_error(
                "product_state_axis",
                "product states must be the lexicographic source-state pairs",
            )
        if (
            self.product.arity != self.left.arity
            or self.product.arity != self.right.arity
        ):
            raise _validation_error(
                "product_signature",
                "Boolean product inputs must share the ranked signature",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class TreeAutomatonComplementResult(TreeAutomatonComplementRequest):
    """Complement automaton with canonical state-axis transport."""

    complement: CompleteDeterministicBottomUpTreeAutomaton
    old_to_new: tuple[int, ...]
    new_to_old: tuple[int, ...]
    transition_count: int = Field(ge=0, le=MAX_TA_TRANSITIONS)

    @model_validator(mode="after")
    def require_axis_maps(self) -> Self:
        state_count = self.automaton.state_count
        expected = tuple(range(state_count))
        if (
            tuple(sorted(self.old_to_new)) != expected
            or len(self.old_to_new) != state_count
        ):
            raise _validation_error(
                "complement_state_map", "old_to_new must be a state permutation"
            )
        if (
            len(self.new_to_old) != state_count
            or tuple(sorted(self.new_to_old)) != expected
            or tuple(self.old_to_new[old] for old in self.new_to_old) != expected
        ):
            raise _validation_error(
                "complement_state_map", "state maps must be mutual inverses"
            )
        if (
            self.complement.state_count != state_count
            or self.complement.arity != self.automaton.arity
        ):
            raise _validation_error(
                "complement_axes",
                "complement preserves the state count and ranked signature",
            )
        if len(self.complement.transitions) != self.transition_count:
            raise _validation_error(
                "complement_transition_count",
                "transition_count must match the complete result table",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class TreeAutomatonCompletionRequest(StrictModel):
    """Complete a partial deterministic machine over its ranked alphabet."""

    automaton: DeterministicBottomUpTreeAutomaton


class TreeAutomatonCompletionResult(TreeAutomatonCompletionRequest):
    """Complete transition table with the source-state map and optional sink."""

    completed: CompleteDeterministicBottomUpTreeAutomaton
    source_to_completed: tuple[int, ...] = Field(max_length=MAX_TA_STATES)
    sink_state: int | None = Field(default=None, ge=0, lt=MAX_TA_STATES)

    @model_validator(mode="after")
    def require_canonical_state_transport(self) -> Self:
        source = self.automaton
        if self.source_to_completed != tuple(range(source.state_count)):
            raise _validation_error(
                "completion_state_map",
                "completion preserves the source-state order",
            )
        if (
            self.completed.arity != source.arity
            or self.completed.final_states != source.final_states
        ):
            raise _validation_error(
                "completion_source_context",
                "completion preserves the ranked alphabet and source final states",
            )
        if self.sink_state is None:
            required = sum(source.state_count**rank for rank in source.arity)
            if (
                self.completed.state_count != source.state_count
                or len(source.transitions) != required
            ):
                raise _validation_error(
                    "completion_missing_sink",
                    "a partial source needs one added sink state",
                )
        elif (
            self.sink_state != source.state_count
            or self.completed.state_count != source.state_count + 1
            or self.sink_state in self.completed.final_states
        ):
            raise _validation_error(
                "completion_sink_state",
                "the sink is the appended nonfinal state",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class TreeAutomatonMinimizeRequest(StrictModel):
    """Minimize a partial or complete deterministic bottom-up automaton."""

    automaton: DeterministicBottomUpTreeAutomaton


class TreeAutomatonMinimizeResult(TreeAutomatonMinimizeRequest):
    """Smallest reachable deterministic quotient and source-state transport.

    A total quotient table is carried as the complete deterministic carrier,
    so it composes directly with complement and Boolean products; a partial
    quotient stays on the partial deterministic carrier.
    """

    minimized: (
        CompleteDeterministicBottomUpTreeAutomaton | DeterministicBottomUpTreeAutomaton
    )
    old_to_new: tuple[int, ...] = Field(max_length=MAX_TA_STATES)
    new_to_old: tuple[int | None, ...] = Field(max_length=MAX_TA_STATES)
    reachable_states: tuple[int, ...] = Field(max_length=MAX_TA_STATES)

    @model_validator(mode="after")
    def require_canonical_minimization_axes(self) -> Self:
        source = self.automaton
        if self.reachable_states != tuple(sorted(set(self.reachable_states))):
            raise _validation_error(
                "minimize_reachable_order", "reachable states must be unique and sorted"
            )
        if any(not 0 <= state < source.state_count for state in self.reachable_states):
            raise _validation_error(
                "minimize_reachable_range", "reachable state is out of range"
            )
        if len(self.old_to_new) != source.state_count:
            raise _validation_error(
                "minimize_old_to_new_axis",
                "old-to-new map must cover every source state",
            )
        if any(
            image != -1 and not 0 <= image < self.minimized.state_count
            for image in self.old_to_new
        ):
            raise _validation_error(
                "minimize_map_range", "state images must be -1 or quotient states"
            )
        if self.minimized.arity != source.arity:
            raise _validation_error(
                "minimize_signature", "minimization preserves the ranked signature"
            )
        non_null_representatives = tuple(
            state for state in self.new_to_old if state is not None
        )
        if (
            len(self.new_to_old) != self.minimized.state_count
            or (
                len(non_null_representatives) == len(self.new_to_old)
                and non_null_representatives
                != tuple(sorted(set(non_null_representatives)))
            )
            or any(
                state is not None and not 0 <= state < source.state_count
                for state in self.new_to_old
            )
        ):
            raise _validation_error(
                "minimize_representatives",
                "representatives must be one ordered source state per quotient state",
            )
        if self.reachable_states:
            if set(self.reachable_states) != {
                state for state, image in enumerate(self.old_to_new) if image >= 0
            }:
                raise _validation_error(
                    "minimize_reachable_map",
                    "reachable states must be exactly the mapped source states",
                )
            if {self.old_to_new[state] for state in self.reachable_states} != set(
                range(self.minimized.state_count)
            ):
                raise _validation_error(
                    "minimize_quotient_coverage",
                    "reachable states must cover every quotient state",
                )
            if any(
                representative is None or self.old_to_new[representative] != quotient
                for quotient, representative in enumerate(self.new_to_old)
            ):
                raise _validation_error(
                    "minimize_representative_map",
                    "each representative must map to its quotient state",
                )
        elif (
            self.minimized.state_count != 1
            or self.minimized.transitions
            or self.minimized.final_states
            or any(image != -1 for image in self.old_to_new)
            or self.new_to_old != (None,)
        ):
            raise _validation_error(
                "minimize_empty_language_shape",
                "an automaton with no ground trees uses one nonfinal state and maps all source states to -1",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


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


class TreeDeterminizeRequest(StrictModel):
    """Determinize a bottom-up tree automaton by subset construction."""

    automaton: BottomUpTreeAutomaton
    max_subset_states: int = Field(default=64, ge=1, le=64)
    sample_max_height: int = Field(default=3, ge=0, le=5)


class TreeDeterminizeResult(TreeDeterminizeRequest):
    """A bounded subset-construction outcome with explicit status semantics.

    Deserialization checks only the canonical shape: the subset map covers
    exactly the deterministic states, the deterministic machine is
    deterministic with finals consistent with the subset map, and the
    equivalence claim agrees with the status. The owner-local kernel
    establishes transition closure and sample acceptance agreement on the
    COMPLETE path; a TRUNCATED construction never claims equivalence.
    """

    status: Literal["COMPLETE", "TRUNCATED"]
    truncation_reason: Literal["NONE", "STATE_BUDGET", "WORK_BUDGET"]
    deterministic: DeterministicBottomUpTreeAutomaton
    subset_map: tuple[tuple[int, ...], ...]
    equivalence_claim: bool
    closure_rows_checked: int = Field(ge=0)
    combos_evaluated: int = Field(ge=0)
    sample_trees_checked: int = Field(ge=0)
    sample_agreement: bool

    @model_validator(mode="after")
    def require_canonical_determinize_shape(self) -> Self:
        if self.deterministic.arity != self.automaton.arity:
            raise _validation_error(
                "determinize_alphabet", "determinization preserves the ranked alphabet"
            )
        if len(self.subset_map) != self.deterministic.state_count:
            raise _validation_error(
                "determinize_subset_axis",
                "the subset map must cover every deterministic state",
            )
        for subset in self.subset_map:
            if subset != tuple(sorted(set(subset))):
                raise _validation_error(
                    "determinize_subset_not_canonical",
                    "subsets must be unique and sorted",
                )
            if any(not 0 <= state < self.automaton.state_count for state in subset):
                raise _validation_error(
                    "determinize_subset_out_of_range",
                    "subsets must use source states",
                )
        keys = [
            (transition.symbol, transition.child_states)
            for transition in self.deterministic.transitions
        ]
        if len(set(keys)) != len(keys):
            raise _validation_error(
                "determinize_not_deterministic",
                "the deterministic machine must have unique transition keys",
            )
        if any(
            not 0 <= transition.target_state < self.deterministic.state_count
            or any(
                not 0 <= child < self.deterministic.state_count
                for child in transition.child_states
            )
            for transition in self.deterministic.transitions
        ):
            raise _validation_error(
                "determinize_transition_out_of_range",
                "deterministic transitions must use deterministic states",
            )
        source_finals = set(self.automaton.final_states)
        if set(self.deterministic.final_states) != {
            index
            for index, subset in enumerate(self.subset_map)
            if set(subset) & source_finals
        }:
            raise _validation_error(
                "determinize_finals_mismatch",
                "deterministic finals must meet the source finals through subsets",
            )
        if self.status == "COMPLETE":
            if (
                self.truncation_reason != "NONE"
                or not self.equivalence_claim
                or not self.sample_agreement
            ):
                raise _validation_error(
                    "determinize_complete_claim",
                    "a complete construction claims equivalence with agreement",
                )
            if self.closure_rows_checked != len(self.deterministic.transitions):
                raise _validation_error(
                    "determinize_closure_count",
                    "a complete construction replays every deterministic row",
                )
        else:
            if (
                self.truncation_reason == "NONE"
                or self.equivalence_claim
                or self.sample_agreement
            ):
                raise _validation_error(
                    "determinize_truncated_claim",
                    "a truncated construction must never claim equivalence",
                )
            if self.closure_rows_checked != 0 or self.sample_trees_checked != 0:
                raise _validation_error(
                    "determinize_truncated_evidence",
                    "a truncated construction carries no replayed evidence",
                )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


__all__ = [
    "AcceptedTreeCountRequest",
    "AcceptedTreeCountResult",
    "RegularTreeGrammarToAutomatonRequest",
    "RegularTreeGrammarToAutomatonResult",
    "TreeAutomatonBooleanProductRequest",
    "TreeAutomatonBooleanProductResult",
    "TreeAutomatonComplementRequest",
    "TreeAutomatonComplementResult",
    "TreeAutomatonMinimizeRequest",
    "TreeAutomatonMinimizeResult",
    "TreeAutomatonReachabilityRequest",
    "TreeAutomatonTrimRequest",
    "TreeAutomatonTrimResult",
    "TreeDeterminizeRequest",
    "TreeDeterminizeResult",
    "TreeRunRequest",
    "TreeRunResult",
    "TreeStateChartEntry",
    "TreeStateWitness",
]
