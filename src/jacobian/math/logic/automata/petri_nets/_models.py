"""Typed wire contracts for Petri net operations."""

from __future__ import annotations

from itertools import combinations
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.logic.automata.petri_nets.values import (
    MAX_PETRI_MARKING,
    MAX_PETRI_PLACES,
    MAX_PETRI_TRANSITIONS,
    MAX_REACHABILITY_FIRING_RECORDS,
    MAX_REACHABILITY_STATES,
    FiringSequence,
    Marking,
    PetriMarkingState,
    PetriNet,
    PetriPlaceSubset,
    PetriReachabilityEdge,
)
from jacobian.math.matrices.values import IntegerMatrix

MAX_SIPHON_TRAP_WORK = 20_000_000
MAX_SIPHON_TRAP_PLACES = 20
MAX_SIPHON_TRAP_FAMILY_OUTPUT_BYTES = 4_000_000
MAX_FIRING_SEQUENCE_LENGTH = 1024
MAX_CONCURRENT_STEP_OCCURRENCES = 1000
MAX_STATE_EQUATION_OCCURRENCES = 1000
MAX_STATE_EQUATION_TARGET_ABS = 64_001_000
MAX_MARKING_CONFLICT_PROFILE_OUTPUT_BYTES = 10 * 1024 * 1024
MAX_MARKING_CONFLICT_PROFILE_PAIRS = (
    MAX_PETRI_TRANSITIONS * (MAX_PETRI_TRANSITIONS - 1) // 2
)
MAX_MARKING_COMMUTATION_PROFILE_OUTPUT_BYTES = 10 * 1024 * 1024
MAX_MARKING_COMMUTATION_PROFILE_WORK = 100_000
MAX_REACHABLE_DEAD_MARKINGS_OUTPUT_BYTES = 10 * 1024 * 1024
MAX_TERMINAL_SCC_PROFILE_WORK = 1_000_000
MAX_TERMINAL_SCC_PROFILE_OUTPUT_BYTES = 10 * 1024 * 1024
TerminalSCCStateIndices = Annotated[
    tuple[int, ...],
    Field(min_length=1, max_length=MAX_REACHABILITY_STATES),
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"petri_net.{reason}", message)


def _require_marking_parent(net: PetriNet, marking: Marking) -> None:
    if not isinstance(marking, Marking):
        raise _validation_error("marking_type", "marking must be a Marking value")
    if marking.net is not None and marking.net != net:
        raise _validation_error(
            "marking_parent", "marking belongs to a different Petri net place axis"
        )


def _require_result_marking(
    net: PetriNet, marking: Marking, *, reason: str = "marking_length"
) -> None:
    _require_marking_parent(net, marking)
    if len(marking.tokens) != net.place_count:
        raise _validation_error(
            reason, "marking length must match the declared net place axis"
        )


class EnabledTransitionsRequest(StrictModel):
    """Find all enabled transitions at a marking."""

    net: PetriNet
    marking: Marking

    @model_validator(mode="after")
    def require_valid_marking_size(self) -> Self:
        _require_marking_parent(self.net, self.marking)
        if len(self.marking.tokens) != self.net.place_count:
            raise _validation_error(
                "marking_length", "marking length must match place_count"
            )
        return self


class EnabledTransitionsResult(StrictModel):
    """The set of enabled transition indices bound to the source marking."""

    net: PetriNet
    marking: Marking
    transitions: tuple[int, ...]

    @model_validator(mode="after")
    def require_source_shape(self) -> Self:
        _require_marking_parent(self.net, self.marking)
        if len(self.marking.tokens) != self.net.place_count:
            raise _validation_error(
                "marking_length", "marking length must match place_count"
            )
        if self.transitions != tuple(sorted(set(self.transitions))) or any(
            not 0 <= transition < self.net.transition_count
            for transition in self.transitions
        ):
            raise _validation_error("transition_axis", "transitions must be canonical")
        return self


class MarkingConflictProfileRequest(StrictModel):
    """Compare individual enabledness with pairwise simultaneous enabling."""

    net: PetriNet
    marking: Marking

    @model_validator(mode="after")
    def require_valid_marking_size(self) -> Self:
        _require_marking_parent(self.net, self.marking)
        if len(self.marking.tokens) != self.net.place_count:
            raise _validation_error(
                "marking_length", "marking length must match place_count"
            )
        return self


class MarkingConflictProfileResult(StrictModel):
    """Complete pairwise resource-conflict profile at one source marking.

    Pairs are unordered and contain distinct transitions that are each enabled
    alone. ``jointly_enabled_pairs`` can fire as one simultaneous step;
    ``conflicting_pairs`` cannot because their aggregate input demand exceeds
    the source marking at one or more places.
    """

    net: PetriNet
    marking: Marking
    enabled_transitions: tuple[int, ...] = Field(max_length=MAX_PETRI_TRANSITIONS)
    jointly_enabled_pairs: tuple[tuple[int, int], ...] = Field(
        max_length=MAX_MARKING_CONFLICT_PROFILE_PAIRS
    )
    conflicting_pairs: tuple[tuple[int, int], ...] = Field(
        max_length=MAX_MARKING_CONFLICT_PROFILE_PAIRS
    )

    @model_validator(mode="after")
    def require_profile_partition(self) -> Self:
        _require_result_marking(self.net, self.marking)
        enabled = self.enabled_transitions
        if len(enabled) > self.net.transition_count:
            raise _validation_error(
                "conflict_profile", "enabled transition count exceeds the net axis"
            )
        if enabled != tuple(sorted(set(enabled))) or any(
            not 0 <= transition < self.net.transition_count for transition in enabled
        ):
            raise _validation_error(
                "transition_axis", "enabled transitions must be canonical"
            )
        pair_bound = len(enabled) * (len(enabled) - 1) // 2
        if (
            len(self.jointly_enabled_pairs) > pair_bound
            or len(self.conflicting_pairs) > pair_bound
            or len(self.jointly_enabled_pairs) + len(self.conflicting_pairs)
            > pair_bound
        ):
            raise _validation_error(
                "conflict_profile", "pair-family cardinality exceeds enabled pairs"
            )
        output_bound = _marking_conflict_profile_output_bound(
            self.net,
            self.marking,
            enabled_count=len(enabled),
            pair_count=pair_bound,
        )
        if output_bound > MAX_MARKING_CONFLICT_PROFILE_OUTPUT_BYTES:
            raise _validation_error(
                "conflict_profile_output",
                "marking conflict profile exceeds the serialized result bound",
            )
        all_pairs = (*self.jointly_enabled_pairs, *self.conflicting_pairs)
        if (
            any(pair[0] >= pair[1] for pair in all_pairs)
            or tuple(sorted(set(self.jointly_enabled_pairs)))
            != self.jointly_enabled_pairs
            or tuple(sorted(set(self.conflicting_pairs))) != self.conflicting_pairs
            or set(self.jointly_enabled_pairs) & set(self.conflicting_pairs)
            or any(
                left not in enabled or right not in enabled for left, right in all_pairs
            )
        ):
            raise _validation_error(
                "conflict_profile",
                "pair families must be canonical subsets of enabled transitions",
            )
        expected = tuple(combinations(enabled, 2))
        if tuple(sorted(all_pairs)) != expected:
            raise _validation_error(
                "conflict_profile", "pair families must partition all enabled pairs"
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build from the admitted conflict-profile kernel without replay."""

        return cls.model_construct(**values)


class MarkingCommutationProfileRequest(StrictModel):
    """Replay both sequential orders of a distinct transition pair."""

    net: PetriNet
    marking: Marking
    transitions: tuple[int, int] = Field(
        description="An ordered pair of distinct transition indices on the net axis."
    )

    @model_validator(mode="after")
    def require_transition_pair(self) -> Self:
        _require_marking_parent(self.net, self.marking)
        if len(self.marking.tokens) != self.net.place_count:
            raise _validation_error(
                "marking_length", "marking length must match place_count"
            )
        first, second = self.transitions
        if first == second or any(
            not 0 <= transition < self.net.transition_count
            for transition in self.transitions
        ):
            raise _validation_error(
                "commutation_transition_pair",
                "commutation profiling requires two distinct transitions on the net axis",
            )
        return self


def _marking_commutation_profile_output_bound(net: PetriNet, marking: Marking) -> int:
    """Bound both two-step replay values before constructing either result."""

    net_bytes = len(net.model_dump_json().encode("utf-8"))
    source_marking_bytes = len(marking.model_dump_json().encode("utf-8"))
    # Every produced marking has the same parent convention and at most four
    # decimal digits per token (MAX_PETRI_MARKING is 1000). Allow each source
    # token spelling to grow to that width.
    derived_marking_bytes = source_marking_bytes + 3 * net.place_count
    places = net.place_count
    transitions = net.transition_count
    replay_bound = (
        net_bytes
        + source_marking_bytes
        + 3 * derived_marking_bytes  # up to two prefixes and one final marking
        + 5 * places  # full deficit profile including delimiters
        + 2 * places  # zero state-equation residual
        + 3 * transitions
        + 1  # two-entry Parikh vector
        + 7  # two transition indices and sequence delimiters
        + 1024  # keys, statuses, blocked metadata, and JSON punctuation
    )
    return 2 * replay_bound + 1024  # two order replays and outer profile fields


def _marking_conflict_profile_output_bound(
    net: PetriNet, marking: Marking, *, enabled_count: int, pair_count: int
) -> int:
    """Conservative UTF-8 JSON bound, computed before pair materialization."""

    source_bytes = len(net.model_dump_json().encode("utf-8")) + len(
        marking.model_dump_json().encode("utf-8")
    )
    # Every pair is two at-most-two-digit transition indices plus delimiters;
    # the extra byte accounts for array separators. The fixed allowance covers
    # object keys, array delimiters, and the empty source fields.
    return source_bytes + 512 + enabled_count * 4 + pair_count * 10


class FireTransitionRequest(StrictModel):
    """Fire one transition at a marking."""

    net: PetriNet
    marking: Marking
    transition: int = Field(ge=0)

    @model_validator(mode="after")
    def require_valid_marking_size(self) -> Self:
        _require_marking_parent(self.net, self.marking)
        if len(self.marking.tokens) != self.net.place_count:
            raise _validation_error(
                "marking_length", "marking length must match place_count"
            )
        if not 0 <= self.transition < self.net.transition_count:
            raise _validation_error("transition_index", "transition index out of range")
        return self


class FireTransitionResult(StrictModel):
    """Result of firing a transition, retaining its source context."""

    net: PetriNet
    marking: Marking
    transition: int = Field(ge=0)
    status: Literal["FIRED", "NOT_ENABLED", "ESCAPES_DECLARED_ENVELOPE"]
    new_marking: Marking | None = None
    envelope_escape: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def require_consistent_outcome(self) -> Self:
        _require_result_marking(self.net, self.marking)
        if not 0 <= self.transition < self.net.transition_count:
            raise _validation_error("transition_index", "transition index out of range")
        if self.status == "ESCAPES_DECLARED_ENVELOPE":
            if self.new_marking is not None or self.envelope_escape is None:
                raise _validation_error(
                    "escape_payload", "envelope escape must carry only the successor"
                )
            if all(token <= MAX_PETRI_MARKING for token in self.envelope_escape):
                raise _validation_error(
                    "escape_bound", "envelope escape must exceed the marking bound"
                )
        elif self.new_marking is None or self.envelope_escape is not None:
            raise _validation_error(
                "ordinary_payload", "ordinary firing outcomes must carry only a marking"
            )
        if self.new_marking is not None:
            _require_result_marking(
                self.net, self.new_marking, reason="new_marking_length"
            )
        if (
            self.envelope_escape is not None
            and len(self.envelope_escape) != self.net.place_count
        ):
            raise _validation_error(
                "escape_length", "envelope escape length must match place_count"
            )
        return self


class ConcurrentStepRequest(StrictModel):
    """Fire a transition multiset simultaneously at one marking."""

    net: PetriNet
    marking: Marking
    transition_counts: tuple[int, ...] = Field(
        max_length=MAX_PETRI_TRANSITIONS,
        description="Nonnegative transition multiplicities; this is one simultaneous step.",
    )

    @model_validator(mode="after")
    def require_step_shape(self) -> Self:
        _require_marking_parent(self.net, self.marking)
        if len(self.marking.tokens) != self.net.place_count:
            raise _validation_error(
                "marking_length", "marking length must match place_count"
            )
        if len(self.transition_counts) != self.net.transition_count:
            raise _validation_error(
                "step_transition_axis", "transition counts must match transition_count"
            )
        if any(count < 0 for count in self.transition_counts):
            raise _validation_error(
                "step_count_sign", "transition counts must be nonnegative"
            )
        return self


class ConcurrentStepResult(StrictModel):
    """Exact simultaneous-step outcome with aggregate resource profile."""

    net: PetriNet
    marking: Marking
    transition_counts: tuple[int, ...]
    required: tuple[int, ...]
    deficit: tuple[int, ...]
    status: Literal["FIRED", "NOT_ENABLED", "ESCAPES_DECLARED_ENVELOPE"]
    new_marking: Marking | None = None
    envelope_escape: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def require_result_shape(self) -> Self:
        _require_result_marking(self.net, self.marking)
        if (
            len(self.transition_counts) != self.net.transition_count
            or any(count < 0 for count in self.transition_counts)
            or sum(self.transition_counts) > MAX_CONCURRENT_STEP_OCCURRENCES
        ):
            raise _validation_error(
                "step_transition_axis",
                "transition counts must be admitted on the net axis",
            )
        if (
            len(self.required) != self.net.place_count
            or len(self.deficit) != self.net.place_count
        ):
            raise _validation_error(
                "step_place_axis",
                "required and deficit profiles must match place_count",
            )
        if any(value < 0 for value in (*self.required, *self.deficit)):
            raise _validation_error(
                "step_profile_sign", "required and deficit entries must be nonnegative"
            )
        expected_deficit = tuple(
            max(0, need - have)
            for need, have in zip(self.required, self.marking.tokens, strict=True)
        )
        if self.deficit != expected_deficit:
            raise _validation_error(
                "step_deficit",
                "deficit must equal required minus available, truncated at zero",
            )
        if self.status == "NOT_ENABLED":
            if (
                not any(self.deficit)
                or self.new_marking is not None
                or self.envelope_escape is not None
            ):
                raise _validation_error(
                    "step_blocked_payload",
                    "NOT_ENABLED requires a deficit and no successor",
                )
        elif self.status == "FIRED":
            if (
                any(self.deficit)
                or self.new_marking is None
                or self.envelope_escape is not None
            ):
                raise _validation_error(
                    "step_fired_payload",
                    "FIRED requires zero deficit and one successor",
                )
            _require_result_marking(
                self.net, self.new_marking, reason="new_marking_length"
            )
        else:
            if (
                any(self.deficit)
                or self.new_marking is not None
                or self.envelope_escape is None
            ):
                raise _validation_error(
                    "step_escape_payload",
                    "envelope escape carries only its successor tuple",
                )
            if len(self.envelope_escape) != self.net.place_count or all(
                token <= MAX_PETRI_MARKING for token in self.envelope_escape
            ):
                raise _validation_error(
                    "step_escape_shape",
                    "escape must exceed the token envelope on the place axis",
                )
        return self


class IncidenceMatrixRequest(StrictModel):
    """Compute the incidence matrix C = Post - Pre."""

    net: PetriNet


class IncidenceMatrixResult(StrictModel):
    """The incidence matrix bound to its net's place/transition axes."""

    net: PetriNet
    incidence: IntegerMatrix

    @model_validator(mode="after")
    def require_source_axes(self) -> Self:
        if (
            self.incidence.row_count != self.net.place_count
            or self.incidence.column_count != self.net.transition_count
        ):
            raise _validation_error(
                "incidence_axes", "incidence axes must match the net"
            )
        return self


class StateEquationRequest(StrictModel):
    """Compute the formal target M0 + C y for a nonnegative count vector."""

    net: PetriNet
    marking: Marking
    transition_counts: tuple[int, ...] = Field(
        max_length=MAX_PETRI_TRANSITIONS,
        description="Nonnegative transition counts; no firing sequence is asserted.",
    )

    @model_validator(mode="after")
    def require_state_equation_axes(self) -> Self:
        _require_marking_parent(self.net, self.marking)
        if len(self.marking.tokens) != self.net.place_count:
            raise _validation_error(
                "marking_length", "marking length must match place_count"
            )
        if len(self.transition_counts) != self.net.transition_count:
            raise _validation_error(
                "state_equation_transition_axis", "counts must match transition_count"
            )
        if any(count < 0 for count in self.transition_counts):
            raise _validation_error(
                "state_equation_count_sign", "transition counts must be nonnegative"
            )
        if sum(self.transition_counts) > MAX_STATE_EQUATION_OCCURRENCES:
            raise _validation_error(
                "state_equation_occurrence_bound",
                "total transition count exceeds the admitted bound",
            )
        return self


class StateEquationResult(StrictModel):
    """Formal integer target M0 + C y, which does not assert reachability."""

    net: PetriNet
    marking: Marking
    transition_counts: tuple[int, ...]
    target: tuple[int, ...] = Field(max_length=MAX_PETRI_PLACES)

    @model_validator(mode="after")
    def require_state_equation_axes(self) -> Self:
        _require_result_marking(self.net, self.marking)
        if (
            len(self.transition_counts) != self.net.transition_count
            or any(count < 0 for count in self.transition_counts)
            or sum(self.transition_counts) > MAX_STATE_EQUATION_OCCURRENCES
        ):
            raise _validation_error(
                "state_equation_transition_axis",
                "counts must be admitted on the net transition axis",
            )
        if len(self.target) != self.net.place_count:
            raise _validation_error(
                "state_equation_place_axis", "target must match the net place axis"
            )
        if any(
            type(coordinate) is not int
            or abs(coordinate) > MAX_STATE_EQUATION_TARGET_ABS
            for coordinate in self.target
        ):
            raise _validation_error(
                "state_equation_target_bound",
                "target coordinates must fit the admitted signed integer envelope",
            )
        return self


class ReachabilityRequest(StrictModel):
    """Compute the bounded reachability graph from an initial marking.

    Bounds the state space to avoid unbounded exploration.
    """

    net: PetriNet
    initial_marking: Marking
    max_states: int = Field(default=10000, ge=1, le=MAX_REACHABILITY_STATES)

    @model_validator(mode="after")
    def require_valid_marking_size(self) -> Self:
        _require_marking_parent(self.net, self.initial_marking)
        if len(self.initial_marking.tokens) != self.net.place_count:
            raise _validation_error(
                "marking_length", "marking length must match place_count"
            )
        return self


class ReachabilityResult(StrictModel):
    """The bounded reachability graph.

    Each state is a marking tuple. The graph is a mapping from marking
    to a list of (transition, resulting_marking) pairs.
    """

    net: PetriNet
    initial_marking: Marking
    max_states: int = Field(ge=1, le=MAX_REACHABILITY_STATES)
    states: tuple[PetriMarkingState, ...] = Field(max_length=MAX_REACHABILITY_STATES)
    edges: tuple[PetriReachabilityEdge, ...] = Field(
        max_length=MAX_REACHABILITY_FIRING_RECORDS
    )
    truncated: bool

    @model_validator(mode="after")
    def require_source_axes(self) -> Self:
        _require_result_marking(self.net, self.initial_marking)
        for state in self.states:
            _require_result_marking(
                self.net, state.marking, reason="state_marking_parent"
            )
        if tuple(state.state_index for state in self.states) != tuple(
            range(len(self.states))
        ):
            raise _validation_error(
                "state_axis", "states must use a complete index axis"
            )
        if any(
            state.place_axis != tuple(range(self.net.place_count))
            for state in self.states
        ):
            raise _validation_error(
                "state_marking_length", "state markings must match place_count"
            )
        if any(
            edge.source_state >= len(self.states)
            or edge.target_state >= len(self.states)
            or edge.transition >= self.net.transition_count
            for edge in self.edges
        ):
            raise _validation_error(
                "edge_axis", "reachability edges must use declared axes"
            )
        return self


class ReachableDeadMarkingsRequest(StrictModel):
    """List dead markings found in the bounded reachability exploration."""

    net: PetriNet
    initial_marking: Marking
    max_states: int = Field(default=10000, ge=1, le=MAX_REACHABILITY_STATES)

    @model_validator(mode="after")
    def require_valid_marking_size(self) -> Self:
        _require_marking_parent(self.net, self.initial_marking)
        if len(self.initial_marking.tokens) != self.net.place_count:
            raise _validation_error(
                "marking_length", "marking length must match place_count"
            )
        return self


class ReachableDeadMarkingsResult(StrictModel):
    """Dead markings among discovered states, with exploration truncation."""

    net: PetriNet
    initial_marking: Marking
    max_states: int = Field(ge=1, le=MAX_REACHABILITY_STATES)
    dead_markings: tuple[tuple[int, ...], ...] = Field(
        max_length=MAX_REACHABILITY_STATES
    )
    truncated: bool

    @model_validator(mode="after")
    def require_canonical_profile(self) -> Self:
        _require_result_marking(self.net, self.initial_marking)
        if any(len(marking) != self.net.place_count for marking in self.dead_markings):
            raise _validation_error(
                "dead_marking_axis", "dead markings must match the net place axis"
            )
        if len(self.dead_markings) > self.max_states:
            raise _validation_error(
                "dead_marking_count", "dead-marking count exceeds explored states"
            )
        if self.dead_markings != tuple(sorted(set(self.dead_markings))):
            raise _validation_error(
                "dead_markings", "dead markings must be sorted and unique"
            )
        if any(
            type(token) is not int or not 0 <= token <= MAX_PETRI_MARKING
            for marking in self.dead_markings
            for token in marking
        ):
            raise _validation_error(
                "dead_marking_token", "dead-marking tokens exceed the marking bound"
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build an admitted bounded profile without replaying enabledness."""

        return cls.model_construct(**values)


class ReachabilityTerminalSCCProfileRequest(StrictModel):
    """Find sink strongly connected components in one represented graph."""

    source_graph: ReachabilityResult


class ReachabilityTerminalSCCProfileResult(StrictModel):
    """Terminal SCCs of the exact represented reachability graph.

    When ``source_graph.truncated`` is true, terminality is only a property of
    the represented partial graph and says nothing about omitted successors.
    """

    source_graph: ReachabilityResult
    terminal_components: tuple[TerminalSCCStateIndices, ...] = Field(
        max_length=MAX_REACHABILITY_STATES
    )

    @model_validator(mode="after")
    def require_canonical_components(self) -> Self:
        state_count = len(self.source_graph.states)
        components = self.terminal_components
        if any(
            not component
            or component != tuple(sorted(set(component)))
            or any(not 0 <= state < state_count for state in component)
            for component in components
        ):
            raise _validation_error(
                "terminal_scc_components",
                "terminal SCCs must be nonempty canonical subsets of source states",
            )
        if components != tuple(sorted(components, key=lambda component: component[0])):
            raise _validation_error(
                "terminal_scc_order",
                "terminal SCCs must be ordered by least state index",
            )
        if len({state for component in components for state in component}) != sum(
            len(component) for component in components
        ):
            raise _validation_error(
                "terminal_scc_overlap", "terminal SCCs must be pairwise disjoint"
            )
        return self


class MarkingReachabilityRequest(StrictModel):
    """Find a firing sequence from an initial marking to a target marking."""

    net: PetriNet
    initial_marking: Marking
    target_marking: Marking
    max_states: int = Field(default=10000, ge=1, le=MAX_REACHABILITY_STATES)

    @model_validator(mode="after")
    def require_valid_marking_axes(self) -> Self:
        for field_name, marking in (
            ("initial_marking", self.initial_marking),
            ("target_marking", self.target_marking),
        ):
            _require_marking_parent(self.net, marking)
            if len(marking.tokens) != self.net.place_count:
                raise _validation_error(
                    "marking_length", f"{field_name} must match place_count"
                )
        return self


class MarkingReachabilityResult(StrictModel):
    """A firing witness or a truthful conclusion from target-directed BFS."""

    net: PetriNet
    initial_marking: Marking
    target_marking: Marking
    max_states: int = Field(ge=1, le=MAX_REACHABILITY_STATES)
    status: Literal["REACHABLE", "UNREACHABLE", "INCOMPLETE"]
    sequence: FiringSequence | None = None
    explored_state_count: int = Field(ge=1, le=MAX_REACHABILITY_STATES)
    incomplete_reasons: tuple[
        Literal["MARKING_LIMIT", "SEQUENCE_LIMIT", "STATE_LIMIT"], ...
    ] = ()

    @model_validator(mode="after")
    def require_result_shape(self) -> Self:
        _require_result_marking(self.net, self.initial_marking)
        _require_result_marking(
            self.net, self.target_marking, reason="target_marking_length"
        )
        if self.explored_state_count > self.max_states:
            raise _validation_error(
                "explored_state_count", "explored states must not exceed max_states"
            )
        if self.incomplete_reasons != tuple(sorted(set(self.incomplete_reasons))):
            raise _validation_error(
                "incomplete_reasons", "incomplete reasons must be sorted and unique"
            )
        if self.status == "REACHABLE":
            if self.sequence is None:
                raise _validation_error(
                    "reachability_witness", "REACHABLE requires a firing sequence"
                )
            if len(self.sequence.transitions) > MAX_FIRING_SEQUENCE_LENGTH:
                raise _validation_error(
                    "reachability_sequence_bound",
                    "sequence exceeds the replay operation bound",
                )
            if any(
                type(transition) is not int
                or not 0 <= transition < self.net.transition_count
                for transition in self.sequence.transitions
            ):
                raise _validation_error(
                    "reachability_sequence_axis",
                    "sequence transitions must use the declared transition axis",
                )
        elif self.sequence is not None:
            raise _validation_error(
                "reachability_sequence",
                "negative and incomplete results have no sequence",
            )
        if self.status == "UNREACHABLE" and self.incomplete_reasons:
            raise _validation_error(
                "unreachable_incomplete", "UNREACHABLE requires exhaustive search"
            )
        if self.status == "INCOMPLETE" and not self.incomplete_reasons:
            raise _validation_error(
                "incomplete_reason", "INCOMPLETE requires an envelope-cut reason"
            )
        return self


class SiphonTrapRequest(StrictModel):
    """Check for siphons and traps in a Petri net."""

    net: PetriNet


class SiphonTrapResult(StrictModel):
    """Minimal siphons and traps of the net.

    Each siphon/trap is represented as a tuple of place indices.
    """

    net: PetriNet
    siphons: tuple[PetriPlaceSubset, ...]
    traps: tuple[PetriPlaceSubset, ...]

    @model_validator(mode="after")
    def require_place_axes(self) -> Self:
        if any(
            place >= self.net.place_count
            for subset in (*self.siphons, *self.traps)
            for place in subset.places
        ):
            raise _validation_error("place_axis", "subsets must use the net place axis")
        return self


class SiphonTrapFamilyRequest(StrictModel):
    """Enumerate every nonempty siphon and trap under a finite envelope."""

    net: PetriNet


class SiphonTrapFamilyResult(StrictModel):
    """Complete nonempty siphon and trap families on one net's place axis."""

    net: PetriNet
    siphons: tuple[PetriPlaceSubset, ...]
    traps: tuple[PetriPlaceSubset, ...]

    @model_validator(mode="after")
    def require_canonical_families(self) -> Self:
        for family in (self.siphons, self.traps):
            keys = tuple((len(item.places), item.places) for item in family)
            if keys != tuple(sorted(set(keys))):
                raise _validation_error(
                    "siphon_trap_family_order",
                    "families must be sorted by cardinality and place indices",
                )
            if any(
                not item.places or item.places[-1] >= self.net.place_count
                for item in family
            ):
                raise _validation_error(
                    "siphon_trap_family_axis",
                    "families must contain nonempty subsets of the net place axis",
                )
        return self


class PlaceSetSupportRequest(StrictModel):
    """Check the exact transition-support profile of a place subset."""

    net: PetriNet
    places: PetriPlaceSubset

    @model_validator(mode="after")
    def require_place_axis(self) -> Self:
        if any(place >= self.net.place_count for place in self.places.places):
            raise _validation_error("place_axis", "subset must use the net place axis")
        return self


class PlaceSetSupportResult(StrictModel):
    """Exact producers/consumers and siphon/trap predicates for one subset."""

    net: PetriNet
    places: PetriPlaceSubset
    producers_into: tuple[int, ...]
    consumers_from: tuple[int, ...]
    siphon_offenders: tuple[int, ...]
    trap_offenders: tuple[int, ...]
    is_siphon: bool
    is_trap: bool

    @model_validator(mode="after")
    def require_canonical_profile(self) -> Self:
        if any(place >= self.net.place_count for place in self.places.places):
            raise _validation_error("place_axis", "subset must use the net place axis")
        for name in (
            "producers_into",
            "consumers_from",
            "siphon_offenders",
            "trap_offenders",
        ):
            values = getattr(self, name)
            if values != tuple(sorted(set(values))) or any(
                transition >= self.net.transition_count for transition in values
            ):
                raise _validation_error(
                    "transition_axis",
                    f"{name} must use the transition axis canonically",
                )
        if self.siphon_offenders != tuple(
            transition
            for transition in self.producers_into
            if transition not in self.consumers_from
        ):
            raise _validation_error(
                "siphon_profile",
                "siphon offenders must be producers not consuming from the subset",
            )
        if self.trap_offenders != tuple(
            transition
            for transition in self.consumers_from
            if transition not in self.producers_into
        ):
            raise _validation_error(
                "trap_profile",
                "trap offenders must be consumers not producing into the subset",
            )
        if self.is_siphon != (not self.siphon_offenders) or self.is_trap != (
            not self.trap_offenders
        ):
            raise _validation_error(
                "predicate_profile", "predicates must match their exact offender sets"
            )
        return self


class PlaceSetInitialMarkingProfileRequest(StrictModel):
    """Profile one selected place subset against a source marking."""

    net: PetriNet
    places: PetriPlaceSubset
    marking: Marking

    @model_validator(mode="after")
    def require_source_axes(self) -> Self:
        _require_marking_parent(self.net, self.marking)
        if len(self.marking.tokens) != self.net.place_count:
            raise _validation_error(
                "marking_length", "marking length must match place_count"
            )
        if any(place >= self.net.place_count for place in self.places.places):
            raise _validation_error("place_axis", "subset must use the net place axis")
        return self


class PlaceSetInitialMarkingProfileResult(StrictModel):
    """A place-set's initial token total and applicable persistence laws."""

    net: PetriNet
    places: PetriPlaceSubset
    marking: Marking
    selected_token_total: int = Field(ge=0)
    support_profile: PlaceSetSupportResult
    preservation_implications: tuple[
        Literal["EMPTY_SIPHON_REMAINS_EMPTY", "MARKED_TRAP_REMAINS_MARKED"], ...
    ]

    @model_validator(mode="after")
    def require_source_axes(self) -> Self:
        _require_marking_parent(self.net, self.marking)
        if len(self.marking.tokens) != self.net.place_count:
            raise _validation_error(
                "marking_length", "marking length must match place_count"
            )
        if any(place >= self.net.place_count for place in self.places.places):
            raise _validation_error("place_axis", "subset must use the net place axis")
        if (
            self.support_profile.net != self.net
            or self.support_profile.places != self.places
        ):
            raise _validation_error(
                "support_source_mismatch",
                "support profile must use the same net and selected places",
            )
        if self.preservation_implications != tuple(
            dict.fromkeys(self.preservation_implications)
        ):
            raise _validation_error(
                "preservation_order", "preservation implications must be unique"
            )
        return self


class FiringSequenceReplayRequest(StrictModel):
    """Replay a bounded transition sequence from a source marking."""

    net: PetriNet
    marking: Marking
    sequence: tuple[int, ...] = Field(
        default=(),
        max_length=MAX_FIRING_SEQUENCE_LENGTH,
        description=(
            "Ordered transition indices to fire from the source marking; "
            "a later transition cannot repair an earlier disabled one."
        ),
    )

    @model_validator(mode="after")
    def require_valid_sequence_axes(self) -> Self:
        _require_marking_parent(self.net, self.marking)
        if len(self.marking.tokens) != self.net.place_count:
            raise _validation_error(
                "marking_length", "marking length must match place_count"
            )
        if any(
            not 0 <= transition < self.net.transition_count
            for transition in self.sequence
        ):
            raise _validation_error(
                "transition_axis", "sequence transitions must use the net axis"
            )
        return self


class FiringSequenceReplayResult(StrictModel):
    """Closed firing-sequence replay: FIRES or first-blocked BLOCKED."""

    net: PetriNet
    marking: Marking
    sequence: tuple[int, ...] = Field(max_length=MAX_FIRING_SEQUENCE_LENGTH)
    status: Literal["FIRES", "BLOCKED"]
    prefix_markings: tuple[Marking, ...]
    final_marking: Marking | None = None
    parikh: tuple[int, ...]
    state_equation_residual: tuple[int, ...]
    blocked_index: int | None = None
    deficit: tuple[int, ...] | None = None
    first_deficient_place: int | None = None

    def _require_shared_axes(self) -> None:
        _require_result_marking(self.net, self.marking)
        for prefix in self.prefix_markings:
            _require_result_marking(self.net, prefix, reason="prefix_marking_parent")
        if self.final_marking is not None:
            _require_result_marking(
                self.net, self.final_marking, reason="final_marking_parent"
            )
        if any(
            not 0 <= transition < self.net.transition_count
            for transition in self.sequence
        ):
            raise _validation_error(
                "transition_axis", "sequence transitions must use the net axis"
            )
        if len(self.parikh) != self.net.transition_count or any(
            count < 0 for count in self.parikh
        ):
            raise _validation_error(
                "parikh_axis", "parikh must be a nonnegative transition multiset"
            )
        if len(self.state_equation_residual) != self.net.place_count:
            raise _validation_error(
                "residual_axis", "state-equation residual must use the place axis"
            )

    def _require_fires_payload(self) -> None:
        if (
            self.final_marking is None
            or self.blocked_index is not None
            or self.deficit is not None
            or self.first_deficient_place is not None
        ):
            raise _validation_error(
                "fires_payload", "FIRES must carry only a final marking"
            )
        if len(self.prefix_markings) != len(self.sequence):
            raise _validation_error(
                "fires_prefix", "FIRES must record one marking per transition"
            )
        if sum(self.parikh) != len(self.sequence):
            raise _validation_error(
                "fires_parikh", "FIRES parikh must count every transition"
            )
        if any(entry != 0 for entry in self.state_equation_residual):
            raise _validation_error(
                "fires_residual", "FIRES residual must witness state-equation equality"
            )
        if self.sequence and self.prefix_markings[-1] != self.final_marking:
            raise _validation_error(
                "fires_final", "FIRES final marking must extend the prefix ledger"
            )
        if not self.sequence and self.final_marking != self.marking:
            raise _validation_error(
                "fires_empty", "an empty sequence leaves the marking unchanged"
            )

    def _require_blocked_payload(self) -> None:
        if (
            self.final_marking is not None
            or self.blocked_index is None
            or self.deficit is None
            or self.first_deficient_place is None
        ):
            raise _validation_error(
                "blocked_payload", "BLOCKED must carry only its obstruction"
            )
        if not 0 <= self.blocked_index < len(self.sequence):
            raise _validation_error(
                "blocked_index", "blocked index must point into the sequence"
            )
        if len(self.prefix_markings) != self.blocked_index:
            raise _validation_error(
                "blocked_prefix", "BLOCKED must record exactly the fired prefix"
            )
        if sum(self.parikh) != self.blocked_index:
            raise _validation_error(
                "blocked_parikh", "BLOCKED parikh must count the fired prefix"
            )
        if len(self.deficit) != self.net.place_count or not any(
            entry > 0 for entry in self.deficit
        ):
            raise _validation_error(
                "blocked_deficit", "BLOCKED deficit must be a nonempty place profile"
            )
        if self.deficit[self.first_deficient_place] <= 0 or any(
            entry > 0 for entry in self.deficit[: self.first_deficient_place]
        ):
            raise _validation_error(
                "blocked_first",
                "first deficient place must be the first positive deficit",
            )

    @model_validator(mode="after")
    def require_branch_consistency(self) -> Self:
        self._require_shared_axes()
        if self.status == "FIRES":
            self._require_fires_payload()
        else:
            self._require_blocked_payload()
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build from the admitted replay kernel without replaying its math."""
        return cls.model_construct(**values)


class PumpingWitnessRequest(StrictModel):
    """Check a supplied firing sequence for strict componentwise growth."""

    net: PetriNet
    marking: Marking
    sequence: tuple[int, ...] = Field(
        default=(),
        max_length=MAX_FIRING_SEQUENCE_LENGTH,
        description="A concrete sequence proposed as a repeatable growth witness.",
    )

    @model_validator(mode="after")
    def require_source_axes(self) -> Self:
        _require_marking_parent(self.net, self.marking)
        if len(self.marking.tokens) != self.net.place_count:
            raise _validation_error("marking_length", "marking must match place_count")
        if any(
            not 0 <= transition < self.net.transition_count
            for transition in self.sequence
        ):
            raise _validation_error(
                "transition_axis", "sequence transitions must use the net axis"
            )
        return self


class PumpingWitnessResult(StrictModel):
    """Replay evidence and the exact componentwise growth conclusion."""

    net: PetriNet
    marking: Marking
    sequence: tuple[int, ...] = Field(max_length=MAX_FIRING_SEQUENCE_LENGTH)
    replay: FiringSequenceReplayResult
    status: Literal["PUMPING_WITNESS", "FIRES_WITHOUT_GROWTH", "BLOCKED"]
    growth: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def require_consistent_growth(self) -> Self:
        _require_result_marking(self.net, self.marking)
        if self.replay.net != self.net or self.replay.marking != self.marking:
            raise _validation_error(
                "pumping_source", "replay must use the same net and source marking"
            )
        if self.replay.sequence != self.sequence:
            raise _validation_error(
                "pumping_sequence", "replay must use the proposed sequence"
            )
        if self.status == "BLOCKED":
            if self.replay.status != "BLOCKED" or self.growth is not None:
                raise _validation_error(
                    "pumping_blocked",
                    "blocked result requires blocked replay and no growth",
                )
        else:
            if self.replay.status != "FIRES" or self.replay.final_marking is None:
                raise _validation_error(
                    "pumping_replay", "growth classification requires a firing replay"
                )
            delta = tuple(
                final - initial
                for initial, final in zip(
                    self.marking.tokens, self.replay.final_marking.tokens, strict=True
                )
            )
            if self.status == "PUMPING_WITNESS":
                if (
                    self.growth != delta
                    or not any(delta)
                    or any(value < 0 for value in delta)
                ):
                    raise _validation_error(
                        "pumping_growth",
                        "witness growth must be nonnegative and nonzero",
                    )
            elif self.growth is not None or (
                all(value >= 0 for value in delta) and any(delta)
            ):
                raise _validation_error(
                    "pumping_non_growth",
                    "non-witness result must not have strict componentwise growth",
                )
        return self


class MarkingCommutationProfileResult(StrictModel):
    """Exact sequential diamond evidence for both orders of one transition pair.

    The two embedded replay values carry the common source net and marking,
    each intermediate marking or first blocking deficit, and each final
    marking when that order fires. This profile makes no simultaneous-step
    or global-confluence claim.
    """

    first_then_second: FiringSequenceReplayResult
    second_then_first: FiringSequenceReplayResult
    both_orders_fire: bool
    same_target: bool

    @model_validator(mode="after")
    def require_exact_local_diamond_profile(self) -> Self:
        first = self.first_then_second
        second = self.second_then_first
        if first.net != second.net or first.marking != second.marking:
            raise _validation_error(
                "commutation_source_mismatch",
                "both replay orders must use the same source net and marking",
            )
        if (
            len(first.sequence) != 2
            or len(second.sequence) != 2
            or first.sequence[0] == first.sequence[1]
            or second.sequence != (first.sequence[1], first.sequence[0])
        ):
            raise _validation_error(
                "commutation_sequence_pair",
                "replay sequences must be the two orders of a distinct transition pair",
            )
        both_fire = first.status == second.status == "FIRES"
        if both_fire and first.final_marking != second.final_marking:
            raise _validation_error(
                "commutation_targets",
                "both executable orders must reach the same additive firing target",
            )
        same_target = bool(both_fire and first.final_marking == second.final_marking)
        if self.both_orders_fire != both_fire or self.same_target != same_target:
            raise _validation_error(
                "commutation_profile_claim",
                "commutation flags must agree with the two replay outcomes",
            )
        if (
            _marking_commutation_profile_output_bound(first.net, first.marking)
            > MAX_MARKING_COMMUTATION_PROFILE_OUTPUT_BYTES
        ):
            raise _validation_error(
                "commutation_profile_output",
                "marking commutation profile exceeds the serialized result bound",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build from two admitted replay kernels without replaying validation math."""

        return cls.model_construct(**values)


class PetriInvariantsRequest(StrictModel):
    """Compute P-invariants and T-invariants as exact integer modules."""

    net: PetriNet


class PetriNetRelabelingRequest(StrictModel):
    """Permute place and transition axes of a weighted Petri net.

    Each map sends a source index to its index on the relabeled axis. Optional
    labels are carried with their elements; unlabeled axes remain unlabeled.
    """

    net: PetriNet
    place_source_to_target: tuple[int, ...] = Field(max_length=MAX_PETRI_PLACES)
    transition_source_to_target: tuple[int, ...] = Field(
        max_length=MAX_PETRI_TRANSITIONS
    )


class PetriNetRelabelingResult(StrictModel):
    """An exact isomorphism between a net and its permuted-axis presentation."""

    source_net: PetriNet
    target_net: PetriNet
    place_source_to_target: tuple[int, ...] = Field(max_length=MAX_PETRI_PLACES)
    transition_source_to_target: tuple[int, ...] = Field(
        max_length=MAX_PETRI_TRANSITIONS
    )

    @model_validator(mode="after")
    def require_relabeling_shape(self) -> Self:
        if (self.source_net.place_count, self.source_net.transition_count) != (
            self.target_net.place_count,
            self.target_net.transition_count,
        ):
            raise _validation_error(
                "relabel_axes", "source and target net axes must have equal sizes"
            )
        for axis_name, mapping, size in (
            ("place", self.place_source_to_target, self.source_net.place_count),
            (
                "transition",
                self.transition_source_to_target,
                self.source_net.transition_count,
            ),
        ):
            if len(mapping) != size or tuple(sorted(mapping)) != tuple(range(size)):
                raise _validation_error(
                    f"relabel_{axis_name}_bijection",
                    f"{axis_name} map must be a bijection of its complete axis",
                )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build the result after the kernel establishes the relabeling law."""
        return cls.model_construct(**values)


class PetriInvariantsResult(PetriInvariantsRequest):
    """P/T-invariant bases with their incidence rank profile.

    Deserialization checks only the canonical shape: incidence axes match
    the net, every vector uses its axis, each basis is sign-normalized,
    sorted, and unique with the nullity the Smith rank predicts. The
    owner-local kernel computes the exact bases.
    """

    incidence: IntegerMatrix
    incidence_rank: int = Field(ge=0)
    p_invariants: tuple[tuple[int, ...], ...] = Field(default=())
    t_invariants: tuple[tuple[int, ...], ...] = Field(default=())

    @model_validator(mode="after")
    def require_canonical_invariant_shape(self) -> Self:
        if (
            self.incidence.row_count != self.net.place_count
            or self.incidence.column_count != self.net.transition_count
        ):
            raise _validation_error(
                "invariants_incidence_axes", "incidence axes must match the net"
            )
        if self.incidence_rank > min(self.net.place_count, self.net.transition_count):
            raise _validation_error(
                "invariants_rank_bound",
                "the incidence rank fits inside the net axes",
            )
        if len(self.p_invariants) != self.net.place_count - self.incidence_rank:
            raise _validation_error(
                "invariants_p_nullity",
                "the P-basis size must match the left nullity",
            )
        if len(self.t_invariants) != self.net.transition_count - self.incidence_rank:
            raise _validation_error(
                "invariants_t_nullity",
                "the T-basis size must match the right nullity",
            )
        for vectors, ambient in (
            (self.p_invariants, self.net.place_count),
            (self.t_invariants, self.net.transition_count),
        ):
            if vectors != tuple(sorted(set(vectors))):
                raise _validation_error(
                    "invariants_not_canonical",
                    "invariant bases must be sorted and unique",
                )
            for vector in vectors:
                if len(vector) != ambient or not any(vector):
                    raise _validation_error(
                        "invariants_axis",
                        "invariant vectors must use their axis nontrivially",
                    )
                if next(value for value in vector if value) < 0:
                    raise _validation_error(
                        "invariants_sign",
                        "invariant vectors must be sign-normalized",
                    )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


__all__ = [
    "MAX_CONCURRENT_STEP_OCCURRENCES",
    "MAX_FIRING_SEQUENCE_LENGTH",
    "MAX_MARKING_COMMUTATION_PROFILE_OUTPUT_BYTES",
    "MAX_MARKING_COMMUTATION_PROFILE_WORK",
    "MAX_REACHABLE_DEAD_MARKINGS_OUTPUT_BYTES",
    "MAX_SIPHON_TRAP_FAMILY_OUTPUT_BYTES",
    "MAX_SIPHON_TRAP_WORK",
    "MAX_STATE_EQUATION_OCCURRENCES",
    "MAX_STATE_EQUATION_TARGET_ABS",
    "ConcurrentStepRequest",
    "ConcurrentStepResult",
    "EnabledTransitionsRequest",
    "EnabledTransitionsResult",
    "FireTransitionRequest",
    "FireTransitionResult",
    "FiringSequenceReplayRequest",
    "FiringSequenceReplayResult",
    "IncidenceMatrixRequest",
    "IncidenceMatrixResult",
    "MarkingCommutationProfileRequest",
    "MarkingCommutationProfileResult",
    "MarkingConflictProfileRequest",
    "MarkingConflictProfileResult",
    "PetriInvariantsRequest",
    "PetriInvariantsResult",
    "PetriMarkingState",
    "PetriNetRelabelingRequest",
    "PetriNetRelabelingResult",
    "PetriPlaceSubset",
    "PetriReachabilityEdge",
    "PlaceSetInitialMarkingProfileRequest",
    "PlaceSetInitialMarkingProfileResult",
    "PlaceSetSupportRequest",
    "PlaceSetSupportResult",
    "ReachabilityRequest",
    "ReachabilityResult",
    "ReachableDeadMarkingsRequest",
    "ReachableDeadMarkingsResult",
    "SiphonTrapFamilyRequest",
    "SiphonTrapFamilyResult",
    "SiphonTrapRequest",
    "SiphonTrapResult",
    "StateEquationRequest",
    "StateEquationResult",
]
