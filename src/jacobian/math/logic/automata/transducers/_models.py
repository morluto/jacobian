"""Typed wire contracts for exact bounded finite-state transducers."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.logic.automata.transducers.values import (
    MAX_FST_ALPHABET_ID_LENGTH,
    MAX_FST_RESULT_WORD_LENGTH,
    MAX_FST_STATES,
    MAX_FST_WORD_LENGTH,
    FiniteAlphabet,
    RationalTransducer,
    SubsequentialTransducer,
    alphabet_parent_mismatch,
)
from jacobian.math.logic.languages.words.values import WordMorphism


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by transducer contracts."""

    return PydanticCustomError(f"finite_state_transducer.{reason}", message)


class SubseqRunRequest(StrictModel):
    transducer: SubsequentialTransducer
    word: tuple[int, ...] = Field(
        max_length=MAX_FST_WORD_LENGTH,
        description=(
            "Input symbol indices in the request's transducer input alphabet order. "
            "This is request-scoped index data, not an alphabet-parented FiniteWord."
        ),
    )


class SubseqIdentityRequest(StrictModel):
    """Construct the identity function on one exact ordered alphabet."""

    alphabet: FiniteAlphabet
    alphabet_id: str | None = Field(default=None, max_length=MAX_FST_ALPHABET_ID_LENGTH)


class WordMorphismToSubseqRequest(StrictModel):
    """A total word morphism to realize as a one-state subsequential machine."""

    morphism: WordMorphism


class SubseqRunResult(SubseqRunRequest):
    """A bounded subsequential-run outcome.

    Deserialization checks only the canonical shape of a claimed outcome.  The
    owner-local verifier replays independently supplied claims; trusted kernel
    output is constructed through ``_from_kernel`` below.
    """

    status: Literal["OUTPUT", "UNDEFINED_TRANSITION", "NONFINAL_DOMAIN_STATE"]
    output: tuple[int, ...] = Field(max_length=MAX_FST_RESULT_WORD_LENGTH)
    final_state: int = Field(ge=0, lt=MAX_FST_STATES)
    undefined_position: int | None = None
    partial_output: tuple[int, ...] = Field(max_length=MAX_FST_RESULT_WORD_LENGTH)
    state_trace: tuple[int, ...] = Field(
        max_length=MAX_FST_WORD_LENGTH + 1,
        description="Initial state followed by the state after each consumed symbol.",
    )
    transition_outputs: tuple[
        Annotated[tuple[int, ...], Field(max_length=MAX_FST_WORD_LENGTH)], ...
    ] = Field(
        max_length=MAX_FST_WORD_LENGTH,
        description="One output word per successfully consumed input symbol.",
    )
    cumulative_outputs: tuple[
        Annotated[tuple[int, ...], Field(max_length=MAX_FST_RESULT_WORD_LENGTH)], ...
    ] = Field(
        max_length=MAX_FST_WORD_LENGTH + 1,
        description=(
            "Transition-output prefix after each input prefix, including the empty "
            "prefix; final output is separate."
        ),
    )
    final_output: tuple[int, ...] = Field(
        max_length=MAX_FST_WORD_LENGTH,
        description="Final-state output on OUTPUT; empty otherwise.",
    )
    obstruction_position: int | None = Field(
        default=None,
        description="Undefined input position, or input length for a nonfinal terminal state.",
    )
    obstruction_state: int | None = Field(default=None)
    obstruction_symbol: int | None = Field(
        default=None, description="Input symbol lacking a transition, when undefined."
    )

    @model_validator(mode="after")
    def require_canonical_outcome_shape(self) -> Self:
        if not 0 <= self.final_state < self.transducer.state_count:
            raise _validation_error(
                "run_final_state_out_of_range", "final state is outside the transducer"
            )
        output_rows = (
            self.output,
            self.partial_output,
            self.final_output,
            *self.transition_outputs,
            *self.cumulative_outputs,
        )
        if any(
            not 0 <= symbol < self.transducer.output_alphabet_size
            for row in output_rows
            for symbol in row
        ):
            raise _validation_error(
                "run_output_symbol_out_of_range",
                "run output contains a symbol outside the output alphabet",
            )
        if any(
            not 0 <= state < self.transducer.state_count for state in self.state_trace
        ):
            raise _validation_error(
                "run_state_trace_out_of_range", "run trace contains an undeclared state"
            )
        if (
            len(self.state_trace) != len(self.transition_outputs) + 1
            or len(self.cumulative_outputs) != len(self.state_trace)
            or not self.state_trace
            or self.state_trace[0] != self.transducer.initial_state
            or self.state_trace[-1] != self.final_state
        ):
            raise _validation_error(
                "run_trace_shape", "run trace arrays must align with consumed prefixes"
            )
        if self.status == "OUTPUT":
            valid = (
                self.undefined_position is None
                and self.obstruction_position is None
                and self.obstruction_state is None
                and self.obstruction_symbol is None
                and len(self.state_trace) == len(self.word) + 1
                and not self.partial_output
                and len(self.transition_outputs) == len(self.word)
                and len(self.final_output) <= MAX_FST_WORD_LENGTH
                and len(self.output) <= MAX_FST_RESULT_WORD_LENGTH
            )
        elif self.status == "UNDEFINED_TRANSITION":
            valid = (
                not self.output
                and self.undefined_position is not None
                and 0 <= self.undefined_position < len(self.word)
                and self.obstruction_position == self.undefined_position
                and self.obstruction_state == self.final_state
                and self.obstruction_symbol == self.word[self.undefined_position]
                and len(self.transition_outputs) == self.undefined_position
                and len(self.state_trace) == self.undefined_position + 1
                and not self.final_output
            )
        else:
            valid = (
                not self.output
                and self.undefined_position is None
                and self.obstruction_position == len(self.word)
                and self.obstruction_state == self.final_state
                and self.obstruction_symbol is None
                and len(self.transition_outputs) == len(self.word)
                and len(self.state_trace) == len(self.word) + 1
                and not self.final_output
            )
        if not valid:
            raise _validation_error(
                "run_outcome_shape",
                "status and run outcome fields have an incompatible shape",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: SubseqRunRequest,
        *,
        status: Literal["OUTPUT", "UNDEFINED_TRANSITION", "NONFINAL_DOMAIN_STATE"],
        output: tuple[int, ...],
        final_state: int,
        undefined_position: int | None,
        partial_output: tuple[int, ...],
        state_trace: tuple[int, ...],
        transition_outputs: tuple[tuple[int, ...], ...],
        cumulative_outputs: tuple[tuple[int, ...], ...],
        final_output: tuple[int, ...],
        obstruction_position: int | None,
        obstruction_state: int | None,
        obstruction_symbol: int | None,
    ) -> Self:
        """Construct a run outcome emitted by the trusted owner-local kernel."""

        return cls.model_construct(
            transducer=request.transducer,
            word=request.word,
            status=status,
            output=output,
            final_state=final_state,
            undefined_position=undefined_position,
            partial_output=partial_output,
            state_trace=state_trace,
            transition_outputs=transition_outputs,
            cumulative_outputs=cumulative_outputs,
            final_output=final_output,
            obstruction_position=obstruction_position,
            obstruction_state=obstruction_state,
            obstruction_symbol=obstruction_symbol,
        )


class ComposeRequest(StrictModel):
    first: SubsequentialTransducer
    second: SubsequentialTransducer

    @model_validator(mode="after")
    def require_composable_alphabets(self) -> Self:
        mismatch = alphabet_parent_mismatch(
            self.first.output_alphabet_id,
            self.first.output_alphabet,
            self.second.input_alphabet_id,
            self.second.input_alphabet,
        )
        if mismatch is not None:
            raise _validation_error(*mismatch)
        if self.first.output_alphabet_size != self.second.input_alphabet_size:
            raise _validation_error(
                "composition_alphabet_mismatch",
                "first output alphabet must match second input alphabet",
            )
        return self


class ComposeResult(ComposeRequest):
    transducer: SubsequentialTransducer

    @model_validator(mode="after")
    def require_composite_shape(self) -> Self:
        if (
            self.transducer.input_alphabet_size != self.first.input_alphabet_size
            or self.transducer.output_alphabet_size != self.second.output_alphabet_size
            or self.transducer.input_alphabet_id != self.first.input_alphabet_id
            or self.transducer.output_alphabet_id != self.second.output_alphabet_id
            or self.transducer.input_alphabet != self.first.input_alphabet
            or self.transducer.output_alphabet != self.second.output_alphabet
            or self.transducer.state_count
            > self.first.state_count * self.second.state_count
        ):
            raise _validation_error(
                "composition_result_shape",
                "composite transducer must retain composition alphabets and product-state bound",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, request: ComposeRequest, *, transducer: SubsequentialTransducer
    ) -> Self:
        """Construct a composition emitted by the trusted owner-local kernel."""

        return cls.model_construct(
            first=request.first,
            second=request.second,
            transducer=transducer,
        )


class TrimRequest(StrictModel):
    transducer: SubsequentialTransducer


class ReachableStatesRequest(StrictModel):
    """Find shortest input paths to each reachable transducer state."""

    transducer: SubsequentialTransducer


class ReachableStateWitness(StrictModel):
    """One shortest path witness from the transducer's initial state."""

    state: int = Field(ge=0, lt=MAX_FST_STATES)
    input_word: tuple[int, ...] = Field(max_length=MAX_FST_STATES - 1)
    output_word: tuple[int, ...] = Field(
        max_length=MAX_FST_STATES * MAX_FST_WORD_LENGTH
    )
    state_trace: tuple[int, ...] = Field(max_length=MAX_FST_STATES)


class ReachableStatesResult(ReachableStatesRequest):
    """Reachable states with canonical shortest transition-output traces.

    Rows are ordered by source state ID. A kernel establishes that each path
    is valid and shortest; this carrier checks only its bounded shape.
    """

    witnesses: tuple[ReachableStateWitness, ...] = Field(max_length=MAX_FST_STATES)

    @model_validator(mode="after")
    def require_canonical_witness_shape(self) -> Self:
        states = tuple(row.state for row in self.witnesses)
        if (
            states != tuple(sorted(set(states)))
            or self.transducer.initial_state not in states
        ):
            raise _validation_error(
                "reachable_witness_order",
                "witnesses must be state-sorted and include the initial state",
            )
        for row in self.witnesses:
            if (
                row.state >= self.transducer.state_count
                or row.state_trace[0] != self.transducer.initial_state
                or row.state_trace[-1] != row.state
                or len(row.state_trace) != len(row.input_word) + 1
                or any(
                    not 0 <= symbol < self.transducer.input_alphabet_size
                    for symbol in row.input_word
                )
                or any(
                    not 0 <= symbol < self.transducer.output_alphabet_size
                    for symbol in row.output_word
                )
                or any(
                    not 0 <= state < self.transducer.state_count
                    for state in row.state_trace
                )
            ):
                raise _validation_error(
                    "reachable_witness_shape",
                    "each witness must have bounded symbols and aligned source-state endpoints",
                )
        initial = next(
            row for row in self.witnesses if row.state == self.transducer.initial_state
        )
        if (
            initial.input_word
            or initial.output_word
            or initial.state_trace != (initial.state,)
        ):
            raise _validation_error(
                "reachable_initial_witness",
                "the initial state witness must be the empty path",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: ReachableStatesRequest,
        *,
        witnesses: tuple[ReachableStateWitness, ...],
    ) -> Self:
        return cls.model_construct(transducer=request.transducer, witnesses=witnesses)


class TrimResult(TrimRequest):
    """A subsequential transducer restricted to its live states.

    Deserialization checks only the canonical shape of a claimed restriction:
    the two state maps are inverse bijections and every retained index is in
    range.  The owner-local kernel establishes that the kept states are exactly
    the reachable and coaccessible ones; trusted kernel output is constructed
    through ``_from_kernel`` below.
    """

    trimmed: SubsequentialTransducer
    old_to_new: tuple[tuple[int, int], ...] = Field(
        description="Old-state to new-state pairs in strictly increasing old-state order."
    )
    new_to_old: tuple[int, ...] = Field(
        description="Old state retained at each new-state index."
    )

    @model_validator(mode="after")
    def require_canonical_trim_shape(self) -> Self:
        source_count = self.transducer.state_count
        if self.new_to_old != tuple(sorted(set(self.new_to_old))):
            raise _validation_error(
                "trim_new_to_old_not_canonical",
                "new_to_old must list each retained old state once in increasing order",
            )
        if any(not 0 <= old < source_count for old in self.new_to_old):
            raise _validation_error(
                "trim_new_to_old_out_of_range",
                "new_to_old lists an old state outside the source transducer",
            )
        olds = [old for old, _ in self.old_to_new]
        news = [new for _, new in self.old_to_new]
        if tuple(olds) != tuple(sorted(set(olds))):
            raise _validation_error(
                "trim_old_to_new_not_canonical",
                "old_to_new must carry each mapped old state once in increasing order",
            )
        if sorted(news) != list(range(len(self.new_to_old))):
            raise _validation_error(
                "trim_new_ids_not_canonical",
                "old_to_new must address exactly the new-state range",
            )
        if set(olds) != set(self.new_to_old):
            raise _validation_error(
                "trim_maps_disagree",
                "old_to_new and new_to_old must describe the same retained states",
            )
        if (
            self.trimmed.input_alphabet_size != self.transducer.input_alphabet_size
            or self.trimmed.output_alphabet_size != self.transducer.output_alphabet_size
            or self.trimmed.input_alphabet_id != self.transducer.input_alphabet_id
            or self.trimmed.output_alphabet_id != self.transducer.output_alphabet_id
            or self.trimmed.input_alphabet != self.transducer.input_alphabet
            or self.trimmed.output_alphabet != self.transducer.output_alphabet
        ):
            raise _validation_error(
                "trim_alphabet_mismatch",
                "the trimmed transducer must retain both source alphabets",
            )
        if not self.new_to_old:
            if (
                self.trimmed.state_count != 1
                or self.trimmed.initial_state != 0
                or self.trimmed.transitions
                or self.trimmed.final_outputs
            ):
                raise _validation_error(
                    "trim_empty_restriction_shape",
                    "an empty restriction must be the canonical single-state transducer",
                )
            return self
        if self.trimmed.state_count != len(self.new_to_old):
            raise _validation_error(
                "trim_state_count_mismatch",
                "the trimmed state count must match the retained-state map",
            )
        forward = dict(self.old_to_new)
        if self.transducer.initial_state not in forward:
            raise _validation_error(
                "trim_initial_state_dropped",
                "a nonempty restriction must retain the source initial state",
            )
        if self.trimmed.initial_state != forward[self.transducer.initial_state]:
            raise _validation_error(
                "trim_initial_state_mismatch",
                "the trimmed initial state must follow the old-to-new map",
            )
        if any(
            not 0 <= transition.source < self.trimmed.state_count
            or not 0 <= transition.target < self.trimmed.state_count
            for transition in self.trimmed.transitions
        ):
            raise _validation_error(
                "trim_transition_out_of_range",
                "a trimmed transition leaves the retained state range",
            )
        if any(
            not 0 <= final.state < self.trimmed.state_count
            for final in self.trimmed.final_outputs
        ):
            raise _validation_error(
                "trim_final_output_out_of_range",
                "a trimmed final output leaves the retained state range",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: TrimRequest,
        *,
        trimmed: SubsequentialTransducer,
        old_to_new: dict[int, int],
    ) -> Self:
        """Construct a restriction emitted by the trusted owner-local kernel."""

        pairs = tuple(sorted(old_to_new.items()))
        new_to_old = tuple(old for _, old in sorted((new, old) for old, new in pairs))
        return cls.model_construct(
            transducer=request.transducer,
            trimmed=trimmed,
            old_to_new=pairs,
            new_to_old=new_to_old,
        )


class StatePairDistinguishability(StrictModel):
    """One row of a Myhill-Nerode-style state-distinguishability table.

    Deserialization checks only the canonical pair shape. The owner-local
    kernel establishes that ``equivalent`` agrees with the minimized
    partition and that every inequivalent pair carries a word separating
    the two states.
    """

    first_state: int = Field(ge=0, lt=MAX_FST_STATES)
    second_state: int = Field(ge=0, lt=MAX_FST_STATES)
    equivalent: bool
    witness_word: tuple[int, ...] = Field(default=(), max_length=MAX_FST_WORD_LENGTH)

    @model_validator(mode="after")
    def require_canonical_pair_shape(self) -> Self:
        if not self.first_state < self.second_state:
            raise _validation_error(
                "distinguishability_pair_order",
                "distinguishability pairs must list two distinct states in order",
            )
        if self.equivalent and self.witness_word:
            raise _validation_error(
                "distinguishability_witness_shape",
                "an equivalent pair cannot carry a separating word",
            )
        return self


class MinimizeRequest(StrictModel):
    transducer: SubsequentialTransducer
    sample_max_length: int = Field(default=5, ge=0, le=12)


class MinimizeResult(MinimizeRequest):
    """A minimized subsequential transducer with its equivalence certificate.

    Deserialization checks only the canonical shape of the quotient: the
    partition covers exactly the mapped (live) states, the state maps are
    inverse on representatives, the distinguishability table covers every
    live pair exactly once with ``equivalent`` agreeing with the partition,
    and the replayed sample size matches the requested sample budget. The
    owner-local kernel establishes that the quotient preserves the realized
    partial function.
    """

    minimized: SubsequentialTransducer
    old_to_new: tuple[int, ...]
    new_to_old: tuple[int, ...]
    partition: tuple[tuple[int, ...], ...]
    distinguishability: tuple[StatePairDistinguishability, ...]
    sample_words_checked: int = Field(ge=0)
    sample_agreement: bool

    @model_validator(mode="after")
    def require_canonical_minimize_shape(self) -> Self:
        if (
            self.minimized.input_alphabet_size != self.transducer.input_alphabet_size
            or self.minimized.output_alphabet_size
            != self.transducer.output_alphabet_size
            or self.minimized.input_alphabet_id != self.transducer.input_alphabet_id
            or self.minimized.output_alphabet_id != self.transducer.output_alphabet_id
            or self.minimized.input_alphabet != self.transducer.input_alphabet
            or self.minimized.output_alphabet != self.transducer.output_alphabet
        ):
            raise _validation_error(
                "minimize_alphabet_mismatch",
                "the minimized transducer must retain both source alphabets",
            )
        if not self.partition:
            if (
                self.minimized.state_count != 1
                or self.minimized.initial_state != 0
                or self.minimized.transitions
                or self.minimized.final_outputs
                or self.old_to_new != (-1,) * self.transducer.state_count
                or self.new_to_old
                or self.distinguishability
            ):
                raise _validation_error(
                    "minimize_empty_restriction_shape",
                    "an empty restriction must be the canonical single-state transducer",
                )
            self._require_sample_agreement()
            return self
        live = self._require_quotient_shape()
        self._require_table_shape(live)
        self._require_sample_agreement()
        return self

    def _require_quotient_shape(self) -> set[int]:
        """Check the quotient maps and partition; return the live states."""

        if not (
            self.minimized.state_count == len(self.partition) == len(self.new_to_old)
        ):
            raise _validation_error(
                "minimize_block_count_mismatch",
                "partition, representatives, and minimized states must agree",
            )
        if len(self.old_to_new) != self.transducer.state_count:
            raise _validation_error(
                "minimize_map_axis",
                "old-to-new map must cover every source state",
            )
        if any(
            new != -1 and not 0 <= new < self.minimized.state_count
            for new in self.old_to_new
        ):
            raise _validation_error(
                "minimize_map_out_of_range",
                "old-to-new entries must be -1 or a minimized state",
            )
        for block in self.partition:
            if block != tuple(sorted(set(block))):
                raise _validation_error(
                    "minimize_partition_not_canonical",
                    "partition blocks must be unique and sorted",
                )
            if any(not 0 <= old < self.transducer.state_count for old in block):
                raise _validation_error(
                    "minimize_partition_out_of_range",
                    "partition blocks must use source states",
                )
        if [min(block) for block in self.partition] != sorted(
            min(block) for block in self.partition
        ) or len({min(block) for block in self.partition}) != len(self.partition):
            raise _validation_error(
                "minimize_partition_order",
                "partition blocks must be ordered by strictly increasing minimum",
            )
        live = {old for block in self.partition for old in block}
        if sum(len(block) for block in self.partition) != len(live):
            raise _validation_error(
                "minimize_partition_overlap",
                "partition blocks must be disjoint",
            )
        if {old for old, new in enumerate(self.old_to_new) if new != -1} != live:
            raise _validation_error(
                "minimize_maps_disagree",
                "mapped states must agree with the partition union",
            )
        if self.new_to_old != tuple(min(block) for block in self.partition):
            raise _validation_error(
                "minimize_representatives",
                "representatives must be the least state of each block",
            )
        block_of = {
            old: index for index, block in enumerate(self.partition) for old in block
        }
        if any(self.old_to_new[old] != block_of[old] for old in live) or any(
            self.old_to_new[new] != index for index, new in enumerate(self.new_to_old)
        ):
            raise _validation_error(
                "minimize_map_block_mismatch",
                "old-to-new must send each block to its own minimized state",
            )
        return live

    def _require_table_shape(self, live: set[int]) -> None:
        """Check the distinguishability table against the partition."""

        block_of = {
            old: index for index, block in enumerate(self.partition) for old in block
        }
        pairs = {(row.first_state, row.second_state) for row in self.distinguishability}
        live_sorted = sorted(live)
        expected = {
            (first, second)
            for pos, first in enumerate(live_sorted)
            for second in live_sorted[pos + 1 :]
        }
        if pairs != expected:
            raise _validation_error(
                "minimize_table_coverage",
                "the distinguishability table must cover every live pair once",
            )
        for row in self.distinguishability:
            same_block = block_of[row.first_state] == block_of[row.second_state]
            if row.equivalent != same_block:
                raise _validation_error(
                    "minimize_table_partition_mismatch",
                    "equivalence flags must agree with the partition",
                )
            # An empty witness word is meaningful: the pair is already
            # distinguished by its final outputs on the empty word.
            if any(
                not 0 <= symbol < self.transducer.input_alphabet_size
                for symbol in row.witness_word
            ):
                raise _validation_error(
                    "minimize_witness_symbol_out_of_range",
                    "separating words must use the input alphabet",
                )
            if len(row.witness_word) > self.transducer.state_count:
                raise _validation_error(
                    "minimize_witness_too_long",
                    "separating words are bounded by the source state count",
                )

    def _require_sample_agreement(self) -> None:
        alphabet = self.transducer.input_alphabet_size
        if alphabet <= 1:
            expected_words = self.sample_max_length + 1
        else:
            expected_words = sum(
                alphabet**length for length in range(self.sample_max_length + 1)
            )
        if self.sample_words_checked != expected_words:
            raise _validation_error(
                "minimize_sample_count",
                "the replayed sample size must match the requested sample budget",
            )
        if not self.sample_agreement:
            raise _validation_error(
                "minimize_sample_disagreement",
                "a minimization result must replay its sample in agreement",
            )

    @classmethod
    def _from_kernel(
        cls,
        *,
        transducer: SubsequentialTransducer,
        sample_max_length: int,
        minimized: SubsequentialTransducer,
        old_to_new: tuple[int, ...],
        new_to_old: tuple[int, ...],
        partition: tuple[tuple[int, ...], ...],
        distinguishability: tuple[StatePairDistinguishability, ...],
        sample_words_checked: int,
        sample_agreement: bool,
    ) -> Self:
        """Construct a minimization emitted by the trusted owner-local kernel."""

        return cls.model_construct(
            transducer=transducer,
            sample_max_length=sample_max_length,
            minimized=minimized,
            old_to_new=old_to_new,
            new_to_old=new_to_old,
            partition=partition,
            distinguishability=distinguishability,
            sample_words_checked=sample_words_checked,
            sample_agreement=sample_agreement,
        )


class RelationPathReplayRequest(StrictModel):
    transducer: RationalTransducer
    initial_state: int = Field(ge=0, lt=MAX_FST_STATES)
    edge_path: tuple[int, ...] = Field(max_length=MAX_FST_WORD_LENGTH)


class RationalRelationInverseRequest(StrictModel):
    """Reverse the input/output coordinates of one finite rational relation."""

    transducer: RationalTransducer


class RelationPathReplayResult(RelationPathReplayRequest):
    status: Literal["ACCEPTING_PAIR", "INVALID_PATH"]
    input_word: tuple[int, ...] = Field(max_length=MAX_FST_RESULT_WORD_LENGTH)
    output_word: tuple[int, ...] = Field(max_length=MAX_FST_RESULT_WORD_LENGTH)
    state_trace: tuple[int, ...] = Field(max_length=MAX_FST_WORD_LENGTH + 1)
    error: str | None = None

    @model_validator(mode="after")
    def require_canonical_replay_shape(self) -> Self:
        if (
            not self.state_trace
            or self.state_trace[0] != self.initial_state
            or len(self.state_trace) > len(self.edge_path) + 1
            or any(
                not 0 <= state < self.transducer.state_count
                for state in self.state_trace
            )
            or any(
                not 0 <= symbol < self.transducer.input_alphabet_size
                for symbol in self.input_word
            )
            or any(
                not 0 <= symbol < self.transducer.output_alphabet_size
                for symbol in self.output_word
            )
        ):
            raise _validation_error(
                "replay_result_shape",
                "path replay fields have an invalid canonical shape",
            )
        if self.status == "ACCEPTING_PAIR" and self.error is not None:
            raise _validation_error(
                "replay_accepting_error", "an accepting path cannot carry an error"
            )
        if self.status == "INVALID_PATH" and not self.error:
            raise _validation_error(
                "replay_invalid_missing_error",
                "an invalid path must explain its failure",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: RelationPathReplayRequest,
        *,
        status: Literal["ACCEPTING_PAIR", "INVALID_PATH"],
        input_word: tuple[int, ...],
        output_word: tuple[int, ...],
        state_trace: tuple[int, ...],
        error: str | None,
    ) -> Self:
        """Construct a path replay emitted by the trusted owner-local kernel."""

        return cls.model_construct(
            transducer=request.transducer,
            initial_state=request.initial_state,
            edge_path=request.edge_path,
            status=status,
            input_word=input_word,
            output_word=output_word,
            state_trace=state_trace,
            error=error,
        )


__all__ = [
    "ComposeRequest",
    "ComposeResult",
    "MinimizeRequest",
    "MinimizeResult",
    "RationalRelationInverseRequest",
    "RelationPathReplayRequest",
    "RelationPathReplayResult",
    "StatePairDistinguishability",
    "SubseqRunRequest",
    "SubseqRunResult",
    "TrimRequest",
    "TrimResult",
]
