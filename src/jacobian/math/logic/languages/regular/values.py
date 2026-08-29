"""Provider-independent values for finite automata and regular languages."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"regular_language.{reason}", message)


MAX_DFA_STATES = 64
MAX_DFA_ALPHABET = 32
MAX_DFA_TRANSITIONS = 4096
MAX_WORD_LENGTH = 1000
# Raw JSON integers remain exactly interoperable through this exponent. FLINT
# accepts the full range, while owner-local work and result admission narrow it.
MAX_COUNT_WORD_LENGTH = (1 << 53) - 1
MAX_COUNT_MATRIX_BIT_WORK = 35_000_000_000
MAX_COUNT_RESULT_DIGITS = 32_768

# A materialized state axis is cheap relative to the transition/profile data;
# operation-owned reachable-work bounds decide which path requests are admitted.
MAX_LABELED_AUTOMATON_STATES = 4_096
MAX_LABELED_AUTOMATON_ALPHABET = 32
MAX_LABELED_AUTOMATON_TRANSITIONS = 4096
# Conservative layer-iteration fallback for the pure-Python recurrence. Derived
# path, DP, intermediate, and output budgets normally reject branching profiles
# much earlier, while deterministic or quickly dead carriers remain useful far
# beyond the ordinary DFA word bound. A future cycle-accelerated kernel could
# raise this ceiling without changing the public postcondition.
MAX_TRANSITION_PROFILE_PATH_LENGTH = 1_000_000
MAX_TRANSITION_PROFILE_ENTRIES = 43_000
MAX_TRANSITION_PROFILE_COUNT_DIGITS = 32_768

TransitionUseCount = Annotated[
    int,
    Field(ge=0, le=MAX_TRANSITION_PROFILE_PATH_LENGTH),
]


class DFATransition(StrictModel):
    """One transition: from ``source`` on ``symbol`` to ``target``."""

    source: int = Field(ge=0, le=MAX_DFA_STATES - 1)
    symbol: int = Field(ge=0, le=MAX_DFA_ALPHABET - 1)
    target: int = Field(ge=0, le=MAX_DFA_STATES - 1)


class DFA(StrictModel):
    """One total deterministic finite automaton over an integer alphabet.

    A valid DFA declares exactly one transition for every ``(state, symbol)``
    pair so that simulation and word counting share one consistent semantics.
    """

    state_count: int = Field(ge=1, le=MAX_DFA_STATES)
    alphabet_size: int = Field(ge=1, le=MAX_DFA_ALPHABET)
    transitions: tuple[DFATransition, ...] = Field(
        min_length=0,
        max_length=MAX_DFA_TRANSITIONS,
    )
    initial_state: int = Field(ge=0, le=MAX_DFA_STATES - 1)
    accepting_states: tuple[int, ...] = Field(
        min_length=0,
        max_length=MAX_DFA_STATES,
    )

    @model_validator(mode="after")
    def require_total_deterministic_dfa(self) -> Self:
        if not 0 <= self.initial_state < self.state_count:
            raise _validation_error(
                "initial_state_out_of_range",
                "initial_state must be in 0..state_count-1",
            )
        if any(not 0 <= state < self.state_count for state in self.accepting_states):
            raise _validation_error(
                "accepting_state_out_of_range",
                "accepting states must be in 0..state_count-1",
            )
        if len(set(self.accepting_states)) != len(self.accepting_states):
            raise _validation_error(
                "accepting_states_not_unique", "accepting states must be unique"
            )
        seen: set[tuple[int, int]] = set()
        for transition in self.transitions:
            if not 0 <= transition.source < self.state_count:
                raise _validation_error(
                    "transition_source_out_of_range",
                    "transition source must be in 0..state_count-1",
                )
            if not 0 <= transition.target < self.state_count:
                raise _validation_error(
                    "transition_target_out_of_range",
                    "transition target must be in 0..state_count-1",
                )
            if not 0 <= transition.symbol < self.alphabet_size:
                raise _validation_error(
                    "transition_symbol_out_of_range",
                    "transition symbol must be in 0..alphabet_size-1",
                )
            key = (transition.source, transition.symbol)
            if key in seen:
                raise _validation_error(
                    "dfa_not_deterministic",
                    "DFA must be deterministic (no duplicate source/symbol)",
                )
            seen.add(key)
        expected = self.state_count * self.alphabet_size
        if len(seen) != expected:
            raise _validation_error(
                "dfa_not_total",
                "DFA must be total: expected one transition for every "
                f"state-symbol pair ({expected}), got {len(seen)}",
            )
        return self


class AutomatonTransition(StrictModel):
    """One identified edge of a finite labeled transition system.

    ``transition_id`` is its stable coordinate on the automaton's ordered
    transition axis. Parallel transitions remain distinct mathematical edges.
    """

    transition_id: int = Field(ge=0, le=MAX_LABELED_AUTOMATON_TRANSITIONS - 1)
    source: int = Field(ge=0, le=MAX_LABELED_AUTOMATON_STATES - 1)
    symbol: int = Field(ge=0, le=MAX_LABELED_AUTOMATON_ALPHABET - 1)
    target: int = Field(ge=0, le=MAX_LABELED_AUTOMATON_STATES - 1)


class FiniteLabeledAutomaton(StrictModel):
    """A finite labeled transition carrier with an explicit edge axis.

    States and symbols use the declared zero-based axes. Transitions are the
    complete ordered transition axis: their identifiers must be exactly
    ``0, ..., len(transitions)-1`` in tuple order. The carrier may be partial,
    nondeterministic, have parallel identified transitions, or have no
    transitions. Path endpoints belong to the operation using the carrier
    rather than being baked into this reusable value.
    """

    state_count: int = Field(ge=1, le=MAX_LABELED_AUTOMATON_STATES)
    alphabet_size: int = Field(ge=0, le=MAX_LABELED_AUTOMATON_ALPHABET)
    transitions: tuple[AutomatonTransition, ...] = Field(
        max_length=MAX_LABELED_AUTOMATON_TRANSITIONS,
        description=(
            "Complete transition axis in transition_id order; IDs must be the "
            "contiguous zero-based positions. Parallel transitions are distinct."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_transition_axis(self) -> Self:
        if tuple(transition.transition_id for transition in self.transitions) != tuple(
            range(len(self.transitions))
        ):
            raise _validation_error(
                "transition_axis_not_contiguous",
                "transition_id values must be the contiguous zero-based axis in "
                "tuple order",
            )
        for transition in self.transitions:
            if not 0 <= transition.source < self.state_count:
                raise _validation_error(
                    "transition_source_out_of_range",
                    "transition source must be in 0..state_count-1",
                )
            if not 0 <= transition.target < self.state_count:
                raise _validation_error(
                    "transition_target_out_of_range",
                    "transition target must be in 0..state_count-1",
                )
            if not 0 <= transition.symbol < self.alphabet_size:
                raise _validation_error(
                    "transition_symbol_out_of_range",
                    "transition symbol must be in 0..alphabet_size-1",
                )
        return self


class TransitionParikhCell(StrictModel):
    """One transition-count vector and its positive exact path multiplicity."""

    transition_counts: tuple[TransitionUseCount, ...] = Field(
        max_length=MAX_LABELED_AUTOMATON_TRANSITIONS,
        description=(
            "Nonnegative transition uses on the source automaton's transition_id "
            "axis; the coordinate sum equals the exact path length."
        ),
    )
    multiplicity: CanonicalInteger = Field(
        max_length=MAX_TRANSITION_PROFILE_COUNT_DIGITS,
        description="Positive exact number of paths with this transition ledger.",
    )

    @model_validator(mode="after")
    def require_positive_multiplicity(self) -> Self:
        if parse_canonical_integer(self.multiplicity) <= 0:
            raise _validation_error(
                "path_multiplicity_not_positive", "path multiplicity must be positive"
            )
        return self


class TransitionParikhProfile(StrictModel):
    """Complete transition-use histogram for one exact endpoint-bound path set."""

    automaton: FiniteLabeledAutomaton
    source_state: int = Field(ge=0, le=MAX_LABELED_AUTOMATON_STATES - 1)
    target_state: int = Field(ge=0, le=MAX_LABELED_AUTOMATON_STATES - 1)
    path_length: int = Field(ge=0, le=MAX_TRANSITION_PROFILE_PATH_LENGTH)
    entries: tuple[TransitionParikhCell, ...] = Field(
        max_length=MAX_TRANSITION_PROFILE_ENTRIES,
        description=(
            "Complete lexicographically ordered map entries from dense transition "
            "count vectors to exact path multiplicities."
        ),
    )
    total_path_count: CanonicalInteger = Field(
        max_length=MAX_TRANSITION_PROFILE_COUNT_DIGITS
    )

    @model_validator(mode="after")
    def require_source_and_canonical_complete_profile(self) -> Self:
        if not 0 <= self.source_state < self.automaton.state_count:
            raise _validation_error(
                "source_state_out_of_range", "source_state must be in 0..state_count-1"
            )
        if not 0 <= self.target_state < self.automaton.state_count:
            raise _validation_error(
                "target_state_out_of_range", "target_state must be in 0..state_count-1"
            )

        transition_count = len(self.automaton.transitions)
        vectors = tuple(entry.transition_counts for entry in self.entries)
        if vectors != tuple(sorted(set(vectors))):
            raise _validation_error(
                "profile_vectors_not_canonical",
                "profile transition-count vectors must be lexicographically sorted "
                "and unique",
            )
        for vector in vectors:
            if len(vector) != transition_count:
                raise _validation_error(
                    "profile_vector_length_mismatch",
                    "every transition-count vector must use the automaton's complete "
                    "transition axis",
                )
            if sum(vector) != self.path_length:
                raise _validation_error(
                    "profile_vector_sum_mismatch",
                    "every transition-count vector must sum to path_length",
                )

        total = sum(
            parse_canonical_integer(entry.multiplicity) for entry in self.entries
        )
        if parse_canonical_integer(self.total_path_count) != total:
            raise _validation_error(
                "profile_total_mismatch",
                "total_path_count must equal the sum of profile multiplicities",
            )

        if self.path_length == 0:
            expected_vectors = (
                (tuple(0 for _ in self.automaton.transitions),)
                if self.source_state == self.target_state
                else ()
            )
            if vectors != expected_vectors or total != len(expected_vectors):
                raise _validation_error(
                    "zero_length_profile_invalid",
                    "length-zero profile must contain the zero vector once exactly "
                    "when source_state equals target_state",
                )

        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        automaton: FiniteLabeledAutomaton,
        source_state: int,
        target_state: int,
        path_length: int,
        entries: tuple[TransitionParikhCell, ...],
        total_path_count: CanonicalInteger,
    ) -> Self:
        """Construct a profile produced by the trusted bounded recurrence."""

        return cls.model_construct(
            automaton=automaton,
            source_state=source_state,
            target_state=target_state,
            path_length=path_length,
            entries=entries,
            total_path_count=total_path_count,
        )


__all__ = [
    "DFA",
    "AutomatonTransition",
    "DFATransition",
    "FiniteLabeledAutomaton",
    "TransitionParikhCell",
    "TransitionParikhProfile",
]
