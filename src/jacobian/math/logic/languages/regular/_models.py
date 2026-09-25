"""Typed wire contracts for exact regular language operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.math.logic.automata.transducers.values import (
    SubsequentialTransducer,
)
from jacobian.math.logic.languages.regular.values import (
    DFA,
    MAX_COUNT_WORD_LENGTH,
    MAX_DFA_EQUIVALENCE_WITNESS_LENGTH,
    MAX_DFA_STATES,
    MAX_LABELED_AUTOMATON_STATES,
    MAX_TRANSITION_PROFILE_PATH_LENGTH,
    MAX_WORD_LENGTH,
    NFA,
    FiniteLabeledAutomaton,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"regular_language.{reason}", message)


class RunRequest(StrictModel):
    """Check if a word is accepted by a DFA."""

    dfa: DFA
    word: tuple[int, ...] = Field(max_length=MAX_WORD_LENGTH)


class CountRequest(StrictModel):
    """Count accepted words of a given length."""

    dfa: DFA
    word_length: int = Field(ge=0, le=MAX_COUNT_WORD_LENGTH)


class ComplementRequest(StrictModel):
    """Compute the complement of a DFA's language."""

    dfa: DFA


class SubsequentialPreimageRequest(StrictModel):
    """Preimage of a DFA language under a parented subsequential transducer."""

    dfa: DFA
    transducer: SubsequentialTransducer

    @model_validator(mode="after")
    def require_same_explicit_output_alphabet(self) -> Self:
        alphabet = self.dfa.alphabet
        output_alphabet = self.transducer.output_alphabet
        if alphabet is None or output_alphabet is None:
            raise _validation_error(
                "preimage_alphabet_context_missing",
                "preimage requires explicit DFA and transducer output alphabet contexts",
            )
        if self.transducer.input_alphabet is None:
            raise _validation_error(
                "preimage_input_alphabet_context_missing",
                "preimage requires an explicit transducer input alphabet context",
            )
        if alphabet != output_alphabet:
            raise _validation_error(
                "preimage_alphabet_context_mismatch",
                "DFA and transducer output alphabet contexts must be identical",
            )
        if (self.dfa.alphabet_id is None) != (
            self.transducer.output_alphabet_id is None
        ) or self.dfa.alphabet_id != self.transducer.output_alphabet_id:
            raise _validation_error(
                "preimage_alphabet_identity_mismatch",
                "DFA and transducer output alphabet identities must be identical",
            )
        if self.dfa.alphabet_size != len(alphabet.symbols):
            raise _validation_error(
                "preimage_alphabet_size_mismatch",
                "DFA alphabet_size must match the explicit alphabet context",
            )
        return self


class SubsequentialImageRequest(StrictModel):
    """Image of a source regular language under a parented subsequential map."""

    dfa: DFA
    transducer: SubsequentialTransducer

    @model_validator(mode="after")
    def require_matching_explicit_input_alphabet(self) -> Self:
        alphabet = self.dfa.alphabet
        input_alphabet = self.transducer.input_alphabet
        if alphabet is None or input_alphabet is None:
            raise _validation_error(
                "image_alphabet_context_missing",
                "image requires explicit DFA and transducer input alphabet contexts",
            )
        if self.transducer.output_alphabet is None:
            raise _validation_error(
                "image_output_alphabet_context_missing",
                "image requires an explicit transducer output alphabet context",
            )
        if alphabet != input_alphabet:
            raise _validation_error(
                "image_alphabet_context_mismatch",
                "DFA and transducer input alphabet contexts must be identical",
            )
        if (self.dfa.alphabet_id is None) != (
            self.transducer.input_alphabet_id is None
        ) or self.dfa.alphabet_id != self.transducer.input_alphabet_id:
            raise _validation_error(
                "image_alphabet_identity_mismatch",
                "DFA and transducer input alphabet identities must be identical",
            )
        if self.dfa.alphabet_size != len(alphabet.symbols):
            raise _validation_error(
                "image_alphabet_size_mismatch",
                "DFA alphabet_size must match the explicit alphabet context",
            )
        return self


class SubsequentialNFAImageRequest(StrictModel):
    """Image of a parented epsilon-NFA language under a subsequential map."""

    nfa: NFA
    transducer: SubsequentialTransducer

    @model_validator(mode="after")
    def require_matching_explicit_input_alphabet(self) -> Self:
        alphabet = self.nfa.alphabet
        input_alphabet = self.transducer.input_alphabet
        if alphabet is None or input_alphabet is None:
            raise _validation_error(
                "nfa_image_alphabet_context_missing",
                "NFA image requires explicit source and transducer input alphabets",
            )
        if self.transducer.output_alphabet is None:
            raise _validation_error(
                "nfa_image_output_alphabet_context_missing",
                "NFA image requires an explicit transducer output alphabet",
            )
        if alphabet != input_alphabet:
            raise _validation_error(
                "nfa_image_alphabet_context_mismatch",
                "NFA and transducer input alphabet contexts must be identical",
            )
        if (self.nfa.alphabet_id is None) != (
            self.transducer.input_alphabet_id is None
        ) or self.nfa.alphabet_id != self.transducer.input_alphabet_id:
            raise _validation_error(
                "nfa_image_alphabet_identity_mismatch",
                "NFA and transducer input alphabet identities must be identical",
            )
        if self.nfa.alphabet_size != len(alphabet.symbols):
            raise _validation_error(
                "nfa_image_alphabet_size_mismatch",
                "NFA alphabet_size must match its explicit alphabet context",
            )
        return self


class NFAMembershipRequest(StrictModel):
    """Decide whether a parented finite word belongs to an NFA language."""

    nfa: NFA
    word: tuple[int, ...] = Field(max_length=MAX_WORD_LENGTH)

    @model_validator(mode="after")
    def require_explicit_alphabet_context(self) -> Self:
        if self.nfa.alphabet is None:
            raise _validation_error(
                "nfa_membership_alphabet_context_missing",
                "NFA membership requires an explicit alphabet context",
            )
        return self


class NFAMembershipResult(StrictModel):
    """Exact NFA membership value bound to its NFA and finite word."""

    nfa: NFA
    word: tuple[int, ...] = Field(max_length=MAX_WORD_LENGTH)
    accepted: bool


class EquivalenceRequest(StrictModel):
    """Two total DFAs over one common ordered alphabet parent."""

    left: DFA = Field(
        description=(
            "Complete deterministic finite automaton; alphabet size, optional "
            "identity, and optional ordered context must equal right."
        )
    )
    right: DFA = Field(
        description=(
            "Complete deterministic finite automaton; alphabet size, optional "
            "identity, and optional ordered context must equal left."
        )
    )

    @model_validator(mode="after")
    def require_same_alphabet_parent(self) -> Self:
        if (
            self.left.alphabet_size != self.right.alphabet_size
            or self.left.alphabet_id != self.right.alphabet_id
            or self.left.alphabet != self.right.alphabet
        ):
            raise _validation_error(
                "alphabet_mismatch",
                "equivalence requires exactly matching alphabet sizes and parents",
            )
        return self


class EquivalenceResult(StrictModel):
    """Exact equivalence decision with a replayable least counterexample."""

    left: DFA
    right: DFA
    equivalent: bool
    distinguishing_word: tuple[int, ...] | None = Field(
        default=None, max_length=MAX_DFA_EQUIVALENCE_WITNESS_LENGTH
    )
    left_state_trace: tuple[int, ...] | None = None
    right_state_trace: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def require_counterexample_shape(self) -> Self:
        if self.left.alphabet_size != self.right.alphabet_size:
            raise _validation_error(
                "alphabet_mismatch",
                "equivalence results must retain DFAs over one common alphabet",
            )
        if (
            self.left.alphabet_id != self.right.alphabet_id
            or self.left.alphabet != self.right.alphabet
        ):
            raise _validation_error(
                "alphabet_mismatch",
                "equivalence results must retain DFAs with identical alphabet parents",
            )
        fields = (
            self.distinguishing_word,
            self.left_state_trace,
            self.right_state_trace,
        )
        if self.equivalent:
            if any(field is not None for field in fields):
                raise _validation_error(
                    "equivalent_counterexample",
                    "equivalent DFAs cannot carry a distinguishing word or traces",
                )
            return self
        distinguishing_word = self.distinguishing_word
        left_state_trace = self.left_state_trace
        right_state_trace = self.right_state_trace
        if (
            distinguishing_word is None
            or left_state_trace is None
            or right_state_trace is None
        ):
            raise _validation_error(
                "missing_counterexample",
                "inequivalent DFAs require a distinguishing word and both traces",
            )
        expected = len(distinguishing_word) + 1
        if len(left_state_trace) != expected or len(right_state_trace) != expected:
            raise _validation_error(
                "counterexample_trace_length",
                "each counterexample trace must contain one state per word prefix",
            )
        if any(
            not 0 <= symbol < self.left.alphabet_size for symbol in distinguishing_word
        ):
            raise _validation_error(
                "counterexample_symbol_out_of_range",
                "distinguishing word contains a symbol outside the common alphabet",
            )
        if (
            left_state_trace[0] != self.left.initial_state
            or right_state_trace[0] != self.right.initial_state
            or any(not 0 <= state < self.left.state_count for state in left_state_trace)
            or any(
                not 0 <= state < self.right.state_count for state in right_state_trace
            )
        ):
            raise _validation_error(
                "counterexample_trace_state",
                "counterexample traces must start at each initial state and stay on "
                "their source state axes",
            )
        return self


class TransitionParikhProfileRequest(StrictModel):
    """Compute a complete transition-use profile for exact automaton paths.

    The automaton's ordered ``transition_id`` axis is authoritative. Admission
    derives path-extension, sparse-DP-cell, vector-coordinate, multiplicity,
    profile-cell, and retained-result bounds before running the recurrence.
    """

    automaton: FiniteLabeledAutomaton = Field(
        description=(
            "Finite labeled transition carrier whose contiguous transition_id "
            "order is the complete profile coordinate axis."
        )
    )
    source_state: int = Field(
        ge=0,
        le=MAX_LABELED_AUTOMATON_STATES - 1,
        description="Path source on the automaton's zero-based state axis.",
    )
    target_state: int = Field(
        ge=0,
        le=MAX_LABELED_AUTOMATON_STATES - 1,
        description="Path target on the automaton's zero-based state axis.",
    )
    path_length: int = Field(
        ge=0,
        le=MAX_TRANSITION_PROFILE_PATH_LENGTH,
        description="Exact nonnegative number of transitions in every path.",
    )


class RunResult(RunRequest):
    """Whether a word was accepted and the final state reached."""

    accepted: bool
    final_state: int = Field(ge=0, le=MAX_DFA_STATES - 1)
    state_trace: tuple[int, ...]

    @model_validator(mode="after")
    def require_run_shape(self) -> Self:
        if len(self.state_trace) != len(self.word) + 1:
            raise _validation_error(
                "run_trace_length_mismatch",
                "DFA run trace must contain the initial state and one state per symbol",
            )
        if (
            self.state_trace[0] != self.dfa.initial_state
            or self.final_state != self.state_trace[-1]
        ):
            raise _validation_error(
                "run_trace_endpoint_mismatch",
                "DFA run trace must begin at the initial state and end at final_state",
            )
        if any(not 0 <= state < self.dfa.state_count for state in self.state_trace):
            raise _validation_error(
                "run_trace_state_out_of_range",
                "DFA run trace contains a state outside its carrier",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: RunRequest,
        *,
        accepted: bool,
        final_state: int,
        state_trace: tuple[int, ...],
    ) -> Self:
        return cls.model_construct(
            dfa=request.dfa,
            word=request.word,
            accepted=accepted,
            final_state=final_state,
            state_trace=state_trace,
        )


class CountResult(CountRequest):
    """Exact count of accepted words of a given length."""

    count: ExactInteger
    word_length: int = Field(ge=0, le=MAX_COUNT_WORD_LENGTH)

    @model_validator(mode="after")
    def require_nonnegative_count(self) -> Self:
        if self.count < 0:
            raise _validation_error("count_negative", "word count must be nonnegative")
        return self

    @classmethod
    def _from_kernel(cls, request: CountRequest, *, count: ExactInteger) -> Self:
        return cls.model_construct(
            dfa=request.dfa,
            word_length=request.word_length,
            count=count,
        )


class ComplementResult(StrictModel):
    """The complement DFA."""

    dfa: DFA


__all__ = [
    "ComplementRequest",
    "ComplementResult",
    "CountRequest",
    "CountResult",
    "EquivalenceRequest",
    "EquivalenceResult",
    "NFAMembershipRequest",
    "NFAMembershipResult",
    "RunRequest",
    "RunResult",
    "SubsequentialImageRequest",
    "SubsequentialPreimageRequest",
    "TransitionParikhProfileRequest",
]
