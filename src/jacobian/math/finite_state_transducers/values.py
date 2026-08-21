"""Provider-independent values for exact finite-state transducers."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_FST_STATES = 64
MAX_FST_ALPHABET = 32
MAX_FST_WORD_LENGTH = 512
MAX_FST_EDGES = 4096
MAX_FST_RESULT_WORD_LENGTH = 4096


class SubseqTransition(StrictModel):
    """One deterministic transition of a subsequential transducer.

    Maps ``(source, input_symbol)`` to ``(target, output_word)`` where
    ``output_word`` is a sequence of output-alphabet symbols.
    """

    source: int = Field(ge=0)
    input_symbol: int = Field(ge=0)
    target: int = Field(ge=0)
    output: tuple[int, ...] = Field(default=())


class SubseqFinalOutput(StrictModel):
    """One final-output word associated with a state."""

    state: int = Field(ge=0)
    output: tuple[int, ...] = Field(default=())


def _check_subseq_transitions(
    transitions: tuple[SubseqTransition, ...],
    state_count: int,
    input_size: int,
    output_size: int,
) -> None:
    seen_pairs: set[tuple[int, int]] = set()
    for tr in transitions:
        if not 0 <= tr.source < state_count:
            raise ValueError("transition source out of range")
        if not 0 <= tr.target < state_count:
            raise ValueError("transition target out of range")
        if not 0 <= tr.input_symbol < input_size:
            raise ValueError("transition input_symbol out of range")
        key = (tr.source, tr.input_symbol)
        if key in seen_pairs:
            raise ValueError("duplicate (source, input_symbol) transition")
        seen_pairs.add(key)
        if any(not 0 <= sym < output_size for sym in tr.output):
            raise ValueError("transition output symbol out of range")
        if len(tr.output) > MAX_FST_WORD_LENGTH:
            raise ValueError("transition output word too long")


def _check_subseq_finals(
    final_outputs: tuple[SubseqFinalOutput, ...],
    state_count: int,
    output_size: int,
) -> None:
    seen_finals: set[int] = set()
    for fo in final_outputs:
        if not 0 <= fo.state < state_count:
            raise ValueError("final output state out of range")
        if fo.state in seen_finals:
            raise ValueError("duplicate final output state")
        seen_finals.add(fo.state)
        if any(not 0 <= sym < output_size for sym in fo.output):
            raise ValueError("final output symbol out of range")
        if len(fo.output) > MAX_FST_WORD_LENGTH:
            raise ValueError("final output word too long")


class SubsequentialTransducer(StrictModel):
    """A deterministic subsequential transducer computing a partial function f: A* -> B*.

    The transition map is partial (sparse omission means undefined, not
    empty output). A word is in the domain exactly when every input transition
    is defined along the path and the final state has a final output.
    """

    input_alphabet_size: int = Field(ge=1, le=MAX_FST_ALPHABET)
    output_alphabet_size: int = Field(ge=1, le=MAX_FST_ALPHABET)
    state_count: int = Field(ge=1, le=MAX_FST_STATES)
    initial_state: int = Field(ge=0)
    transitions: tuple[SubseqTransition, ...] = Field(
        min_length=0, max_length=MAX_FST_EDGES
    )
    final_outputs: tuple[SubseqFinalOutput, ...] = Field(
        min_length=0, max_length=MAX_FST_STATES
    )

    @model_validator(mode="after")
    def require_valid_transducer(self) -> Self:
        if not 0 <= self.initial_state < self.state_count:
            raise ValueError("initial_state must be in 0..state_count-1")
        _check_subseq_transitions(
            self.transitions,
            self.state_count,
            self.input_alphabet_size,
            self.output_alphabet_size,
        )
        _check_subseq_finals(
            self.final_outputs,
            self.state_count,
            self.output_alphabet_size,
        )
        return self


class RationalEdge(StrictModel):
    """One edge of a nondeterministic rational transducer.

    Carries finite input and output label words over separate alphabets.
    ``(input, output)`` must not both be empty.
    """

    source: int = Field(ge=0)
    target: int = Field(ge=0)
    input_label: tuple[int, ...] = Field(default=())
    output_label: tuple[int, ...] = Field(default=())


def _check_rational_edges(
    edges: tuple[RationalEdge, ...],
    state_count: int,
    input_size: int,
    output_size: int,
) -> None:
    for edge in edges:
        if not 0 <= edge.source < state_count:
            raise ValueError("edge source out of range")
        if not 0 <= edge.target < state_count:
            raise ValueError("edge target out of range")
        for sym in edge.input_label:
            if not 0 <= sym < input_size:
                raise ValueError("edge input label out of range")
        for sym in edge.output_label:
            if not 0 <= sym < output_size:
                raise ValueError("edge output label out of range")
        if not edge.input_label and not edge.output_label:
            raise ValueError("edge with both labels empty is forbidden")
        if (
            len(edge.input_label) > MAX_FST_WORD_LENGTH
            or len(edge.output_label) > MAX_FST_WORD_LENGTH
        ):
            raise ValueError("edge label too long")


class RationalTransducer(StrictModel):
    """A finite-state transducer defining a rational relation R subseteq A* x B*.

    Edges are a multigraph (parallel edges preserved by identity). Each edge
    has finite input and output label words; ``(u, v)`` both empty is forbidden.
    """

    input_alphabet_size: int = Field(ge=1, le=MAX_FST_ALPHABET)
    output_alphabet_size: int = Field(ge=1, le=MAX_FST_ALPHABET)
    state_count: int = Field(ge=1, le=MAX_FST_STATES)
    initial_states: tuple[int, ...] = Field(min_length=1)
    accepting_states: tuple[int, ...] = Field(min_length=0)
    edges: tuple[RationalEdge, ...] = Field(min_length=0, max_length=MAX_FST_EDGES)

    @model_validator(mode="after")
    def require_valid_relation(self) -> Self:
        if len(self.initial_states) > MAX_FST_STATES:
            raise ValueError("too many initial states")
        if len(set(self.initial_states)) != len(self.initial_states):
            raise ValueError("initial states must be distinct")
        if len(set(self.accepting_states)) != len(self.accepting_states):
            raise ValueError("accepting states must be distinct")
        if any(not 0 <= s < self.state_count for s in self.initial_states):
            raise ValueError("initial state out of range")
        if any(not 0 <= s < self.state_count for s in self.accepting_states):
            raise ValueError("accepting state out of range")
        _check_rational_edges(
            self.edges,
            self.state_count,
            self.input_alphabet_size,
            self.output_alphabet_size,
        )
        return self


__all__ = [
    "MAX_FST_ALPHABET",
    "MAX_FST_EDGES",
    "MAX_FST_RESULT_WORD_LENGTH",
    "MAX_FST_STATES",
    "MAX_FST_WORD_LENGTH",
    "RationalEdge",
    "RationalTransducer",
    "SubseqFinalOutput",
    "SubseqTransition",
    "SubsequentialTransducer",
]
