"""Exact symbol-level Parikh profiles for accepted DFA words."""

from math import comb
from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.languages.regular.values import (
    DFA,
    MAX_COUNT_RESULT_DIGITS,
    MAX_DFA_ALPHABET,
)

MAX_SYMBOL_PARIKH_LENGTH = 1_000
MAX_SYMBOL_PARIKH_DP_WORK = 2_000_000
MAX_SYMBOL_PARIKH_CELLS = 43_000


class SymbolParikhProfileRequest(StrictModel):
    dfa: DFA
    word_length: StrictInt = Field(ge=0, le=MAX_SYMBOL_PARIKH_LENGTH)


class SymbolParikhCell(StrictModel):
    symbol_counts: tuple[StrictInt, ...] = Field(max_length=MAX_DFA_ALPHABET)
    multiplicity: ExactInteger = Field(ge=1)


class SymbolParikhProfileResult(StrictModel):
    dfa: DFA
    alphabet: tuple[StrictInt, ...] = Field(max_length=MAX_DFA_ALPHABET)
    word_length: StrictInt = Field(ge=0, le=MAX_SYMBOL_PARIKH_LENGTH)
    cells: tuple[SymbolParikhCell, ...] = Field(max_length=MAX_SYMBOL_PARIKH_CELLS)
    total_accepted_words: ExactInteger

    @model_validator(mode="after")
    def require_canonical_cells(self) -> Self:
        if self.alphabet != tuple(range(self.dfa.alphabet_size)):
            raise ValueError("symbol-Parikh alphabet must be the DFA's ordered axis")
        vectors = tuple(cell.symbol_counts for cell in self.cells)
        if vectors != tuple(sorted(set(vectors))):
            raise ValueError(
                "symbol-Parikh cells must be lexicographically sorted and unique"
            )
        for vector in vectors:
            if len(vector) != self.dfa.alphabet_size:
                raise ValueError(
                    "symbol-Parikh vectors must use the complete alphabet axis"
                )
            if any(count < 0 for count in vector) or sum(vector) != self.word_length:
                raise ValueError(
                    "symbol-Parikh vectors must be nonnegative and sum to word_length"
                )
        if self.total_accepted_words != sum(cell.multiplicity for cell in self.cells):
            raise ValueError(
                "total_accepted_words must equal the sum of cell multiplicities"
            )
        return self


def symbol_parikh_profile(
    request: SymbolParikhProfileRequest,
) -> SymbolParikhProfileResult:
    dfa = request.dfa
    length = request.word_length
    alphabet_size = dfa.alphabet_size
    transitions = {
        (transition.source, transition.symbol): transition.target
        for transition in dfa.transitions
    }
    reachable = {dfa.initial_state}
    frontier = [dfa.initial_state]
    while frontier:
        source = frontier.pop()
        for symbol in range(alphabet_size):
            target = transitions[(source, symbol)]
            if target not in reachable:
                reachable.add(target)
                frontier.append(target)
    output_bound = comb(length + alphabet_size - 1, alphabet_size - 1)
    # The layer at step t contains weak compositions of t, so only layers
    # t=0..length-1 are extended.  The final layer is materialized separately
    # into profile cells and must not be charged as another transition layer.
    extension_cells = len(reachable) * comb(length + alphabet_size - 1, alphabet_size)
    extension_coordinate_work = extension_cells * alphabet_size * max(1, alphabet_size)
    output_materialization_work = len(reachable) * output_bound * max(1, alphabet_size)
    reachability_work = len(reachable) * alphabet_size
    work_bound = (
        reachability_work + extension_coordinate_work + output_materialization_work
    )
    if output_bound > MAX_SYMBOL_PARIKH_CELLS or work_bound > MAX_SYMBOL_PARIKH_DP_WORK:
        raise OperationResourceAdmissionError(
            location=("word_length",),
            code="regular_language.symbol_parikh.profile_bound",
            message="symbol-Parikh DP or output exceeds its exact admitted bound",
        )
    if length * max(1, len(str(alphabet_size))) > MAX_COUNT_RESULT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("word_length",),
            code="regular_language.symbol_parikh.count_bound",
            message="symbol-Parikh multiplicities exceed the exact integer digit bound",
        )
    zero = (0,) * alphabet_size
    layer: dict[tuple[int, tuple[int, ...]], int] = {(dfa.initial_state, zero): 1}
    for _step in range(length):
        next_layer: dict[tuple[int, tuple[int, ...]], int] = {}
        for (state, counts), multiplicity in layer.items():
            for symbol in range(alphabet_size):
                target = transitions[(state, symbol)]
                if target not in reachable:
                    continue
                updated = (*counts[:symbol], counts[symbol] + 1, *counts[symbol + 1 :])
                key = (target, updated)
                next_layer[key] = next_layer.get(key, 0) + multiplicity
        layer = next_layer
    profile: dict[tuple[int, ...], int] = {}
    accepting = set(dfa.accepting_states)
    for (state, counts), multiplicity in layer.items():
        if state in accepting:
            profile[counts] = profile.get(counts, 0) + multiplicity
    total = sum(profile.values())
    return SymbolParikhProfileResult(
        dfa=dfa,
        alphabet=tuple(range(alphabet_size)),
        word_length=length,
        cells=tuple(
            SymbolParikhCell(symbol_counts=counts, multiplicity=multiplicity)
            for counts, multiplicity in sorted(profile.items())
        ),
        total_accepted_words=total,
    )
