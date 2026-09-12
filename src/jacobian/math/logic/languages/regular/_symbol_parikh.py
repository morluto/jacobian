"""Exact symbol-level Parikh profiles for accepted DFA words."""

from math import comb
from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.languages.regular.values import (
    DFA,
    MAX_COUNT_RESULT_DIGITS,
    MAX_DFA_ALPHABET,
)

MAX_SYMBOL_PARIKH_LENGTH = 1_000
MAX_SYMBOL_PARIKH_DP_WORK = 2_000_000
MAX_SYMBOL_PARIKH_CELLS = 43_000


class SymbolParikhProfileRequest(StrictModel):
    """Request for the complete accepted-word symbol-count profile."""

    dfa: DFA = Field(
        description=(
            "Complete deterministic finite automaton; symbol coordinates use its "
            "ordered zero-based alphabet axis."
        )
    )
    word_length: StrictInt = Field(
        ge=0,
        le=MAX_SYMBOL_PARIKH_LENGTH,
        description="Exact nonnegative length of every counted accepted word.",
    )


class SymbolParikhCell(StrictModel):
    """One canonical symbol-count vector and its positive exact multiplicity."""

    symbol_counts: tuple[StrictInt, ...] = Field(
        max_length=MAX_DFA_ALPHABET,
        description=(
            "Dense nonnegative counts on the DFA alphabet axis; coordinates sum "
            "to the requested word length."
        ),
    )
    multiplicity: ExactInteger = Field(
        ge=1,
        description="Positive exact number of accepted words with this vector.",
    )


class SymbolParikhProfileResult(StrictModel):
    """Complete canonical map from symbol-count vectors to accepted words."""

    dfa: DFA = Field(description="The source DFA retained for composition.")
    alphabet: tuple[StrictInt, ...] = Field(
        max_length=MAX_DFA_ALPHABET,
        description="Ordered zero-based alphabet axis retained by every vector.",
    )
    word_length: StrictInt = Field(
        ge=0,
        le=MAX_SYMBOL_PARIKH_LENGTH,
        description="Exact length shared by every profile vector.",
    )
    cells: tuple[SymbolParikhCell, ...] = Field(
        max_length=MAX_SYMBOL_PARIKH_CELLS,
        description="Lexicographically sorted, unique nonzero profile cells.",
    )
    total_accepted_words: ExactInteger = Field(
        description="Exact sum of all cell multiplicities.",
    )

    @classmethod
    def _from_kernel(
        cls,
        request: SymbolParikhProfileRequest,
        *,
        cells: tuple[SymbolParikhCell, ...],
        total_accepted_words: ExactInteger,
    ) -> Self:
        """Construct a profile after the trusted DP established its invariants."""

        return cls.model_construct(
            dfa=request.dfa,
            alphabet=tuple(range(request.dfa.alphabet_size)),
            word_length=request.word_length,
            cells=cells,
            total_accepted_words=total_accepted_words,
        )

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


def _build_transition_index(
    dfa: DFA,
) -> dict[tuple[int, int], int]:
    return {
        (transition.source, transition.symbol): transition.target
        for transition in dfa.transitions
    }


def _reachable_states_without_index(dfa: DFA) -> set[int]:
    reachable = {dfa.initial_state}
    frontier = [dfa.initial_state]
    while frontier:
        source = frontier.pop()
        for transition in dfa.transitions:
            if transition.source != source or transition.target in reachable:
                continue
            reachable.add(transition.target)
            frontier.append(transition.target)
    return reachable


def _symbol_parikh_profile_request(
    request: SymbolParikhProfileRequest,
) -> SymbolParikhProfileResult:
    dfa = request.dfa
    length = request.word_length
    alphabet_size = dfa.alphabet_size
    output_bound = comb(length + alphabet_size - 1, alphabet_size - 1)
    transition_count = dfa.state_count * alphabet_size
    # The layer at step t contains weak compositions of t, so only layers
    # t=0..length-1 are extended.  The final layer is materialized separately
    # into profile cells and must not be charged as another transition layer.
    # Reachability scans the raw transition axis once per discovered source.
    # Admit its declared-dimension upper bound before doing that scan or
    # building the full transition index.
    preflight_scan_work = dfa.state_count * transition_count
    if output_bound > MAX_SYMBOL_PARIKH_CELLS or (
        transition_count + preflight_scan_work > MAX_SYMBOL_PARIKH_DP_WORK
    ):
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
    reachable = _reachable_states_without_index(dfa)
    extension_cells = len(reachable) * comb(length + alphabet_size - 1, alphabet_size)
    extension_coordinate_work = extension_cells * alphabet_size * max(1, alphabet_size)
    output_materialization_work = len(reachable) * output_bound * max(1, alphabet_size)
    # The transition index is built from every DFA edge, including edges from
    # states that are unreachable from the initial state.
    transition_index_work = transition_count
    reachability_scan_work = len(reachable) * transition_count
    work_bound = (
        transition_index_work
        + reachability_scan_work
        + extension_coordinate_work
        + output_materialization_work
    )
    if work_bound > MAX_SYMBOL_PARIKH_DP_WORK:
        raise OperationResourceAdmissionError(
            location=("word_length",),
            code="regular_language.symbol_parikh.profile_bound",
            message="symbol-Parikh DP or output exceeds its exact admitted bound",
        )
    transitions = _build_transition_index(dfa)
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
    return SymbolParikhProfileResult._from_kernel(
        request,
        cells=tuple(
            SymbolParikhCell(symbol_counts=counts, multiplicity=multiplicity)
            for counts, multiplicity in sorted(profile.items())
        ),
        total_accepted_words=total,
    )


def symbol_parikh_profile(dfa: DFA, word_length: int) -> SymbolParikhProfileResult:
    """Return the accepted-word symbol Parikh profile for one exact length."""

    if not isinstance(dfa, DFA):
        raise OperationDomainValidationError(
            location=("dfa",),
            code="regular_language.symbol_parikh.dfa_type",
            message="dfa must be a canonical DFA value",
        )
    if (
        type(word_length) is not int
        or word_length < 0
        or word_length > MAX_SYMBOL_PARIKH_LENGTH
    ):
        raise OperationDomainValidationError(
            location=("word_length",),
            code="regular_language.symbol_parikh.word_length",
            message=(
                "word_length must be an integer from 0 through "
                f"{MAX_SYMBOL_PARIKH_LENGTH}"
            ),
        )
    return _symbol_parikh_profile_request(
        SymbolParikhProfileRequest(dfa=dfa, word_length=word_length)
    )
