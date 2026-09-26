"""Typed wire contracts for exact algebraic combinatorics operations."""

from __future__ import annotations

from itertools import pairwise
from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, WithJsonSchema, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS, ExactInteger
from jacobian._models import StrictModel
from jacobian.math.combinatorics.algebraic.values import (
    MAX_RSK_ROW_SEARCH_COMPARISONS,
    MAX_RSK_WORD_LENGTH,
    MAX_RSK_WORD_PAYLOAD_SCALARS,
    RSKConvention,
    RSKInsertionEvent,
    RSKReverseInsertionEvent,
    RSKTableauPair,
)
from jacobian.math.combinatorics.symmetric_functions.values import (
    MAX_PARTITION_SIZE as MAX_CANONICAL_PARTITION_SIZE,
)
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    SemistandardYoungTableau,
    StandardYoungTableau,
    TableauCandidate,
)
from jacobian.math.logic.languages.words.values import FiniteWord, Symbol

# A permutation of length N inserts N ranks through the same _row_insert
# kernel and produces two N-cell tableaux, so the canonical tableau cell
# budget derives the permutation envelope.
MAX_RSK_PERMUTATION_LENGTH = MAX_RSK_WORD_LENGTH
MAX_RSK_TRACE_RESULT_BYTES = 8_000_000
MAX_RSK_TRACE_WORK = 2_000_000

_POSITIVE_EXACT_INTEGER_SCHEMA = {
    "type": "string",
    "pattern": rf"^[1-9][0-9]{{0,{MAX_CANONICAL_INTEGER_DIGITS - 1}}}(?![\s\S])",
    "maxLength": MAX_CANONICAL_INTEGER_DIGITS,
    "examples": ["2"],
}
_NONNEGATIVE_EXACT_INTEGER_SCHEMA = {
    "type": "string",
    "pattern": rf"^(?:0|[1-9][0-9]{{0,{MAX_CANONICAL_INTEGER_DIGITS - 1}}})(?![\s\S])",
    "maxLength": MAX_CANONICAL_INTEGER_DIGITS,
    "examples": ["0", "2"],
}

_PositiveExactInteger = Annotated[
    ExactInteger,
    Field(ge=1),
    WithJsonSchema(_POSITIVE_EXACT_INTEGER_SCHEMA),
]
_NonnegativeExactInteger = Annotated[
    ExactInteger,
    Field(ge=0),
    WithJsonSchema(_NONNEGATIVE_EXACT_INTEGER_SCHEMA),
]


class HookLengthRequest(StrictModel):
    """Compute the hook lengths of a partition."""

    partition: IntegerPartition


class StandardYoungTableauCountRequest(StrictModel):
    """Count standard Young tableaux of a given shape."""

    partition: IntegerPartition


class ConjugatePartitionRequest(StrictModel):
    """Compute the conjugate (transpose) partition."""

    partition: IntegerPartition


class HookLengthResult(StrictModel):
    """Hook lengths as a flat list of row-indexed values."""

    hooks: tuple[tuple[int, ...], ...]
    total_product: ExactInteger = Field(description="Product of all hook lengths.")


class StandardYoungTableauCountResult(StrictModel):
    """The number of standard Young tableaux of a given shape."""

    count: ExactInteger = Field(description="Number of standard Young tableaux.")
    n: int = Field(ge=0, le=MAX_CANONICAL_PARTITION_SIZE)


class ConjugatePartitionResult(StrictModel):
    """The conjugate (transpose) partition."""

    conjugate: IntegerPartition


class SemistandardYoungTableauCountRequest(StrictModel):
    """Count semistandard tableaux with entries in ``1..alphabet_size``."""

    partition: IntegerPartition
    # The alphabet is bounded by the operation's derived admission estimate,
    # not by the number of cells in the partition.  In particular, a one-cell
    # shape has an exact count equal to the alphabet size.
    # The alphabet is an exact mathematical parameter and may exceed the
    # interoperable JSON number range.  ExactInteger keeps Python values as
    # ints while encoding JSON values as canonical decimal strings.
    alphabet_size: _PositiveExactInteger

    @model_validator(mode="after")
    def require_positive_alphabet(self) -> Self:
        if self.alphabet_size < 1:
            raise PydanticCustomError(
                "algebraic_combinatorics.alphabet_size_not_positive",
                "alphabet_size must be positive",
            )
        return self


class SemistandardYoungTableauCountResult(StrictModel):
    """Exact SSYT count bound to its source shape and alphabet."""

    partition: IntegerPartition
    alphabet_size: _PositiveExactInteger
    count: _NonnegativeExactInteger


class StandardTableauCheckResult(StrictModel):
    """Standard-tableau membership bound to the checked structural source."""

    tableau: TableauCandidate
    is_member: bool


class SemistandardTableauCheckResult(StrictModel):
    """Semistandard-tableau membership bound to the checked structural source."""

    tableau: TableauCandidate
    is_member: bool


class PartitionDominanceRequest(StrictModel):
    """Compare two equal-size partitions in dominance order."""

    left: IntegerPartition
    right: IntegerPartition


DominanceRelation = Literal[
    "LEFT_DOMINATES",
    "RIGHT_DOMINATES",
    "EQUAL",
    "INCOMPARABLE",
    "NOT_COMPARABLE_DIFFERENT_SIZE",
]


class PartitionDominanceResult(StrictModel):
    """Dominance relation bound to the compared partitions."""

    left: IntegerPartition
    right: IntegerPartition
    relation: DominanceRelation


class StandardTableauCheckRequest(StrictModel):
    """Check a candidate standard Young tableau."""

    tableau: TableauCandidate


class SemistandardTableauCheckRequest(StrictModel):
    """Check a candidate semistandard Young tableau."""

    tableau: TableauCandidate


# ---------------------------------------------------------------------------
# RSK operations
# ---------------------------------------------------------------------------


class RSKPermutationRequest(StrictModel):
    __doc__ = f"""One strict bounded permutation for ordinary row-insertion RSK.

    Forward insertion performs at most
    ``N(N-1)/2 <= {
        MAX_RSK_PERMUTATION_LENGTH * (MAX_RSK_PERMUTATION_LENGTH - 1) // 2
    }`` binary row searches, with at most
    {MAX_RSK_ROW_SEARCH_COMPARISONS} integer comparisons per search for
    ``N <= {MAX_RSK_PERMUTATION_LENGTH}``.
    """

    permutation: tuple[StrictInt, ...] = Field(
        min_length=0, max_length=MAX_RSK_PERMUTATION_LENGTH
    )
    convention: RSKConvention = "ROW_INSERTION_RSK_V1"


class RSKResult(StrictModel):
    """Canonical tableaux produced by one admitted permutation-RSK kernel."""

    permutation: tuple[StrictInt, ...] = Field(
        min_length=0,
        max_length=MAX_RSK_PERMUTATION_LENGTH,
        description="The exact source permutation of 1 through n.",
    )
    p_tableau: StandardYoungTableau
    q_tableau: StandardYoungTableau
    shape: IntegerPartition
    lis_length: StrictInt = Field(ge=0, le=MAX_RSK_PERMUTATION_LENGTH)
    lds_length: StrictInt = Field(ge=0, le=MAX_RSK_PERMUTATION_LENGTH)
    convention: RSKConvention = "ROW_INSERTION_RSK_V1"

    @model_validator(mode="after")
    def require_structural_consistency(self) -> Self:
        if self.p_tableau.shape != self.shape or self.q_tableau.shape != self.shape:
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_shape_mismatch",
                "tableaux and shape must agree",
            )
        if sum(self.shape.parts) != len(self.permutation):
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_size_mismatch",
                "tableau shape size must equal permutation length",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: RSKPermutationRequest,
        *,
        insertion_rows: tuple[tuple[int, ...], ...],
        recording_rows: tuple[tuple[int, ...], ...],
    ) -> Self:
        """Build one result after the admitted RSK kernel established it."""

        shape = IntegerPartition(parts=tuple(len(row) for row in insertion_rows))
        return cls.model_construct(
            permutation=request.permutation,
            p_tableau=StandardYoungTableau(rows=insertion_rows),
            q_tableau=StandardYoungTableau(rows=recording_rows),
            shape=shape,
            lis_length=shape.parts[0] if shape.parts else 0,
            lds_length=len(shape.parts),
            convention=request.convention,
        )


class RSKWordRequest(StrictModel):
    __doc__ = f"""One bounded word under the ordinary row-insertion convention.

    Forward insertion performs at most
    ``N(N-1)/2 <= {MAX_RSK_WORD_LENGTH * (MAX_RSK_WORD_LENGTH - 1) // 2}``
    binary row searches, with at most
    {MAX_RSK_ROW_SEARCH_COMPARISONS} integer comparisons per search for
    ``N <= {MAX_RSK_WORD_LENGTH}``.  The compact result contains exactly
    ``2N`` tableau cells; no insertion ledger is materialized.
    """

    word: FiniteWord = Field(
        description=(
            "A finite word over an explicit ordered tuple of unique strings; "
            "every positioned letter must be one of those exact symbols. The "
            f"word has at most {MAX_RSK_WORD_LENGTH} letters and the alphabet "
            "plus positioned letters carry at most "
            f"{MAX_RSK_WORD_PAYLOAD_SCALARS} Unicode scalar values."
        )
    )
    convention: RSKConvention = "ROW_INSERTION_RSK_V1"


class RSKInverseWordRequest(StrictModel):
    __doc__ = f"""One compatible compact word-RSK pair of at most
    {MAX_RSK_WORD_LENGTH} cells to invert.

    Reverse insertion performs at most ``N(A-1)`` binary row searches for
    ``N`` cells and an alphabet of ``A`` ranks, with at most
    {MAX_RSK_ROW_SEARCH_COMPARISONS} integer comparisons per search.
    """

    pair: RSKTableauPair
    convention: RSKConvention = "ROW_INSERTION_RSK_V1"


class RSKWordTraceRequest(StrictModel):
    """Trace every insertion in one bounded ordinary row-insertion word RSK.

    The trace includes at most ``n(n-1)/2`` bump steps. Admission bounds the
    exact row-search work and a conservative serialized result estimate
    before constructing any tableaux or ledger rows.
    """

    word: FiniteWord = Field(
        description=(
            "A finite word over an explicit ordered alphabet; letters are "
            "ranked by their positions in that alphabet. The complete trace "
            "must fit the operation's pre-admitted work and output bounds."
        )
    )
    convention: RSKConvention = "ROW_INSERTION_RSK_V1"


class RSKWordTraceResult(StrictModel):
    """Source-bound final RSK pair and one insertion event per source letter."""

    word: FiniteWord
    tableau_pair: RSKTableauPair
    insertion_events: tuple[RSKInsertionEvent, ...] = Field(
        max_length=MAX_RSK_WORD_LENGTH
    )
    convention: RSKConvention = "ROW_INSERTION_RSK_V1"

    @model_validator(mode="after")
    def require_trace_source_alignment(self) -> Self:
        if self.tableau_pair.alphabet != self.word.alphabet:
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_trace_alphabet_mismatch",
                "the final tableau pair must retain the source word alphabet",
            )
        if self.tableau_pair.convention != self.convention:
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_trace_convention_mismatch",
                "the trace and final tableau pair must use the same convention",
            )
        if len(self.insertion_events) != len(self.word.letters) or tuple(
            event.position for event in self.insertion_events
        ) != tuple(range(1, len(self.word.letters) + 1)):
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_trace_event_positions",
                "the trace must contain one event for each source position in order",
            )
        if any(
            event.letter != self.word.letters[event.position - 1]
            for event in self.insertion_events
        ):
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_trace_letter_mismatch",
                "each event must retain the letter at its source position",
            )
        bump_bound = sum(
            min(prefix_length, len(self.word.alphabet))
            for prefix_length in range(len(self.word.letters))
        )
        if sum(len(event.bump_path) for event in self.insertion_events) > bump_bound:
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_trace_total_bumps",
                "the total bump path exceeds the alphabet-height bound",
            )
        previous_lengths: tuple[int, ...] = ()
        for event in self.insertion_events:
            lengths = event.row_lengths
            if len(lengths) not in (len(previous_lengths), len(previous_lengths) + 1):
                raise PydanticCustomError(
                    "algebraic_combinatorics.rsk_trace_prefix_shape",
                    "successive trace shapes must grow by one cell in one row",
                )
            previous = previous_lengths + (
                (0,) if len(lengths) > len(previous_lengths) else ()
            )
            deltas = tuple(
                current - prior
                for prior, current in zip(previous, lengths, strict=True)
            )
            if (
                any(delta not in (0, 1) for delta in deltas)
                or sum(deltas) != 1
                or deltas[event.added_row] != 1
                or previous[event.added_row] != event.added_column
            ):
                raise PydanticCustomError(
                    "algebraic_combinatorics.rsk_trace_prefix_shape",
                    "successive trace shapes must grow by one cell in one row",
                )
            previous_lengths = lengths
        if previous_lengths != self.tableau_pair.shape.parts:
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_trace_final_shape",
                "the final trace prefix must match the tableau pair shape",
            )
        rank_by_letter = {
            letter: rank for rank, letter in enumerate(self.word.alphabet, start=1)
        }
        for event in self.insertion_events:
            carried = rank_by_letter[event.letter]
            for step in event.bump_path:
                if step.bumped_entry <= carried:
                    raise PydanticCustomError(
                        "algebraic_combinatorics.rsk_trace_entry_chain",
                        "each bumped entry must be strictly larger than the carried entry",
                    )
                carried = step.bumped_entry
            if event.added_entry != carried:
                raise PydanticCustomError(
                    "algebraic_combinatorics.rsk_trace_entry_chain",
                    "the terminal entry must equal the final carried entry",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: RSKWordTraceRequest,
        tableau_pair: RSKTableauPair,
        insertion_events: tuple[RSKInsertionEvent, ...],
    ) -> Self:
        """Build the result after one admitted traced insertion pass."""
        return cls.model_construct(
            word=request.word,
            tableau_pair=tableau_pair,
            insertion_events=insertion_events,
            convention=request.convention,
        )


class RSKWordInverseTraceResult(StrictModel):
    """The reconstructed source word and the exact reverse-insertion ledger."""

    word: FiniteWord
    tableau_pair: RSKTableauPair
    reverse_insertion_events: tuple[RSKReverseInsertionEvent, ...] = Field(
        max_length=MAX_RSK_WORD_LENGTH
    )
    convention: RSKConvention = "ROW_INSERTION_RSK_V1"

    @model_validator(mode="after")
    def require_trace_source_alignment(self) -> Self:
        count = sum(self.tableau_pair.shape.parts)
        if (
            self.word.alphabet != self.tableau_pair.alphabet
            or self.convention != self.tableau_pair.convention
        ):
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_reverse_trace_source",
                "the trace word, tableau pair, and convention must agree",
            )
        if len(self.word.letters) != count or tuple(
            event.position for event in self.reverse_insertion_events
        ) != tuple(range(count, 0, -1)):
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_reverse_trace_positions",
                "the reverse trace must contain one descending event per cell",
            )
        if any(
            event.letter != self.word.letters[event.position - 1]
            for event in self.reverse_insertion_events
        ):
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_reverse_trace_letters",
                "each event letter must match its reconstructed word position",
            )
        current_shape = self.tableau_pair.shape.parts
        for event in self.reverse_insertion_events:
            after_shape = event.row_lengths
            if event.removed_row == len(after_shape):
                prior_shape = (*after_shape, 1)
            elif event.removed_row < len(after_shape):
                prior = list(after_shape)
                prior[event.removed_row] += 1
                prior_shape = tuple(prior)
                previous_length = (
                    after_shape[event.removed_row - 1]
                    if event.removed_row > 0
                    else None
                )
                if (
                    previous_length is not None
                    and previous_length <= after_shape[event.removed_row]
                ):
                    raise PydanticCustomError(
                        "algebraic_combinatorics.rsk_reverse_trace_shape_chain",
                        "each removed cell must be a corner of the prior partition",
                    )
            else:
                raise PydanticCustomError(
                    "algebraic_combinatorics.rsk_reverse_trace_shape_chain",
                    "each removed cell must belong to the prior partition",
                )
            if prior_shape != current_shape:
                raise PydanticCustomError(
                    "algebraic_combinatorics.rsk_reverse_trace_shape_chain",
                    "reverse event shapes must form one removal chain from the tableau pair",
                )
            current_shape = after_shape
        if current_shape:
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_reverse_trace_shape_chain",
                "the reverse removal chain must end at the empty partition",
            )
        rank_by_letter = {
            letter: rank
            for rank, letter in enumerate(self.tableau_pair.alphabet, start=1)
        }
        if any(
            event.removed_entry > len(rank_by_letter)
            or event.output_entry > len(rank_by_letter)
            or any(
                step.displaced_entry > len(rank_by_letter)
                for step in event.reverse_bump_path
            )
            or rank_by_letter.get(event.letter) != event.output_entry
            for event in self.reverse_insertion_events
        ):
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_reverse_trace_ranks",
                "event ranks must belong to the retained alphabet and identify the emitted letter",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: RSKInverseWordRequest,
        word: FiniteWord,
        events: tuple[RSKReverseInsertionEvent, ...],
    ) -> Self:
        return cls.model_construct(
            word=word,
            tableau_pair=request.pair,
            reverse_insertion_events=events,
            convention=request.convention,
        )


MAX_LIS_WORD_LENGTH = MAX_RSK_WORD_LENGTH
MAX_LIS_WORD_PAYLOAD_SCALARS = MAX_RSK_WORD_PAYLOAD_SCALARS
MAX_LIS_DP_WORK = MAX_LIS_WORD_LENGTH * (MAX_LIS_WORD_LENGTH - 1) // 2
# The exact result retains the source word payload and copies at most n of
# its own letters as the witness values, so the emitted letter payload is
# bounded by the second copy of the source envelope; the n witness positions
# and the one length are integers not exceeding the word length and therefore
# carry at most MAX_LIS_INDEX_DIGITS decimal digits each.
MAX_LIS_INDEX_DIGITS = len(str(MAX_LIS_WORD_LENGTH))
MAX_LIS_OUTPUT_SCALARS = (
    2 * MAX_LIS_WORD_PAYLOAD_SCALARS + (MAX_LIS_WORD_LENGTH + 1) * MAX_LIS_INDEX_DIGITS
)


class LongestIncreasingSubsequenceRequest(StrictModel):
    """Compute a strict longest increasing subsequence under the word order."""

    word: FiniteWord = Field(
        description=(
            "A finite ordered-alphabet word. The strict convention requires "
            "each selected letter to be strictly greater than its predecessor. "
            f"The word has at most {MAX_LIS_WORD_LENGTH} letters and its "
            "alphabet plus positioned letters carry at most "
            f"{MAX_LIS_WORD_PAYLOAD_SCALARS} Unicode scalar values; the exact "
            "result is bounded by the same retained-word cardinality plus "
            f"{MAX_LIS_INDEX_DIGITS}-digit indices."
        )
    )


class LongestIncreasingSubsequenceResult(StrictModel):
    """Exact length and a deterministic source-index witness for strict LIS."""

    source_word: FiniteWord
    length: StrictInt = Field(ge=0, le=MAX_LIS_WORD_LENGTH)
    indices: tuple[StrictInt, ...] = Field(max_length=MAX_LIS_WORD_LENGTH)
    values: tuple[Symbol, ...] = Field(max_length=MAX_LIS_WORD_LENGTH)

    @model_validator(mode="after")
    def require_valid_witness(self) -> Self:
        if len(self.indices) != self.length or len(self.values) != self.length:
            raise PydanticCustomError(
                "algebraic_combinatorics.lis_witness_length",
                "LIS witness length must equal the reported length",
            )
        if any(
            index < 0
            or index >= len(self.source_word.letters)
            or self.source_word.letters[index] != value
            for index, value in zip(self.indices, self.values, strict=True)
        ):
            raise PydanticCustomError(
                "algebraic_combinatorics.lis_witness_source",
                "LIS witness positions and values must replay in the source word",
            )
        if any(left >= right for left, right in pairwise(self.indices)):
            raise PydanticCustomError(
                "algebraic_combinatorics.lis_witness_indices",
                "LIS witness positions must be strictly increasing",
            )
        ranks = {letter: rank for rank, letter in enumerate(self.source_word.alphabet)}
        if any(ranks[left] >= ranks[right] for left, right in pairwise(self.values)):
            raise PydanticCustomError(
                "algebraic_combinatorics.lis_witness_order",
                "LIS witness values must be strictly increasing in the word alphabet",
            )
        return self


class LongestDecreasingSubsequenceRequest(StrictModel):
    """Compute a strict longest decreasing subsequence under the word order."""

    word: FiniteWord = Field(
        description=(
            "A finite ordered-alphabet word. Strict decrease requires each "
            "selected letter to be strictly smaller than its predecessor. "
            f"The word has at most {MAX_LIS_WORD_LENGTH} letters and its "
            "alphabet plus positioned letters carry at most "
            f"{MAX_LIS_WORD_PAYLOAD_SCALARS} Unicode scalar values; the exact "
            "result is bounded by the same retained-word cardinality plus "
            f"{MAX_LIS_INDEX_DIGITS}-digit indices."
        )
    )


class LongestDecreasingSubsequenceResult(StrictModel):
    """Exact length and deterministic source-index witness for strict LDS."""

    source_word: FiniteWord
    length: StrictInt = Field(ge=0, le=MAX_LIS_WORD_LENGTH)
    indices: tuple[StrictInt, ...] = Field(max_length=MAX_LIS_WORD_LENGTH)
    values: tuple[Symbol, ...] = Field(max_length=MAX_LIS_WORD_LENGTH)

    @model_validator(mode="after")
    def require_valid_witness(self) -> Self:
        if len(self.indices) != self.length or len(self.values) != self.length:
            raise PydanticCustomError(
                "algebraic_combinatorics.lds_witness_length",
                "LDS witness length must equal the reported length",
            )
        if any(
            index < 0
            or index >= len(self.source_word.letters)
            or self.source_word.letters[index] != value
            for index, value in zip(self.indices, self.values, strict=True)
        ):
            raise PydanticCustomError(
                "algebraic_combinatorics.lds_witness_source",
                "LDS witness positions and values must replay in the source word",
            )
        if any(left >= right for left, right in pairwise(self.indices)):
            raise PydanticCustomError(
                "algebraic_combinatorics.lds_witness_indices",
                "LDS witness positions must be strictly increasing",
            )
        ranks = {letter: rank for rank, letter in enumerate(self.source_word.alphabet)}
        if any(ranks[left] <= ranks[right] for left, right in pairwise(self.values)):
            raise PydanticCustomError(
                "algebraic_combinatorics.lds_witness_order",
                "LDS witness values must be strictly decreasing in the word alphabet",
            )
        return self


KnuthRelation = Literal["K1", "K2"]


class KnuthNeighbor(StrictModel):
    """One one-step Knuth neighbor with its local position and relation."""

    position: StrictInt = Field(
        ge=0,
        description="Left index of the length-three window carrying the relation.",
    )
    relation: KnuthRelation = Field(
        description=(
            "K1 swaps the first two letters of xzy<->zxy with x<=y<z; "
            "K2 swaps the last two letters of yxz<->yzx with x<y<=z, "
            "compared in the word's explicit alphabet order."
        )
    )
    neighbor: FiniteWord = Field(
        description="The full word after applying the relation once."
    )


class KnuthMovesRequest(StrictModel):
    __doc__ = f"""One bounded ordered word whose Knuth neighborhood is materialized.

    The word has at most {MAX_RSK_WORD_LENGTH} letters; the neighborhood holds
    at most one row per length-three window, so output is linear in the word.
    """

    word: FiniteWord = Field(
        description=(
            "A finite word over an explicit ordered tuple of unique strings; "
            "alphabet order determines every relation side condition."
        )
    )
    convention: RSKConvention = "ROW_INSERTION_RSK_V1"


class KnuthMovesResult(StrictModel):
    """Every valid one-step Knuth neighbor bound to its source word."""

    word: FiniteWord
    neighbors: tuple[KnuthNeighbor, ...]
    neighbor_count: StrictInt = Field(ge=0)
    convention: RSKConvention = "ROW_INSERTION_RSK_V1"

    @model_validator(mode="after")
    def require_structural_neighbors(self) -> Self:
        if self.neighbor_count != len(self.neighbors):
            raise PydanticCustomError(
                "algebraic_combinatorics.knuth_count_mismatch",
                "neighbor count must equal the number of neighbor rows",
            )
        positions = [neighbor.position for neighbor in self.neighbors]
        if positions != sorted(positions) or len(set(positions)) != len(positions):
            raise PydanticCustomError(
                "algebraic_combinatorics.knuth_positions_not_canonical",
                "neighbor positions must be strictly increasing",
            )
        for neighbor in self.neighbors:
            if neighbor.position + 2 >= len(self.word.letters):
                raise PydanticCustomError(
                    "algebraic_combinatorics.knuth_position_out_of_range",
                    "neighbor position must start a length-three window",
                )
            if neighbor.neighbor.alphabet != self.word.alphabet or len(
                neighbor.neighbor.letters
            ) != len(self.word.letters):
                raise PydanticCustomError(
                    "algebraic_combinatorics.knuth_neighbor_shape_mismatch",
                    "every neighbor must share the source alphabet and length",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: KnuthMovesRequest,
        *,
        neighbors: tuple[KnuthNeighbor, ...],
    ) -> Self:
        """Build one result after the admitted Knuth kernel established it."""

        return cls.model_construct(
            word=request.word,
            neighbors=neighbors,
            neighbor_count=len(neighbors),
            convention=request.convention,
        )


class PlacticNormalFormRequest(StrictModel):
    """Compute the row-reading representative of one word's plactic class."""

    word: FiniteWord
    convention: RSKConvention = "ROW_INSERTION_RSK_V1"


class PlacticNormalFormResult(StrictModel):
    """Source-bound insertion tableau and canonical row-reading word.

    The canonical representative reads each tableau row left-to-right, from
    the bottom row to the top row. Tableau entries are one-based ranks in the
    exact alphabet retained by ``normal_form``.
    """

    source_word: FiniteWord
    insertion_tableau: SemistandardYoungTableau
    normal_form: FiniteWord
    convention: RSKConvention = "ROW_INSERTION_RSK_V1"

    @model_validator(mode="after")
    def require_row_reading_word(self) -> Self:
        if self.source_word.alphabet != self.normal_form.alphabet:
            raise PydanticCustomError(
                "algebraic_combinatorics.plactic_alphabet_mismatch",
                "source and normal-form words must use the same ordered alphabet",
            )
        if len(self.source_word.letters) != sum(
            len(row) for row in self.insertion_tableau.rows
        ):
            raise PydanticCustomError(
                "algebraic_combinatorics.plactic_source_size_mismatch",
                "source word length must equal the insertion-tableau cell count",
            )
        expected = tuple(
            self.normal_form.alphabet[entry - 1]
            for row in reversed(self.insertion_tableau.rows)
            for entry in row
        )
        if self.normal_form.letters != expected:
            raise PydanticCustomError(
                "algebraic_combinatorics.plactic_row_reading_mismatch",
                "normal_form must be the bottom-to-top, left-to-right row reading",
            )
        return self


class PlacticEquivalenceRequest(StrictModel):
    """Compare two words in one explicitly ordered plactic alphabet."""

    left: FiniteWord
    right: FiniteWord
    convention: RSKConvention = "ROW_INSERTION_RSK_V1"

    @model_validator(mode="after")
    def require_shared_alphabet(self) -> Self:
        if self.left.alphabet != self.right.alphabet:
            raise PydanticCustomError(
                "algebraic_combinatorics.plactic_alphabet_mismatch",
                "plactic equivalence requires the same ordered alphabet",
            )
        return self


class PlacticEquivalenceResult(StrictModel):
    """Both source-bound canonical forms and their exact plactic relation."""

    left: PlacticNormalFormResult
    right: PlacticNormalFormResult
    equivalent: bool
    convention: RSKConvention = "ROW_INSERTION_RSK_V1"

    @model_validator(mode="after")
    def require_shared_alphabet_and_relation(self) -> Self:
        if self.left.source_word.alphabet != self.right.source_word.alphabet:
            raise PydanticCustomError(
                "algebraic_combinatorics.plactic_alphabet_mismatch",
                "plactic equivalence requires the same ordered alphabet",
            )
        if self.equivalent != (
            self.left.insertion_tableau == self.right.insertion_tableau
        ):
            raise PydanticCustomError(
                "algebraic_combinatorics.plactic_relation_mismatch",
                "equivalent must agree with equality of insertion tableaux",
            )
        if (
            self.left.convention != self.convention
            or self.right.convention != self.convention
        ):
            raise PydanticCustomError(
                "algebraic_combinatorics.plactic_convention_mismatch",
                "both normal forms must use the declared RSK convention",
            )
        return self


# ---------------------------------------------------------------------------
# Skew Littlewood-Richardson checking
# ---------------------------------------------------------------------------

SkewReadingConvention = Literal["READING_WORD_RL_TOP_V1"]

SkewLRFailureKind = Literal[
    "OK",
    "CELL_COVERAGE",
    "SEMISTANDARD_ROW",
    "SEMISTANDARD_COLUMN",
    "CONTENT",
    "LATTICE",
]


class SkewLittlewoodRichardsonCheckRequest(StrictModel):
    """Check LR-tableau membership on a skew shape under one reading convention.

    ``tableau`` rows carry only skew cells: row ``i`` holds exactly
    ``outer.parts[i] - inner.parts[i]`` entries (with ``inner`` padded by
    zeros), left to right within the skew row. The reading word concatenates
    rows top to bottom, each right to left, under READING_WORD_RL_TOP_V1.
    """

    outer: IntegerPartition
    inner: IntegerPartition
    tableau: TableauCandidate
    content: IntegerPartition = Field(
        description=(
            "Claimed content nu as a partition: exactly nu_i copies of value i."
        ),
    )
    convention: SkewReadingConvention = "READING_WORD_RL_TOP_V1"


class SkewLittlewoodRichardsonCheckResult(StrictModel):
    """Skew-LR membership bound to the checked skew source.

    Deserialization establishes only the retained source and bounded result
    shape. Kernel output uses ``_from_kernel`` after its trusted bounded
    replay of cell coverage, semistandardity, content, and every prefix
    lattice inequality.
    """

    outer: IntegerPartition
    inner: IntegerPartition
    tableau: TableauCandidate
    content: IntegerPartition
    is_member: bool
    reading_word: tuple[StrictInt, ...] = Field(
        max_length=MAX_CANONICAL_PARTITION_SIZE,
        description="Reading word under the declared convention.",
    )
    failure_kind: SkewLRFailureKind = Field(
        description=(
            "OK for members, else the first failed replay stage: cell "
            "coverage, row semistandardity, column semistandardity, content, "
            "or a prefix lattice inequality."
        ),
    )
    failed_row: StrictInt = Field(
        ge=-1,
        description="Zero-based failing row, or -1 when not a row/cell failure.",
    )
    failed_column: StrictInt = Field(
        ge=-1,
        description="Zero-based failing skew-row position, or -1 otherwise.",
    )
    failed_value: StrictInt = Field(
        ge=-1,
        description="Failing entry value for content failures, else -1.",
    )
    failed_prefix_length: StrictInt = Field(
        ge=0,
        description="Length of the first lattice-violating prefix, else 0.",
    )
    convention: SkewReadingConvention = "READING_WORD_RL_TOP_V1"

    @classmethod
    def _from_kernel(
        cls,
        outer: IntegerPartition,
        inner: IntegerPartition,
        tableau: TableauCandidate,
        content: IntegerPartition,
        convention: SkewReadingConvention,
        *,
        is_member: bool,
        reading_word: tuple[int, ...],
        failure_kind: SkewLRFailureKind,
        failed_row: int = -1,
        failed_column: int = -1,
        failed_value: int = -1,
        failed_prefix_length: int = 0,
    ) -> Self:
        """Build one result after the admitted skew-LR kernel established it."""

        return cls.model_construct(
            outer=outer,
            inner=inner,
            tableau=tableau,
            content=content,
            convention=convention,
            is_member=is_member,
            reading_word=reading_word,
            failure_kind=failure_kind,
            failed_row=failed_row,
            failed_column=failed_column,
            failed_value=failed_value,
            failed_prefix_length=failed_prefix_length,
        )


__all__ = [
    "ConjugatePartitionRequest",
    "ConjugatePartitionResult",
    "DominanceRelation",
    "HookLengthRequest",
    "HookLengthResult",
    "KnuthMovesRequest",
    "KnuthMovesResult",
    "KnuthNeighbor",
    "KnuthRelation",
    "PartitionDominanceRequest",
    "PartitionDominanceResult",
    "PlacticNormalFormRequest",
    "PlacticNormalFormResult",
    "RSKInverseWordRequest",
    "RSKPermutationRequest",
    "RSKResult",
    "RSKWordInverseTraceResult",
    "RSKWordRequest",
    "RSKWordTraceRequest",
    "RSKWordTraceResult",
    "SemistandardTableauCheckRequest",
    "SemistandardTableauCheckResult",
    "SemistandardYoungTableauCountRequest",
    "SemistandardYoungTableauCountResult",
    "SkewLRFailureKind",
    "SkewLittlewoodRichardsonCheckRequest",
    "SkewLittlewoodRichardsonCheckResult",
    "SkewReadingConvention",
    "StandardTableauCheckRequest",
    "StandardTableauCheckResult",
    "StandardYoungTableauCountRequest",
    "StandardYoungTableauCountResult",
]
