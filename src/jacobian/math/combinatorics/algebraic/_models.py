"""Typed wire contracts for exact algebraic combinatorics operations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, WithJsonSchema, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS, ExactInteger
from jacobian._models import StrictModel
from jacobian.math.combinatorics.algebraic.values import (
    MAX_RSK_ROW_SEARCH_COMPARISONS,
    MAX_RSK_WORD_BYTES,
    MAX_RSK_WORD_LENGTH,
    RSKConvention,
    RSKTableauPair,
)
from jacobian.math.combinatorics.symmetric_functions.values import (
    MAX_PARTITION_SIZE as MAX_CANONICAL_PARTITION_SIZE,
)
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    StandardYoungTableau,
    TableauCandidate,
)
from jacobian.math.logic.languages.words.values import FiniteWord

# A permutation of length N inserts N ranks through the same _row_insert
# kernel and produces two N-cell tableaux, so the canonical tableau cell
# budget derives the permutation envelope.
MAX_RSK_PERMUTATION_LENGTH = MAX_RSK_WORD_LENGTH

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
            f"{MAX_RSK_WORD_BYTES} UTF-8 bytes."
        )
    )
    convention: RSKConvention = "ROW_INSERTION_RSK_V1"


class RSKInverseWordRequest(StrictModel):
    __doc__ = f"""One compatible compact word-RSK pair of at most
    {MAX_RSK_WORD_LENGTH} cells to invert.

    Reverse insertion and its forward replay each perform at most
    ``N(N-1)/2 <= {MAX_RSK_WORD_LENGTH * (MAX_RSK_WORD_LENGTH - 1) // 2}``
    binary row searches, with at most
    {MAX_RSK_ROW_SEARCH_COMPARISONS} integer comparisons per search.
    """

    pair: RSKTableauPair
    convention: RSKConvention = "ROW_INSERTION_RSK_V1"


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
        request: SkewLittlewoodRichardsonCheckRequest,
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
            outer=request.outer,
            inner=request.inner,
            tableau=request.tableau,
            content=request.content,
            is_member=is_member,
            reading_word=reading_word,
            failure_kind=failure_kind,
            failed_row=failed_row,
            failed_column=failed_column,
            failed_value=failed_value,
            failed_prefix_length=failed_prefix_length,
            convention=request.convention,
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
    "RSKInverseWordRequest",
    "RSKPermutationRequest",
    "RSKResult",
    "RSKWordRequest",
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
