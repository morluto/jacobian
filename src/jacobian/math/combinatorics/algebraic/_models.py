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
    SemistandardYoungTableau,
    StandardYoungTableau,
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

    tableau: StandardYoungTableau
    is_member: bool


class SemistandardTableauCheckResult(StrictModel):
    """Semistandard-tableau membership bound to the checked structural source."""

    tableau: SemistandardYoungTableau
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

    tableau: StandardYoungTableau


class SemistandardTableauCheckRequest(StrictModel):
    """Check a candidate semistandard Young tableau."""

    tableau: SemistandardYoungTableau


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


__all__ = [
    "ConjugatePartitionRequest",
    "ConjugatePartitionResult",
    "DominanceRelation",
    "HookLengthRequest",
    "HookLengthResult",
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
    "StandardTableauCheckRequest",
    "StandardTableauCheckResult",
    "StandardYoungTableauCountRequest",
    "StandardYoungTableauCountResult",
]
