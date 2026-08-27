"""Typed wire contracts for exact algebraic combinatorics operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.math.algebraic_combinatorics.values import (
    MAX_RSK_ROW_SEARCH_COMPARISONS,
    MAX_RSK_WORD_BYTES,
    MAX_RSK_WORD_LENGTH,
    RSKConvention,
    RSKTableauPair,
)
from jacobian.math.symmetric_functions.values import (
    MAX_PARTITION_SIZE as MAX_CANONICAL_PARTITION_SIZE,
)
from jacobian.math.symmetric_functions.values import (
    IntegerPartition,
    StandardYoungTableau,
)
from jacobian.math.words.values import FiniteWord

# A permutation of length N inserts N ranks through the same _row_insert
# kernel and produces two N-cell tableaux, so the canonical tableau cell
# budget derives the permutation envelope.
MAX_RSK_PERMUTATION_LENGTH = MAX_RSK_WORD_LENGTH


def _require_permutation(permutation: tuple[int, ...]) -> None:
    if not permutation:
        return
    if sorted(permutation) != list(range(1, len(permutation) + 1)):
        raise PydanticCustomError(
            "algebraic_combinatorics.permutation_invalid",
            "permutation must be a permutation of 1..n",
        )


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
    total_product: CanonicalInteger = Field(description="Product of all hook lengths.")
    method: Literal["HOOK_FORMULA"] = "HOOK_FORMULA"


class StandardYoungTableauCountResult(StrictModel):
    """The number of standard Young tableaux of a given shape."""

    count: CanonicalInteger = Field(description="Number of standard Young tableaux.")
    n: int = Field(ge=0, le=MAX_CANONICAL_PARTITION_SIZE)
    method: Literal["HOOK_LENGTH_FORMULA"] = "HOOK_LENGTH_FORMULA"


class ConjugatePartitionResult(StrictModel):
    """The conjugate (transpose) partition."""

    conjugate: IntegerPartition
    method: Literal["FERRERS_TRANSPOSE"] = "FERRERS_TRANSPOSE"


# ---------------------------------------------------------------------------
# RSK operations
# ---------------------------------------------------------------------------


class RSKPermutationRequest(StrictModel):
    __doc__ = f"""One strict bounded permutation for ordinary row-insertion RSK.

    Forward and replayed reverse insertion each perform at most
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

    @model_validator(mode="after")
    def require_valid_permutation(self) -> Self:
        _require_permutation(self.permutation)
        return self


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
        _require_permutation(self.permutation)
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
        expected_lis = self.shape.parts[0] if self.shape.parts else 0
        expected_lds = len(self.shape.parts)
        if self.lis_length != expected_lis or self.lds_length != expected_lds:
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_lengths_mismatch",
                "LIS/LDS lengths do not match the tableau shape",
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
        return cls(
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

    Forward and replayed reverse insertion each perform at most
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
    "HookLengthRequest",
    "HookLengthResult",
    "RSKInverseWordRequest",
    "RSKPermutationRequest",
    "RSKResult",
    "RSKWordRequest",
    "StandardYoungTableauCountRequest",
    "StandardYoungTableauCountResult",
]
