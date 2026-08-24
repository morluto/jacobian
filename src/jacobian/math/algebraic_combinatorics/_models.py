"""Typed wire contracts for exact algebraic combinatorics operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.math.algebraic_combinatorics._rsk import require_rsk_word_budget
from jacobian.math.algebraic_combinatorics.values import (
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
        raise ValueError("permutation must be a permutation of 1..n")


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
    """One strict bounded permutation for ordinary row-insertion RSK.

    Forward and replayed reverse insertion each perform at most
    ``N(N-1)/2 <= 124750`` binary row searches, with at most nine integer
    comparisons per search for ``N <= 500``.
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
    """Source-bound canonical tableaux from permutation RSK.

    Construction and result replay each perform at most
    ``N(N-1)/2 <= 124750`` binary row searches, with at most nine integer
    comparisons per search for ``N <= 500``.
    """

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
    def replay_permutation_rsk(self) -> Self:
        from jacobian.math.algebraic_combinatorics._rsk import _row_insert

        _require_permutation(self.permutation)
        insertion_rows, recording_rows = _row_insert(self.permutation)
        expected_p = StandardYoungTableau(rows=insertion_rows)
        expected_q = StandardYoungTableau(rows=recording_rows)
        expected_shape = expected_p.shape
        expected_lis = len(insertion_rows[0]) if insertion_rows else 0
        expected_lds = len(insertion_rows)
        if self.p_tableau != expected_p or self.q_tableau != expected_q:
            raise ValueError("permutation tableaux do not match exact row insertion")
        if self.shape != expected_shape or self.q_tableau.shape != expected_shape:
            raise ValueError("permutation tableaux and shape must agree")
        if self.lis_length != expected_lis or self.lds_length != expected_lds:
            raise ValueError("LIS/LDS lengths do not match the exact RSK shape")
        return self


class RSKWordRequest(StrictModel):
    """One bounded word under the ordinary row-insertion convention.

    Forward and replayed reverse insertion each perform at most
    ``N(N-1)/2 <= 124750`` binary row searches, with at most nine integer
    comparisons per search for ``N <= 500``.  The compact result contains
    exactly ``2N`` tableau cells; no insertion ledger is materialized.
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

    @model_validator(mode="after")
    def require_complete_budget(self) -> Self:
        require_rsk_word_budget(self.word)
        return self


class RSKInverseWordRequest(StrictModel):
    """One compatible compact word-RSK pair of at most 500 cells to invert.

    Reverse insertion and its forward replay each perform at most
    ``N(N-1)/2 <= 124750`` binary row searches, with at most nine integer
    comparisons per search.
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
