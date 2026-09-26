"""Exact ordinary row-insertion RSK kernels."""

from __future__ import annotations

from bisect import bisect_left, bisect_right

from jacobian._execution import request_checkpoint
from jacobian.math.combinatorics.algebraic.values import (
    MAX_RSK_WORD_LENGTH,
    MAX_RSK_WORD_PAYLOAD_SCALARS,
    FinitePermutation,
    PermutationRSKPair,
    RSKTableauPair,
)
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    SemistandardYoungTableau,
    StandardYoungTableau,
    require_semistandard,
    require_standard,
)
from jacobian.math.logic.languages.words.values import FiniteWord


def word_payload_scalars(word: FiniteWord) -> int:
    """Return the Unicode scalar values carried by the payload.

    The count covers the alphabet and every positioned letter. A surrogate
    code point is not a Unicode scalar value, so a payload containing one is
    rejected before any count is returned.
    """
    payload = 0
    for symbol in (*word.alphabet, *word.letters):
        if any("\ud800" <= character <= "\udfff" for character in symbol):
            raise ValueError(
                "RSK word symbols must be Unicode scalar values without surrogates"
            )
        payload += len(symbol)
    return payload


def require_rsk_word_budget(word: FiniteWord) -> None:
    """Validate the complete work and source-payload envelope before insertion."""
    if len(word.letters) > MAX_RSK_WORD_LENGTH:
        raise ValueError(f"RSK word length must not exceed {MAX_RSK_WORD_LENGTH}")
    if word_payload_scalars(word) > MAX_RSK_WORD_PAYLOAD_SCALARS:
        raise ValueError(
            "RSK word payload must not exceed "
            f"{MAX_RSK_WORD_PAYLOAD_SCALARS} Unicode scalar values"
        )


def _row_insert(
    entries: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Insert ranks left-to-right, bumping the first strictly greater entry."""
    insertion: list[list[int]] = []
    recording: list[list[int]] = []

    for position, entry in enumerate(entries, start=1):
        current = entry
        row_index = 0
        while row_index < len(insertion):
            row = insertion[row_index]
            column = bisect_right(row, current)
            if column == len(row):
                row.append(current)
                recording[row_index].append(position)
                break
            row[column], current = current, row[column]
            row_index += 1
        else:
            insertion.append([current])
            recording.append([position])

    return (
        tuple(tuple(row) for row in insertion),
        tuple(tuple(row) for row in recording),
    )


def _forward(word: FiniteWord) -> RSKTableauPair:
    require_rsk_word_budget(word)
    rank = {symbol: index for index, symbol in enumerate(word.alphabet, start=1)}
    insertion_rows, recording_rows = _row_insert(
        tuple(rank[letter] for letter in word.letters)
    )
    shape = IntegerPartition(parts=tuple(len(row) for row in insertion_rows))
    return RSKTableauPair(
        alphabet=word.alphabet,
        insertion_tableau=SemistandardYoungTableau(rows=insertion_rows),
        recording_tableau=StandardYoungTableau(rows=recording_rows),
        shape=shape,
    )


def _inverse(pair: RSKTableauPair) -> FiniteWord:
    require_semistandard(pair.insertion_tableau)
    require_standard(pair.recording_tableau)
    if any(
        entry > len(pair.alphabet)
        for row in pair.insertion_tableau.rows
        for entry in row
    ):
        raise ValueError("insertion tableau entry is outside the ordered alphabet")
    reversed_ranks = _reverse_insert_ranks(
        pair.insertion_tableau.rows, pair.recording_tableau.rows
    )
    letters = tuple(pair.alphabet[rank - 1] for rank in reversed(reversed_ranks))
    return FiniteWord(alphabet=pair.alphabet, letters=letters)


def _reverse_insert_ranks(
    insertion_rows: tuple[tuple[int, ...], ...],
    recording_rows: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    """Return output ranks in reverse source order for a compatible pair."""

    cell_count = sum(map(len, recording_rows))
    insertion = [list(row) for row in insertion_rows]
    label_rows_by_entry = [0] * cell_count
    for row_index, row in enumerate(recording_rows):
        for label in row:
            label_rows_by_entry[label - 1] = row_index
    reversed_ranks: list[int] = []

    for label in range(cell_count, 0, -1):
        request_checkpoint("during reverse row insertion")
        row_index = label_rows_by_entry[label - 1]
        if row_index >= len(insertion) or not insertion[row_index]:
            raise ValueError("recording tableau does not select an outer corner")
        current = insertion[row_index].pop()
        if not insertion[row_index]:
            if row_index != len(insertion) - 1:
                raise ValueError("reverse insertion produced a non-partition shape")
            insertion.pop()

        for upper_index in range(row_index - 1, -1, -1):
            upper_row = insertion[upper_index]
            column = bisect_left(upper_row, current) - 1
            if column < 0:
                raise ValueError("tableau pair failed reverse row insertion")
            upper_row[column], current = current, upper_row[column]
        reversed_ranks.append(current)

    if insertion:
        raise ValueError("reverse insertion did not remove every tableau cell")
    return tuple(reversed_ranks)


def row_insertion_rsk(word: FiniteWord) -> RSKTableauPair:
    """Compute compact ordinary RSK using ``ROW_INSERTION_RSK_V1``.

    Letters are replaced by their one-based ranks in the word's explicit
    alphabet. Each rank is inserted left-to-right, bumping the first entry
    strictly greater than it.
    """
    return _forward(word)


def inverse_row_insertion_rsk(pair: RSKTableauPair) -> FiniteWord:
    """Invert one compatible ordinary-word RSK tableau pair exactly.

    Recording labels are removed from largest to smallest.  At each preceding
    row, reverse insertion bumps the rightmost entry strictly smaller than the
    current entry.
    """
    return _inverse(pair)


def inverse_permutation_rsk(pair: PermutationRSKPair) -> FinitePermutation:
    """Invert a standard-tableau permutation pair by reverse row insertion."""

    reversed_images = _reverse_insert_ranks(pair.p_tableau.rows, pair.q_tableau.rows)
    return FinitePermutation._from_kernel(tuple(reversed(reversed_images)))


__all__ = [
    "inverse_permutation_rsk",
    "inverse_row_insertion_rsk",
    "row_insertion_rsk",
]
