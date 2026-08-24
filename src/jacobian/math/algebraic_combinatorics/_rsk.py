"""Exact ordinary row-insertion RSK kernels."""

from __future__ import annotations

from bisect import bisect_left, bisect_right

from jacobian.math.algebraic_combinatorics.values import (
    MAX_RSK_WORD_BYTES,
    MAX_RSK_WORD_LENGTH,
    RSKTableauPair,
)
from jacobian.math.symmetric_functions.values import (
    IntegerPartition,
    SemistandardYoungTableau,
    StandardYoungTableau,
)
from jacobian.math.words.values import FiniteWord


def word_payload_bytes(word: FiniteWord) -> int:
    """Return the UTF-8 bytes carried by the alphabet and positioned letters."""
    try:
        alphabet_bytes = sum(len(symbol.encode("utf-8")) for symbol in word.alphabet)
        letter_bytes = sum(len(letter.encode("utf-8")) for letter in word.letters)
    except UnicodeEncodeError as error:
        raise ValueError(
            "RSK word symbols must be Unicode scalar values without surrogates"
        ) from error
    return alphabet_bytes + letter_bytes


def require_rsk_word_budget(word: FiniteWord) -> None:
    """Validate the complete work and source-payload envelope before insertion."""
    if len(word.letters) > MAX_RSK_WORD_LENGTH:
        raise ValueError(f"RSK word length must not exceed {MAX_RSK_WORD_LENGTH}")
    if word_payload_bytes(word) > MAX_RSK_WORD_BYTES:
        raise ValueError(
            f"RSK word payload must not exceed {MAX_RSK_WORD_BYTES} UTF-8 bytes"
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


def _forward_without_replay(word: FiniteWord) -> RSKTableauPair:
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


def _inverse_without_replay(pair: RSKTableauPair) -> FiniteWord:
    cell_count = sum(pair.shape.parts)
    insertion = [list(row) for row in pair.insertion_tableau.rows]
    label_rows_by_entry = [0] * cell_count
    for row_index, row in enumerate(pair.recording_tableau.rows):
        for label in row:
            label_rows_by_entry[label - 1] = row_index
    reversed_ranks: list[int] = []

    for label in range(cell_count, 0, -1):
        row_index = label_rows_by_entry[label - 1]
        current = insertion[row_index].pop()
        if not insertion[row_index]:
            if row_index != len(insertion) - 1:
                raise RuntimeError("reverse insertion produced a non-partition shape")
            insertion.pop()

        for upper_index in range(row_index - 1, -1, -1):
            upper_row = insertion[upper_index]
            column = bisect_left(upper_row, current) - 1
            if column < 0:
                raise RuntimeError("semistandard pair failed reverse row insertion")
            upper_row[column], current = current, upper_row[column]
        reversed_ranks.append(current)

    letters = tuple(pair.alphabet[rank - 1] for rank in reversed(reversed_ranks))
    return FiniteWord(alphabet=pair.alphabet, letters=letters)


def row_insertion_rsk(word: FiniteWord) -> RSKTableauPair:
    """Compute compact ordinary RSK using ``ROW_INSERTION_RSK_V1``.

    Letters are replaced by their one-based ranks in the word's explicit
    alphabet.  Each rank is inserted left-to-right, bumping the first entry
    strictly greater than it.  The result is replayed through inverse RSK
    before it is returned.
    """
    pair = _forward_without_replay(word)
    if _inverse_without_replay(pair) != word:
        raise RuntimeError("row-insertion RSK failed its inverse replay")
    return pair


def inverse_row_insertion_rsk(pair: RSKTableauPair) -> FiniteWord:
    """Invert one compatible ordinary-word RSK tableau pair exactly.

    Recording labels are removed from largest to smallest.  At each preceding
    row, reverse insertion bumps the rightmost entry strictly smaller than the
    current entry.  The reconstructed word is replayed through forward RSK
    before it is returned.
    """
    word = _inverse_without_replay(pair)
    if _forward_without_replay(word) != pair:
        raise RuntimeError("inverse row insertion failed its forward replay")
    return word


__all__ = ["inverse_row_insertion_rsk", "row_insertion_rsk"]
