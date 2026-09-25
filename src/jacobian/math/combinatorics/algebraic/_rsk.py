"""Exact ordinary row-insertion RSK kernels."""

from __future__ import annotations

from bisect import bisect_left, bisect_right

from pydantic import ValidationError

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.algebraic._models import (
    MAX_RSK_TRACE_RESULT_BYTES,
    MAX_RSK_TRACE_WORK,
    RSKInverseWordRequest,
    RSKWordInverseTraceResult,
    RSKWordTraceRequest,
    RSKWordTraceResult,
)
from jacobian.math.combinatorics.algebraic.values import (
    MAX_RSK_ROW_SEARCH_COMPARISONS,
    MAX_RSK_WORD_LENGTH,
    MAX_RSK_WORD_PAYLOAD_SCALARS,
    RSKBumpStep,
    RSKInsertionEvent,
    RSKReverseBumpStep,
    RSKReverseInsertionEvent,
    RSKTableauPair,
)
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    SemistandardYoungTableau,
    StandardYoungTableau,
    require_semistandard,
    require_standard,
)
from jacobian.math.logic.languages.words.values import MAX_SYMBOL_LENGTH, FiniteWord


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


def _admit_inverse_trace(pair: RSKTableauPair) -> tuple[RSKInverseWordRequest, int]:
    """Validate the typed pair and admit all reverse-trace work and output."""
    if type(pair) is not RSKTableauPair:
        raise OperationDomainValidationError(
            location=("pair",),
            code="algebraic_combinatorics.rsk_inverse_trace_pair",
            message="expected a canonical RSK tableau pair",
        )
    try:
        pair = RSKTableauPair.model_validate(pair.model_dump(mode="python"))
        require_semistandard(pair.insertion_tableau)
        require_standard(pair.recording_tableau)
        if any(
            entry > len(pair.alphabet)
            for row in pair.insertion_tableau.rows
            for entry in row
        ):
            raise ValueError("insertion tableau entry is outside the ordered alphabet")
    except (ValidationError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("pair",),
            code="algebraic_combinatorics.rsk_inverse_trace_pair",
            message="expected a compatible semistandard and standard tableau pair",
        ) from exc

    cell_count = sum(pair.shape.parts)
    if cell_count > MAX_RSK_WORD_LENGTH:
        raise OperationResourceAdmissionError(
            location=("pair", "shape"),
            code="algebraic_combinatorics.rsk_inverse_trace_cells",
            message="the tableau pair exceeds the inverse RSK trace cell bound",
        )
    alphabet_scalars = sum(len(symbol) for symbol in pair.alphabet)
    if alphabet_scalars + cell_count * MAX_SYMBOL_LENGTH > MAX_RSK_WORD_PAYLOAD_SCALARS:
        raise OperationResourceAdmissionError(
            location=("pair", "alphabet"),
            code="algebraic_combinatorics.rsk_inverse_trace_payload",
            message="the reconstructed word exceeds the admitted text payload",
        )
    height = len(pair.alphabet)
    work = cell_count * max(0, height - 1) * MAX_RSK_ROW_SEARCH_COMPARISONS
    if work > MAX_RSK_TRACE_WORK:
        raise OperationResourceAdmissionError(
            location=("pair",),
            code="algebraic_combinatorics.rsk_inverse_trace_work",
            message="the complete reverse-insertion trace exceeds the work bound",
        )
    bump_count = cell_count * max(0, height - 1)
    output_bound = (
        2048
        + 6 * (alphabet_scalars + 2 * cell_count * MAX_SYMBOL_LENGTH)
        + 192 * cell_count
        + 72 * bump_count
        + 16 * cell_count * height
    )
    if output_bound > MAX_RSK_TRACE_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("pair",),
            code="algebraic_combinatorics.rsk_inverse_trace_output",
            message="the complete reverse-insertion ledger exceeds the 8 MB result bound",
        )

    return RSKInverseWordRequest.model_construct(pair=pair), cell_count


def inverse_row_insertion_rsk_trace(
    pair: RSKTableauPair,
) -> RSKWordInverseTraceResult:
    """Reconstruct a word together with every reverse-insertion event."""
    request, cell_count = _admit_inverse_trace(pair)
    insertion = [list(row) for row in pair.insertion_tableau.rows]
    label_cells: list[tuple[int, int] | None] = [None] * cell_count
    for row_index, row in enumerate(pair.recording_tableau.rows):
        for column, label in enumerate(row):
            label_cells[label - 1] = (row_index, column)
    events: list[RSKReverseInsertionEvent] = []
    for position in range(cell_count, 0, -1):
        if position == cell_count or position % 16 == 0:
            request_checkpoint("during reverse row-insertion RSK trace")
        cell = label_cells[position - 1]
        if cell is None:
            raise RuntimeError("recording tableau is missing a label")
        row_index, column = cell
        if column != len(insertion[row_index]) - 1:
            raise RuntimeError("recording label is not at an outer corner")
        removed_entry = insertion[row_index].pop()
        if not insertion[row_index]:
            if row_index != len(insertion) - 1:
                raise RuntimeError("reverse insertion produced a non-partition shape")
            insertion.pop()
        current = removed_entry
        path: list[RSKReverseBumpStep] = []
        for upper_index in range(row_index - 1, -1, -1):
            upper_row = insertion[upper_index]
            target = bisect_left(upper_row, current) - 1
            if target < 0:
                raise RuntimeError("semistandard pair failed reverse row insertion")
            upper_row[target], current = current, upper_row[target]
            path.append(
                RSKReverseBumpStep(
                    row=upper_index, column=target, displaced_entry=current
                )
            )
        events.append(
            RSKReverseInsertionEvent(
                position=position,
                letter=pair.alphabet[current - 1],
                removed_row=row_index,
                removed_column=column,
                removed_entry=removed_entry,
                output_entry=current,
                reverse_bump_path=tuple(path),
                row_lengths=tuple(len(row) for row in insertion),
            )
        )
    letters = [""] * cell_count
    for event in events:
        letters[event.position - 1] = event.letter
    word = FiniteWord(alphabet=pair.alphabet, letters=tuple(letters))
    return RSKWordInverseTraceResult._from_kernel(request, word, tuple(events))


def _trace_output_size_bound(word: FiniteWord, payload_scalars: int) -> int:
    """Conservatively bound the JSON size of a full insertion ledger."""
    length = len(word.letters)
    # Columns in a semistandard tableau strictly increase, so its height is
    # at most the number of ranks in the source alphabet.
    bump_steps = sum(
        min(prefix_length, len(word.alphabet)) for prefix_length in range(length)
    )
    alphabet_scalars = sum(len(symbol) for symbol in word.alphabet)
    retained_scalars = 2 * payload_scalars + alphabet_scalars
    return (
        2048
        + 6 * retained_scalars
        + 16 * (length + len(word.alphabet))
        + 68 * bump_steps
        + 512 * length
        + 16 * length
    )


def _admit_rsk_trace(
    request: RSKWordTraceRequest,
) -> tuple[RSKWordTraceRequest, dict[str, int]]:
    """Canonicalize once, then admit trace work and result growth."""
    if type(request) is not RSKWordTraceRequest or request.convention != (
        "ROW_INSERTION_RSK_V1"
    ):
        raise OperationDomainValidationError(
            location=("request",),
            code="algebraic_combinatorics.rsk_trace_request",
            message="expected a canonical row-insertion RSK trace request",
        )
    if type(request.word) is not FiniteWord:
        raise OperationDomainValidationError(
            location=("word",),
            code="algebraic_combinatorics.rsk_trace_word",
            message="expected a canonical finite word",
        )
    try:
        word = FiniteWord(
            alphabet=request.word.alphabet,
            letters=request.word.letters,
        )
        payload_scalars = word_payload_scalars(word)
    except (ValidationError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("word",),
            code="algebraic_combinatorics.rsk_trace_word",
            message="word must use distinct Unicode symbols from its ordered alphabet",
        ) from exc
    length = len(word.letters)
    if length > MAX_RSK_WORD_LENGTH or payload_scalars > MAX_RSK_WORD_PAYLOAD_SCALARS:
        raise OperationResourceAdmissionError(
            location=("word",),
            code="algebraic_combinatorics.rsk_trace_word_limit",
            message="the word exceeds the admitted RSK trace envelope",
        )
    bump_steps = sum(
        min(prefix_length, len(word.alphabet)) for prefix_length in range(length)
    )
    work = bump_steps * MAX_RSK_ROW_SEARCH_COMPARISONS
    if work > MAX_RSK_TRACE_WORK:
        raise OperationResourceAdmissionError(
            location=("word",),
            code="algebraic_combinatorics.rsk_trace_work",
            message="the complete insertion trace exceeds the admitted work bound",
        )
    output_bytes = _trace_output_size_bound(word, payload_scalars)
    if output_bytes > MAX_RSK_TRACE_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("word",),
            code="algebraic_combinatorics.rsk_trace_output",
            message=(
                "the complete insertion ledger exceeds the admitted 8 MB result bound"
            ),
        )
    canonical_request = RSKWordTraceRequest.model_construct(
        word=word,
        convention=request.convention,
    )
    rank = {symbol: index for index, symbol in enumerate(word.alphabet, 1)}
    return canonical_request, rank


def _trace_validated_rsk_word(
    request: RSKWordTraceRequest,
) -> RSKWordTraceResult:
    """Execute one request after its owner admission."""
    canonical_request, rank = _admit_rsk_trace(request)
    word = canonical_request.word
    insertion: list[list[int]] = []
    recording: list[list[int]] = []
    events: list[RSKInsertionEvent] = []

    for position, letter in enumerate(word.letters, start=1):
        if position == 1 or position % 16 == 0:
            request_checkpoint("during row-insertion RSK trace")
        current = rank[letter]
        row_index = 0
        bump_path: list[RSKBumpStep] = []
        while row_index < len(insertion):
            row = insertion[row_index]
            column = bisect_right(row, current)
            if column == len(row):
                row.append(current)
                recording[row_index].append(position)
                added_row, added_column, added_entry = row_index, column, current
                break
            bumped = row[column]
            row[column] = current
            bump_path.append(
                RSKBumpStep(row=row_index, column=column, bumped_entry=bumped)
            )
            current = bumped
            row_index += 1
        else:
            insertion.append([current])
            recording.append([position])
            added_row, added_column, added_entry = row_index, 0, current
        events.append(
            RSKInsertionEvent(
                position=position,
                letter=letter,
                bump_path=tuple(bump_path),
                added_row=added_row,
                added_column=added_column,
                added_entry=added_entry,
                row_lengths=tuple(len(row) for row in insertion),
            )
        )

    insertion_rows = tuple(tuple(row) for row in insertion)
    recording_rows = tuple(tuple(row) for row in recording)
    shape = IntegerPartition(parts=tuple(len(row) for row in insertion_rows))
    pair = RSKTableauPair(
        alphabet=word.alphabet,
        insertion_tableau=SemistandardYoungTableau(rows=insertion_rows),
        recording_tableau=StandardYoungTableau(rows=recording_rows),
        shape=shape,
        convention=canonical_request.convention,
    )
    return RSKWordTraceResult._from_kernel(canonical_request, pair, tuple(events))


def row_insertion_rsk_trace(word: FiniteWord) -> RSKWordTraceResult:
    """Return the ordinary word-RSK pair with its exact insertion ledger."""
    if type(word) is not FiniteWord:
        raise OperationDomainValidationError(
            location=("word",),
            code="algebraic_combinatorics.rsk_trace_word",
            message="expected a canonical finite word",
        )
    request = RSKWordTraceRequest.model_construct(word=word)
    return _trace_validated_rsk_word(request)


__all__ = [
    "inverse_row_insertion_rsk",
    "inverse_row_insertion_rsk_trace",
    "row_insertion_rsk",
    "row_insertion_rsk_trace",
]
