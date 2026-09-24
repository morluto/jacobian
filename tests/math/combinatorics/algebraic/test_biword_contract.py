from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.algebraic.biword import (
    Biword,
    BiwordNormalizeRequest,
    BiwordRSKPair,
    NonnegativeIntegerMatrix,
)
from jacobian.math.combinatorics.algebraic.biword_ops import (
    greene,
    inverse_biword,
    matrix_biword,
    normalize_biword,
    rsk_biword,
)
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    SemistandardYoungTableau,
)
from jacobian.math.logic.languages.words.values import FiniteWord


def test_normalization_rejects_oversized_source_before_sorting() -> None:
    request = BiwordNormalizeRequest.model_construct(
        top_alphabet=("a",),
        bottom_alphabet=("a",),
        top=("a",) * 501,
        bottom=("a",) * 501,
    )
    with pytest.raises(OperationDomainValidationError):
        normalize_biword(request)


def test_native_algebraic_boundaries_reject_raw_invalid_values() -> None:
    with pytest.raises(OperationDomainValidationError):
        rsk_biword(None)  # type: ignore[arg-type]
    with pytest.raises(OperationDomainValidationError):
        greene(None)
    forged = Biword.model_construct(
        top_alphabet=("a",), bottom_alphabet=("a",), top=("a",), bottom=("a",)
    )
    assert rsk_biword(forged).shape.parts == (1,)
    with pytest.raises(OperationDomainValidationError):
        greene(FiniteWord.model_construct(alphabet=("a",), letters=("b",)))


def test_greene_decreasing_invariants_are_cumulative_column_totals() -> None:
    result = greene(
        FiniteWord(
            alphabet=("1", "2", "3"),
            letters=("2", "1", "3"),
        ),
        requested_k=2,
    )

    assert result.shape.parts == (2, 1)
    assert result.increasing_totals == (2, 3)
    assert result.decreasing_totals == (2, 3)


def _reference_matrix_rsk(
    row_labels: tuple[str, ...],
    column_labels: tuple[str, ...],
    entries: tuple[tuple[int, ...], ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Small independent row-insertion reference for labelled matrices."""
    insertion: list[list[int]] = []
    recording: list[list[int]] = []
    for row, cells in enumerate(entries):
        for column, multiplicity in enumerate(cells):
            for _ in range(multiplicity):
                bumped = column + 1
                current_row = 0
                while current_row < len(insertion):
                    values = insertion[current_row]
                    target = next(
                        (index for index, value in enumerate(values) if value > bumped),
                        len(values),
                    )
                    if target == len(values):
                        values.append(bumped)
                        recording[current_row].append(row + 1)
                        break
                    values[target], bumped = bumped, values[target]
                    current_row += 1
                else:
                    insertion.append([bumped])
                    recording.append([row + 1])
    # The labels are included in the arguments to make their role explicit;
    # tableau cells carry their one-based ranks in those ordered axes.
    assert len(entries) == len(row_labels)
    assert all(len(row) == len(column_labels) for row in entries)
    return tuple(map(tuple, insertion)), tuple(map(tuple, recording))


@pytest.mark.parametrize(
    ("row_labels", "column_labels", "entries"),
    [
        (("r0", "r1"), ("c0", "c1"), ((1, 2), (0, 1))),
        (("north", "south", "west"), ("x", "y"), ((0, 1), (2, 0), (1, 1))),
        (("only-row",), ("first", "second", "third"), ((2, 1, 0),)),
    ],
)
def test_matrix_transposition_swaps_tableaux_against_independent_reference(
    row_labels: tuple[str, ...],
    column_labels: tuple[str, ...],
    entries: tuple[tuple[int, ...], ...],
) -> None:
    source = NonnegativeIntegerMatrix(
        row_labels=row_labels, column_labels=column_labels, entries=entries
    )
    transposed_entries = tuple(
        tuple(entries[row][column] for row in range(len(row_labels)))
        for column in range(len(column_labels))
    )
    transpose = NonnegativeIntegerMatrix(
        row_labels=column_labels,
        column_labels=row_labels,
        entries=transposed_entries,
    )

    source_pair = matrix_biword(source)
    transpose_pair = matrix_biword(transpose)
    expected_p, expected_q = _reference_matrix_rsk(row_labels, column_labels, entries)
    expected_transpose_p, expected_transpose_q = _reference_matrix_rsk(
        column_labels, row_labels, transposed_entries
    )

    assert source_pair.insertion_tableau.rows == expected_p
    assert source_pair.recording_tableau.rows == expected_q
    assert transpose_pair.insertion_tableau.rows == expected_transpose_p
    assert transpose_pair.recording_tableau.rows == expected_transpose_q
    assert transpose_pair.insertion_tableau.rows == source_pair.recording_tableau.rows
    assert transpose_pair.recording_tableau.rows == source_pair.insertion_tableau.rows


def test_inverse_rejects_forged_pair_content_at_native_boundary() -> None:
    pair = BiwordRSKPair.model_construct(
        top_alphabet=(),
        bottom_alphabet=(),
        insertion_tableau=SemistandardYoungTableau.model_construct(rows=((1,),)),
        recording_tableau=SemistandardYoungTableau.model_construct(rows=((1,),)),
        shape=IntegerPartition.model_construct(parts=(1,)),
        source_kind="BIWORD",
    )
    with pytest.raises(OperationDomainValidationError):
        inverse_biword(pair)
