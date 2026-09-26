"""Exact fixed-content semistandard tableau counts (Kostka numbers)."""

from __future__ import annotations

from itertools import permutations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.semistandard_tableaux import content_count as kernel
from jacobian.math.combinatorics.semistandard_tableaux._models import (
    FixedContentCountRequest,
)
from jacobian.math.combinatorics.semistandard_tableaux.content_count import (
    fixed_content_count,
)
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    TableauContent,
)


def _partitions(size: int, largest: int | None = None):
    if size == 0:
        yield ()
        return
    maximum = min(size, largest if largest is not None else size)
    for first in range(maximum, 0, -1):
        for rest in _partitions(size - first, first):
            yield (first, *rest)


def _content_vectors(total: int, labels: int):
    if labels == 0:
        if total == 0:
            yield ()
        return
    for first in range(total + 1):
        for rest in _content_vectors(total - first, labels - 1):
            yield (first, *rest)


def _reference_count(parts: tuple[int, ...], labels: tuple[int, ...]) -> int:
    """Enumerate unique multiset words and check every tableau adjacency."""
    cell_positions: list[tuple[int, int]] = []
    position = 0
    for row, width in enumerate(parts):
        cell_positions.extend((row, column) for column in range(width))
        position += width

    words = set(permutations(labels))
    valid = 0
    for word in words:
        rows = [[0] * width for width in parts]
        for cell, entry in zip(cell_positions, word, strict=True):
            row, column = cell
            rows[row][column] = entry
        if any(
            rows[row][column] > rows[row][column + 1]
            for row, width in enumerate(parts)
            for column in range(width - 1)
        ):
            continue
        if any(
            rows[row][column] >= rows[row + 1][column]
            for row in range(len(parts) - 1)
            for column in range(parts[row + 1])
        ):
            continue
        valid += 1
    return valid


def _request(parts: tuple[int, ...], weighted_labels: tuple[tuple[int, int], ...]):
    return FixedContentCountRequest(
        partition=IntegerPartition(parts=parts),
        content=TableauContent(
            terms=tuple(
                {"entry": entry, "multiplicity": multiplicity}
                for entry, multiplicity in weighted_labels
            )
        ),
    )


def test_fixed_content_count_matches_independent_exhaustive_oracle() -> None:
    for size in range(6):
        for parts in _partitions(size):
            for labels_count in range(1, 4):
                for weights in _content_vectors(size, labels_count):
                    weighted_labels = tuple(
                        (label, weight)
                        for label, weight in enumerate(weights, start=1)
                        if weight
                    )
                    expanded = tuple(
                        label
                        for label, multiplicity in weighted_labels
                        for _ in range(multiplicity)
                    )
                    request = _request(parts, weighted_labels)
                    result = fixed_content_count(request)
                    assert result.count == _reference_count(parts, expanded)
                    assert result.partition == request.partition
                    assert result.content == request.content


def test_content_labels_are_sparse_exact_and_never_reindexed() -> None:
    original = _request((3, 2), ((2, 2), (5, 2), (9, 1)))
    relabelled = _request((3, 2), ((11, 2), (30, 2), (100, 1)))

    first = fixed_content_count(original)
    assert fixed_content_count(original.partition, original.content) == first
    second = fixed_content_count(relabelled)

    assert first.count == second.count
    assert first.content == original.content
    assert second.content == relabelled.content
    assert tuple(term.entry for term in second.content.terms) == (11, 30, 100)


def test_one_row_accepts_a_large_fixed_content_without_multiset_expansion() -> None:
    request = _request((500,), ((9, 500),))
    assert fixed_content_count(request).count == 1


def test_standard_content_reuses_the_hook_length_count() -> None:
    request = _request((2, 2), ((10, 1), (20, 1), (30, 1), (40, 1)))
    assert fixed_content_count(request).count == 2


def test_wrong_total_and_too_few_distinct_labels_are_exact_zero() -> None:
    wrong_total = _request((2, 1), ((1, 1),))
    insufficient_height = _request((1, 1), ((7, 2),))

    assert fixed_content_count(wrong_total).count == 0
    assert fixed_content_count(insufficient_height).count == 0


def test_multiset_prefix_work_is_rejected_before_search(monkeypatch) -> None:
    def unexpected_search(*_args, **_kwargs):
        pytest.fail("search began before fixed-content work admission")

    monkeypatch.setattr(kernel, "_count_by_row_major_search", unexpected_search)
    request = _request((11, 9), ((1, 10), (2, 10)))
    with pytest.raises(OperationResourceAdmissionError):
        fixed_content_count(request)


def test_forged_oversized_carriers_are_rejected_before_dump(monkeypatch) -> None:
    partition = IntegerPartition.model_construct(parts=(1,) * 1_000_000)
    content = TableauContent(terms=())

    def forbidden_dump(self, *args, **kwargs):
        pytest.fail("oversized forged carrier was dumped before envelope check")

    monkeypatch.setattr(IntegerPartition, "model_dump", forbidden_dump)
    with pytest.raises(OperationDomainValidationError):
        fixed_content_count(partition, content)


def test_content_terms_must_be_unique_and_in_increasing_label_order() -> None:
    with pytest.raises(ValueError):
        TableauContent(
            terms=({"entry": 3, "multiplicity": 1}, {"entry": 2, "multiplicity": 1})
        )
    with pytest.raises(ValueError):
        TableauContent(
            terms=({"entry": 2, "multiplicity": 1}, {"entry": 2, "multiplicity": 1})
        )


def test_narrow_two_row_weight_is_accepted_and_counted_without_word_expansion() -> None:
    request = _request((499, 1), ((1, 499), (2, 1)))
    assert fixed_content_count(request).count == 1


def test_multiset_search_work_admits_its_exact_bound(monkeypatch) -> None:
    request = _request((19, 1), ((4, 18), (9, 2)))
    exact_work_bound = 21 * 190 * (2 + 2)
    monkeypatch.setattr(kernel, "MAX_KOSTKA_SEARCH_WORK", exact_work_bound)
    assert fixed_content_count(request).count == 1

    monkeypatch.setattr(kernel, "MAX_KOSTKA_SEARCH_WORK", exact_work_bound - 1)
    with pytest.raises(OperationResourceAdmissionError):
        fixed_content_count(request)
