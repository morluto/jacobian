"""Exact LR coefficients, independently checked by character theory."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from functools import cache
from itertools import product
from math import factorial

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics.symmetric_functions._models import (
    MAX_LR_SEARCH_STATES,
    MAX_LR_SKEW_CELLS,
    LittlewoodRichardsonCoefficientRequest,
    SchurProductRequest,
)
from jacobian.math.combinatorics.symmetric_functions._tools import TOOLS
from jacobian.math.combinatorics.symmetric_functions.littlewood_richardson import (
    littlewood_richardson_coefficient,
    schur_product,
)
from jacobian.math.combinatorics.symmetric_functions.values import IntegerPartition


def _partitions(total: int, maximum: int | None = None) -> tuple[tuple[int, ...], ...]:
    if total == 0:
        return ((),)
    limit = total if maximum is None else min(total, maximum)
    return tuple(
        (first, *rest)
        for first in range(limit, 0, -1)
        for rest in _partitions(total - first, first)
    )


def _border_strip_sign(
    outer: tuple[int, ...], inner: tuple[int, ...], strip_size: int
) -> int | None:
    """Return the Murnaghan-Nakayama sign if outer/inner is a rim hook."""
    cells = {
        (row, column)
        for row, width in enumerate(outer)
        for column in range(1, width + 1)
        if column > (inner[row] if row < len(inner) else 0)
    }
    if len(cells) != strip_size or not cells:
        return None
    reached = {next(iter(cells))}
    pending = list(reached)
    while pending:
        row, column = pending.pop()
        for neighbor in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            if neighbor in cells and neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    if reached != cells:
        return None
    if any(
        all(
            cell in cells
            for cell in (
                (row, column),
                (row + 1, column),
                (row, column + 1),
                (row + 1, column + 1),
            )
        )
        for row, column in cells
    ):
        return None
    occupied_rows = {row for row, _ in cells}
    return -1 if (len(occupied_rows) - 1) % 2 else 1


@cache
def _irreducible_character(shape: tuple[int, ...], cycle_type: tuple[int, ...]) -> int:
    """Compute an S_n character value by Murnaghan-Nakayama rim-hook removal."""
    if not cycle_type:
        return int(not shape)
    first, *rest = cycle_type
    remainder_size = sum(shape) - first
    if remainder_size < 0:
        return 0
    value = 0
    for remainder in _partitions(remainder_size):
        if len(remainder) > len(shape) or any(
            part > shape[index] for index, part in enumerate(remainder)
        ):
            continue
        sign = _border_strip_sign(shape, remainder, first)
        if sign is not None:
            value += sign * _irreducible_character(remainder, tuple(rest))
    return value


def _centralizer_size(cycle_type: tuple[int, ...]) -> int:
    multiplicities = Counter(cycle_type)
    answer = 1
    for part, multiplicity in multiplicities.items():
        answer *= part**multiplicity * factorial(multiplicity)
    return answer


def _independent_lr_character_oracle(
    outer: tuple[int, ...], inner: tuple[int, ...], content: tuple[int, ...]
) -> int:
    """Use symmetric-group induction characters, independent of LR fillings.

    Frobenius' character formula gives the multiplicity of ``S_outer`` in
    ``Ind(S_inner x S_content)`` as a sum over conjugacy classes. The
    Littlewood-Richardson rule identifies that multiplicity with the requested
    coefficient.
    """
    numerator = Fraction(0)
    for alpha in _partitions(sum(inner)):
        for beta in _partitions(sum(content)):
            union = tuple(sorted((*alpha, *beta), reverse=True))
            numerator += Fraction(
                (
                    _irreducible_character(inner, alpha)
                    * _irreducible_character(content, beta)
                    * _irreducible_character(outer, union)
                ),
                _centralizer_size(alpha) * _centralizer_size(beta),
            )
    assert numerator.denominator == 1
    return numerator.numerator


def _request(
    outer: tuple[int, ...], inner: tuple[int, ...], content: tuple[int, ...]
) -> LittlewoodRichardsonCoefficientRequest:
    return LittlewoodRichardsonCoefficientRequest(
        outer=IntegerPartition(parts=outer),
        inner=IntegerPartition(parts=inner),
        content=IntegerPartition(parts=content),
    )


def _coefficient(
    outer: tuple[int, ...], inner: tuple[int, ...], content: tuple[int, ...]
) -> int:
    request = _request(outer, inner, content)
    return littlewood_richardson_coefficient(
        request.outer, request.inner, request.content
    ).coefficient


def test_lr_coefficient_uses_fixed_reverse_row_lattice_convention() -> None:
    result = _request((3, 2, 1), (2, 1), (2, 1))
    result = littlewood_richardson_coefficient(
        result.outer, result.inner, result.content
    )
    assert result.coefficient == 2
    assert (result.outer.parts, result.inner.parts, result.content.parts) == (
        (3, 2, 1),
        (2, 1),
        (2, 1),
    )


def test_lr_coefficients_match_independent_character_oracle_exhaustively() -> None:
    # Exhaust every coefficient c^lambda_{mu,nu} with |mu|+|nu| <= 5.
    # The implementation enumerates skew tableaux; this oracle instead uses
    # irreducible symmetric-group characters and Frobenius' inner product.
    for inner_size in range(6):
        for content_size in range(6 - inner_size):
            total = inner_size + content_size
            for inner, content, outer in product(
                _partitions(inner_size),
                _partitions(content_size),
                _partitions(total),
            ):
                actual = _coefficient(outer, inner, content)
                expected = _independent_lr_character_oracle(outer, inner, content)
                assert actual == expected, (outer, inner, content)


def test_lr_zero_cases_and_empty_partition_unit() -> None:
    assert _coefficient((3, 1), (2,), (1,)) == 0  # size mismatch
    assert _coefficient((2, 1), (3,), ()) == 0  # inner not contained
    assert _coefficient((), (), ()) == 1


def test_lr_request_enforces_exact_cell_and_search_envelopes() -> None:
    assert MAX_LR_SKEW_CELLS == 8
    assert MAX_LR_SEARCH_STATES == 100_000
    assert _coefficient((8,), (), (8,)) == 1
    with pytest.raises(ValidationError, match="outer partition size"):
        _request((9,), (), (9,))
    with pytest.raises(ValidationError, match="prefix bound"):
        _request((8,), (), (1,) * 8)


def test_lr_operation_is_published_in_its_owner_manifest() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id
        == "symmetric_function.littlewood_richardson.coefficient.compute"
    )
    example = tool.examples[0]
    result = tool.run(tool.request_type.model_validate(example.input))
    assert result.coefficient == 2


def test_schur_products_match_independent_character_oracle_exhaustively() -> None:
    # Check every product through total degree four. The oracle is the
    # independent Frobenius character inner product above, not LR tableaux.
    for left_size in range(5):
        for right_size in range(5 - left_size):
            for left, right in product(_partitions(left_size), _partitions(right_size)):
                expansion = schur_product(
                    IntegerPartition(parts=left), IntegerPartition(parts=right)
                )
                actual = {
                    term.partition.parts: term.coefficient for term in expansion.terms
                }
                expected = {
                    outer: _independent_lr_character_oracle(outer, left, right)
                    for outer in _partitions(left_size + right_size)
                }
                assert actual == {
                    shape: count for shape, count in expected.items() if count
                }
                assert expansion.left.parts == left
                assert expansion.right.parts == right


def test_schur_product_unit_and_operation_example() -> None:
    empty = IntegerPartition(parts=())
    unit = schur_product(empty, IntegerPartition(parts=(2,)))
    assert [(term.partition.parts, term.coefficient) for term in unit.terms] == [
        ((2,), 1)
    ]
    zero = schur_product(empty, empty)
    assert [(term.partition.parts, term.coefficient) for term in zero.terms] == [
        ((), 1)
    ]
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "symmetric_function.schur.product.compute"
    )
    result = tool.run(tool.request_type.model_validate(tool.examples[0].input))
    assert [(term.partition.parts, term.coefficient) for term in result.terms] == [
        ((3,), 1),
        ((2, 1), 1),
    ]


def test_schur_product_admits_total_degree_before_candidate_search() -> None:
    with pytest.raises(ValidationError, match="total degree"):
        SchurProductRequest(
            left=IntegerPartition(parts=(9,)), right=IntegerPartition(parts=())
        )
    with pytest.raises(ValidationError, match="content-prefix bound"):
        SchurProductRequest(
            left=IntegerPartition(parts=()),
            right=IntegerPartition(parts=(1,) * 8),
        )
