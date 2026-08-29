"""Tests for Latin square operations."""

import pytest
from pydantic import ValidationError
from typing_extensions import TypedDict

from jacobian.canonical import encode_strict_json, loads_strict_json
from jacobian.math.combinatorics.designs.latin_squares import (
    is_latin_square,
    orthogonality_profile,
    transpose,
)
from jacobian.math.combinatorics.designs.latin_squares._models import (
    LatinSquare,
    LatinSquareCandidate,
    LatinSquareRequest,
    LatinSquareTransposeResult,
    OrthogonalityRequest,
    TransposeRequest,
)
from jacobian.math.combinatorics.designs.latin_squares._tools import (
    TOOLS,
    compute_latin_square_check,
    compute_latin_square_transpose,
    compute_orthogonality,
)
from jacobian.math.combinatorics.designs.latin_squares.operations import (
    MAX_LATIN_ORTHOGONALITY_PAIR_CELLS,
)


class _RawSquare(TypedDict):
    order: int
    cells: list[list[int]]


def _latin_square(
    order: int,
    cells: tuple[tuple[int, ...], ...],
) -> LatinSquare:
    return LatinSquare(order=order, cells=cells)


def _candidate_square(
    order: int,
    cells: tuple[tuple[int, ...], ...],
) -> LatinSquareCandidate:
    return LatinSquareCandidate(order=order, cells=cells)


Z2 = _latin_square(2, ((0, 1), (1, 0)))


def _cyclic_square(order: int, multiplier: int = 1) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple((row + multiplier * column) % order for column in range(order))
        for row in range(order)
    )


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "latin_square.check",
        "latin_square.orthogonality.check",
        "latin_square.transpose.compute",
    }


def test_latin_square_check_valid() -> None:
    request = LatinSquareRequest(square=_candidate_square(2, ((0, 1), (1, 0))))
    result = compute_latin_square_check(request)
    assert result.is_latin is True
    assert is_latin_square(request.square)


def test_latin_square_check_invalid() -> None:
    request = LatinSquareRequest(square=_candidate_square(2, ((0, 0), (1, 1))))
    result = compute_latin_square_check(request)
    assert result.is_latin is False


def test_orthogonality_identical_not_orthogonal() -> None:
    request = OrthogonalityRequest(
        square_a=Z2,
        square_b=Z2,
    )
    result = compute_orthogonality(request)
    assert result.is_orthogonal is False


def test_orthogonality_orthogonal() -> None:
    request = OrthogonalityRequest(
        square_a=_latin_square(3, ((0, 1, 2), (1, 2, 0), (2, 0, 1))),
        square_b=_latin_square(3, ((0, 1, 2), (2, 0, 1), (1, 2, 0))),
    )
    result = compute_orthogonality(request)
    assert result.is_orthogonal is True
    assert result.pair_count == 9


def test_transpose() -> None:
    request = TransposeRequest(square=Z2)
    result = compute_latin_square_transpose(request)
    assert result.transposed == Z2
    native = transpose(Z2)
    assert native == result.transposed
    assert isinstance(native, LatinSquare)
    # Native and MCP producers share the canonical LatinSquare carrier so the
    # native result composes unchanged into transpose and orthogonality.
    assert transpose(native) == Z2
    assert orthogonality_profile(native, Z2) == (False, 2)
    restored = LatinSquareTransposeResult.model_validate(
        loads_strict_json(encode_strict_json(result.model_dump(mode="json")))
    )
    relayed = TransposeRequest(square=restored.transposed)
    assert compute_latin_square_transpose(relayed).transposed == Z2
    assert orthogonality_profile(Z2, Z2) == (False, 2)


def test_orthogonality_rejects_non_latin_square() -> None:
    """Orthogonality must only be computed on actual Latin squares."""
    non_latin: _RawSquare = {"order": 2, "cells": [[0, 0], [1, 1]]}
    with pytest.raises(ValidationError):
        OrthogonalityRequest.model_validate({"square_a": non_latin, "square_b": Z2})
    with pytest.raises(ValidationError):
        OrthogonalityRequest.model_validate({"square_a": Z2, "square_b": non_latin})


def test_transpose_rejects_non_latin_square() -> None:
    """Transpose must only be computed on actual Latin squares."""
    non_latin: _RawSquare = {"order": 2, "cells": [[0, 0], [1, 1]]}
    with pytest.raises(ValidationError):
        TransposeRequest.model_validate({"square": non_latin})


def test_check_accepts_non_latin_square() -> None:
    """The check operation must accept non-Latin squares to test them."""
    request = LatinSquareRequest(square=_candidate_square(2, ((0, 0), (1, 1))))
    result = compute_latin_square_check(request)
    assert result.is_latin is False


def test_operations_admit_materialized_squares_beyond_order_32() -> None:
    order = 127
    left_cells = _cyclic_square(order)
    right_cells = _cyclic_square(order, 2)

    check = compute_latin_square_check(
        LatinSquareRequest(square=_candidate_square(order, left_cells))
    )
    left = _latin_square(order, left_cells)
    right = _latin_square(order, right_cells)
    orthogonality = compute_orthogonality(
        OrthogonalityRequest(square_a=left, square_b=right)
    )
    transposed = compute_latin_square_transpose(TransposeRequest(square=left))

    assert check.is_latin is True
    assert orthogonality.is_orthogonal is True
    assert orthogonality.pair_count == order * order
    assert all(
        transposed.transposed.cells[row][column] == left.cells[column][row]
        for row in range(order)
        for column in range(order)
    )


def test_large_negative_orthogonality_uses_compact_pair_storage() -> None:
    order = 513
    square = _latin_square(order, _cyclic_square(order))

    assert order * order > 262_144
    assert order * order <= MAX_LATIN_ORTHOGONALITY_PAIR_CELLS
    assert orthogonality_profile(square, square) == (False, order)


def test_pair_count_covers_all_cells_after_first_duplicate() -> None:
    left = _latin_square(4, _cyclic_square(4))
    right = _latin_square(
        4,
        (
            (0, 1, 3, 2),
            (1, 2, 0, 3),
            (2, 3, 1, 0),
            (3, 0, 2, 1),
        ),
    )

    assert orthogonality_profile(left, right) == (False, 12)
