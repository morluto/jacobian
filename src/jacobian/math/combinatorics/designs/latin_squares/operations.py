"""Exact Latin-square operations."""

from jacobian.canonical import CanonicalLimits
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.designs.latin_squares._models import (
    LatinSquare,
    LatinSquareCandidate,
)

MAX_LATIN_CHECK_CELLS = 1_048_576
MAX_LATIN_ORTHOGONALITY_PAIR_CELLS = 1_048_576
MAX_LATIN_TRANSPOSE_CELLS = 1_048_576
_LATIN_TRANSPOSE_RESULT_RESERVE_BYTES = 4_096


def _reject(location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(
        location=location,
        code=f"combinatorics.latin_square.{code}",
        message=message,
    )


def _square_cells(square: LatinSquare | LatinSquareCandidate) -> int:
    return square.order * square.order


def is_latin_square(square: LatinSquareCandidate) -> bool:
    """Return whether every row and column contains every symbol once."""

    cells = _square_cells(square)
    if cells > MAX_LATIN_CHECK_CELLS:
        _reject(
            ("square", "cells"),
            "check_cells_exceeded",
            "Latin-square check exceeds the source-cell work budget",
        )
    expected = set(range(square.order))
    return all(set(row) == expected for row in square.cells) and all(
        {square.cells[row][column] for row in range(square.order)} == expected
        for column in range(square.order)
    )


def orthogonality_profile(
    left: LatinSquare,
    right: LatinSquare,
) -> tuple[bool, int]:
    """Return orthogonality and the number of distinct aligned symbol pairs."""

    if left.order != right.order:
        raise OperationDomainValidationError(
            location=("right", "order"),
            code="combinatorics.latin_square.order",
            message="squares must have the same order",
        )
    pair_cells = _square_cells(left)
    if pair_cells > MAX_LATIN_ORTHOGONALITY_PAIR_CELLS:
        _reject(
            ("left", "cells"),
            "orthogonality_pair_cells_exceeded",
            "orthogonality check exceeds the distinct-pair memory budget",
        )
    # Symbols already lie in ``range(order)``, so the aligned pair has a dense
    # integer address.  A byte table keeps the complete positive-decision
    # envelope bounded without paying Python tuple/set overhead for each pair.
    seen = bytearray(pair_cells)
    pair_count = 0
    for row in range(left.order):
        for column in range(left.order):
            pair_index = left.cells[row][column] * left.order + right.cells[row][column]
            if not seen[pair_index]:
                seen[pair_index] = 1
                pair_count += 1
    return pair_count == pair_cells, pair_count


def transpose(square: LatinSquare) -> LatinSquare:
    """Return the exact transpose of one Latin square."""

    cells = _square_cells(square)
    if cells > MAX_LATIN_TRANSPOSE_CELLS:
        _reject(
            ("square", "cells"),
            "transpose_cells_exceeded",
            "Latin-square transpose exceeds the source-cell work budget",
        )
    symbol_digits = len(str(square.order - 1))
    predicted_bytes = (
        cells * (symbol_digits + 1)
        + 2 * square.order
        + _LATIN_TRANSPOSE_RESULT_RESERVE_BYTES
    )
    if predicted_bytes > CanonicalLimits().max_output_bytes:
        _reject(
            ("square", "cells"),
            "transpose_result_bytes_exceeded",
            "Latin-square transpose exceeds the canonical output-byte limit",
        )
    # The transpose of a Latin square is Latin, so construct the canonical
    # carrier without re-running the Latin-property validator.
    return LatinSquare.model_construct(
        order=square.order,
        cells=tuple(
            tuple(square.cells[row][column] for row in range(square.order))
            for column in range(square.order)
        ),
    )


__all__ = ["is_latin_square", "orthogonality_profile", "transpose"]
