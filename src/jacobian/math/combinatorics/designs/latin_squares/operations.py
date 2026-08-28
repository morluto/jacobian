"""Exact Latin-square operations."""

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.designs.latin_squares._models import (
    LatinSquare,
    LatinSquareCandidate,
)


def is_latin_square(square: LatinSquareCandidate) -> bool:
    """Return whether every row and column contains every symbol once."""

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
    pairs: set[tuple[int, int]] = set()
    for row in range(left.order):
        for column in range(left.order):
            pair = (left.cells[row][column], right.cells[row][column])
            if pair in pairs:
                return False, len(pairs)
            pairs.add(pair)
    return True, len(pairs)


def transpose(square: LatinSquare) -> tuple[tuple[int, ...], ...]:
    """Return the exact transpose of one Latin square."""

    return tuple(
        tuple(square.cells[row][column] for row in range(square.order))
        for column in range(square.order)
    )


__all__ = ["is_latin_square", "orthogonality_profile", "transpose"]
