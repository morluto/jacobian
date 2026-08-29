"""Private python-flint kernels for exact finite-frame arithmetic."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flint import fmpz_mat


def _canonical_rows(matrix: fmpz_mat) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(matrix[row, column]) for column in range(matrix.ncols()))
        for row in range(matrix.nrows())
    )


def integer_rank(vectors: tuple[tuple[int, ...], ...]) -> int:
    """Return the exact rank of the integer vector family."""

    from flint import fmpz_mat

    return int(fmpz_mat(vectors).rank())


def integer_gram(
    vectors: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    """Return the exact row Gram matrix ``V V^T``."""

    from flint import fmpz_mat

    source = fmpz_mat(vectors)
    return _canonical_rows(source * source.transpose())


def integer_gram_and_rank(
    vectors: tuple[tuple[int, ...], ...],
) -> tuple[int, tuple[tuple[int, ...], ...] | None]:
    """Return rank and, only when admitted, the Gram matrix from one source."""

    from flint import fmpz_mat

    source = fmpz_mat(vectors)
    rank = int(source.rank())
    if rank != len(vectors[0]):
        return rank, None
    return rank, _canonical_rows(source * source.transpose())


__all__ = ["integer_gram", "integer_gram_and_rank", "integer_rank"]
