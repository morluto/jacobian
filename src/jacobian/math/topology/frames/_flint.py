"""Private python-flint kernels for exact finite-frame arithmetic."""

from flint import fmpz_mat


def _canonical_rows(matrix: fmpz_mat) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(matrix[row, column]) for column in range(matrix.ncols()))
        for row in range(matrix.nrows())
    )


def integer_gram(
    vectors: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    """Return the exact row Gram matrix ``V V^T``."""

    source = fmpz_mat(vectors)
    return _canonical_rows(source * source.transpose())


def integer_gram_and_rank(
    vectors: tuple[tuple[int, ...], ...],
) -> tuple[tuple[tuple[int, ...], ...], int]:
    """Return the exact row Gram matrix and rank from one FLINT source."""

    source = fmpz_mat(vectors)
    return _canonical_rows(source * source.transpose()), int(source.rank())


__all__ = ["integer_gram", "integer_gram_and_rank"]
