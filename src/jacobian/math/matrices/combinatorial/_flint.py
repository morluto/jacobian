"""Private python-flint kernels for exact combinatorial matrices."""

from flint import fmpz_mat


def integer_gram(
    rows: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    """Return ``rows * rows^T`` through FLINT's exact integer matrices."""

    source = fmpz_mat(rows)
    product = source * source.transpose()
    return tuple(
        tuple(int(product[i, j]) for j in range(product.ncols()))
        for i in range(product.nrows())
    )


__all__ = ["integer_gram"]
