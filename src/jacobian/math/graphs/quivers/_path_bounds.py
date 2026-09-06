"""Pre-execution envelopes for fixed-length quiver path counts."""

from __future__ import annotations

from dataclasses import dataclass

MAX_PATH_MATRIX_SCALAR_PRODUCTS = 10_000_000


@dataclass(frozen=True)
class FixedLengthPathsEnvelope:
    """Conservative work and intermediate bounds."""

    path_count_bound: int
    maximum_entry_digits: int
    matrix_scalar_products: int


def fixed_length_paths_envelope(
    *, vertex_count: int, arrow_count: int, length: int
) -> FixedLengthPathsEnvelope:
    """Bound the complete fixed-length matrix-power computation before it runs.

    There are at most ``arrow_count ** length`` composable arrow sequences.
    Thus that same quantity bounds every matrix-power entry, their aggregate,
    and every nonnegative intermediate in the repeated multiplication path.
    The matrix construction performs ``length - 1`` dense products after the
    adjacency matrix is built; their scalar-product count is retained here as
    the operation's explicit work ledger.
    """

    if type(length) is not int or not 0 <= length <= 32:
        raise ValueError("path length must be an integer from 0 through 32")
    path_count_bound = vertex_count if length == 0 else pow(arrow_count, length)
    entry_digits = len(str(path_count_bound))
    matrix_scalar_products = max(length - 1, 0) * vertex_count**3
    if matrix_scalar_products > MAX_PATH_MATRIX_SCALAR_PRODUCTS:
        raise ValueError(
            "fixed-length matrix powers exceed the "
            f"{MAX_PATH_MATRIX_SCALAR_PRODUCTS}-product work bound"
        )
    return FixedLengthPathsEnvelope(
        path_count_bound=path_count_bound,
        maximum_entry_digits=entry_digits,
        matrix_scalar_products=matrix_scalar_products,
    )


__all__ = [
    "MAX_PATH_MATRIX_SCALAR_PRODUCTS",
    "FixedLengthPathsEnvelope",
    "fixed_length_paths_envelope",
]
