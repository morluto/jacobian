"""Pre-execution envelopes for fixed-length quiver path counts."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.canonical import CanonicalLimits

# Strict JSON transports integer scalars only through IEEE-754's interoperable
# integer range.  ``FixedLengthPathsResult`` deliberately exposes its counts
# as integer scalars (rather than an operation-specific decimal wrapper), so
# every materialized matrix entry and the aggregate count must fit this range.
MAX_TRANSPORTABLE_PATH_COUNT = (1 << 53) - 1


@dataclass(frozen=True)
class FixedLengthPathsEnvelope:
    """Conservative work, intermediate, and strict-JSON result bounds."""

    path_count_bound: int
    maximum_entry_digits: int
    matrix_scalar_products: int
    result_json_bytes: int


def _matrix_json_bound(vertex_count: int, entry_digits: int) -> int:
    """Return a byte bound for a matrix of nonnegative decimal integers."""

    # ``[[e,...,e],...]``: every row has two brackets, n entries, and n - 1
    # commas; the outer array adds two brackets and n - 1 separators.
    return (
        2
        + vertex_count * (2 + vertex_count * entry_digits + vertex_count - 1)
        + vertex_count
        - 1
    )


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

    path_count_bound = vertex_count if length == 0 else pow(arrow_count, length)
    if path_count_bound > MAX_TRANSPORTABLE_PATH_COUNT:
        raise ValueError(
            "fixed-length path count can exceed the interoperable JSON integer "
            f"bound of {MAX_TRANSPORTABLE_PATH_COUNT}"
        )

    entry_digits = len(str(path_count_bound))
    matrix_bytes = _matrix_json_bound(vertex_count, entry_digits)
    # Exact field punctuation plus the maximum decimal total and fixed method.
    result_json_bytes = (
        len('{"path_matrix":')
        + matrix_bytes
        + len(',"total_paths":')
        + entry_digits
        + len(',"method":"MATRIX_POWER"}')
    )
    output_limit = CanonicalLimits().max_output_bytes
    if result_json_bytes > output_limit:
        raise ValueError(
            "fixed-length path result can exceed the "
            f"{output_limit}-byte canonical JSON output limit"
        )

    return FixedLengthPathsEnvelope(
        path_count_bound=path_count_bound,
        maximum_entry_digits=entry_digits,
        matrix_scalar_products=max(length - 1, 0) * vertex_count**3,
        result_json_bytes=result_json_bytes,
    )


__all__ = [
    "MAX_TRANSPORTABLE_PATH_COUNT",
    "FixedLengthPathsEnvelope",
    "fixed_length_paths_envelope",
]
