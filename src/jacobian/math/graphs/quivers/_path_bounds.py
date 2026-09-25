"""Pre-execution envelopes for fixed-length quiver path counts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MAX_PATH_SCALAR_UPDATES = 10_000_000


@dataclass(frozen=True)
class FixedLengthPathsEnvelope:
    """Conservative work and intermediate bounds."""

    path_count_bound: int
    maximum_entry_digits: int
    method: Literal["dense", "sparse"]
    scalar_updates: int


def fixed_length_paths_envelope(
    *, vertex_count: int, arrow_count: int, length: int
) -> FixedLengthPathsEnvelope:
    """Bound dense powers or arrow-wise recurrence before either runs.

    There are at most ``arrow_count ** length`` positive-length arrow walks.
    That quantity bounds every result entry and intermediate of positive
    degree; the sparse recurrence starts from an identity matrix with entries
    at most one. Dense powers perform ``length - 1`` cubic products; the sparse
    recurrence applies each arrow to every source row at every step. Matrix
    allocation is bounded separately by the 128-vertex and 32-step structural
    caps: at most 33 * 128**2 cells are initialized in either regime. With at
    most 16,384 arrows, every output entry has at most 135 decimal digits;
    the retained source and 128**2-entry matrix serialize within 3 MiB, below
    the 10 MiB canonical output limit.
    """

    if type(length) is not int or not 0 <= length <= 32:
        raise ValueError("path length must be an integer from 0 through 32")
    path_count_bound = vertex_count if length == 0 else pow(arrow_count, length)
    entry_digits = len(str(path_count_bound))
    if length == 0:
        return FixedLengthPathsEnvelope(
            path_count_bound=path_count_bound,
            maximum_entry_digits=entry_digits,
            method="dense",
            scalar_updates=0,
        )
    dense_work = max(length - 1, 0) * vertex_count**3
    sparse_work = length * vertex_count * arrow_count
    method: Literal["dense", "sparse"] = (
        "sparse" if sparse_work < dense_work else "dense"
    )
    scalar_updates = min(dense_work, sparse_work)
    if scalar_updates > MAX_PATH_SCALAR_UPDATES:
        raise ValueError(
            "fixed-length path counts exceed the "
            f"{MAX_PATH_SCALAR_UPDATES}-update work bound"
        )
    return FixedLengthPathsEnvelope(
        path_count_bound=path_count_bound,
        maximum_entry_digits=entry_digits,
        method=method,
        scalar_updates=scalar_updates,
    )


__all__ = [
    "MAX_PATH_SCALAR_UPDATES",
    "FixedLengthPathsEnvelope",
    "fixed_length_paths_envelope",
]
