"""Domain-owned linear matroid operations over the shared GF(p) kernels."""

from __future__ import annotations

from jacobian.math.combinatorics.matroids._models import (
    LinearMatroid,
    validate_subset_indices,
)
from jacobian.math.matrices.finite_fields.linear_algebra import (
    PrimeFieldMatrix,
)
from jacobian.math.matrices.finite_fields.linear_algebra import (
    rank as pf_rank,
)


def _selected_columns_matrix(
    matroid: LinearMatroid, column_indices: list[int]
) -> PrimeFieldMatrix:
    """The canonical matrix restricted to the selected ground elements."""
    rows = matroid.matrix.entries
    selected = [tuple(row[j] for j in column_indices) for row in rows]
    return PrimeFieldMatrix(
        prime=matroid.matrix.prime,
        entries=tuple(selected),
        columns=len(column_indices),
    )


def matroid_rank(matroid: LinearMatroid) -> int:
    """Exact rank of the linear matroid (dimension of the column span).

    Routes through the maintained shared ``rank`` kernel so the matroid
    domain never maintains a divergent elimination implementation.
    """
    return pf_rank(matroid.matrix)


def _closure_invariant(
    matroid: LinearMatroid, subset: list[int]
) -> tuple[tuple[int, ...], int]:
    """Pure closure core: the flat of ``subset`` and its rank.

    An element e joins the closure exactly when adding it does not raise
    the subset's rank; every intermediate rank routes through the shared
    kernel. Returns ``(sorted_closure, subset_rank)``.
    """
    subset_rank = pf_rank(_selected_columns_matrix(matroid, subset))
    closure = set(subset)
    for element in range(matroid.ground_size):
        if element in closure:
            continue
        test = sorted(closure | {element})
        if pf_rank(_selected_columns_matrix(matroid, test)) == subset_rank:
            closure.add(element)
    return tuple(sorted(closure)), subset_rank


def compute_matroid_closure(
    matroid: LinearMatroid, subset: list[int]
) -> tuple[tuple[int, ...], int]:
    """Public native entry: exact closure and subset rank.

    Applies the same subset admission as the wire request so negative or
    out-of-range indices never reach the kernel through Python indexing.
    """
    validate_subset_indices(matroid, subset)
    return _closure_invariant(matroid, subset)
