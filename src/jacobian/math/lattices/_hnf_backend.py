"""Owner-local python-flint adapter for exact row Hermite normal form."""

from typing import Any


def flint_row_hnf(entries: list[list[int]]) -> tuple[Any, Any]:
    """Return the row HNF ``H`` and unimodular ``U`` with ``H = U*A``.

    The owner admits the canonical integer matrix before this adapter is
    entered.  FLINT owns algorithm selection inside its maintained exact HNF
    kernel; Jacobian retains the mathematical contract and canonical result
    construction.
    """
    from flint import fmpz_mat

    source = fmpz_mat(entries)
    return source.hnf(True)
