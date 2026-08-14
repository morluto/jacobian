"""Exact lattice-basis reduction on canonical integer matrix inputs.

This is the supported public API for ``jacobian.math.lattices``.  The FLINT
backend is private to this module and loaded lazily.
"""

from __future__ import annotations

from typing import Any

__all__ = ["hermite_normal_form", "reduce_basis"]


def hermite_normal_form(entries: list[list[int]]) -> tuple[Any, Any]:
    """Return the row Hermite normal form and its left transformation."""

    import flint

    return flint.fmpz_mat(entries).hnf(True)


def reduce_basis(
    entries: list[list[int]],
    *,
    delta: float = 0.99,
    eta: float = 0.51,
) -> tuple[Any, Any, int]:
    """Reduce one integer lattice basis with exact-gram LLL.

    Accepts a row-major integer list-of-lists and returns the reduced basis,
    the left transformation, and the rank.  FLINT rejects one-row bases, so
    that mathematically valid boundary case is preserved with the identity
    transformation.
    """

    import flint

    source = flint.fmpz_mat(entries)
    if source.nrows() == 1:
        reduced = source
        transformation = flint.fmpz_mat([[1]])
    else:
        reduced, transformation = source.lll(
            True,
            delta,
            eta,
            "zbasis",
            "exact",
        )
    if transformation * source != reduced:
        raise ValueError("The LLL left transformation does not bind the source basis.")
    return reduced, transformation, int(reduced.rank())
