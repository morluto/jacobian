"""Exact Boolean truth-table operations backed by SymPy."""

from __future__ import annotations

from collections.abc import Sequence

from jacobian.catalog.models import OperationDomainValidationError

__all__ = ["walsh_hadamard_transform"]


def walsh_hadamard_transform(truth_table: Sequence[int]) -> list[int]:
    """Return the exact Boolean Walsh spectrum of a Boolean function's truth table.

    The truth table is a list of ``0``/``1`` values whose length is a positive
    power of two.  The result is the exact integer Walsh spectrum in Hadamard
    order, computed by applying the fast Walsh-Hadamard transform (``fwht``) to
    the **sign vector** ``(-1)^f`` — i.e. ``1 - 2*f`` — rather than the raw
    ``0``/``1`` truth table.

    For a Boolean function ``f : F_2^n -> F_2``, the Walsh transform is
    ``W_f(u) = sum_x (-1)^(f(x) + u·x)``, so the FWHT input must be the sign
    vector ``(-1)^f = 1 - 2f``, not the ``0``/``1`` truth vector.

    No floating-point arithmetic is involved.
    """

    from sympy.discrete.transforms import fwht

    if not truth_table:
        raise OperationDomainValidationError(
            location=("truth_table",),
            code="boolean.fourier.walsh_transform.nonempty",
            message="truth table must not be empty",
        )
    n = len(truth_table)
    if n & (n - 1) != 0:
        raise OperationDomainValidationError(
            location=("truth_table",),
            code="boolean.fourier.walsh_transform.power_of_two",
            message="truth table length must be a power of two",
        )
    if any(value not in (0, 1) for value in truth_table):
        raise OperationDomainValidationError(
            location=("truth_table",),
            code="boolean.fourier.walsh_transform.boolean_entries",
            message="truth table entries must be 0 or 1",
        )
    sign_vector = [1 - 2 * bit for bit in truth_table]
    return [int(coefficient) for coefficient in fwht(sign_vector)]
