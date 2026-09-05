"""Admission for the full-row-rank augmented HNF representation."""

from math import factorial

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.values import (
    MAX_INTEGER_MATRIX_ORDER,
    MAX_MATRIX_SCALAR_DIGITS,
)

MAX_HNF_INPUT_DIGITS = 256
MAX_HNF_DIGIT_WORK = 100_000_000


def admit_hermite_normal_form(entries: list[list[int]]) -> None:
    """Admit [A | I_m], including the entire m by m transformation.

    Every maximal minor of [A | I_m] is, up to sign, a minor of A of
    order at most r=min(m,n). With d input digits, Leibniz bounds these
    minors by r! * 10**(r*d). Full-row-rank row HNF entries are bounded
    by the largest maximal minor. This applies to H and U together,
    including singular, tall and wide A: the augmented matrix always
    has rank m. No rank computation or HNF is replayed in admission.

    Use m**2*(n+m) times the minor digit height as the conservative
    admission cost estimate used by the existing invariant-form graph-HNF
    envelope. This is a shape/height policy, not a count of FLINT's internal
    arithmetic operations. The structural axes bound both H and U by 128
    per axis and their combined cells by 32,768; no extra transport-size
    cap is imposed here.
    """
    rows = len(entries)
    columns = len(entries[0]) if rows else 0
    if (
        not 1 <= rows <= MAX_INTEGER_MATRIX_ORDER
        or not 1 <= columns <= MAX_INTEGER_MATRIX_ORDER
        or any(len(row) != columns for row in entries)
    ):
        raise OperationDomainValidationError(
            location=("matrix",),
            code="lattice.budget_exceeded",
            message=f"HNF requires a nonempty rectangular matrix with axes at most {MAX_INTEGER_MATRIX_ORDER}",
        )
    if any(type(value) is not int for row in entries for value in row):
        raise TypeError("Hermite normal form entries must be integers")
    if any(abs(value) >= 10**MAX_HNF_INPUT_DIGITS for row in entries for value in row):
        raise OperationDomainValidationError(
            location=("matrix",),
            code="lattice.budget_exceeded",
            message=f"HNF input scalars are limited to {MAX_HNF_INPUT_DIGITS} decimal digits",
        )
    digits = max(len(str(abs(value))) for row in entries for value in row)
    rank_bound = min(rows, columns)
    minor_digits = rank_bound * digits + len(str(factorial(rank_bound)))
    if minor_digits > MAX_MATRIX_SCALAR_DIGITS:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="lattice.budget_exceeded",
            message="HNF augmented minor height exceeds the canonical scalar envelope",
        )
    cells = rows * (columns + rows)
    work = rows * cells * minor_digits
    if work > MAX_HNF_DIGIT_WORK:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="lattice.budget_exceeded",
            message=f"HNF augmented minor-height work {work:,} exceeds {MAX_HNF_DIGIT_WORK:,} digit-work units",
        )
