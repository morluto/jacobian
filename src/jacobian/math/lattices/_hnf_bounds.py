"""Bounds for fraction-free elimination, modular HNF, and exact lifting.

The bounds follow the explicitly selected SymPy 1.14 algorithms documented
in ``_hnf_backend``. Arithmetic counts and operand heights are independent;
no time measurement or output-height proxy stands in for intermediate work.
"""

from dataclasses import dataclass

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.values import (
    MAX_INTEGER_MATRIX_ORDER,
    MAX_MATRIX_SCALAR_DIGITS,
)

MAX_HNF_INPUT_DIGITS = 256
MAX_HNF_WORK_UNITS = 250_000_000
MAX_HNF_INTERMEDIATE_BITS = 262_144
MAX_HNF_COEFFICIENT_BITS = 12_000_000_000


@dataclass(frozen=True, slots=True)
class HNFAdmission:
    """One source-derived envelope for all mandatory mathematical phases."""

    rows: int
    columns: int
    minor_bits: int
    intermediate_bits: int
    work_units: int
    coefficient_bits: int


def _reject(message: str) -> None:
    raise OperationDomainValidationError(
        location=("matrix",), code="lattice.budget_exceeded", message=message
    )


def admit_hermite_normal_form(entries: list[list[int]]) -> HNFAdmission:
    """Bound every phase before constructing the augmented backend matrix.

    Let P=2**b bound all minors of [A|I]. Each is a minor of A, so
    Hadamard bounds it by the product of the largest min(m,n) row norms
    of A (norms smaller than one are replaced by one). Squared row norms
    give integer-only upper bounds on b, without rank or elimination.

    SymPy's FF_dense Gauss-Jordan stores minors and forms differences of
    two products before exact division: stored entries <=P, temporaries
    <=2*P**2. There are at most m pivots and 6*m*m*(n+m) scalar operations,
    including nonpivot-column updates and pivot searches.

    The square modular HNF uses at most m*(m+1)/2 extended gcds on operands
    <=P and 14*m**3 other scalar operations. Euclidean remainders halve
    every two steps, so charge 16*b+24 abstract arithmetic units per gcd.
    ZZ.gcdex uses the maintained scalar backend: this charge and its bounded
    operands do not purport to count native instructions. Modular
    column updates form products of operands <=P then reduce modulo R<=P.
    Above-pivot reductions in W are not modular: a column undergoes at
    most m reductions. If its current height is S>=1, one reduction has
    height <=S+(S+1)*P <=(P+2)*S. Starting at P, (m+2)*(b+1)+2 bits bound
    even the last pre-subtraction product. This separately accounts for
    the intermediates that are larger than the final HNF.

    Lifting multiplies the square HNF by the fraction-free RREF numerator,
    then divides exactly by its denominator. At most 3*m*m*(n+m) scalar
    operations suffice; partial dot products are <=m*P**2. The final [H|U]
    is the full-row-rank HNF of [A|I], hence its entries are <=P.

    Coefficient storage reserves four augmented arrays at elimination/lift
    height and four square arrays at modular height, plus scalar scratch.
    This bounds mathematical coefficient bits, not allocator or process RSS.
    The loops, matrix cells, and operand heights are all bounded separately.
    """
    rows = len(entries)
    columns = len(entries[0]) if rows else 0
    if (
        not 1 <= rows <= MAX_INTEGER_MATRIX_ORDER
        or not 1 <= columns <= MAX_INTEGER_MATRIX_ORDER
        or any(len(row) != columns for row in entries)
    ):
        _reject(
            "HNF requires a nonempty rectangular matrix with axes at most "
            f"{MAX_INTEGER_MATRIX_ORDER}"
        )
    if any(type(value) is not int for row in entries for value in row):
        raise TypeError("Hermite normal form entries must be integers")
    scalar_limit = 10**MAX_HNF_INPUT_DIGITS
    if any(abs(value) >= scalar_limit for row in entries for value in row):
        _reject(
            f"HNF input scalars are limited to {MAX_HNF_INPUT_DIGITS} decimal digits"
        )

    # A squared norm <2**k gives a norm <=2**ceil(k/2).
    row_bits = sorted(
        (
            (sum(value * value for value in row).bit_length() + 1) // 2
            for row in entries
        ),
        reverse=True,
    )
    minor_bits = max(1, sum(row_bits[: min(rows, columns)]))
    minor_digits = minor_bits * 30_103 // 100_000 + 1
    if minor_digits > MAX_MATRIX_SCALAR_DIGITS:
        _reject("HNF augmented minor height exceeds the canonical scalar envelope")

    cells = rows * (columns + rows)
    lift_bits = 2 * minor_bits + rows.bit_length() + 2
    modular_bits = (rows + 2) * (minor_bits + 1) + 2
    intermediate_bits = max(lift_bits, modular_bits)
    if intermediate_bits > MAX_HNF_INTERMEDIATE_BITS:
        _reject(
            f"HNF intermediate height {intermediate_bits:,} exceeds "
            f"{MAX_HNF_INTERMEDIATE_BITS:,} bits"
        )
    gcd_calls = rows * (rows + 1) // 2
    work = (
        9 * rows * cells
        + 14 * rows**3
        + gcd_calls * (16 * minor_bits + 24)
        + 2 * rows * columns  # squared norms in admission
    )
    if work > MAX_HNF_WORK_UNITS:
        _reject(
            f"HNF scalar-operation work {work:,} exceeds {MAX_HNF_WORK_UNITS:,} units"
        )
    coefficient_bits = (
        4 * cells * lift_bits + 4 * rows * rows * modular_bits + 16 * intermediate_bits
    )
    if coefficient_bits > MAX_HNF_COEFFICIENT_BITS:
        _reject(
            f"HNF retained coefficient storage {coefficient_bits:,} exceeds "
            f"{MAX_HNF_COEFFICIENT_BITS:,} bits"
        )
    return HNFAdmission(
        rows=rows,
        columns=columns,
        minor_bits=minor_bits,
        intermediate_bits=intermediate_bits,
        work_units=work,
        coefficient_bits=coefficient_bits,
    )
