"""One exact facial reduction, using the matrix owner's rational kernels."""

import time
from fractions import Fraction

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices._flint import rational_matrix_product, rational_rref
from jacobian.math.matrices.analysis.operations import _symmetric_inertia
from jacobian.math.matrices.semidefinite.values import (
    RationalSemidefiniteSystem,
    SemidefiniteFaceReduction,
)
from jacobian.math.matrices.values import rational_matrix_from_fractions

_MAX_CELLS = 131_072
_MAX_BIT_WORK = 2_000_000_000
_MAX_OUTPUT_DIGITS = 8_000_000
_WALL_SECONDS = 3600.0


def _reject(message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("system",), code="matrix.face_reduction_budget", message=message
    )


def _admit(
    system: RationalSemidefiniteSystem, multipliers: tuple[CanonicalRational, ...]
) -> None:
    n, m = system.order, len(system.matrices)
    if len(multipliers) != m:
        raise OperationDomainValidationError(
            location=("multipliers",),
            code="matrix.shape_mismatch",
            message="multipliers must index every equality",
        )
    # Dense source inspection, exposing matrix, basis and compressed output.
    cells = (2 * m + 3) * n * n + 2 * m
    if cells > _MAX_CELLS:
        _reject("source and reduced matrices exceed the dense cell envelope")
    scalars = (
        *(
            entry
            for matrix in system.matrices
            for row in matrix.entries
            for entry in row
        ),
        *system.rhs,
        *multipliers,
    )
    # Clear all input denominators by one common Q. Its bit length is bounded
    # without constructing Q. Repeated denominators are counted once.
    denominator_bits = sum(
        (den - 1).bit_length() for den in {q.den for q in scalars} if den != 1
    )
    input_bits = (
        denominator_bits
        + max((abs(q.num).bit_length() for q in scalars), default=1)
        + 1
    )
    # Q^2 W is integral: each entry sums m products of Q-scaled inputs.
    exposing_bits = 2 * input_bits + max(1, m.bit_length())
    # Hadamard bounds every minor of Q^2 W. RREF coordinates have a common
    # pivot-minor denominator and numerators bounded by these same minors.
    # A nullspace basis adds identity coordinates. All compression products
    # therefore share Q * pivot_minor^2 as a denominator, rather than a
    # product of unrelated denominators for each summand.
    minor_bits = max(1, n * (exposing_bits + n.bit_length()))
    result_bits = input_bits + 2 * minor_bits + 2 * n.bit_length() + 2
    # Congruence Schur entries are ratios of bordered minors; factor four
    # covers unreduced multiply/add operands in 1x1 and 2x2 pivot updates.
    intermediate_bits = 4 * max(result_bits, minor_bits)
    digits = (result_bits * 30103 + 99999) // 100000 + 1
    if digits > MAX_CANONICAL_RATIONAL_DIGITS:
        _reject("exact compression exceeds the canonical rational height envelope")
    if cells * 2 * digits > _MAX_OUTPUT_DIGITS:
        _reject("source-bound reduction exceeds the exact output digit envelope")
    work = (2 * m + 4) * n**3 + m * n * n + m
    if work * intermediate_bits > _MAX_BIT_WORK:
        _reject("rational elimination and compression exceed the bit-work envelope")


def _invalid_relation(message: str) -> None:
    raise OperationDomainValidationError(
        location=("multipliers",),
        code="matrix.invalid_exposing_relation",
        message=message,
    )


def reduce_exposed_face(
    system: RationalSemidefiniteSystem, multipliers: tuple[CanonicalRational, ...]
) -> SemidefiniteFaceReduction:
    """Recognize y and return equivalent PSD equalities on ker(sum y_i A_i).

    The embedding uses the RREF fundamental kernel basis, with free columns
    in source-axis order. No minimal face or feasibility claim is made.
    """
    execution = current_request_execution()
    started = execution.started_at if execution is not None else time.monotonic()
    deadline = started + _WALL_SECONDS
    if execution is not None and execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)

    def checkpoint(phase: str) -> None:
        request_checkpoint(phase)
        if time.monotonic() >= deadline:
            raise OperationExecutionTimeoutError(
                "semidefinite face reduction timed out"
            )

    checkpoint("before face reduction admission")
    _admit(system, multipliers)
    checkpoint("after face reduction admission")
    n = system.order
    y = tuple(q.as_fraction() for q in multipliers)
    if sum(
        (a * b.as_fraction() for a, b in zip(y, system.rhs, strict=True)), Fraction()
    ):
        _invalid_relation("the exposing relation must satisfy y^T b = 0")
    matrices = tuple(
        tuple(tuple(q.as_fraction() for q in row) for row in matrix.entries)
        for matrix in system.matrices
    )
    exposing = tuple(
        tuple(
            sum(
                (a * matrix[i][j] for a, matrix in zip(y, matrices, strict=True)),
                Fraction(),
            )
            for j in range(n)
        )
        for i in range(n)
    )
    checkpoint("after exposing matrix construction")
    if not any(entry for row in exposing for entry in row):
        _invalid_relation("the exposing matrix must be nonzero")
    _, negative, _ = _symmetric_inertia(
        [list(row) for row in exposing], checkpoint=checkpoint
    )
    if negative:
        _invalid_relation("the exposing matrix must be positive semidefinite")
    reduced_rows, rank = rational_rref(exposing)
    checkpoint("after exposing kernel elimination")
    pivots = tuple(
        next(j for j, q in enumerate(row) if q) for row in reduced_rows[:rank]
    )
    free = tuple(j for j in range(n) if j not in pivots)
    width = len(free)
    basis = [[Fraction(int(i == j)) for j in free] for i in range(n)]
    for row, pivot in enumerate(pivots):
        basis[pivot] = [-reduced_rows[row][j] for j in free]
    embedding_entries = tuple(tuple(row) for row in basis)
    transposed = tuple(tuple(basis[i][j] for i in range(n)) for j in range(width))
    compressed = []
    for matrix in matrices:
        checkpoint("before equality compression")
        entries = (
            rational_matrix_product(
                rational_matrix_product(transposed, matrix), embedding_entries
            )
            if width
            else ()
        )
        compressed.append(rational_matrix_from_fractions(entries, column_count=width))
    checkpoint("before face reduction result construction")
    result = SemidefiniteFaceReduction(
        source=system,
        multipliers=multipliers,
        exposing_matrix=rational_matrix_from_fractions(exposing),
        embedding=rational_matrix_from_fractions(embedding_entries, column_count=width),
        reduced=RationalSemidefiniteSystem(
            order=width, matrices=tuple(compressed), rhs=system.rhs
        ),
    )
    checkpoint("after face reduction result construction")
    return result


__all__ = ["reduce_exposed_face"]
