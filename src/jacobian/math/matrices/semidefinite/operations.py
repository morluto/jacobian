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
    MAX_SEMIDEFINITE_CELLS,
    RationalSemidefiniteSystem,
    SemidefiniteFaceReduction,
)
from jacobian.math.matrices.values import rational_matrix_from_fractions

_MAX_CELLS = MAX_SEMIDEFINITE_CELLS
_MAX_BIT_WORK = 2_000_000_000
_MAX_OUTPUT_DIGITS = 8_000_000
_WALL_SECONDS = 3600.0


def _reject(message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("system",), code="matrix.face_reduction_budget", message=message
    )


def _admit(
    system: RationalSemidefiniteSystem, multipliers: tuple[CanonicalRational, ...]
) -> bool:
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
    diagonal = all(
        not entry.num
        for matrix in system.matrices
        for i, row in enumerate(matrix.entries)
        for j, entry in enumerate(row)
        if i != j
    )
    if diagonal:
        # A diagonal exposing matrix selects coordinate axes: no elimination,
        # division or matrix products are needed. Charge only the scalar dot
        # products and retained source/selected submatrices, including y^T b.
        exposing_bounds = tuple(
            _dot_product_bits(
                tuple(
                    (y, matrix.entries[i][i])
                    for y, matrix in zip(multipliers, system.matrices, strict=True)
                )
            )
            for i in range(n)
        )
        result_bits = max(
            max(
                (max(abs(q.num).bit_length(), q.den.bit_length()) for q in scalars),
                default=1,
            ),
            _dot_product_bits(tuple(zip(multipliers, system.rhs, strict=True))),
            max(exposing_bounds, default=1),
        )
        # Copies keep their own scalar heights; a single large diagonal does
        # not enlarge the many zero entries in the retained dense matrices.
        copied_digits = 2 * sum(
            _component_digits(abs(component).bit_length())
            for q in scalars
            for component in (q.num, q.den)
        )
        output_digits = (
            copied_digits
            + 4 * n * n
            + sum(2 * _component_digits(bits) for bits in exposing_bounds)
        )
        _check_budgets(
            cells,
            result_bits,
            4 * result_bits,
            cells + m * (n + 1),
            output_digits=output_digits,
        )
        return True
    active = tuple(
        (multiplier, matrix)
        for multiplier, matrix in zip(multipliers, system.matrices, strict=True)
        if multiplier.num
    )
    exposing_scalars = (
        *(entry for _, matrix in active for row in matrix.entries for entry in row),
        *(multiplier for multiplier, _ in active),
        *(
            rhs
            for multiplier, rhs in zip(multipliers, system.rhs, strict=True)
            if multiplier.num
        ),
    )
    denominator_bits = sum(
        (den - 1).bit_length() for den in {q.den for q in exposing_scalars} if den != 1
    )
    input_bits = (
        denominator_bits
        + max((abs(q.num).bit_length() for q in exposing_scalars), default=1)
        + 1
    )
    exposing_bits = 2 * input_bits + max(1, max(len(active), 1).bit_length())
    minor_bits = max(1, n * (exposing_bits + n.bit_length()))
    matrix_bits = []
    for matrix, multiplier in zip(system.matrices, multipliers, strict=True):
        dens = {entry.den for row in matrix.entries for entry in row}
        aggregate_denominators = sum((den - 1).bit_length() for den in dens if den != 1)
        entry_bits = max(
            (
                abs(entry.num).bit_length() + entry.den.bit_length()
                for row in matrix.entries
                for entry in row
            ),
            default=1,
        )
        denominator_growth = aggregate_denominators
        matrix_bits.append(
            entry_bits + denominator_growth + 2 * minor_bits + (n * n).bit_length()
        )
    result_bits = max(
        input_bits + 2 * minor_bits + 2 * n.bit_length() + 2,
        max(matrix_bits, default=1),
    )
    intermediate_bits = 4 * max(result_bits, minor_bits)
    work = (2 * m + 4) * n**3 + m * n * n + m
    retained_scalars = (
        *(
            entry
            for matrix in system.matrices
            for row in matrix.entries
            for entry in row
        ),
        *system.rhs,
        *multipliers,
    )
    output_digits = 2 * sum(
        _component_digits(abs(component).bit_length())
        for q in retained_scalars
        for component in (q.num, q.den)
    ) + 2 * n * n * _component_digits(result_bits)
    _check_budgets(
        cells, result_bits, intermediate_bits, work, output_digits=output_digits
    )
    return False


def _dot_product_bits(
    pairs: tuple[tuple[CanonicalRational, CanonicalRational], ...],
) -> int:
    active = tuple((a, b) for a, b in pairs if a.num and b.num)
    if not active:
        return 1
    # The product of every operand denominator is a common denominator for
    # every partial sum. Each scaled numerator is bounded by that denominator
    # times the largest numerator product; summing k terms adds ceil(log2 k).
    denominator_bits = sum(
        (a.den - 1).bit_length() + (b.den - 1).bit_length() for a, b in active
    )
    numerator_bits = max(
        abs(a.num).bit_length() + abs(b.num).bit_length() for a, b in active
    )
    return denominator_bits + numerator_bits + (len(active) - 1).bit_length() + 1


def _component_digits(bits: int) -> int:
    return max(1, (bits * 30103 + 99999) // 100000)


def _check_budgets(
    cells: int,
    result_bits: int,
    intermediate_bits: int,
    work: int,
    *,
    output_digits: int | None = None,
) -> None:
    digits = _component_digits(result_bits)
    if digits > MAX_CANONICAL_RATIONAL_DIGITS:
        _reject("exact compression exceeds the canonical rational height envelope")
    if (
        cells * 2 * digits if output_digits is None else output_digits
    ) > _MAX_OUTPUT_DIGITS:
        _reject("source-bound reduction exceeds the exact output digit envelope")
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
    diagonal = _admit(system, multipliers)
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
            Fraction()
            if diagonal and i != j
            else sum(
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
    pivots: tuple[int, ...]
    reduced_rows: tuple[tuple[Fraction, ...], ...]
    if diagonal:
        if any(exposing[i][i] < 0 for i in range(n)):
            _invalid_relation("the exposing matrix must be positive semidefinite")
        free = tuple(i for i in range(n) if not exposing[i][i])
        pivots = ()
        reduced_rows = ()
    else:
        _, negative, _ = _symmetric_inertia(
            [list(row) for row in exposing], checkpoint=checkpoint
        )
        if negative:
            _invalid_relation("the exposing matrix must be positive semidefinite")
        reduced_rows, rank = rational_rref(exposing)
        pivots = tuple(
            next(j for j, q in enumerate(row) if q) for row in reduced_rows[:rank]
        )
        free = tuple(j for j in range(n) if j not in pivots)
    checkpoint("after exposing kernel computation")
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
            tuple(tuple(matrix[i][j] for j in free) for i in free)
            if diagonal
            else (
                rational_matrix_product(
                    rational_matrix_product(transposed, matrix), embedding_entries
                )
                if width
                else ()
            )
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
