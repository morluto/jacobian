"""Certified integral homology for finite based chain complexes.

The two Smith reductions have deliberately separate roles.  The outgoing
reduction supplies a saturated integral cycle basis; the incoming reduction
computes the quotient of that cycle lattice by the boundary lattice.  SymPy
owns both exact Smith decompositions.  This module owns admission, source-basis
reconstruction, and the public height/result contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from time import monotonic

from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
)
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.certified_snf.operations import (
    Matrix,
    SmithReduction,
    certificate_from_reduction,
    inverse_unimodular,
    matrix_columns,
    matrix_multiply,
    matrix_vector_multiply,
    smith_reduce,
)
from jacobian.math.matrices.certified_snf.values import CertifiedIntegerMatrix
from jacobian.math.topology.chain_complexes.values import (
    INTEGRAL_HOMOLOGY_WALL_SECONDS,
    MAX_INTEGRAL_HOMOLOGY_CHAIN_RANK,
    MAX_INTEGRAL_HOMOLOGY_INPUT_DIGITS,
    MAX_INTEGRAL_HOMOLOGY_MATRIX_CELLS,
    MAX_INTEGRAL_HOMOLOGY_OUTPUT_BITS,
    MAX_INTEGRAL_HOMOLOGY_OUTPUT_SCALARS,
    MAX_INTEGRAL_HOMOLOGY_TOTAL_CHAIN_RANK,
    MAX_INTEGRAL_HOMOLOGY_WORK_UNITS,
    ChainComplexValue,
    CoefficientRing,
    IntegralFreeGenerator,
    IntegralHomologyGroupValue,
    IntegralTorsionGenerator,
    IntegralVector,
)


@dataclass(frozen=True, slots=True)
class SmithHeightBound:
    """A source-derived upper bound for one pinned SymPy decomposition."""

    left_bits: int
    right_bits: int
    diagonal_bits: int
    intermediate_bits: int
    work_units: int
    transformations_are_identity: bool = False

    @property
    def maximum_bits(self) -> int:
        return max(
            self.left_bits,
            self.right_bits,
            self.diagonal_bits,
            self.intermediate_bits,
        )


@dataclass(frozen=True, slots=True)
class IntegralHomologyDegreePlan:
    degree: int
    chain_rank: int
    incoming_chain_rank: int
    outgoing: Matrix
    incoming: Matrix
    outgoing_height: SmithHeightBound
    incoming_heights_by_cycle_rank: tuple[SmithHeightBound, ...]
    coordinate_bits_by_cycle_rank: tuple[int, ...]
    output_bits_by_cycle_rank: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class IntegralHomologyExecutionPlan:
    source: ChainComplexValue
    degrees: tuple[IntegralHomologyDegreePlan, ...]
    parse_work: int
    square_zero_work: int
    smith_work: int
    reconstruction_work: int
    output_scalar_count: int
    total_work: int
    deadline: float


def _domain_error(code: str, message: str) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("complex",),
        code=f"chain_complex.{code}",
        message=message,
    )


def _capped(value: int) -> int:
    return min(value, MAX_INTEGRAL_HOMOLOGY_OUTPUT_BITS + 1)


def _add_bits(*values: int) -> int:
    total = sum(values)
    return _capped(total)


def _ceil_log2(value: int) -> int:
    return 0 if value <= 1 else (value - 1).bit_length()


def _factorial_bits(size: int) -> int:
    """A cheap upper bound for ``ceil(log2(size!))``."""

    return sum(_ceil_log2(value) for value in range(2, size + 1))


def _matrix_bits(matrix: Matrix) -> int:
    return max(
        (abs(value).bit_length() for row in matrix for value in row),
        default=1,
    )


def _include_smith_postcheck_bound(
    bound: SmithHeightBound,
    *,
    rows: int,
    columns: int,
    input_bits: int,
) -> SmithHeightBound:
    """Include ``smith_reduce`` reconstruction and determinant checks.

    The maintained primitive verifies ``D = U A V`` and both unimodular
    determinants after SymPy returns. Matrix-product heights add operand
    heights and the inner-dimension sum; determinant intermediates are bounded
    by the Leibniz/Hadamard envelope already used for inverse cofactors.
    """

    left_source_bits = _add_bits(_ceil_log2(rows), bound.left_bits, input_bits, 1)
    reconstruction_bits = _add_bits(
        _ceil_log2(columns), left_source_bits, bound.right_bits, 1
    )
    left_determinant_bits = _add_bits(rows * bound.left_bits, _factorial_bits(rows), 1)
    right_determinant_bits = _add_bits(
        columns * bound.right_bits, _factorial_bits(columns), 1
    )
    return SmithHeightBound(
        left_bits=bound.left_bits,
        right_bits=bound.right_bits,
        diagonal_bits=bound.diagonal_bits,
        intermediate_bits=max(
            bound.intermediate_bits,
            reconstruction_bits,
            left_determinant_bits,
            right_determinant_bits,
        ),
        work_units=bound.work_units,
        transformations_are_identity=bound.transformations_are_identity,
    )


def _is_positive_smith_diagonal(matrix: Matrix, *, rows: int, columns: int) -> bool:
    """Recognize the exact fixed point of SymPy's positive Smith convention."""

    diagonal_count = min(rows, columns)
    diagonal = [matrix[index][index] for index in range(diagonal_count)]
    if any(
        matrix[row][column] != 0
        for row in range(rows)
        for column in range(columns)
        if row != column
    ):
        return False
    nonzero = [value for value in diagonal if value]
    return (
        diagonal == nonzero + [0] * (diagonal_count - len(nonzero))
        and all(value > 0 for value in nonzero)
        and all(right % left == 0 for left, right in pairwise(nonzero))
    )


def _one_dimensional_smith_bound(
    rows: int, columns: int, input_bits: int
) -> SmithHeightBound:
    """Bound the extended-gcd transform for an ``n x 1`` or ``1 x n`` matrix.

    Each successive extended-gcd update has coefficients bounded by the
    current source height.  Across ``length - 1`` updates, every accumulated
    transformation entry is therefore bounded by ``(2B)^length``.  This is
    intentionally looser than the classical Bezout bound but depends only on
    the admitted source height and covers the pinned implementation.
    """

    length = max(rows, columns)
    transform_bits = _add_bits(length * (input_bits + 1), _ceil_log2(length + 1))
    work = max(1, length - 1) * max(1, input_bits) * max(1, length)
    return SmithHeightBound(
        left_bits=transform_bits if columns == 1 else 1,
        right_bits=transform_bits if rows == 1 else 1,
        diagonal_bits=input_bits,
        intermediate_bits=transform_bits,
        work_units=work,
    )


def _general_smith_bound(
    rows: int,
    columns: int,
    input_bits: int,
) -> SmithHeightBound:
    """Conservatively bound SymPy 1.14's recursive Smith transformations.

    In one pivot level, every elementary row/column coefficient is bounded by
    the current entry height ``B``: it is a quotient, a Bezout coefficient, or
    a quotient by a gcd.  One update therefore maps ``B`` to at most ``2B^2``.
    A pass performs at most ``rows + columns - 2`` updates.  Every additional
    pass strictly replaces the positive pivot by a proper divisor, so there
    are at most ``input_bits + 1`` passes.  The recurrence below is the closed
    bit-height bound for those updates, followed by the recursively embedded
    lower-right decomposition and the final divisibility repairs.  It is
    deliberately conservative; requests exceeding it are rejected before
    SymPy runs.
    """

    updates = (rows + columns - 2) * (input_bits + 1)
    if updates >= MAX_INTEGRAL_HOMOLOGY_OUTPUT_BITS.bit_length():
        cleared_bits = MAX_INTEGRAL_HOMOLOGY_OUTPUT_BITS + 1
    else:
        cleared_bits = _capped((input_bits + 1) * (1 << updates))
    if cleared_bits > MAX_INTEGRAL_HOMOLOGY_OUTPUT_BITS:
        return SmithHeightBound(
            left_bits=cleared_bits,
            right_bits=cleared_bits,
            diagonal_bits=cleared_bits,
            intermediate_bits=cleared_bits,
            work_units=MAX_INTEGRAL_HOMOLOGY_WORK_UNITS + 1,
        )

    recursive = _smith_height_bound_from_shape(
        rows - 1,
        columns - 1,
        cleared_bits,
    )
    matrix_product_bits = _ceil_log2(max(rows, columns))
    left_bits = _add_bits(cleared_bits, recursive.left_bits, matrix_product_bits)
    right_bits = _add_bits(cleared_bits, recursive.right_bits, matrix_product_bits)
    factor_bits = _add_bits(
        min(rows, columns) * input_bits,
        _factorial_bits(min(rows, columns)),
        1,
    )
    repairs = max(0, min(rows, columns) - 1)
    left_bits = _add_bits(left_bits, 3 * repairs * (factor_bits + 1))
    right_bits = _add_bits(right_bits, 2 * repairs * (factor_bits + 1))
    operation_cells = rows * columns + rows * rows + columns * columns
    work = (
        recursive.work_units
        + updates * max(1, operation_cells)
        + repairs * 5 * max(1, operation_cells)
    )
    return SmithHeightBound(
        left_bits=left_bits,
        right_bits=right_bits,
        diagonal_bits=factor_bits,
        intermediate_bits=max(cleared_bits, recursive.intermediate_bits),
        work_units=work,
    )


def _small_ternary_smith_bound(
    rows: int,
    columns: int,
) -> SmithHeightBound:
    """Bound the pinned reduction on a matrix of ``-1, 0, 1`` entries.

    This branch depends only on shape and coefficient height, not on matrix
    provenance: it admits every small ternary matrix, including but not limited
    to simplicial incidence matrices, without pretending arbitrary larger
    signed matrices have the same envelope. For a matrix with at most three
    rows and columns, the first nonzero pivot selected by SymPy is a unit.
    Clearing its row and column takes one pass: the lower-right entries have
    magnitude at most two and both first-level transformations have entries of
    magnitude at most one. If the first row and column are zero, recursion
    starts with the unchanged ternary lower-right block, which is smaller
    still. Applying the general recurrence only to that at-most ``2 x 2``
    block therefore bounds every intermediate before execution.

    The final divisibility repairs use minors of the original ternary matrix;
    Hadamard's determinant bound supplies ``factor_bits`` below.
    """

    recursive = _smith_height_bound_from_shape(rows - 1, columns - 1, 2)
    product_bits = _ceil_log2(max(rows, columns))
    left_bits = _add_bits(1, recursive.left_bits, product_bits)
    right_bits = _add_bits(1, recursive.right_bits, product_bits)
    factor_bits = _add_bits(
        min(rows, columns),
        _factorial_bits(min(rows, columns)),
        1,
    )
    repairs = max(0, min(rows, columns) - 1)
    left_bits = _add_bits(left_bits, 3 * repairs * (factor_bits + 1))
    right_bits = _add_bits(right_bits, 2 * repairs * (factor_bits + 1))
    operation_cells = rows * columns + rows * rows + columns * columns
    work = (
        recursive.work_units
        + (rows + columns) * max(1, operation_cells)
        + repairs * 5 * max(1, operation_cells)
    )
    return SmithHeightBound(
        left_bits=left_bits,
        right_bits=right_bits,
        diagonal_bits=factor_bits,
        intermediate_bits=max(2, recursive.intermediate_bits),
        work_units=work,
    )


def _smith_height_bound_from_shape(
    rows: int,
    columns: int,
    input_bits: int,
) -> SmithHeightBound:
    if rows == 0 or columns == 0:
        core = SmithHeightBound(1, 1, 1, 1, 1, True)
    elif min(rows, columns) == 1:
        core = _one_dimensional_smith_bound(rows, columns, input_bits)
    elif rows <= 3 and columns <= 3 and input_bits <= 1:
        core = _small_ternary_smith_bound(rows, columns)
    else:
        core = _general_smith_bound(rows, columns, input_bits)
    return _include_smith_postcheck_bound(
        core,
        rows=rows,
        columns=columns,
        input_bits=input_bits,
    )


def _smith_height_bound(
    matrix: Matrix,
    *,
    rows: int,
    columns: int,
) -> SmithHeightBound:
    input_bits = _matrix_bits(matrix)
    if rows == 0 or columns == 0 or not any(value for row in matrix for value in row):
        core = SmithHeightBound(1, 1, 1, 1, 1, True)
        return _include_smith_postcheck_bound(
            core,
            rows=rows,
            columns=columns,
            input_bits=input_bits,
        )
    if _is_positive_smith_diagonal(matrix, rows=rows, columns=columns):
        core = SmithHeightBound(
            1,
            1,
            input_bits,
            input_bits,
            rows * columns,
            True,
        )
        return _include_smith_postcheck_bound(
            core,
            rows=rows,
            columns=columns,
            input_bits=input_bits,
        )
    return _smith_height_bound_from_shape(rows, columns, input_bits)


def _known_rank_before_smith(
    matrix: Matrix,
    *,
    rows: int,
    columns: int,
    bound: SmithHeightBound,
) -> int | None:
    """Return a rank established without running a decomposition."""

    if bound.transformations_are_identity:
        return sum(
            1 for index in range(min(rows, columns)) if matrix[index][index] != 0
        )
    if min(rows, columns) == 1:
        # Zero matrices took the identity branch above.
        return 1
    return None


def _unit_row_preserves_kernel_coordinate_height(matrix: Matrix) -> bool:
    """Whether SymPy's one-row unit clearing only selects source coordinates.

    With a nonzero row of ``-1, 0, 1`` entries, the pinned implementation
    swaps a unit into the first position and clears every other entry by a
    unit column addition.  In the inverse transform, the non-pivot rows are
    therefore a permutation of the corresponding source-coordinate rows.
    Because ``d^2 = 0`` makes the pivot coordinate zero, the retained cycle
    coordinates have no more height than the incoming differential itself.
    """

    return len(matrix) == 1 and all(abs(value) <= 1 for value in matrix[0])


def _inverse_unimodular_bits(size: int, entry_bits: int) -> int:
    if size <= 1:
        return 1
    return _add_bits((size - 1) * entry_bits, _factorial_bits(size - 1), 1)


def _matrix_product_bits(inner: int, left_bits: int, right_bits: int) -> int:
    if inner == 0:
        return 1
    return _add_bits(_ceil_log2(inner), left_bits, right_bits, 1)


def _require_height(bound: SmithHeightBound, *, label: str) -> None:
    if bound.maximum_bits > MAX_INTEGRAL_HOMOLOGY_OUTPUT_BITS:
        raise _domain_error(
            "integral_homology_height_budget_exceeded",
            f"{label} has a conservative Smith transformation-height bound above "
            f"{MAX_INTEGRAL_HOMOLOGY_OUTPUT_BITS} bits; reduce the matrix dimension "
            "or coefficient height",
        )


def _parse_integer_differentials(source: ChainComplexValue) -> tuple[Matrix, ...]:
    parsed: list[Matrix] = []
    for matrix in source.differential_matrices:
        parsed.append(
            [[parse_canonical_integer(value) for value in row] for row in matrix]
        )
    return tuple(parsed)


def _require_integral_source_bounds(source: ChainComplexValue) -> None:
    if source.coefficient_ring is not CoefficientRing.INTEGER:
        raise _domain_error(
            "integral_homology_ring_mismatch",
            "the integral Smith kernel requires coefficient ring ZZ",
        )
    if any(size > MAX_INTEGRAL_HOMOLOGY_CHAIN_RANK for size in source.basis_sizes):
        raise _domain_error(
            "integral_homology_chain_rank_exceeded",
            "integral homology requires every chain rank to be at most "
            f"{MAX_INTEGRAL_HOMOLOGY_CHAIN_RANK}",
        )
    if sum(source.basis_sizes) > MAX_INTEGRAL_HOMOLOGY_TOTAL_CHAIN_RANK:
        raise _domain_error(
            "integral_homology_total_rank_exceeded",
            "integral homology requires total chain rank at most "
            f"{MAX_INTEGRAL_HOMOLOGY_TOTAL_CHAIN_RANK}",
        )
    cells = sum(
        source.basis_sizes[index] * source.basis_sizes[index + 1]
        for index in range(len(source.basis_sizes) - 1)
    )
    if cells > MAX_INTEGRAL_HOMOLOGY_MATRIX_CELLS:
        raise _domain_error(
            "integral_homology_matrix_cells_exceeded",
            "integral homology differentials contain more than "
            f"{MAX_INTEGRAL_HOMOLOGY_MATRIX_CELLS} cells",
        )
    if any(
        len(value.lstrip("-")) > MAX_INTEGRAL_HOMOLOGY_INPUT_DIGITS
        for matrix in source.differential_matrices
        for row in matrix
        for value in row
    ):
        raise _domain_error(
            "integral_homology_input_digits_exceeded",
            "integral homology coefficients may contain at most "
            f"{MAX_INTEGRAL_HOMOLOGY_INPUT_DIGITS} decimal digits",
        )


def _require_square_zero(
    source: ChainComplexValue,
    differentials: tuple[Matrix, ...],
) -> int:
    work = 0
    for index in range(len(differentials) - 1):
        left = differentials[index]
        right = differentials[index + 1]
        middle = source.basis_sizes[index + 1]
        columns = source.basis_sizes[index + 2]
        product_bits = _matrix_product_bits(
            middle,
            _matrix_bits(left),
            _matrix_bits(right),
        )
        if product_bits > MAX_INTEGRAL_HOMOLOGY_OUTPUT_BITS:
            raise _domain_error(
                "differential_product_height_exceeded",
                "the conservative d^2 intermediate-height bound exceeds the "
                f"{MAX_INTEGRAL_HOMOLOGY_OUTPUT_BITS}-bit execution envelope",
            )
        work += source.basis_sizes[index] * middle * columns
        product = matrix_multiply(left, right, right_columns_if_empty=columns)
        if any(value for row in product for value in row):
            raise _domain_error(
                "differential_not_square_zero",
                "chain complex violates d^2=0 at chain degree "
                f"{source.degree_min + index + 1}",
            )
    return work


def _output_scalar_count(
    chain_rank: int,
    outgoing_rows: int,
    incoming_chain_rank: int,
) -> int:
    """Worst-case scalar count for one integral group and its certificates."""

    cycle_rank = chain_rank
    outgoing_certificate = (
        2 * outgoing_rows * chain_rank
        + outgoing_rows * outgoing_rows
        + chain_rank * chain_rank
    )
    incoming_certificate = (
        3 * cycle_rank * incoming_chain_rank
        + cycle_rank * cycle_rank
        + incoming_chain_rank * incoming_chain_rank
    )
    generators = cycle_rank * (chain_rank + cycle_rank + incoming_chain_rank + 1)
    metadata_and_factors = (
        32 + min(outgoing_rows, chain_rank) + 2 * min(cycle_rank, incoming_chain_rank)
    )
    return (
        outgoing_certificate + incoming_certificate + generators + metadata_and_factors
    )


def _reconstruction_work(
    chain_rank: int,
    outgoing_rows: int,
    incoming_chain_rank: int,
    cycle_rank: int,
) -> int:
    """Bound certificate replay, inverses, coordinates, and representatives."""

    outgoing_certificate = (
        outgoing_rows * outgoing_rows * chain_rank
        + outgoing_rows * chain_rank * chain_rank
        + outgoing_rows**3
        + chain_rank**3
    )
    cycle_coordinates = chain_rank**3 + (chain_rank * chain_rank * incoming_chain_rank)
    incoming_certificate = (
        cycle_rank * cycle_rank * incoming_chain_rank
        + cycle_rank * incoming_chain_rank * incoming_chain_rank
        + cycle_rank**3
        + incoming_chain_rank**3
    )
    representatives = (
        cycle_rank**3
        + cycle_rank * chain_rank * cycle_rank
        + cycle_rank * chain_rank * incoming_chain_rank
    )
    return (
        outgoing_certificate
        + cycle_coordinates
        + incoming_certificate
        + representatives
    )


def _deadline() -> float:
    execution = current_request_execution()
    started = execution.started_at if execution is not None else monotonic()
    deadline = started + INTEGRAL_HOMOLOGY_WALL_SECONDS
    bind_request_deadline(deadline)
    return deadline


def _require_deadline(deadline: float, stage: str) -> None:
    if monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            f"integral homology deadline expired {stage}"
        )


def admit_integral_homology(source: ChainComplexValue) -> IntegralHomologyExecutionPlan:
    """Admit and prepare one exact integral-homology computation."""

    deadline = _deadline()
    _require_deadline(deadline, "before source admission")
    _require_integral_source_bounds(source)
    matrix_cells = sum(
        source.basis_sizes[index] * source.basis_sizes[index + 1]
        for index in range(len(source.basis_sizes) - 1)
    )
    parse_work = (
        len(source.basis_sizes)
        + matrix_cells
        + sum(
            len(value)
            for matrix in source.differential_matrices
            for row in matrix
            for value in row
        )
    )
    differentials = _parse_integer_differentials(source)
    square_zero_work = _require_square_zero(source, differentials)
    _require_deadline(deadline, "after d^2 admission")

    degree_plans: list[IntegralHomologyDegreePlan] = []
    smith_work = 0
    reconstruction_work = 0
    # The result retains the complete source complex in addition to the
    # per-degree certificates and representatives.
    output_scalar_count = matrix_cells + len(source.basis_sizes) + 8
    for index, chain_rank in enumerate(source.basis_sizes):
        outgoing_rows = source.basis_sizes[index - 1] if index > 0 else 0
        outgoing = differentials[index - 1] if index > 0 else []
        incoming_chain_rank = (
            source.basis_sizes[index + 1] if index + 1 < len(source.basis_sizes) else 0
        )
        incoming = (
            differentials[index]
            if index < len(differentials)
            else [[] for _ in range(chain_rank)]
        )
        outgoing_height = _smith_height_bound(
            outgoing,
            rows=outgoing_rows,
            columns=chain_rank,
        )
        _require_height(
            outgoing_height,
            label=f"degree {source.degree_min + index} outgoing differential",
        )
        smith_work += outgoing_height.work_units

        coordinate_bounds: list[int] = []
        incoming_bounds: list[SmithHeightBound] = []
        result_bounds: list[int] = []
        known_outgoing_rank = _known_rank_before_smith(
            outgoing,
            rows=outgoing_rows,
            columns=chain_rank,
            bound=outgoing_height,
        )
        known_cycle_rank = (
            None if known_outgoing_rank is None else chain_rank - known_outgoing_rank
        )
        incoming_bits = _matrix_bits(incoming)
        cycle_ranks = (
            (known_cycle_rank,)
            if known_cycle_rank is not None
            else tuple(range(chain_rank + 1))
        )
        bounds_by_cycle_rank: dict[int, tuple[int, SmithHeightBound, int]] = {}
        for cycle_rank in cycle_ranks:
            if (
                known_cycle_rank == cycle_rank
                and outgoing_height.transformations_are_identity
            ):
                outgoing_rank = chain_rank - cycle_rank
                known_incoming_coordinates = incoming[outgoing_rank:]
                coordinate_bits = incoming_bits
                incoming_height = _smith_height_bound(
                    known_incoming_coordinates,
                    rows=cycle_rank,
                    columns=incoming_chain_rank,
                )
            elif (
                known_cycle_rank == cycle_rank
                and _unit_row_preserves_kernel_coordinate_height(outgoing)
            ):
                coordinate_bits = incoming_bits
                incoming_height = _smith_height_bound_from_shape(
                    cycle_rank,
                    incoming_chain_rank,
                    coordinate_bits,
                )
            else:
                right_inverse_bits = _inverse_unimodular_bits(
                    chain_rank, outgoing_height.right_bits
                )
                coordinate_bits = _matrix_product_bits(
                    chain_rank,
                    right_inverse_bits,
                    incoming_bits,
                )
                incoming_height = _smith_height_bound_from_shape(
                    cycle_rank,
                    incoming_chain_rank,
                    coordinate_bits,
                )
            _require_height(
                incoming_height,
                label=(
                    f"degree {source.degree_min + index} boundary-in-cycle "
                    f"Smith reduction at cycle rank {cycle_rank}"
                ),
            )
            cycle_coordinate_bits = _inverse_unimodular_bits(
                cycle_rank, incoming_height.left_bits
            )
            cycle_bits = _matrix_product_bits(
                cycle_rank,
                outgoing_height.right_bits,
                cycle_coordinate_bits,
            )
            bounding_relation_bits = _matrix_product_bits(
                incoming_chain_rank,
                incoming_bits,
                incoming_height.right_bits,
            )
            torsion_multiple_bits = _add_bits(
                incoming_height.diagonal_bits,
                cycle_bits,
                1,
            )
            result_bits = max(
                coordinate_bits,
                cycle_coordinate_bits,
                cycle_bits,
                incoming_height.right_bits,
                incoming_height.diagonal_bits,
                bounding_relation_bits,
                torsion_multiple_bits,
            )
            if result_bits > MAX_INTEGRAL_HOMOLOGY_OUTPUT_BITS:
                raise _domain_error(
                    "integral_homology_output_height_exceeded",
                    f"degree {source.degree_min + index} has a conservative generator/output height "
                    f"above {MAX_INTEGRAL_HOMOLOGY_OUTPUT_BITS} bits",
                )
            bounds_by_cycle_rank[cycle_rank] = (
                coordinate_bits,
                incoming_height,
                result_bits,
            )
        # Pre-Smith exact-rank cases prove the unique cycle rank. The repeated
        # tuple shape keeps execution lookup constant-time without charging or
        # claiming admission for impossible ranks.
        if known_cycle_rank is not None:
            known_bounds = bounds_by_cycle_rank[known_cycle_rank]
            bounds_by_cycle_rank = dict.fromkeys(range(chain_rank + 1), known_bounds)
        for cycle_rank in range(chain_rank + 1):
            coordinate_bits, incoming_height, result_bits = bounds_by_cycle_rank[
                cycle_rank
            ]
            coordinate_bounds.append(coordinate_bits)
            incoming_bounds.append(incoming_height)
            result_bounds.append(result_bits)
        smith_work += max(bound.work_units for bound in incoming_bounds)
        reconstruction_work += max(
            _reconstruction_work(
                chain_rank,
                outgoing_rows,
                incoming_chain_rank,
                cycle_rank,
            )
            for cycle_rank in cycle_ranks
        )
        output_scalar_count += _output_scalar_count(
            chain_rank,
            outgoing_rows,
            incoming_chain_rank,
        )
        degree_plans.append(
            IntegralHomologyDegreePlan(
                degree=source.degree_min + index,
                chain_rank=chain_rank,
                incoming_chain_rank=incoming_chain_rank,
                outgoing=outgoing,
                incoming=incoming,
                outgoing_height=outgoing_height,
                incoming_heights_by_cycle_rank=tuple(incoming_bounds),
                coordinate_bits_by_cycle_rank=tuple(coordinate_bounds),
                output_bits_by_cycle_rank=tuple(result_bounds),
            )
        )

    # One ledger covers parsing, d^2, both reductions in every degree, and
    # construction/serialization of the complete exact scalar payload.
    total_work = (
        parse_work
        + square_zero_work
        + smith_work
        + reconstruction_work
        + output_scalar_count
    )
    if total_work > MAX_INTEGRAL_HOMOLOGY_WORK_UNITS:
        raise _domain_error(
            "integral_homology_work_budget_exceeded",
            "the conservative d^2 and Smith work ledger exceeds "
            f"{MAX_INTEGRAL_HOMOLOGY_WORK_UNITS} units",
        )
    if output_scalar_count > MAX_INTEGRAL_HOMOLOGY_OUTPUT_SCALARS:
        raise _domain_error(
            "integral_homology_output_size_exceeded",
            "the complete certificate, generator, and bounding-chain result "
            f"has a conservative bound of {output_scalar_count} integer scalars, above the "
            f"{MAX_INTEGRAL_HOMOLOGY_OUTPUT_SCALARS}-scalar output envelope",
        )
    _require_deadline(deadline, "after complete admission")
    return IntegralHomologyExecutionPlan(
        source=source,
        degrees=tuple(degree_plans),
        parse_work=parse_work,
        square_zero_work=square_zero_work,
        smith_work=smith_work,
        reconstruction_work=reconstruction_work,
        output_scalar_count=output_scalar_count,
        total_work=total_work,
        deadline=deadline,
    )


def _actual_reduction_bits(reduction: SmithReduction) -> int:
    return max(
        _matrix_bits(reduction.diagonal),
        _matrix_bits(reduction.left),
        _matrix_bits(reduction.right),
    )


def _require_reduction_bound(
    reduction: SmithReduction,
    bound: SmithHeightBound,
    *,
    label: str,
) -> None:
    if _actual_reduction_bits(reduction) > bound.maximum_bits:
        raise ArithmeticError(
            f"{label} exceeded the admitted SymPy transformation-height proof"
        )


def _integer_matrix(
    entries: Matrix, *, rows: int, columns: int
) -> CertifiedIntegerMatrix:
    return CertifiedIntegerMatrix(
        row_count=rows,
        column_count=columns,
        entries=tuple(
            tuple(format_canonical_integer(value) for value in row) for row in entries
        ),
    )


def _integral_vector(values: list[int]) -> IntegralVector:
    return IntegralVector(
        coefficients=tuple(format_canonical_integer(value) for value in values)
    )


def _integer_sequence_bits(values: list[int]) -> int:
    return max((abs(value).bit_length() for value in values), default=1)


def compute_integral_homology(
    plan: IntegralHomologyExecutionPlan,
) -> tuple[IntegralHomologyGroupValue, ...]:
    """Execute the two certified Smith reductions in every admitted degree."""

    groups: list[IntegralHomologyGroupValue] = []
    for degree_plan in plan.degrees:
        _require_deadline(
            plan.deadline,
            f"before degree {degree_plan.degree} outgoing Smith reduction",
        )
        outgoing_reduction = smith_reduce(
            degree_plan.outgoing,
            row_count=(
                plan.source.basis_sizes[degree_plan.degree - plan.source.degree_min - 1]
                if degree_plan.degree > plan.source.degree_min
                else 0
            ),
            column_count=degree_plan.chain_rank,
        )
        _require_reduction_bound(
            outgoing_reduction,
            degree_plan.outgoing_height,
            label=f"degree {degree_plan.degree} outgoing Smith reduction",
        )
        outgoing_rank = outgoing_reduction.rank
        cycle_rank = degree_plan.chain_rank - outgoing_rank
        cycle_basis = matrix_columns(outgoing_reduction.right, start=outgoing_rank)
        right_inverse = inverse_unimodular(outgoing_reduction.right)
        all_cycle_coordinates = matrix_multiply(
            right_inverse,
            degree_plan.incoming,
            right_columns_if_empty=degree_plan.incoming_chain_rank,
        )
        if any(value for row in all_cycle_coordinates[:outgoing_rank] for value in row):
            raise ArithmeticError("incoming boundary is not in the outgoing kernel")
        incoming_coordinates = all_cycle_coordinates[outgoing_rank:]
        coordinate_bound = degree_plan.coordinate_bits_by_cycle_rank[cycle_rank]
        if _matrix_bits(incoming_coordinates) > coordinate_bound:
            raise ArithmeticError(
                "boundary-in-cycle coordinates exceeded the admitted height proof"
            )

        _require_deadline(
            plan.deadline,
            f"before degree {degree_plan.degree} incoming Smith reduction",
        )
        incoming_reduction = smith_reduce(
            incoming_coordinates,
            row_count=cycle_rank,
            column_count=degree_plan.incoming_chain_rank,
        )
        incoming_height = degree_plan.incoming_heights_by_cycle_rank[cycle_rank]
        _require_reduction_bound(
            incoming_reduction,
            incoming_height,
            label=f"degree {degree_plan.degree} incoming Smith reduction",
        )
        incoming_rank = incoming_reduction.rank
        incoming_left_inverse = inverse_unimodular(incoming_reduction.left)

        free_generators: list[IntegralFreeGenerator] = []
        generator_values: list[int] = []
        for index in range(incoming_rank, cycle_rank):
            coordinate = [
                incoming_left_inverse[row][index] for row in range(cycle_rank)
            ]
            cycle = matrix_vector_multiply(cycle_basis, coordinate)
            generator_values.extend(coordinate)
            generator_values.extend(cycle)
            free_generators.append(
                IntegralFreeGenerator(
                    cycle=_integral_vector(cycle),
                    cycle_coordinates=_integral_vector(coordinate),
                )
            )

        torsion_generators: list[IntegralTorsionGenerator] = []
        for index, factor in enumerate(incoming_reduction.invariant_factors):
            if factor == 1:
                continue
            coordinate = [
                incoming_left_inverse[row][index] for row in range(cycle_rank)
            ]
            cycle = matrix_vector_multiply(cycle_basis, coordinate)
            bounding_chain = [
                incoming_reduction.right[row][index]
                for row in range(degree_plan.incoming_chain_rank)
            ]
            if matrix_vector_multiply(degree_plan.incoming, bounding_chain) != [
                factor * value for value in cycle
            ]:
                raise ArithmeticError("torsion bounding-chain relation is invalid")
            generator_values.extend((factor, *coordinate, *cycle, *bounding_chain))
            torsion_generators.append(
                IntegralTorsionGenerator(
                    order=format_canonical_integer(factor),
                    cycle=_integral_vector(cycle),
                    cycle_coordinates=_integral_vector(coordinate),
                    bounding_chain=_integral_vector(bounding_chain),
                )
            )

        if (
            _integer_sequence_bits(generator_values)
            > (degree_plan.output_bits_by_cycle_rank[cycle_rank])
        ):
            raise ArithmeticError(
                "homology generators exceeded the admitted output-height proof"
            )

        groups.append(
            IntegralHomologyGroupValue(
                degree=degree_plan.degree,
                chain_rank=degree_plan.chain_rank,
                incoming_chain_rank=degree_plan.incoming_chain_rank,
                outgoing_boundary_rank=outgoing_rank,
                cycle_rank=cycle_rank,
                incoming_boundary_rank=incoming_rank,
                free_rank=cycle_rank - incoming_rank,
                torsion_invariant_factors=tuple(
                    format_canonical_integer(factor)
                    for factor in incoming_reduction.invariant_factors
                    if factor > 1
                ),
                free_generators=tuple(free_generators),
                torsion_generators=tuple(torsion_generators),
                outgoing_smith_certificate=certificate_from_reduction(
                    outgoing_reduction
                ),
                boundary_in_cycle_coordinates=_integer_matrix(
                    incoming_coordinates,
                    rows=cycle_rank,
                    columns=degree_plan.incoming_chain_rank,
                ),
                incoming_smith_certificate=certificate_from_reduction(
                    incoming_reduction
                ),
            )
        )
        _require_deadline(
            plan.deadline, f"after degree {degree_plan.degree} result construction"
        )
    return tuple(groups)


__all__ = [
    "IntegralHomologyExecutionPlan",
    "admit_integral_homology",
    "compute_integral_homology",
]
