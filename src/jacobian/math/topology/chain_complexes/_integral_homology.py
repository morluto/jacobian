"""Certified integral homology for finite based chain complexes.

The two Smith reductions have deliberately separate roles.  The outgoing
reduction supplies a saturated integral cycle basis; the incoming reduction
computes the quotient of that cycle lattice by the boundary lattice.  A bounded
unit presolve owns structurally trivial reductions, and a killable SymPy child
owns the remaining exact decompositions.  This module owns admission,
source-basis reconstruction, and the public height/result contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from time import monotonic

from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
)
from jacobian.canonical import parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.certified_snf.operations import (
    Matrix,
    SmithReduction,
    certificate_from_reduction,
    identity_matrix,
    matrix_columns,
    matrix_multiply,
    matrix_vector_multiply,
)
from jacobian.math.matrices.values import IntegerMatrix
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
    outgoing_presolve: SmithReductionData | None
    incoming_heights_by_cycle_rank: tuple[SmithHeightBound, ...]
    incoming_presolves_by_cycle_rank: tuple[SmithReductionData | None, ...]
    coordinate_bits_by_cycle_rank: tuple[int, ...]
    output_bits_by_cycle_rank: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class IntegralHomologyExecutionPlan:
    source: ChainComplexValue
    degrees: tuple[IntegralHomologyDegreePlan, ...]
    parse_work: int
    square_zero_work: int
    smith_work: int
    result_construction_work: int
    output_scalar_count: int
    total_work: int
    deadline: float


@dataclass(frozen=True, slots=True)
class SmithReductionData:
    """One exact Smith reduction with both inverse transformations."""

    reduction: SmithReduction
    left_inverse: Matrix
    right_inverse: Matrix
    work_units: int
    intermediate_bits: int


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


def _fraction_free_inverse_work(size: int) -> int:
    """Bound scalar arithmetic in SymPy's fraction-free inverse path.

    ``DomainMatrix.inv_den(method="rref")`` row-reduces an ``n x 2n``
    augmented matrix.  At each of at most ``n`` pivots, scanning and updating
    every other row touches at most ``2n`` entries with a multiply, subtract,
    and exact division.  The factor below also covers pivot search, slicing,
    and dense conversion.
    """

    return 8 * size**3 + 8 * size**2 + 1


def _determinant_work(size: int) -> int:
    """Conservatively price one maintained exact determinant computation."""

    return 4 * size**4 + 8 * size**2 + 1


def _maintained_worker_projection_work(rows: int, columns: int) -> int:
    """Price work executed around the maintained Smith decomposition.

    The Smith recurrence itself is charged by the core height functions.  A
    general worker additionally computes two exact determinant signs, obtains
    the two inverses needed for homology coordinates with fraction-free RREF,
    and converts a bounded projection across the process boundary.
    """

    projected_scalars = 2 * rows * columns + 3 * rows * rows + 3 * columns * columns
    return (
        _fraction_free_inverse_work(rows)
        + _fraction_free_inverse_work(columns)
        + _determinant_work(rows)
        + _determinant_work(columns)
        + 8 * (projected_scalars + rows + columns + 1)
    )


def _include_maintained_worker_bound(
    bound: SmithHeightBound,
    *,
    rows: int,
    columns: int,
) -> SmithHeightBound:
    """Include the actual inverse, determinant, and projection phases."""

    left_inverse_bits = _inverse_unimodular_bits(rows, bound.left_bits)
    right_inverse_bits = _inverse_unimodular_bits(columns, bound.right_bits)
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
            left_inverse_bits,
            right_inverse_bits,
            left_determinant_bits,
            right_determinant_bits,
        ),
        work_units=(
            bound.work_units + _maintained_worker_projection_work(rows, columns)
        ),
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


def _maintained_smith_structural_work(rows: int, columns: int) -> int:
    operation_cells = rows * columns + rows * rows + columns * columns
    pivot_scans = sum(
        (rows - pivot) * (columns - pivot) for pivot in range(min(rows, columns))
    )
    return 8 * (operation_cells + 1) + pivot_scans


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
    operation_cells = rows * columns + rows * rows + columns * columns
    updates = max(1, length - 1) * (input_bits + 1)
    work = _maintained_smith_structural_work(rows, columns) + 12 * updates * max(
        1, operation_cells
    )
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

    recursive = _smith_core_bound_from_shape(
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
        + _maintained_smith_structural_work(rows, columns)
        + 12 * updates * max(1, operation_cells)
        + 60 * repairs * max(1, operation_cells)
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

    recursive = _smith_core_bound_from_shape(rows - 1, columns - 1, 2)
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
        + _maintained_smith_structural_work(rows, columns)
        + 12 * (rows + columns) * max(1, operation_cells)
        + 60 * repairs * max(1, operation_cells)
    )
    return SmithHeightBound(
        left_bits=left_bits,
        right_bits=right_bits,
        diagonal_bits=factor_bits,
        intermediate_bits=max(2, recursive.intermediate_bits),
        work_units=work,
    )


def _smith_core_bound_from_shape(
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
    return core


def _smith_height_bound_from_shape(
    rows: int,
    columns: int,
    input_bits: int,
) -> SmithHeightBound:
    return _include_maintained_worker_bound(
        _smith_core_bound_from_shape(rows, columns, input_bits),
        rows=rows,
        columns=columns,
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
        return core
    if _is_positive_smith_diagonal(matrix, rows=rows, columns=columns):
        core = SmithHeightBound(
            1,
            1,
            input_bits,
            input_bits,
            rows * columns,
            True,
        )
        return core
    return _smith_height_bound_from_shape(rows, columns, input_bits)


class _PresolveHeightExceededError(Exception):
    """The visible-pivot presolve cannot stay inside the integer envelope."""


def _checked_linear_update(value: int, factor: int, source: int) -> int:
    """Return ``value + factor * source`` without creating an over-height value."""

    predicted_bits = max(
        abs(value).bit_length(),
        abs(factor).bit_length() + abs(source).bit_length() + 1,
    )
    if predicted_bits > MAX_INTEGRAL_HOMOLOGY_OUTPUT_BITS:
        raise _PresolveHeightExceededError
    result = value + factor * source
    if abs(result).bit_length() > MAX_INTEGRAL_HOMOLOGY_OUTPUT_BITS:
        raise _PresolveHeightExceededError
    return result


def _swap_rows(matrix: Matrix, left: int, right: int) -> int:
    if left == right:
        return 0
    matrix[left], matrix[right] = matrix[right], matrix[left]
    return len(matrix[left])


def _swap_columns(matrix: Matrix, left: int, right: int) -> int:
    if left == right:
        return 0
    for row in matrix:
        row[left], row[right] = row[right], row[left]
    return len(matrix)


def _scale_row(matrix: Matrix, row: int, factor: int) -> int:
    matrix[row] = [factor * value for value in matrix[row]]
    return len(matrix[row])


def _scale_column(matrix: Matrix, column: int, factor: int) -> int:
    for row in matrix:
        row[column] *= factor
    return len(matrix)


def _add_row_multiple(matrix: Matrix, *, target: int, source: int, factor: int) -> int:
    matrix[target] = [
        _checked_linear_update(value, factor, source_value)
        for value, source_value in zip(matrix[target], matrix[source], strict=True)
    ]
    return 2 * len(matrix[target])


def _add_column_multiple(
    matrix: Matrix, *, target: int, source: int, factor: int
) -> int:
    for row in matrix:
        row[target] = _checked_linear_update(row[target], factor, row[source])
    return 2 * len(matrix)


def _identity_reduction(
    source: Matrix,
    *,
    rows: int,
    columns: int,
    diagonal: Matrix,
    rank: int,
    factors: tuple[int, ...],
) -> SmithReductionData:
    left = identity_matrix(rows)
    right = identity_matrix(columns)
    reduction = SmithReduction(
        source=[row[:] for row in source],
        diagonal=diagonal,
        left=left,
        right=right,
        rank=rank,
        invariant_factors=factors,
        left_determinant=1,
        right_determinant=1,
    )
    return SmithReductionData(
        reduction=reduction,
        left_inverse=[row[:] for row in left],
        right_inverse=[row[:] for row in right],
        # Cover source/diagonal scans and copies plus construction, comparison,
        # and retention of both square transformations and their inverses.
        # In particular, a 0 x n or n x 0 reduction still constructs two n x n
        # identity matrices and must not be priced as one unit.
        work_units=_identity_reduction_work(rows, columns),
        intermediate_bits=max(1, _matrix_bits(source)),
    )


def _identity_reduction_work(rows: int, columns: int) -> int:
    """Charge scans, copies, and retained identities for a zero reduction."""

    return 8 * (rows * columns + rows * rows + columns * columns + 1)


def _lazy_zero_smith_height(rows: int, columns: int) -> SmithHeightBound:
    """Describe a zero Smith reduction without allocating its matrices."""

    return SmithHeightBound(
        left_bits=1,
        right_bits=1,
        diagonal_bits=1,
        intermediate_bits=1,
        work_units=_identity_reduction_work(rows, columns),
        transformations_are_identity=True,
    )


def _presolve_structural_work(rows: int, columns: int) -> int:
    """Charge retained matrices independent of exact pivot outcomes."""

    return 8 * (rows * columns + rows * rows + columns * columns + 1)


def _presolve_attempt_work_bound(
    rows: int,
    columns: int,
) -> int:
    """Bound one exact visible-pivot attempt before general Smith work."""

    work = _presolve_structural_work(rows, columns)
    state_scalars = rows * columns + 2 * rows * rows + 2 * columns * columns
    for pivot in range(min(rows, columns)):
        remaining_cells = (rows - pivot) * (columns - pivot)
        # One scan chooses a minimum-magnitude nonzero entry. Unless it is a
        # unit, one further scan decides whether it divides the active block.
        work += 2 * remaining_cells
        remaining_rows = rows - pivot - 1
        remaining_columns = columns - pivot - 1
        # At most one row swap, one column swap, and one sign normalization.
        work += 5 * rows + 4 * columns
        # Every clearing update touches the working matrix, its accumulated
        # transformation, and the inverse transformation.
        work += remaining_rows * (2 * columns + 4 * rows)
        work += remaining_columns * (2 * rows + 4 * columns)
        # Retain the largest intermediate after row and column clearing.
        work += 2 * state_scalars
    return work


def _select_visible_divisor_pivot(
    working: Matrix,
    *,
    pivot: int,
) -> tuple[tuple[int, int] | None, int, bool]:
    """Select a minimum-magnitude entry when it divides the trailing block.

    If any entry of the block divides every entry, its magnitude equals the
    minimum nonzero magnitude. Thus testing one minimum is complete for this
    presolve class and avoids an exhaustive scan over candidate pivots.
    """

    rows = len(working)
    columns = len(working[0]) if working else 0
    work = 0
    selected: tuple[int, int] | None = None
    minimum = 0
    for row in range(pivot, rows):
        for column in range(pivot, columns):
            work += 1
            value = working[row][column]
            magnitude = abs(value)
            if magnitude and (selected is None or magnitude < minimum):
                selected = (row, column)
                minimum = magnitude
    if selected is None:
        return None, work, False
    if minimum == 1:
        return selected, work, True

    divisor = working[selected[0]][selected[1]]
    for row in range(pivot, rows):
        for column in range(pivot, columns):
            work += 1
            if working[row][column] % divisor:
                return None, work, True
    return selected, work, True


def _position_visible_pivot(
    working: Matrix,
    left: Matrix,
    right: Matrix,
    left_inverse: Matrix,
    right_inverse: Matrix,
    *,
    pivot: int,
    selected_row: int,
    selected_column: int,
) -> tuple[int, int, int]:
    """Move one selected divisor to the positive diagonal position."""

    work = 0
    left_determinant = 1
    right_determinant = 1
    if selected_row != pivot:
        work += _swap_rows(working, pivot, selected_row)
        work += _swap_rows(left, pivot, selected_row)
        work += _swap_columns(left_inverse, pivot, selected_row)
        left_determinant = -1
    if selected_column != pivot:
        work += _swap_columns(working, pivot, selected_column)
        work += _swap_columns(right, pivot, selected_column)
        work += _swap_rows(right_inverse, pivot, selected_column)
        right_determinant = -1
    if working[pivot][pivot] < 0:
        work += _scale_row(working, pivot, -1)
        work += _scale_row(left, pivot, -1)
        work += _scale_column(left_inverse, pivot, -1)
        left_determinant *= -1
    return work, left_determinant, right_determinant


def _clear_visible_pivot(
    working: Matrix,
    left: Matrix,
    right: Matrix,
    left_inverse: Matrix,
    right_inverse: Matrix,
    *,
    pivot: int,
) -> tuple[int, int]:
    """Clear the row and column of one positive divisor pivot."""

    work = 0
    pivot_value = working[pivot][pivot]
    state_scalars = (
        len(working) * len(right)
        + 2 * len(left) * len(left)
        + 2 * len(right) * len(right)
    )
    for row in range(pivot + 1, len(working)):
        value = working[row][pivot]
        if value == 0:
            continue
        coefficient, remainder = divmod(value, pivot_value)
        if remainder:
            raise ArithmeticError("visible Smith pivot does not divide its column")
        work += _add_row_multiple(
            working, target=row, source=pivot, factor=-coefficient
        )
        work += _add_row_multiple(left, target=row, source=pivot, factor=-coefficient)
        work += _add_column_multiple(
            left_inverse, target=pivot, source=row, factor=coefficient
        )
    maximum_bits = max(
        _matrix_bits(matrix)
        for matrix in (working, left, right, left_inverse, right_inverse)
    )
    work += state_scalars
    column_count = len(right)
    for column in range(pivot + 1, column_count):
        value = working[pivot][column]
        if value == 0:
            continue
        coefficient, remainder = divmod(value, pivot_value)
        if remainder:
            raise ArithmeticError("visible Smith pivot does not divide its row")
        work += _add_column_multiple(
            working, target=column, source=pivot, factor=-coefficient
        )
        work += _add_column_multiple(
            right, target=column, source=pivot, factor=-coefficient
        )
        work += _add_row_multiple(
            right_inverse, target=pivot, source=column, factor=coefficient
        )
    maximum_bits = max(
        maximum_bits,
        *(
            _matrix_bits(matrix)
            for matrix in (working, left, right, left_inverse, right_inverse)
        ),
    )
    work += state_scalars
    return work, maximum_bits


def _presolve_visible_smith(
    source: Matrix,
    *,
    rows: int,
    columns: int,
) -> SmithReductionData | None:
    """Exactly reduce matrices cleared by visible divisibility pivots.

    A pivot that divides its complete trailing block clears its row and column
    using integral elementary operations. Selecting such a pivot at every step
    also proves the positive divisibility order of the resulting diagonal.
    A minimum-magnitude candidate decides this presolve class in two scans of
    each trailing block.
    """

    if rows == 0 or columns == 0 or not any(value for row in source for value in row):
        return _identity_reduction(
            source,
            rows=rows,
            columns=columns,
            diagonal=[[0 for _ in range(columns)] for _ in range(rows)],
            rank=0,
            factors=(),
        )
    if _is_positive_smith_diagonal(source, rows=rows, columns=columns):
        diagonal_factors = tuple(
            source[index][index]
            for index in range(min(rows, columns))
            if source[index][index] != 0
        )
        return _identity_reduction(
            source,
            rows=rows,
            columns=columns,
            diagonal=[row[:] for row in source],
            rank=len(diagonal_factors),
            factors=diagonal_factors,
        )

    working = [row[:] for row in source]
    left = identity_matrix(rows)
    right = identity_matrix(columns)
    left_inverse = identity_matrix(rows)
    right_inverse = identity_matrix(columns)
    left_determinant = 1
    right_determinant = 1
    work = _presolve_structural_work(rows, columns)
    factors: list[int] = []
    maximum_bits = _matrix_bits(source)
    try:
        for pivot in range(min(rows, columns)):
            selected, selection_work, has_nonzero = _select_visible_divisor_pivot(
                working,
                pivot=pivot,
            )
            work += selection_work
            if selected is None:
                if has_nonzero:
                    return None
                break
            selected_row, selected_column = selected
            positioned_work, left_sign, right_sign = _position_visible_pivot(
                working,
                left,
                right,
                left_inverse,
                right_inverse,
                pivot=pivot,
                selected_row=selected_row,
                selected_column=selected_column,
            )
            work += positioned_work
            left_determinant *= left_sign
            right_determinant *= right_sign
            cleared_work, cleared_bits = _clear_visible_pivot(
                working,
                left,
                right,
                left_inverse,
                right_inverse,
                pivot=pivot,
            )
            work += cleared_work
            maximum_bits = max(maximum_bits, cleared_bits)
            factors.append(working[pivot][pivot])
    except _PresolveHeightExceededError:
        return None

    source_bits = _matrix_bits(source)
    left_bits = _matrix_bits(left)
    right_bits = _matrix_bits(right)
    left_inverse_bits = _matrix_bits(left_inverse)
    right_inverse_bits = _matrix_bits(right_inverse)
    maximum_bits = max(
        maximum_bits,
        source_bits,
        _matrix_bits(working),
        left_bits,
        right_bits,
        left_inverse_bits,
        right_inverse_bits,
    )
    if maximum_bits > MAX_INTEGRAL_HOMOLOGY_OUTPUT_BITS:
        return None
    return SmithReductionData(
        reduction=SmithReduction(
            source=[row[:] for row in source],
            diagonal=working,
            left=left,
            right=right,
            rank=len(factors),
            invariant_factors=tuple(factors),
            left_determinant=left_determinant,
            right_determinant=right_determinant,
        ),
        left_inverse=left_inverse,
        right_inverse=right_inverse,
        work_units=work,
        intermediate_bits=maximum_bits,
    )


def _smith_admission(
    matrix: Matrix,
    *,
    rows: int,
    columns: int,
) -> tuple[SmithHeightBound, SmithReductionData | None]:
    attempt_work = _presolve_attempt_work_bound(rows, columns)
    presolved = _presolve_visible_smith(
        matrix,
        rows=rows,
        columns=columns,
    )
    if presolved is None:
        bound = _smith_height_bound(matrix, rows=rows, columns=columns)
        return (
            SmithHeightBound(
                left_bits=bound.left_bits,
                right_bits=bound.right_bits,
                diagonal_bits=bound.diagonal_bits,
                intermediate_bits=bound.intermediate_bits,
                work_units=bound.work_units + attempt_work,
                transformations_are_identity=bound.transformations_are_identity,
            ),
            None,
        )
    reduction = presolved.reduction
    return (
        SmithHeightBound(
            left_bits=_matrix_bits(reduction.left),
            right_bits=_matrix_bits(reduction.right),
            diagonal_bits=_matrix_bits(reduction.diagonal),
            intermediate_bits=presolved.intermediate_bits,
            work_units=presolved.work_units,
            transformations_are_identity=(
                reduction.left == identity_matrix(rows)
                and reduction.right == identity_matrix(columns)
            ),
        ),
        presolved,
    )


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
        left_cells = source.basis_sizes[index] * middle
        right_cells = middle * columns
        product_cells = source.basis_sizes[index] * columns
        work += left_cells + right_cells + product_cells
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


def _result_construction_work(
    chain_rank: int,
    incoming_chain_rank: int,
    cycle_rank: int,
) -> int:
    """Bound executed coordinate projection and representative construction."""

    coordinate_projection = (
        chain_rank * chain_rank * incoming_chain_rank + chain_rank * incoming_chain_rank
    )
    cycle_basis_copy = chain_rank * cycle_rank
    representatives = cycle_rank * (
        cycle_rank + chain_rank * cycle_rank + chain_rank + incoming_chain_rank + 1
    )
    return coordinate_projection + cycle_basis_copy + representatives + 1


def _deadline() -> float:
    execution = current_request_execution()
    started = execution.started_at if execution is not None else monotonic()
    owner_deadline = started + INTEGRAL_HOMOLOGY_WALL_SECONDS
    deadline = (
        min(owner_deadline, execution.deadline)
        if execution is not None and execution.deadline is not None
        else owner_deadline
    )
    bind_request_deadline(deadline)
    return deadline


def _require_deadline(deadline: float, stage: str) -> None:
    request_checkpoint(stage)
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
    result_construction_work = 0
    # The result retains the complete source complex in addition to the
    # per-degree certificates and representatives.
    output_scalar_count = matrix_cells + len(source.basis_sizes) + 8
    for index, chain_rank in enumerate(source.basis_sizes):
        degree = source.degree_min + index
        _require_deadline(deadline, f"before degree {degree} admission")
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
        outgoing_height, outgoing_presolve = _smith_admission(
            outgoing,
            rows=outgoing_rows,
            columns=chain_rank,
        )
        _require_height(
            outgoing_height,
            label=f"degree {source.degree_min + index} outgoing differential",
        )
        smith_work += outgoing_height.work_units
        _require_deadline(deadline, f"after degree {degree} outgoing Smith admission")

        coordinate_bounds: list[int] = []
        incoming_bounds: list[SmithHeightBound] = []
        incoming_presolves: list[SmithReductionData | None] = []
        result_bounds: list[int] = []
        known_outgoing_rank = (
            outgoing_presolve.reduction.rank
            if outgoing_presolve is not None
            else _known_rank_before_smith(
                outgoing,
                rows=outgoing_rows,
                columns=chain_rank,
                bound=outgoing_height,
            )
        )
        known_cycle_rank = (
            None if known_outgoing_rank is None else chain_rank - known_outgoing_rank
        )
        incoming_bits = _matrix_bits(incoming)
        incoming_is_zero = not any(value for row in incoming for value in row)
        cycle_ranks = (
            (known_cycle_rank,)
            if known_cycle_rank is not None
            else tuple(range(chain_rank + 1))
        )
        exact_incoming_coordinates: Matrix | None = None
        if outgoing_presolve is not None and known_outgoing_rank is not None:
            all_coordinates = matrix_multiply(
                outgoing_presolve.right_inverse,
                incoming,
                right_columns_if_empty=incoming_chain_rank,
            )
            if any(
                value for row in all_coordinates[:known_outgoing_rank] for value in row
            ):
                raise ArithmeticError(
                    "visible-pivot Smith presolve produced invalid cycle coordinates"
                )
            exact_incoming_coordinates = all_coordinates[known_outgoing_rank:]
        bounds_by_cycle_rank: dict[
            int, tuple[int, SmithHeightBound, SmithReductionData | None, int]
        ] = {}
        for cycle_rank in cycle_ranks:
            _require_deadline(
                deadline,
                f"during degree {degree} cycle-rank {cycle_rank} admission",
            )
            if incoming_is_zero:
                coordinate_bits = 1
                incoming_height = _lazy_zero_smith_height(
                    cycle_rank,
                    incoming_chain_rank,
                )
                incoming_presolve = None
            elif exact_incoming_coordinates is not None:
                coordinate_bits = _matrix_bits(exact_incoming_coordinates)
                incoming_height, incoming_presolve = _smith_admission(
                    exact_incoming_coordinates,
                    rows=cycle_rank,
                    columns=incoming_chain_rank,
                )
            elif (
                known_cycle_rank == cycle_rank
                and outgoing_height.transformations_are_identity
            ):
                outgoing_rank = chain_rank - cycle_rank
                known_incoming_coordinates = incoming[outgoing_rank:]
                coordinate_bits = incoming_bits
                incoming_height, incoming_presolve = _smith_admission(
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
                incoming_presolve = None
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
                incoming_presolve = None
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
                _matrix_bits(outgoing),
                incoming_bits,
                outgoing_height.left_bits,
                outgoing_height.right_bits,
                outgoing_height.diagonal_bits,
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
                incoming_presolve,
                result_bits,
            )
        # Pre-Smith exact-rank cases prove the unique cycle rank. The repeated
        # tuple shape keeps execution lookup constant-time without charging or
        # claiming admission for impossible ranks.
        if known_cycle_rank is not None:
            known_bounds = bounds_by_cycle_rank[known_cycle_rank]
            bounds_by_cycle_rank = dict.fromkeys(range(chain_rank + 1), known_bounds)
        for cycle_rank in range(chain_rank + 1):
            (
                coordinate_bits,
                incoming_height,
                incoming_presolve,
                result_bits,
            ) = bounds_by_cycle_rank[cycle_rank]
            coordinate_bounds.append(coordinate_bits)
            incoming_bounds.append(incoming_height)
            incoming_presolves.append(incoming_presolve)
            result_bounds.append(result_bits)
        smith_work += max(bound.work_units for bound in incoming_bounds)
        result_construction_work += max(
            _result_construction_work(
                chain_rank,
                incoming_chain_rank,
                cycle_rank,
            )
            for cycle_rank in cycle_ranks
        )
        degree_output_scalars = _output_scalar_count(
            chain_rank,
            outgoing_rows,
            incoming_chain_rank,
        )
        output_scalar_count += degree_output_scalars
        degree_plans.append(
            IntegralHomologyDegreePlan(
                degree=degree,
                chain_rank=chain_rank,
                incoming_chain_rank=incoming_chain_rank,
                outgoing=outgoing,
                incoming=incoming,
                outgoing_height=outgoing_height,
                outgoing_presolve=outgoing_presolve,
                incoming_heights_by_cycle_rank=tuple(incoming_bounds),
                incoming_presolves_by_cycle_rank=tuple(incoming_presolves),
                coordinate_bits_by_cycle_rank=tuple(coordinate_bounds),
                output_bits_by_cycle_rank=tuple(result_bounds),
            )
        )
        _require_deadline(deadline, f"after degree {degree} admission")

    # One ledger covers parsing, d^2, both reductions in every degree, and
    # construction/serialization of the complete exact scalar payload.
    total_work = (
        parse_work
        + square_zero_work
        + smith_work
        + result_construction_work
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
        result_construction_work=result_construction_work,
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


def _integer_matrix(entries: Matrix, *, rows: int, columns: int) -> IntegerMatrix:
    return IntegerMatrix(
        row_count=rows,
        column_count=columns,
        entries=tuple(tuple(int(value) for value in row) for row in entries),
    )


def _integral_vector(values: list[int]) -> IntegralVector:
    return IntegralVector(coefficients=tuple(values))


def _integer_sequence_bits(values: list[int]) -> int:
    return max((abs(value).bit_length() for value in values), default=1)


def _execute_smith_reduction(
    source: Matrix,
    *,
    rows: int,
    columns: int,
    presolved: SmithReductionData | None,
    height: SmithHeightBound,
    deadline: float,
) -> SmithReductionData:
    """Return one admitted reduction, reusing exact presolve when available."""

    if presolved is not None:
        return presolved
    if rows == 0 or columns == 0 or not any(value for row in source for value in row):
        return _identity_reduction(
            source,
            rows=rows,
            columns=columns,
            diagonal=[[0 for _ in range(columns)] for _ in range(rows)],
            rank=0,
            factors=(),
        )
    from jacobian.math.topology.chain_complexes._smith_process import (
        smith_reduce_in_worker,
    )

    completed = smith_reduce_in_worker(
        source,
        rows=rows,
        columns=columns,
        deadline=deadline,
        left_bits=height.left_bits,
        right_bits=height.right_bits,
        diagonal_bits=height.diagonal_bits,
        left_inverse_bits=_inverse_unimodular_bits(rows, height.left_bits),
        right_inverse_bits=_inverse_unimodular_bits(columns, height.right_bits),
    )
    return SmithReductionData(
        reduction=completed.reduction,
        left_inverse=completed.left_inverse,
        right_inverse=completed.right_inverse,
        work_units=0,
        intermediate_bits=_actual_reduction_bits(completed.reduction),
    )


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
        outgoing_execution = _execute_smith_reduction(
            degree_plan.outgoing,
            rows=(
                plan.source.basis_sizes[degree_plan.degree - plan.source.degree_min - 1]
                if degree_plan.degree > plan.source.degree_min
                else 0
            ),
            columns=degree_plan.chain_rank,
            presolved=degree_plan.outgoing_presolve,
            height=degree_plan.outgoing_height,
            deadline=plan.deadline,
        )
        outgoing_reduction = outgoing_execution.reduction
        _require_reduction_bound(
            outgoing_reduction,
            degree_plan.outgoing_height,
            label=f"degree {degree_plan.degree} outgoing Smith reduction",
        )
        outgoing_rank = outgoing_reduction.rank
        cycle_rank = degree_plan.chain_rank - outgoing_rank
        cycle_basis = matrix_columns(outgoing_reduction.right, start=outgoing_rank)
        right_inverse = outgoing_execution.right_inverse
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
        incoming_execution = _execute_smith_reduction(
            incoming_coordinates,
            rows=cycle_rank,
            columns=degree_plan.incoming_chain_rank,
            presolved=degree_plan.incoming_presolves_by_cycle_rank[cycle_rank],
            height=degree_plan.incoming_heights_by_cycle_rank[cycle_rank],
            deadline=plan.deadline,
        )
        incoming_reduction = incoming_execution.reduction
        incoming_height = degree_plan.incoming_heights_by_cycle_rank[cycle_rank]
        _require_reduction_bound(
            incoming_reduction,
            incoming_height,
            label=f"degree {degree_plan.degree} incoming Smith reduction",
        )
        incoming_rank = incoming_reduction.rank
        incoming_left_inverse = incoming_execution.left_inverse

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
            generator_values.extend((factor, *coordinate, *cycle, *bounding_chain))
            torsion_generators.append(
                IntegralTorsionGenerator(
                    order=factor,
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
                kind="FINITELY_GENERATED_ABELIAN_GROUP",
                degree=degree_plan.degree,
                chain_rank=degree_plan.chain_rank,
                incoming_chain_rank=degree_plan.incoming_chain_rank,
                outgoing_boundary_rank=outgoing_rank,
                cycle_rank=cycle_rank,
                incoming_boundary_rank=incoming_rank,
                free_rank=cycle_rank - incoming_rank,
                torsion_invariant_factors=tuple(
                    factor
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
