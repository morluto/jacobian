"""Code theory operations via exact enumeration."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from itertools import product

from sympy import isprime

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.codes.general._models import (
    EXACT_ENUMERATION_PASSES,
    MAX_COVERING_RADIUS_STATES_PER_PASS,
    MAX_COVERING_RADIUS_TRANSITIONS,
    MAX_EXACT_CODEWORD_EVALUATIONS,
    SYNDROME_BFS_PASSES,
    CoveringRadiusRequest,
    LinearCodeRequest,
    _matrix_rank_mod_prime,
)

__all__ = [
    "GeneratorMatrix",
    "covering_radius",
    "minimum_distance",
    "weight_distribution",
]


GeneratorMatrix = tuple[tuple[int, ...], ...]


def _admit_prime_field_matrix(
    field_order: int,
    generator_matrix: GeneratorMatrix,
) -> int:
    if not isprime(field_order):
        raise OperationDomainValidationError(
            location=("field_order",),
            code="code_theory.field_order_not_prime",
            message="field_order must be prime for this prime-field operation",
        )
    width = len(generator_matrix[0])
    if width == 0 or width > 256:
        raise OperationDomainValidationError(
            location=("generator_matrix",),
            code="code_theory.generator_width_out_of_bounds",
            message="generator rows must have between one and 256 entries",
        )
    if any(len(row) != width for row in generator_matrix):
        raise OperationDomainValidationError(
            location=("generator_matrix",),
            code="code_theory.generator_rows_unequal",
            message="generator matrix rows must have equal length",
        )
    if any(not 0 <= entry < field_order for row in generator_matrix for entry in row):
        raise OperationDomainValidationError(
            location=("generator_matrix",),
            code="code_theory.generator_entry_not_canonical",
            message="generator entries must be canonical field residues",
        )
    return width


def _admit_enumeration(request: LinearCodeRequest) -> None:
    _admit_prime_field_matrix(request.field_order, request.generator_matrix)
    if (
        EXACT_ENUMERATION_PASSES * request.field_order ** len(request.generator_matrix)
        > MAX_EXACT_CODEWORD_EVALUATIONS
    ):
        raise OperationDomainValidationError(
            location=("generator_matrix",),
            code="code_theory.enumeration_work_exceeded",
            message="generator matrix exceeds the exact enumeration bound",
        )


def _codewords(
    generator_matrix: GeneratorMatrix, field_order: int
) -> Iterator[tuple[int, ...]]:
    from flint import nmod_mat

    n_rows = len(generator_matrix)
    generator = nmod_mat(generator_matrix, field_order)
    seen = set()
    for coeffs in product(range(field_order), repeat=n_rows):
        coefficient_row = nmod_mat([list(coeffs)], field_order)
        codeword = tuple(
            int(value) for value in (coefficient_row * generator).tolist()[0]
        )
        if codeword not in seen:
            seen.add(codeword)
            yield codeword


def minimum_distance(generator_matrix: GeneratorMatrix, field_order: int) -> int:
    """Return the exact minimum nonzero codeword weight.

    For the zero code (rank 0) no nonzero codeword exists; the code
    length is returned by the empty-code convention.
    """
    request = LinearCodeRequest(
        generator_matrix=generator_matrix, field_order=field_order
    )
    _admit_enumeration(request)
    min_dist = float("inf")
    for codeword in _codewords(request.generator_matrix, request.field_order):
        weight = sum(1 for c in codeword if c != 0)
        if weight > 0 and weight < min_dist:
            min_dist = weight
    return int(min_dist) if min_dist != float("inf") else len(generator_matrix[0])


def weight_distribution(
    generator_matrix: GeneratorMatrix, field_order: int
) -> list[tuple[int, int]]:
    from collections import Counter

    request = LinearCodeRequest(
        generator_matrix=generator_matrix, field_order=field_order
    )
    _admit_enumeration(request)

    weights: Counter[int] = Counter()
    for codeword in _codewords(request.generator_matrix, request.field_order):
        weight = sum(1 for c in codeword if c != 0)
        weights[weight] += 1
    return sorted(weights.items())


def _parity_check_matrix(
    generator_matrix: GeneratorMatrix, field_order: int
) -> list[list[int]]:
    """Return a basis of the generator matrix's right nullspace."""
    rows = [list(row) for row in generator_matrix]
    row_count = len(rows)
    column_count = len(rows[0])
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(column_count):
        pivot = next(
            (
                index
                for index in range(pivot_row, row_count)
                if rows[index][column] % field_order != 0
            ),
            None,
        )
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        inverse = pow(rows[pivot_row][column] % field_order, -1, field_order)
        rows[pivot_row] = [value * inverse % field_order for value in rows[pivot_row]]
        for index, row in enumerate(rows):
            if index == pivot_row:
                continue
            factor = row[column] % field_order
            if factor == 0:
                continue
            rows[index] = [
                (left - factor * right) % field_order
                for left, right in zip(row, rows[pivot_row], strict=True)
            ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == row_count:
            break

    pivot_set = set(pivot_columns)
    free_columns = [column for column in range(column_count) if column not in pivot_set]
    check_rows: list[list[int]] = []
    for free_column in free_columns:
        vector = [0] * column_count
        vector[free_column] = 1
        for index, pivot_column in enumerate(pivot_columns):
            vector[pivot_column] = (-rows[index][free_column]) % field_order
        check_rows.append(vector)
    return check_rows


def covering_radius(generator_matrix: GeneratorMatrix, field_order: int) -> int:
    """Compute a linear code's covering radius by syndrome-space BFS.

    One graph step adds a nonzero scalar multiple of one parity-check column,
    exactly corresponding to changing one coordinate of an error vector.
    Therefore graph distance from the zero syndrome is minimum coset-leader
    weight, and the maximum distance is the covering radius.
    """
    request = CoveringRadiusRequest(
        generator_matrix=generator_matrix, field_order=field_order
    )
    width = _admit_prime_field_matrix(request.field_order, request.generator_matrix)
    rank = _matrix_rank_mod_prime(request.generator_matrix, request.field_order)
    state_count = request.field_order ** (width - rank)
    if state_count > MAX_COVERING_RADIUS_STATES_PER_PASS:
        raise OperationDomainValidationError(
            location=("generator_matrix",),
            code="code_theory.syndrome_state_bound_exceeded",
            message="syndrome space exceeds the exact state bound",
        )
    move_count_bound = min(width * (request.field_order - 1), max(state_count - 1, 0))
    if (
        SYNDROME_BFS_PASSES * state_count * move_count_bound
        > MAX_COVERING_RADIUS_TRANSITIONS
    ):
        raise OperationDomainValidationError(
            location=("generator_matrix",),
            code="code_theory.syndrome_transition_bound_exceeded",
            message="syndrome graph exceeds the exact transition bound",
        )
    check_rows = _parity_check_matrix(
        request.generator_matrix,
        request.field_order,
    )
    if not check_rows:
        return 0

    syndrome_dimension = len(check_rows)
    column_count = len(check_rows[0])
    zero = (0,) * syndrome_dimension
    move_set = {
        tuple(
            scalar * check_rows[row][column] % request.field_order
            for row in range(syndrome_dimension)
        )
        for column in range(column_count)
        for scalar in range(1, request.field_order)
    }
    move_set.discard(zero)
    moves = tuple(sorted(move_set))

    distances = {zero: 0}
    queue = deque([zero])
    radius = 0
    while queue:
        syndrome = queue.popleft()
        next_distance = distances[syndrome] + 1
        for move in moves:
            neighbor = tuple(
                (left + right) % request.field_order
                for left, right in zip(syndrome, move, strict=True)
            )
            if neighbor in distances:
                continue
            distances[neighbor] = next_distance
            radius = max(radius, next_distance)
            queue.append(neighbor)

    expected_states = request.field_order**syndrome_dimension
    if len(distances) != expected_states:
        raise ArithmeticError("parity-check columns did not span the syndrome space")
    return radius
