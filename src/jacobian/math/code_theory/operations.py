"""Code theory operations via exact enumeration."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from itertools import product

__all__ = [
    "GeneratorMatrix",
    "covering_radius",
    "minimum_distance",
    "weight_distribution",
]


GeneratorMatrix = tuple[tuple[int, ...], ...]


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
    from jacobian.math.code_theory._models import LinearCodeRequest

    request = LinearCodeRequest(
        generator_matrix=generator_matrix, field_order=field_order
    )
    min_dist = float("inf")
    for codeword in _codewords(request.generator_matrix, request.field_order):
        weight = sum(1 for c in codeword if c != 0)
        if weight > 0 and weight < min_dist:
            min_dist = weight
    return int(min_dist) if min_dist != float("inf") else 0


def weight_distribution(
    generator_matrix: GeneratorMatrix, field_order: int
) -> list[tuple[int, int]]:
    from collections import Counter

    from jacobian.math.code_theory._models import LinearCodeRequest

    request = LinearCodeRequest(
        generator_matrix=generator_matrix, field_order=field_order
    )

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
    from jacobian.math.code_theory._models import CoveringRadiusRequest

    request = CoveringRadiusRequest(
        generator_matrix=generator_matrix, field_order=field_order
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
