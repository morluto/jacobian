"""Exact HNF-based kernel for affine fixed loci on standard real tori."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import lcm, prod

from flint import fmpq, fmpq_mat, fmpz_mat

from jacobian.math.geometry.affine_tori._kernel_types import (
    AffineTorusKernelSource,
    EmptyFixedLocusKernel,
    FixedLocusKernel,
    NonemptyFixedLocusKernel,
)


@dataclass(frozen=True, slots=True)
class _SaturatedKernel:
    basis: fmpz_mat
    transpose_hnf: fmpz_mat
    transpose_transform: fmpz_mat


def _integer_rank(matrix: fmpz_mat) -> int:
    return int(matrix.rank())


def _integer_hnf(matrix: fmpz_mat) -> fmpz_mat:
    return matrix.hnf()


def _rational_solve(left: fmpq_mat, right: fmpq_mat) -> fmpq_mat:
    return left.solve(right)


def _integer_snf(matrix: fmpz_mat) -> fmpz_mat:
    return matrix.snf()


def _rational_inverse(matrix: fmpq_mat) -> fmpq_mat:
    return matrix.inv()


def _integer_multiply(left: fmpz_mat, right: fmpz_mat) -> fmpz_mat:
    return left * right


def _integer_matrix(
    rows: int, columns: int, entries: tuple[tuple[int, ...], ...]
) -> fmpz_mat:
    if len(entries) != rows or any(len(row) != columns for row in entries):
        raise AssertionError("integer matrix entries disagree with their shape")
    return fmpz_mat(rows, columns, [value for row in entries for value in row])


def _zero_integer_matrix(rows: int, columns: int) -> fmpz_mat:
    return fmpz_mat(rows, columns, [0] * (rows * columns))


def _zero_rational_matrix(rows: int, columns: int) -> fmpq_mat:
    return fmpq_mat(rows, columns, [fmpq(0)] * (rows * columns))


def _integer_rows(matrix: fmpz_mat) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(matrix[row, column]) for column in range(matrix.ncols()))
        for row in range(matrix.nrows())
    )


def _submatrix(
    matrix: fmpz_mat, rows: tuple[int, ...], columns: tuple[int, ...]
) -> fmpz_mat:
    return fmpz_mat(
        len(rows),
        len(columns),
        [matrix[row, column] for row in rows for column in columns],
    )


def _augmented_hnf_transform(source: fmpz_mat) -> tuple[fmpz_mat, fmpz_mat]:
    """Return ``H, U`` from the canonical HNF of ``[source | I]``.

    FLINT's transform is mathematically non-unique. Extracting it as the right
    block of one canonical augmented HNF makes both its identity and its height
    follow from the admitted source minors: ``[H | U] = HNF([source | I])``.
    """

    rows = source.nrows()
    columns = source.ncols()
    augmented = fmpz_mat(
        rows,
        columns + rows,
        [
            (source[row, column] if column < columns else int(column - columns == row))
            for row in range(rows)
            for column in range(columns + rows)
        ],
    )
    augmented_hnf = _integer_hnf(augmented)
    hnf = _submatrix(
        augmented_hnf,
        tuple(range(rows)),
        tuple(range(columns)),
    )
    transform = _submatrix(
        augmented_hnf,
        tuple(range(rows)),
        tuple(range(columns, columns + rows)),
    )
    return hnf, transform


def _saturated_integer_kernel(
    source: fmpz_mat,
    *,
    rank: int,
) -> _SaturatedKernel:
    """Return the canonical saturated column basis of ``ker_Z(source)``.

    In the canonical row HNF ``[H | T] = HNF([source^t | I])``, the final
    ``q-rank(source)`` rows have zero left block. Their right block is already
    the canonical saturated row basis of the left kernel of ``source^t``;
    transposing yields the desired integer kernel.
    """

    transpose_hnf, transform = _augmented_hnf_transform(source.transpose())
    nullity = source.ncols() - rank
    raw_rows = fmpz_mat(
        nullity,
        source.ncols(),
        [
            transform[row, column]
            for row in range(rank, source.ncols())
            for column in range(source.ncols())
        ],
    )
    # The zero-left-block rows are already the row-HNF basis of the saturated
    # kernel: deleting the preceding pivot rows and leading zero columns from
    # HNF([source^t | I]) preserves the row-HNF conditions.
    basis = raw_rows.transpose()
    return _SaturatedKernel(
        basis=basis,
        transpose_hnf=transpose_hnf,
        transpose_transform=transform,
    )


def _first_rank_minor(
    matrix: fmpz_mat, rank: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Select the lexicographic greedy full-rank row/column minor."""

    if rank == 0:
        return (), ()
    all_columns = tuple(range(matrix.ncols()))
    rows: list[int] = []
    current_rank = 0
    for row in range(matrix.nrows()):
        candidate = (*rows, row)
        candidate_rank = _integer_rank(_submatrix(matrix, candidate, all_columns))
        if candidate_rank > current_rank:
            rows.append(row)
            current_rank = candidate_rank
            if current_rank == rank:
                break
    columns: list[int] = []
    current_rank = 0
    for column in range(matrix.ncols()):
        candidate = (*columns, column)
        candidate_rank = _integer_rank(_submatrix(matrix, tuple(rows), candidate))
        if candidate_rank > current_rank:
            columns.append(column)
            current_rank = candidate_rank
            if current_rank == rank:
                break
    if len(rows) != rank or len(columns) != rank:
        raise ArithmeticError("rank-minor selection did not reach the matrix rank")
    return tuple(rows), tuple(columns)


def _solve_integral_character_system(
    kernel: _SaturatedKernel,
    right_hand_side: tuple[int, ...],
) -> tuple[int, ...]:
    """Solve ``W z = h`` using the already-computed HNF of ``W^t``."""

    ambient_dimension = kernel.transpose_hnf.nrows()
    equation_count = kernel.transpose_hnf.ncols()
    if len(right_hand_side) != equation_count:
        raise AssertionError("character right-hand side has the wrong dimension")
    if equation_count == 0:
        return (0,) * ambient_dimension
    if any(
        int(kernel.transpose_hnf[row, column]) != int(row == column)
        for row in range(equation_count)
        for column in range(equation_count)
    ):
        raise ArithmeticError(
            "saturated character lattice did not have identity HNF leading block"
        )
    transformed = fmpz_mat(
        ambient_dimension,
        1,
        [
            right_hand_side[index] if index < equation_count else 0
            for index in range(ambient_dimension)
        ],
    )
    solution = _integer_multiply(kernel.transpose_transform.transpose(), transformed)
    return tuple(int(solution[index, 0]) for index in range(ambient_dimension))


def _require_integral(matrix: fmpq_mat, *, label: str) -> fmpz_mat:
    entries: list[int] = []
    for row in range(matrix.nrows()):
        for column in range(matrix.ncols()):
            value = matrix[row, column]
            if value.q != 1:
                raise ArithmeticError(f"{label} unexpectedly has a nonintegral entry")
            entries.append(int(value.p))
    return fmpz_mat(matrix.nrows(), matrix.ncols(), entries)


def _mod_one(value: Fraction) -> Fraction:
    return value % 1


def compute_fixed_locus_kernel(
    source: AffineTorusKernelSource,
) -> FixedLocusKernel:
    """Compute one exact private projection inside the bounded worker."""

    dimension = source.dimension
    if (
        len(source.linear_part) != dimension
        or any(len(row) != dimension for row in source.linear_part)
        or len(source.translation) != dimension
    ):
        raise AssertionError("affine-torus kernel source has inconsistent dimensions")
    linear = _integer_matrix(
        dimension,
        dimension,
        source.linear_part,
    )
    identity = fmpz_mat(
        dimension,
        dimension,
        [int(i == j) for i in range(dimension) for j in range(dimension)],
    )
    displacement = linear - identity
    translation = source.translation
    rank = _integer_rank(displacement)

    identity_kernel = _saturated_integer_kernel(
        displacement,
        rank=rank,
    )
    character_columns = _saturated_integer_kernel(
        displacement.transpose(),
        rank=rank,
    )
    characters = character_columns.basis.transpose()
    image_kernel = _saturated_integer_kernel(
        characters,
        rank=characters.nrows(),
    )
    image_saturation = image_kernel.basis

    pairings = tuple(
        sum(
            (
                int(characters[row, column]) * translation[column]
                for column in range(dimension)
            ),
            Fraction(0),
        )
        for row in range(characters.nrows())
    )
    for row, pairing in enumerate(pairings):
        residue = _mod_one(pairing)
        if residue:
            character = tuple(
                int(characters[row, column]) for column in range(dimension)
            )
            return EmptyFixedLocusKernel(character=character, pairing=residue)

    rows, columns = _first_rank_minor(displacement, rank)
    character_target = tuple(int(pairing) for pairing in pairings)
    integer_lift = _solve_integral_character_system(image_kernel, character_target)
    image_rhs = tuple(
        Fraction(integer_lift[index]) - translation[index] for index in range(dimension)
    )

    if rank == 0:
        component_lifts = _zero_rational_matrix(dimension, 0)
        image_coordinates = _zero_integer_matrix(0, dimension)
        relation_matrix = _zero_integer_matrix(0, 0)
        base_solution = _zero_rational_matrix(dimension, 1)
    else:
        minor = _submatrix(displacement, rows, columns)
        saturation_rows = _submatrix(image_saturation, rows, tuple(range(rank)))
        selected_lifts = _rational_solve(fmpq_mat(minor), fmpq_mat(saturation_rows))
        component_lifts = _zero_rational_matrix(dimension, rank)
        for row_index, ambient_row in enumerate(columns):
            for column in range(rank):
                component_lifts[ambient_row, column] = selected_lifts[row_index, column]
        saturation_minor = _submatrix(image_saturation, rows, tuple(range(rank)))
        displacement_rows = _submatrix(displacement, rows, tuple(range(dimension)))
        coordinates_rational = _rational_solve(
            fmpq_mat(saturation_minor), fmpq_mat(displacement_rows)
        )
        image_coordinates = _require_integral(
            coordinates_rational, label="image-coordinate matrix"
        )
        relation_rows = _integer_hnf(image_coordinates.transpose())
        relation_matrix = fmpz_mat(
            rank,
            rank,
            [
                relation_rows[column, row]
                for row in range(rank)
                for column in range(rank)
            ],
        )
        rhs_minor = fmpq_mat(
            rank,
            1,
            [
                fmpq(
                    image_rhs[ambient_row].numerator, image_rhs[ambient_row].denominator
                )
                for ambient_row in rows
            ],
        )
        selected_base = _rational_solve(fmpq_mat(minor), rhs_minor)
        base_solution = _zero_rational_matrix(dimension, 1)
        for row_index, ambient_row in enumerate(columns):
            base_solution[ambient_row, 0] = selected_base[row_index, 0]

    base_point = tuple(
        _mod_one(Fraction(int(base_solution[row, 0].p), int(base_solution[row, 0].q)))
        for row in range(dimension)
    )

    diagonal = _integer_snf(relation_matrix)
    diagonal_factors = tuple(abs(int(diagonal[index, index])) for index in range(rank))
    if any(value <= 0 for value in diagonal_factors):
        raise ArithmeticError("relation lattice is not full rank")
    invariant_factors = tuple(value for value in diagonal_factors if value > 1)
    component_count = prod(diagonal_factors)

    if rank == 0:
        generator_orders: tuple[int, ...] = ()
    else:
        inverse_relations = _rational_inverse(fmpq_mat(relation_matrix))
        generator_orders = tuple(
            lcm(*(int(inverse_relations[row, column].q) for row in range(rank)))
            for column in range(rank)
        )

    generators = tuple(
        tuple(
            _mod_one(
                Fraction(
                    int(component_lifts[row, column].p),
                    int(component_lifts[row, column].q),
                )
            )
            for row in range(dimension)
        )
        for column in range(rank)
    )
    return NonemptyFixedLocusKernel(
        base_point=base_point,
        identity_embedding=_integer_rows(identity_kernel.basis),
        component_generators=generators,
        relation_matrix=_integer_rows(relation_matrix),
        generator_orders=generator_orders,
        invariant_factors=invariant_factors,
        component_count=component_count,
    )


__all__ = [
    "compute_fixed_locus_kernel",
]
