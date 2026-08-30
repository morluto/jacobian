"""Exact HNF-based kernel for affine fixed loci on standard real tori."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import lcm, prod

from flint import fmpq, fmpq_mat, fmpz_mat

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.geometry.affine_tori._bounds import (
    MAX_AFFINE_TORUS_BASE_POINT_DIGITS,
    MAX_AFFINE_TORUS_MINOR_DIGITS,
    AffineTorusFixedLocusPlan,
    require_affine_torus_deadline,
)
from jacobian.math.geometry.affine_tori.values import RationalAffineTorusMap


@dataclass(frozen=True, slots=True)
class EmptyFixedLocusKernel:
    character: tuple[int, ...]
    pairing: Fraction


@dataclass(frozen=True, slots=True)
class NonemptyFixedLocusKernel:
    base_point: tuple[Fraction, ...]
    identity_embedding: tuple[tuple[int, ...], ...]
    component_generators: tuple[tuple[Fraction, ...], ...]
    relation_matrix: tuple[tuple[int, ...], ...]
    generator_orders: tuple[int, ...]
    invariant_factors: tuple[int, ...]
    component_count: int


type FixedLocusKernel = EmptyFixedLocusKernel | NonemptyFixedLocusKernel


@dataclass(frozen=True, slots=True)
class _SaturatedKernel:
    basis: fmpz_mat
    transpose_hnf: fmpz_mat
    transpose_transform: fmpz_mat


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


def _backend_checkpoint(plan: AffineTorusFixedLocusPlan, stage: str) -> None:
    require_affine_torus_deadline(plan.deadline, stage)


def _saturated_integer_kernel(
    source: fmpz_mat,
    plan: AffineTorusFixedLocusPlan,
    *,
    label: str,
) -> _SaturatedKernel:
    """Return the canonical saturated column basis of ``ker_Z(source)``.

    If ``H = T source^t`` is row HNF, the final ``q-rank(source)`` rows of
    ``T`` are a saturated row basis of the left kernel of ``source^t``.
    Transposing yields the desired integer kernel; a final row HNF of its
    transpose canonicalizes the column lattice without changing it.
    """

    _backend_checkpoint(plan, f"before {label} HNF with transformation")
    transpose_hnf, transform = source.transpose().hnf(True)
    _backend_checkpoint(plan, f"after {label} HNF with transformation")
    rank = source.rank()
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
    _backend_checkpoint(plan, f"before {label} canonical HNF")
    canonical_rows = raw_rows.hnf()
    _backend_checkpoint(plan, f"after {label} canonical HNF")
    basis = canonical_rows.transpose()
    if source * basis != _zero_integer_matrix(source.nrows(), nullity):
        raise ArithmeticError(f"{label} HNF basis does not lie in the integer kernel")
    if basis.rank() != nullity:
        raise ArithmeticError(f"{label} HNF basis has the wrong rank")
    return _SaturatedKernel(
        basis=basis,
        transpose_hnf=transpose_hnf,
        transpose_transform=transform,
    )


def _first_rank_minor(
    matrix: fmpz_mat, rank: int, plan: AffineTorusFixedLocusPlan
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Select the lexicographic greedy full-rank row/column minor."""

    if rank == 0:
        return (), ()
    all_columns = tuple(range(matrix.ncols()))
    rows: list[int] = []
    current_rank = 0
    for row in range(matrix.nrows()):
        candidate = (*rows, row)
        _backend_checkpoint(plan, "during first rank-increasing row selection")
        candidate_rank = _submatrix(matrix, candidate, all_columns).rank()
        if candidate_rank > current_rank:
            rows.append(row)
            current_rank = candidate_rank
            if current_rank == rank:
                break
    columns: list[int] = []
    current_rank = 0
    for column in range(matrix.ncols()):
        candidate = (*columns, column)
        _backend_checkpoint(plan, "during first rank-increasing column selection")
        candidate_rank = _submatrix(matrix, tuple(rows), candidate).rank()
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
    plan: AffineTorusFixedLocusPlan,
) -> tuple[int, ...]:
    """Solve ``W z = h`` using the already-computed HNF of ``W^t``."""

    ambient_dimension = kernel.transpose_hnf.nrows()
    equation_count = kernel.transpose_hnf.ncols()
    if len(right_hand_side) != equation_count:
        raise AssertionError("character right-hand side has the wrong dimension")
    if equation_count == 0:
        return (0,) * ambient_dimension
    leading = fmpq_mat(
        equation_count,
        equation_count,
        [
            kernel.transpose_hnf[column, row]
            for row in range(equation_count)
            for column in range(equation_count)
        ],
    )
    rhs = fmpq_mat(
        equation_count,
        1,
        [fmpq(value) for value in right_hand_side],
    )
    _backend_checkpoint(plan, "before the integral character solve")
    leading_solution = leading.solve(rhs)
    _backend_checkpoint(plan, "after the integral character solve")
    if any(leading_solution[index, 0].q != 1 for index in range(equation_count)):
        raise ArithmeticError(
            "saturated character lattice did not give an integer lift"
        )
    transformed = fmpz_mat(
        ambient_dimension,
        1,
        [
            int(leading_solution[index, 0]) if index < equation_count else 0
            for index in range(ambient_dimension)
        ],
    )
    solution = kernel.transpose_transform.transpose() * transformed
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


def _digits(value: int) -> int:
    return len(format_canonical_integer(abs(value)))


def _require_predicted_heights(
    *,
    integer_matrices: tuple[fmpz_mat, ...],
    generators: tuple[tuple[Fraction, ...], ...],
    base_point: tuple[Fraction, ...],
) -> None:
    if any(
        _digits(int(matrix[row, column])) > MAX_AFFINE_TORUS_MINOR_DIGITS
        for matrix in integer_matrices
        for row in range(matrix.nrows())
        for column in range(matrix.ncols())
    ):
        raise ArithmeticError("affine-torus minor-height proof was violated")
    if any(
        max(_digits(value.numerator), _digits(value.denominator))
        > MAX_AFFINE_TORUS_MINOR_DIGITS
        for generator in generators
        for value in generator
    ):
        raise ArithmeticError("affine-torus generator-height proof was violated")
    if any(
        max(_digits(value.numerator), _digits(value.denominator))
        > MAX_AFFINE_TORUS_BASE_POINT_DIGITS
        for value in base_point
    ):
        raise ArithmeticError("affine-torus base-point height proof was violated")


def compute_fixed_locus_kernel(
    source: RationalAffineTorusMap,
    plan: AffineTorusFixedLocusPlan,
) -> FixedLocusKernel:
    """Compute the exact fixed-locus postcondition for one admitted source map."""

    dimension = source.torus.dimension
    if plan.dimension != dimension:
        raise AssertionError("affine-torus plan does not belong to its source")
    linear = _integer_matrix(
        dimension,
        dimension,
        tuple(
            tuple(parse_canonical_integer(value) for value in row)
            for row in source.linear_part.entries
        ),
    )
    identity = fmpz_mat(
        dimension,
        dimension,
        [int(i == j) for i in range(dimension) for j in range(dimension)],
    )
    displacement = linear - identity
    translation = tuple(
        coordinate.as_fraction() for coordinate in source.translation.coordinates
    )

    identity_kernel = _saturated_integer_kernel(
        displacement, plan, label="fixed identity-component"
    )
    character_columns = _saturated_integer_kernel(
        displacement.transpose(), plan, label="invariant-character"
    )
    characters = character_columns.basis.transpose()
    image_kernel = _saturated_integer_kernel(characters, plan, label="saturated-image")
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
            _backend_checkpoint(plan, "after empty-locus result normalization")
            return EmptyFixedLocusKernel(character=character, pairing=residue)

    rank = displacement.rank()
    rows, columns = _first_rank_minor(displacement, rank, plan)
    character_target = tuple(int(pairing) for pairing in pairings)
    integer_lift = _solve_integral_character_system(
        image_kernel, character_target, plan
    )
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
        _backend_checkpoint(plan, "before the component-lift rational solve")
        selected_lifts = fmpq_mat(minor).solve(fmpq_mat(saturation_rows))
        _backend_checkpoint(plan, "after the component-lift rational solve")
        component_lifts = _zero_rational_matrix(dimension, rank)
        for row_index, ambient_row in enumerate(columns):
            for column in range(rank):
                component_lifts[ambient_row, column] = selected_lifts[row_index, column]
        if fmpq_mat(displacement) * component_lifts != fmpq_mat(image_saturation):
            raise ArithmeticError("component lifts do not reconstruct image saturation")

        saturation_minor = _submatrix(image_saturation, rows, tuple(range(rank)))
        displacement_rows = _submatrix(displacement, rows, tuple(range(dimension)))
        _backend_checkpoint(plan, "before the image-coordinate rational solve")
        coordinates_rational = fmpq_mat(saturation_minor).solve(
            fmpq_mat(displacement_rows)
        )
        _backend_checkpoint(plan, "after the image-coordinate rational solve")
        image_coordinates = _require_integral(
            coordinates_rational, label="image-coordinate matrix"
        )
        if image_saturation * image_coordinates != displacement:
            raise ArithmeticError("image coordinates do not reconstruct A-I")

        _backend_checkpoint(plan, "before relation-lattice canonical HNF")
        relation_rows = image_coordinates.transpose().hnf()
        _backend_checkpoint(plan, "after relation-lattice canonical HNF")
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
        _backend_checkpoint(plan, "before the base-point rational solve")
        selected_base = fmpq_mat(minor).solve(rhs_minor)
        _backend_checkpoint(plan, "after the base-point rational solve")
        base_solution = _zero_rational_matrix(dimension, 1)
        for row_index, ambient_row in enumerate(columns):
            base_solution[ambient_row, 0] = selected_base[row_index, 0]

    base_point = tuple(
        _mod_one(Fraction(int(base_solution[row, 0].p), int(base_solution[row, 0].q)))
        for row in range(dimension)
    )
    if any(
        sum(
            (
                int(displacement[row, column]) * base_point[column]
                for column in range(dimension)
            ),
            translation[row],
        ).denominator
        != 1
        for row in range(dimension)
    ):
        raise ArithmeticError("base point does not satisfy the fixed-point congruence")

    _backend_checkpoint(plan, "before component Smith invariant factors")
    diagonal = relation_matrix.snf()
    _backend_checkpoint(plan, "after component Smith invariant factors")
    diagonal_factors = tuple(abs(int(diagonal[index, index])) for index in range(rank))
    if any(value <= 0 for value in diagonal_factors):
        raise ArithmeticError("relation lattice is not full rank")
    invariant_factors = tuple(value for value in diagonal_factors if value > 1)
    component_count = abs(int(relation_matrix.det()))
    if component_count != prod(diagonal_factors):
        raise ArithmeticError("Smith invariants do not recover the component count")

    if rank == 0:
        generator_orders: tuple[int, ...] = ()
    else:
        _backend_checkpoint(plan, "before component-generator order solve")
        inverse_relations = fmpq_mat(relation_matrix).inv()
        _backend_checkpoint(plan, "after component-generator order solve")
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
    _require_predicted_heights(
        integer_matrices=(
            identity_kernel.basis,
            image_saturation,
            image_coordinates,
            relation_matrix,
        ),
        generators=generators,
        base_point=base_point,
    )
    _backend_checkpoint(plan, "after exact result normalization")
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
    "EmptyFixedLocusKernel",
    "FixedLocusKernel",
    "NonemptyFixedLocusKernel",
    "compute_fixed_locus_kernel",
]
