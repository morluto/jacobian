"""Exact HNF-based kernel for affine fixed loci on standard real tori."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import lcm, prod

from flint import fmpq, fmpq_mat, fmpz_mat

from jacobian.canonical import parse_canonical_integer
from jacobian.math.geometry.affine_tori._bounds import (
    AffineTorusFixedLocusPlan,
    AffineTorusRankBounds,
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
    augmented_hnf = augmented.hnf()
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
    if transform * source != hnf or abs(int(transform.det())) != 1:
        raise ArithmeticError("augmented HNF did not recover a unimodular transform")
    return hnf, transform


def _saturated_integer_kernel(
    source: fmpz_mat,
    plan: AffineTorusFixedLocusPlan,
    *,
    label: str,
    rank: int,
) -> _SaturatedKernel:
    """Return the canonical saturated column basis of ``ker_Z(source)``.

    In the canonical row HNF ``[H | T] = HNF([source^t | I])``, the final
    ``q-rank(source)`` rows have zero left block. Their right block is already
    the canonical saturated row basis of the left kernel of ``source^t``;
    transposing yields the desired integer kernel.
    """

    _backend_checkpoint(plan, f"before {label} augmented HNF")
    transpose_hnf, transform = _augmented_hnf_transform(source.transpose())
    _backend_checkpoint(plan, f"after {label} augmented HNF")
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
    bounds: AffineTorusRankBounds,
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
    leading_solution_height = max(map(abs, right_hand_side), default=0)
    _require_height(
        leading_solution_height,
        bounds.leading_solution_height,
        label="integral-character leading solution",
    )
    transformed = fmpz_mat(
        ambient_dimension,
        1,
        [
            right_hand_side[index] if index < equation_count else 0
            for index in range(ambient_dimension)
        ],
    )
    _backend_checkpoint(plan, "before the integral character lift")
    solution = kernel.transpose_transform.transpose() * transformed
    _backend_checkpoint(plan, "after the integral character lift")
    _require_height(
        _integer_matrix_height(solution),
        bounds.integral_lift_height,
        label="integral-character lift",
    )
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


def _integer_matrix_height(matrix: fmpz_mat) -> int:
    return max(
        (
            abs(int(matrix[row, column]))
            for row in range(matrix.nrows())
            for column in range(matrix.ncols())
        ),
        default=0,
    )


def _rational_matrix_height(matrix: fmpq_mat) -> int:
    return max(
        (
            max(abs(int(matrix[row, column].p)), int(matrix[row, column].q))
            for row in range(matrix.nrows())
            for column in range(matrix.ncols())
        ),
        default=0,
    )


def _fraction_height(values: tuple[Fraction, ...]) -> int:
    return max(
        (max(abs(value.numerator), value.denominator) for value in values),
        default=0,
    )


def _require_height(actual: int, bound: int, *, label: str) -> None:
    """Fail closed if a derived theorem/adapter invariant is ever violated."""

    if actual > bound:
        raise ArithmeticError(f"{label} exceeded its source-derived height proof")


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
    _backend_checkpoint(plan, "before the source rank")
    rank = displacement.rank()
    _backend_checkpoint(plan, "after the source rank")
    bounds = plan.bounds_for_rank(rank)

    identity_kernel = _saturated_integer_kernel(
        displacement,
        plan,
        label="fixed identity-component",
        rank=rank,
    )
    character_columns = _saturated_integer_kernel(
        displacement.transpose(),
        plan,
        label="invariant-character",
        rank=rank,
    )
    for label, kernel in (
        ("fixed identity-component", identity_kernel),
        ("invariant-character", character_columns),
    ):
        _require_height(
            _integer_matrix_height(kernel.transpose_transform),
            bounds.source_hnf_transform_height,
            label=f"{label} augmented-HNF transform",
        )
        _require_height(
            _integer_matrix_height(kernel.basis),
            bounds.source_minor_height,
            label=f"{label} saturated-kernel basis",
        )
    characters = character_columns.basis.transpose()
    image_kernel = _saturated_integer_kernel(
        characters,
        plan,
        label="saturated-image",
        rank=characters.nrows(),
    )
    image_saturation = image_kernel.basis
    _require_height(
        _integer_matrix_height(image_kernel.transpose_transform),
        bounds.character_hnf_transform_height,
        label="saturated-image augmented-HNF transform",
    )
    _require_height(
        _integer_matrix_height(image_saturation),
        bounds.image_saturation_height,
        label="saturated-image basis",
    )

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
            _require_height(
                _fraction_height((residue,)),
                bounds.base_point_component_height,
                label="empty-locus obstruction pairing",
            )
            _backend_checkpoint(plan, "after empty-locus result normalization")
            return EmptyFixedLocusKernel(character=character, pairing=residue)

    rows, columns = _first_rank_minor(displacement, rank, plan)
    character_target = tuple(int(pairing) for pairing in pairings)
    integer_lift = _solve_integral_character_system(
        image_kernel, character_target, plan, bounds
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
        _require_height(
            _rational_matrix_height(selected_lifts),
            bounds.rational_intermediate_height,
            label="component-lift rational solve",
        )
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
        _require_height(
            _rational_matrix_height(coordinates_rational),
            bounds.rational_intermediate_height,
            label="image-coordinate rational solve",
        )
        image_coordinates = _require_integral(
            coordinates_rational, label="image-coordinate matrix"
        )
        _require_height(
            _integer_matrix_height(image_coordinates),
            bounds.image_coordinate_height,
            label="image-coordinate matrix",
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
        _require_height(
            _integer_matrix_height(relation_matrix),
            bounds.source_minor_height,
            label="component relation HNF",
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
        _require_height(
            _rational_matrix_height(selected_base),
            bounds.rational_intermediate_height,
            label="base-point rational solve",
        )
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
    _require_height(
        _integer_matrix_height(diagonal),
        bounds.source_minor_height,
        label="component Smith factors",
    )
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
        _require_height(
            _rational_matrix_height(inverse_relations),
            bounds.rational_intermediate_height,
            label="component-generator order solve",
        )
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
    _require_height(
        _fraction_height(
            tuple(value for generator in generators for value in generator)
        ),
        bounds.source_minor_height,
        label="component generators",
    )
    _require_height(
        _fraction_height(base_point),
        bounds.base_point_component_height,
        label="base point",
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
