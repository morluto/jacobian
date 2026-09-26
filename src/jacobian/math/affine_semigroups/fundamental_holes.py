"""Exact fundamental holes of bounded rank-two affine semigroups."""

from __future__ import annotations

from collections.abc import Iterator
from math import gcd
from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.affine_semigroups.group_lattice import compute_group_lattice
from jacobian.math.affine_semigroups.holes import (
    MAX_AFFINE_HOLE_CANDIDATES,
    MAX_AFFINE_HOLE_WORK,
    _preflight_hole_semigroup,
)
from jacobian.math.affine_semigroups.semigroup import (
    MAX_AFFINE_DIGITS,
    AffineConfiguration,
    PositiveAffineSemigroup,
    _hilbert_rays,
)
from jacobian.math.lattices.operations import hermite_normal_form

MAX_AFFINE_FUNDAMENTAL_HOLE_OUTPUT_CELLS = 2 * MAX_AFFINE_HOLE_CANDIDATES
MAX_AFFINE_FUNDAMENTAL_HOLE_OUTPUT_DIGITS = 3 * MAX_AFFINE_DIGITS + 2


class AffineSemigroupFundamentalHoles(StrictModel):
    """The finite set of holes minimal under addition by the source semigroup."""

    semigroup: PositiveAffineSemigroup
    holes: tuple[tuple[ExactInteger, ExactInteger], ...]

    @model_validator(mode="after")
    def require_canonical_holes(self) -> Self:
        if self.semigroup.configuration.rows != 2:
            raise PydanticCustomError(
                "affine_semigroup.fundamental_holes_rows",
                "fundamental-hole values retain a two-coordinate ambient row axis",
            )
        if len(self.holes) > MAX_AFFINE_HOLE_CANDIDATES:
            raise PydanticCustomError(
                "affine_semigroup.fundamental_holes_size",
                "fundamental-hole output exceeds the admitted candidate count",
            )
        if self.holes != tuple(sorted(set(self.holes))):
            raise PydanticCustomError(
                "affine_semigroup.fundamental_holes_order",
                "fundamental holes must be distinct and in lexicographic order",
            )
        return self


class AffineSemigroupFundamentalHolesRequest(StrictModel):
    """Request the complete finite fundamental-hole set of one semigroup."""

    semigroup: PositiveAffineSemigroup


def _extreme_source_generators(
    configuration: AffineConfiguration,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Choose actual semigroup generators on the two cone rays."""
    lower, upper = _hilbert_rays(configuration)
    columns = configuration.columns_vectors

    def on_ray(ray: tuple[int, int]) -> tuple[int, int]:
        matches = [
            (column[0], column[1])
            for column in columns
            if (
                ray[0] * column[1] - ray[1] * column[0] == 0
                and ray[0] * column[0] + ray[1] * column[1] > 0
            )
        ]
        if not matches:
            raise ArithmeticError("a cone ray has no source generator")
        return min(matches, key=lambda column: column[0] ** 2 + column[1] ** 2)

    first, second = on_ray(lower), on_ray(upper)
    determinant = first[0] * second[1] - first[1] * second[0]
    if determinant == 0:
        raise ValueError("fundamental holes require a full-dimensional cone")
    return (first, second) if determinant > 0 else (second, first)


def _det(left: tuple[int, int], right: tuple[int, int]) -> int:
    return left[0] * right[1] - left[1] * right[0]


def _iter_parallelogram_lattice_points(
    first: tuple[int, int],
    second: tuple[int, int],
) -> Iterator[tuple[int, int]]:
    """Enumerate half-open parallelogram representatives in O(|det|) points.

    Row HNF supplies a rectangular representative set for ``Z^2 / <u,v>``.
    Exact floor division translates each representative into the unique point
    with ray coordinates in ``[0,1)``.
    """
    determinant = first[0] * second[1] - first[1] * second[0]
    hnf, _ = hermite_normal_form([list(first), list(second)])
    if hnf.nrows() != 2 or hnf.ncols() != 2:
        raise ArithmeticError("extreme-ray HNF lost full rank")
    first_period = int(hnf[0, 0])
    second_period = int(hnf[1, 1])
    if (
        first_period <= 0
        or second_period <= 0
        or int(hnf[1, 0]) != 0
        or not 0 <= int(hnf[0, 1]) < second_period
        or first_period * second_period != determinant
        or any(
            abs(int(hnf[row, column])) > determinant
            for row in range(2)
            for column in range(2)
        )
    ):
        raise ArithmeticError("extreme-ray HNF has unexpected row form")
    for x in range(first_period):
        for y in range(second_period):
            first_numerator = x * second[1] - y * second[0]
            second_numerator = first[0] * y - first[1] * x
            first_floor = first_numerator // determinant
            second_floor = second_numerator // determinant
            yield (
                x - first_floor * first[0] - second_floor * second[0],
                y - first_floor * first[1] - second_floor * second[1],
            )


def fundamental_holes(
    semigroup: PositiveAffineSemigroup,
) -> AffineSemigroupFundamentalHoles:
    """Return all ``Q``-minimal holes in ``cone(Q) ∩ gp(Q)``.

    The admitted domain is a full-rank pointed cone in two ambient dimensions.
    If ``u`` and ``v`` are source generators on its extreme rays, every
    fundamental hole has unique coordinates ``h = λu + μv`` with
    ``0 ≤ λ, μ < 1``. Otherwise subtraction by the relevant ray generator
    leaves a smaller saturation hole. The half-open parallelogram is therefore
    a complete finite search region, not a degree cutoff.
    """
    source = _preflight_hole_semigroup(semigroup)
    source_generators = source.configuration.columns_vectors
    lattice_index = gcd(
        *(
            abs(
                source_generators[left][0] * source_generators[right][1]
                - source_generators[left][1] * source_generators[right][0]
            )
            for left in range(len(source_generators))
            for right in range(left + 1, len(source_generators))
        )
    )
    if lattice_index == 0:
        raise OperationDomainValidationError(
            location=("semigroup", "configuration"),
            code="affine_semigroup.fundamental_holes_rank",
            message="fundamental holes require a full-rank generated lattice",
        )
    group = compute_group_lattice(source.configuration)
    basis = group.lattice.basis.entries
    if len(basis) != 2 or any(len(row) != 2 for row in basis):
        raise OperationDomainValidationError(
            location=("semigroup", "configuration"),
            code="affine_semigroup.fundamental_holes_rank",
            message="fundamental holes require a full-rank generated lattice",
        )
    # For entries |a_ij|<M, every 2x2 minor is <2M^2, and the lattice index
    # is their gcd. In canonical 2D row HNF, every basis entry is at most that
    # index (or one in the index-one case). Cramer's rule then bounds each
    # source generator coordinate by 2M, independently of the HNF algorithm.
    basis_limit = max(1, lattice_index)
    if any(
        type(value) is not int or abs(value) > basis_limit
        for row in basis
        for value in row
    ):
        raise OperationResourceAdmissionError(
            location=("semigroup", "configuration"),
            code="affine_semigroup.fundamental_hole_basis_digits",
            message="generated-lattice basis exceeds its exact minor-index bound",
        )
    generator_coordinates = group.generator_lattice_coordinates.entries
    if len(generator_coordinates) != source.configuration.columns or any(
        len(row) != 2 for row in generator_coordinates
    ):
        raise ArithmeticError("generated-lattice coordinates lost source axes")
    if any(
        abs(value) >= 2 * 10**MAX_AFFINE_DIGITS
        for row in generator_coordinates
        for value in row
    ):
        raise OperationResourceAdmissionError(
            location=("semigroup", "configuration"),
            code="affine_semigroup.fundamental_hole_coordinates",
            message="generated-lattice coordinates exceed the Cramer bound",
        )
    coordinate_configuration = AffineConfiguration(
        row_labels=("lattice_0", "lattice_1"),
        generator_labels=source.configuration.generator_labels,
        entries=tuple(
            tuple(
                generator_coordinates[column][row]
                for column in range(len(generator_coordinates))
            )
            for row in range(2)
        ),
    )

    first, second = _extreme_source_generators(coordinate_configuration)
    # Source generator coordinates are <2M, so parallelogram vertex
    # coordinates are <4M and the two-ray determinant is <8M^2.
    coordinate_bound = max(
        abs(value)
        for value in (
            first[0],
            first[1],
            second[0],
            second[1],
            first[0] + second[0],
            first[1] + second[1],
        )
    )
    determinant_bound = abs(_det(first, second))
    if coordinate_bound >= 4 * 10**MAX_AFFINE_DIGITS or determinant_bound >= 8 * 10 ** (
        2 * MAX_AFFINE_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("semigroup", "configuration"),
            code="affine_semigroup.fundamental_hole_intermediate_digits",
            message="parallelogram coordinates exceed the admitted exact-height bound",
        )

    candidate_count = determinant_bound
    if candidate_count > MAX_AFFINE_HOLE_CANDIDATES:
        raise OperationResourceAdmissionError(
            location=("semigroup", "configuration"),
            code="affine_semigroup.fundamental_hole_candidates",
            message=(
                f"fundamental-parallelogram lattice point count {candidate_count} exceeds "
                f"{MAX_AFFINE_HOLE_CANDIDATES} candidates"
            ),
        )
    if 2 * candidate_count > MAX_AFFINE_FUNDAMENTAL_HOLE_OUTPUT_CELLS:
        raise OperationResourceAdmissionError(
            location=("semigroup", "configuration"),
            code="affine_semigroup.fundamental_hole_output_cells",
            message="fundamental-hole tuple cells exceed their admitted count",
        )

    # HNF gives periods at most D, hence representatives have coordinates <D.
    # Cramer's numerators are <2*D*(4M), while the floor coefficients are
    # <8M. Translation products and their sums are <64*M^2. Guard the largest
    # temporary (the Cramer numerators) before asking the HNF backend to expand.
    source_scalar_limit = 10**MAX_AFFINE_DIGITS
    numerator_bound = 64 * source_scalar_limit**3
    translated_coordinate_bound = 64 * source_scalar_limit**2
    intermediate_limit = 10 ** (3 * MAX_AFFINE_DIGITS + 2)
    if max(numerator_bound, translated_coordinate_bound) >= intermediate_limit:
        raise OperationResourceAdmissionError(
            location=("semigroup", "configuration"),
            code="affine_semigroup.fundamental_hole_intermediate_digits",
            message="HNF representative translation exceeds its exact-height bound",
        )

    generators = tuple(dict.fromkeys(coordinate_configuration.columns_vectors))
    work_bound = candidate_count * (3 * len(generators) + 4)
    if work_bound > MAX_AFFINE_HOLE_WORK:
        raise OperationResourceAdmissionError(
            location=("semigroup", "configuration"),
            code="affine_semigroup.fundamental_hole_work",
            message=(
                f"fundamental-hole work {work_bound} exceeds "
                f"{MAX_AFFINE_HOLE_WORK} units"
            ),
        )

    coordinate_bounds = tuple(abs(first[axis]) + abs(second[axis]) for axis in range(2))
    ambient_bounds = tuple(
        sum(coordinate_bounds[axis] * abs(basis[axis][row]) for axis in range(2))
        for row in range(2)
    )
    if any(
        value >= 10**MAX_AFFINE_FUNDAMENTAL_HOLE_OUTPUT_DIGITS
        for value in ambient_bounds
    ):
        raise OperationResourceAdmissionError(
            location=("semigroup", "configuration"),
            code="affine_semigroup.fundamental_hole_output_digits",
            message="fundamental-hole coordinates exceed the 26-digit scalar envelope",
        )

    candidates = tuple(_iter_parallelogram_lattice_points(first, second))
    candidate_set = set(candidates)
    reachable = {(0, 0)}
    pending = [(0, 0)]
    while pending:
        point = pending.pop()
        for generator in generators:
            neighbor = (point[0] + generator[0], point[1] + generator[1])
            if neighbor in candidate_set and neighbor not in reachable:
                reachable.add(neighbor)
                pending.append(neighbor)

    holes = {point for point in candidates if point not in reachable}
    fundamental = tuple(
        sorted(
            (
                basis[0][0] * point[0] + basis[1][0] * point[1],
                basis[0][1] * point[0] + basis[1][1] * point[1],
            )
            for point in holes
            if all(
                (point[0] - generator[0], point[1] - generator[1]) not in holes
                for generator in generators
            )
        )
    )
    return AffineSemigroupFundamentalHoles(semigroup=source, holes=fundamental)


__all__ = [
    "AffineSemigroupFundamentalHoles",
    "AffineSemigroupFundamentalHolesRequest",
    "fundamental_holes",
]
