"""Private exact kernels for rational fan recognition and orbit profiles.

All arithmetic is exact: integer ray coordinates, ``Fraction`` linear
programming through a compact Fourier-Motzkin elimination, and python-flint
Smith/rank kernels. No floating-point value participates in any decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import gcd

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.matrices._flint import (
    integer_smith_normal_form,
    rational_rref,
)

MAX_TORIC_FM_TABLEAU_ROWS = 4_096

# One exact row ``coefficients . x (rel) rhs`` with rel in {eq, ge, gt}.
_LpRow = tuple[tuple[Fraction, ...], str, Fraction]


def _reject_budget(message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("fan",),
        code="toric.resource_budget_exceeded",
        message=message,
    )


def _fm_feasible(rows: list[_LpRow], variables: int) -> bool:
    """Decide exact rational feasibility by Fourier-Motzkin elimination.

    Equalities are substituted first, which keeps the inequality tableau
    small; remaining variables are eliminated by pairing positive and
    negative inequality coefficients. The tableau is deduplicated after every
    step and bounded by ``MAX_TORIC_FM_TABLEAU_ROWS``.
    """

    for index in range(variables):
        rows = _substitute_equalities(rows, index)
        positive: list[_LpRow] = []
        negative: list[_LpRow] = []
        neutral: list[_LpRow] = []
        for row in rows:
            coefficient = row[0][index]
            if coefficient > 0:
                positive.append(row)
            elif coefficient < 0:
                negative.append(row)
            else:
                neutral.append(row)
        combined: list[_LpRow] = list(neutral)
        for upper in positive:
            scale_up = upper[0][index]
            for lower in negative:
                scale_down = -lower[0][index]
                coefficients = tuple(
                    scale_down * a + scale_up * b
                    for a, b in zip(upper[0], lower[0], strict=True)
                )
                kind = (
                    "gt"
                    if "gt" in (upper[1], lower[1])
                    else "ge"
                )
                combined.append(
                    (
                        coefficients,
                        kind,
                        scale_down * upper[2] + scale_up * lower[2],
                    )
                )
        rows = _deduplicate_rows(combined)
        if len(rows) > MAX_TORIC_FM_TABLEAU_ROWS:
            _reject_budget(
                "exact feasibility tableau exceeds "
                f"{MAX_TORIC_FM_TABLEAU_ROWS} rows"
            )
    for coefficients, kind, rhs in rows:
        if all(value == 0 for value in coefficients):
            if kind == "eq":
                if rhs != 0:
                    return False
            elif kind == "gt":
                if rhs >= 0:
                    return False
            elif rhs > 0:
                return False
    return True


def _substitute_equalities(rows: list[_LpRow], index: int) -> list[_LpRow]:
    """Eliminate variable ``index`` through every equality that mentions it."""

    while True:
        pivot: _LpRow | None = None
        for row in rows:
            if row[1] == "eq" and row[0][index] != 0:
                pivot = row
                break
        if pivot is None:
            return rows
        scale = Fraction(1, 1) / pivot[0][index]
        solved = tuple(scale * value for value in pivot[0])
        solved_rhs = scale * pivot[2]
        remaining: list[_LpRow] = []
        for row in rows:
            if row is pivot:
                continue
            coefficient = row[0][index]
            if coefficient == 0:
                remaining.append(row)
                continue
            remaining.append(
                (
                    tuple(a - coefficient * b for a, b in zip(row[0], solved, strict=True)),
                    row[1],
                    row[2] - coefficient * solved_rhs,
                )
            )
        rows = remaining


def _deduplicate_rows(rows: list[_LpRow]) -> list[_LpRow]:
    """Scale each row to a canonical form and drop duplicates."""

    unique: dict[_LpRow, None] = {}
    for coefficients, kind, rhs in rows:
        scale: Fraction | None = None
        for value in coefficients:
            if value != 0:
                # Scaling by a negative factor would flip an inequality, so
                # canonicalize only by the positive magnitude.
                scale = Fraction(1, 1) / abs(value)
                break
        if scale is None:
            normalized: _LpRow = (
                tuple(Fraction(0) for _ in coefficients),
                kind,
                rhs,
            )
        else:
            normalized = (
                tuple(scale * value for value in coefficients),
                kind,
                scale * rhs,
            )
        unique.setdefault(normalized, None)
    return list(unique)


def _feasible(
    equalities: list[tuple[tuple[int, ...], int]],
    strict_rows: list[tuple[int, ...]],
    nonnegative_variables: int,
    variables: int,
) -> bool:
    """Solve one bounded exact feasibility problem over ``Fraction``."""

    rows: list[_LpRow] = [
        (tuple(Fraction(value) for value in coefficients), "eq", Fraction(rhs))
        for coefficients, rhs in equalities
    ]
    rows.extend(
        (tuple(Fraction(value) for value in coefficients), "gt", Fraction(0))
        for coefficients in strict_rows
    )
    for variable in range(nonnegative_variables):
        unit = [Fraction(0)] * variables
        unit[variable] = Fraction(1)
        rows.append((tuple(unit), "ge", Fraction(0)))
    return _fm_feasible(rows, variables)


@dataclass(frozen=True, slots=True)
class RecognizedCone:
    """One validated cone with its exact dimensional and smoothness facts."""

    cone_id: int
    ray_indices: tuple[int, ...]
    dimension: int
    is_simplicial: bool
    is_smooth: bool


@dataclass(frozen=True, slots=True)
class FanObstruction:
    """The first exact recognition failure of a proposed fan."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class RecognizedFan:
    """Every validated fact reused by the published toric operations."""

    rays: tuple[tuple[int, ...], ...]
    cones: tuple[RecognizedCone, ...]
    face_relations: tuple[tuple[int, int], ...]
    obstruction: FanObstruction | None


def _matrix_rank(rows: tuple[tuple[int, ...], ...]) -> int:
    if not rows:
        return 0
    _, rank = rational_rref(
        tuple(tuple(Fraction(value) for value in row) for row in rows)
    )
    return rank


def _cone_dimension(ray_indices: tuple[int, ...], rays: tuple[tuple[int, ...], ...]) -> int:
    return _matrix_rank(tuple(rays[index] for index in ray_indices))


def _is_smooth(ray_indices: tuple[int, ...], rays: tuple[tuple[int, ...], ...], dimension: int) -> bool:
    if len(ray_indices) != dimension:
        return False
    diagonal = integer_smith_normal_form(tuple(rays[index] for index in ray_indices))
    return all(diagonal[row][row] == 1 for row in range(dimension))


def _ray_in_cone(ray: tuple[int, ...], generators: tuple[tuple[int, ...], ...]) -> bool:
    """Decide exact nonnegative-combination membership of one ray."""

    if not generators:
        return all(value == 0 for value in ray)
    variables = len(generators)
    equalities = [
        (tuple(generator[coordinate] for generator in generators), ray[coordinate])
        for coordinate in range(len(ray))
    ]
    return _feasible(equalities, [], variables, variables)


def _subset_is_face(
    subset: frozenset[int],
    cone_rays: tuple[int, ...],
    rays: tuple[tuple[int, ...], ...],
    rank: int,
) -> bool:
    """Decide whether ``cone(subset)`` is the face exposed by some ``m``."""

    strict_rows = [rays[index] for index in cone_rays if index not in subset]
    equalities = [(rays[index], 0) for index in sorted(subset)]
    return _feasible(equalities, strict_rows, 0, rank)


def recognize_fan(
    rays: tuple[tuple[int, ...], ...],
    cones: tuple[tuple[int, ...], ...],
    lattice_rank: int,
) -> RecognizedFan:
    """Run the complete exact fan recognition, retaining the first obstruction.

    Checks, in order: ray nonzeroness and primitivity, ray uniqueness, cone
    strong convexity, extreme-ray generation, complete face closure, and
    pairwise common-face intersections.
    """

    def obstruct(code: str, message: str) -> RecognizedFan:
        return RecognizedFan(
            rays=rays,
            cones=(),
            face_relations=(),
            obstruction=FanObstruction(code=code, message=message),
        )

    for ray_index, ray in enumerate(rays):
        if all(value == 0 for value in ray):
            return obstruct(
                "toric.ray_zero", f"ray {ray_index} is the zero vector"
            )
        divisor = 0
        for value in ray:
            divisor = gcd(divisor, abs(value))
        if divisor != 1:
            return obstruct(
                "toric.ray_not_primitive",
                f"ray {ray_index} has lattice length {divisor}, not 1",
            )
    if len(set(rays)) != len(rays):
        return obstruct("toric.duplicate_rays", "declared rays are not distinct")

    cone_generators = tuple(tuple(sorted(set(cone))) for cone in cones)
    declared = {cone: cone_id for cone_id, cone in enumerate(cone_generators)}

    for cone_id, cone in enumerate(cone_generators):
        generators = tuple(rays[index] for index in cone)
        if not _feasible(
            [], list(generators), 0, lattice_rank
        ):
            return obstruct(
                "toric.cone_not_strongly_convex",
                f"cone {cone_id} admits no character positive on every generator",
            )
        for index in cone:
            others = tuple(rays[other] for other in cone if other != index)
            if _ray_in_cone(rays[index], others):
                return obstruct(
                    "toric.generator_not_extreme",
                    f"generator {index} of cone {cone_id} is not an extreme ray",
                )

    for ray_index in range(len(rays)):
        if (ray_index,) not in declared:
            return obstruct(
                "toric.ray_not_a_cone",
                f"ray {ray_index} does not span a declared one-dimensional cone",
            )

    for cone_id, cone in enumerate(cone_generators):
        for size in range(len(cone) + 1):
            for subset in combinations(cone, size):
                if not _subset_is_face(frozenset(subset), cone, rays, lattice_rank):
                    continue
                face = tuple(sorted(subset))
                if face not in declared:
                    return obstruct(
                        "toric.missing_face_cone",
                        f"cone {cone_id} has an undeclared face on rays {face}",
                    )

    recognized: list[RecognizedCone] = []
    for cone_id, cone in enumerate(cone_generators):
        dimension = _cone_dimension(cone, rays)
        recognized.append(
            RecognizedCone(
                cone_id=cone_id,
                ray_indices=cone,
                dimension=dimension,
                is_simplicial=len(cone) == dimension,
                is_smooth=_is_smooth(cone, rays, dimension),
            )
        )

    for first in range(len(cone_generators)):
        for second in range(first + 1, len(cone_generators)):
            left = cone_generators[first]
            right = cone_generators[second]
            inward = tuple(
                index
                for index in left
                if _ray_in_cone(rays[index], tuple(rays[j] for j in right))
            )
            outward = tuple(
                index
                for index in right
                if _ray_in_cone(rays[index], tuple(rays[j] for j in left))
            )
            common_face = inward == outward and _subset_is_face(
                frozenset(inward), left, rays, lattice_rank
            )
            if not common_face:
                return obstruct(
                    "toric.intersection_not_common_face",
                    f"cones {first} and {second} do not intersect in a common face",
                )

    face_relations: list[tuple[int, int]] = []
    for tau_id in range(len(cone_generators)):
        tau = frozenset(cone_generators[tau_id])
        for sigma_id in range(tau_id, len(cone_generators)):
            if tau <= frozenset(cone_generators[sigma_id]):
                face_relations.append((tau_id, sigma_id))

    return RecognizedFan(
        rays=rays,
        cones=tuple(recognized),
        face_relations=tuple(face_relations),
        obstruction=None,
    )


__all__ = [
    "MAX_TORIC_FM_TABLEAU_ROWS",
    "FanObstruction",
    "RecognizedCone",
    "RecognizedFan",
    "recognize_fan",
]
