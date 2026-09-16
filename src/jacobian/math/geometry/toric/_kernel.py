"""Private exact kernels for rational fan recognition and orbit profiles.

All arithmetic is exact: integer ray coordinates, ``Fraction`` linear
programming through a compact Fourier-Motzkin elimination, and python-flint
Smith/rank kernels. No floating-point value participates in any decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from math import gcd
from time import monotonic
from typing import NoReturn

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.geometry.toric._models import (
    MAX_TORIC_CHART_BOX,
    MAX_TORIC_CHART_CANDIDATES,
    MAX_TORIC_CHART_COMPONENT_DIGITS,
    MAX_TORIC_CHART_DUAL_RAYS,
    MAX_TORIC_CHART_GENERATORS,
    MAX_TORIC_CHART_RELATIONS,
)
from jacobian.math.matrices._flint import (
    integer_smith_normal_form,
    rational_rref,
)

MAX_TORIC_FM_TABLEAU_ROWS = 4_096

# One exact row ``coefficients . x (rel) rhs`` with rel in {eq, ge, gt}.
_LpRow = tuple[tuple[Fraction, ...], str, Fraction]


def _reject_budget(message: str) -> NoReturn:
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
                kind = "gt" if "gt" in (upper[1], lower[1]) else "ge"
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
                f"exact feasibility tableau exceeds {MAX_TORIC_FM_TABLEAU_ROWS} rows"
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
                    tuple(
                        a - coefficient * b for a, b in zip(row[0], solved, strict=True)
                    ),
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


def _cone_dimension(
    ray_indices: tuple[int, ...], rays: tuple[tuple[int, ...], ...]
) -> int:
    return _matrix_rank(tuple(rays[index] for index in ray_indices))


def _is_smooth(
    ray_indices: tuple[int, ...], rays: tuple[tuple[int, ...], ...], dimension: int
) -> bool:
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


def recognize_fan(  # noqa: C901
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
            return obstruct("toric.ray_zero", f"ray {ray_index} is the zero vector")
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
        if not _feasible([], list(generators), 0, lattice_rank):
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


def _dot(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return sum(first * second for first, second in zip(left, right, strict=True))


def _primitive(values: tuple[int, ...]) -> tuple[int, ...] | None:
    divisor = 0
    for value in values:
        divisor = gcd(divisor, abs(value))
    if divisor == 0:
        return None
    return tuple(value // divisor for value in values)


def _one_dimensional_kernel(
    rows: list[tuple[int, ...]], columns: int
) -> tuple[int, ...] | None:
    """Return the primitive generator of a one-dimensional integer kernel."""

    if not rows:
        return (1,) if columns == 1 else None
    from flint import fmpz_mat

    basis, nullity = fmpz_mat([list(row) for row in rows]).nullspace()
    if int(nullity) != 1 or columns < 1:
        return None
    return tuple(int(basis[index, 0]) for index in range(columns))


def _dual_cone_extreme_rays(
    rays: tuple[tuple[int, ...], ...], dimension: int
) -> tuple[tuple[int, ...], ...]:
    """Extreme rays of ``{m : <m, v> >= 0 for every generator v}``.

    Every one-dimensional face of the dual cone is the exact nullspace of a
    rank ``dimension - 1`` facet subset; the double-description enumeration
    orients each candidate primitive generator into the dual cone. This is the
    shipped exact double description, reused unchanged for every chart.
    """

    if not rays:
        return ()
    collected: dict[tuple[int, ...], None] = {}
    for subset in combinations(range(len(rays)), dimension - 1):
        candidate = _one_dimensional_kernel(
            [rays[index] for index in subset], dimension
        )
        if candidate is None:
            continue
        oriented: tuple[int, ...] | None = None
        if all(_dot(ray, candidate) >= 0 for ray in rays):
            oriented = candidate
        else:
            negated = tuple(-value for value in candidate)
            if all(_dot(ray, negated) >= 0 for ray in rays):
                oriented = negated
        if oriented is None:
            continue
        primitive = _primitive(oriented)
        if primitive is not None:
            collected.setdefault(primitive, None)
    return tuple(collected)


def _exact_feasible(
    equalities: list[tuple[tuple[int, ...], int]],
    inequalities: list[tuple[tuple[int, ...], int, bool]],
    variables: int,
) -> bool:
    """Decide one bounded exact rational feasibility problem over ``Fraction``."""

    rows: list[_LpRow] = [
        (tuple(Fraction(value) for value in coefficients), "eq", Fraction(rhs))
        for coefficients, rhs in equalities
    ]
    rows.extend(
        (
            tuple(Fraction(value) for value in coefficients),
            "gt" if strict else "ge",
            Fraction(rhs),
        )
        for coefficients, rhs, strict in inequalities
    )
    return _fm_feasible(rows, variables)


def _parallelepiped_points(
    dual_rays: tuple[tuple[int, ...], ...],
    cone_rays: tuple[tuple[int, ...], ...],
    dimension: int,
) -> tuple[tuple[int, ...], ...] | None:
    """Enumerate the lattice points of the dual cone's fundamental parallelepiped.

    The half-open parallelepiped ``{sum lambda_i r_i : 0 <= lambda_i < 1}`` over
    the primitive extreme rays ``r_i`` is contained in the coordinate box whose
    ``c``-th bound is ``sum_i |r_i[c]|``. Admission bounds that box before the
    enumeration; ``None`` reports a box outside the admitted envelope. Every
    candidate is confirmed by exact rational feasibility, so a member is never
    inferred from the box alone.
    """

    if not dual_rays:
        return ()
    bounds = tuple(
        sum(abs(ray[coordinate]) for ray in dual_rays)
        for coordinate in range(dimension)
    )
    box_size = 1
    for bound in bounds:
        box_size *= 2 * bound + 1
        if box_size > MAX_TORIC_CHART_BOX:
            return None
    equalities_by_coordinate = [
        (tuple(ray[coordinate] for ray in dual_rays), 0)
        for coordinate in range(dimension)
    ]
    inequalities: list[tuple[tuple[int, ...], int, bool]] = []
    for index in range(len(dual_rays)):
        unit = tuple(
            1 if position == index else 0 for position in range(len(dual_rays))
        )
        inequalities.append((unit, 0, False))
        inequalities.append((tuple(-value for value in unit), -1, True))
    points: list[tuple[int, ...]] = []
    for point in product(*(range(-bound, bound + 1) for bound in bounds)):
        if not all(_dot(ray, point) >= 0 for ray in cone_rays):
            continue
        equalities = [
            (
                equalities_by_coordinate[coordinate][0],
                point[coordinate],
            )
            for coordinate in range(dimension)
        ]
        if _exact_feasible(equalities, inequalities, len(dual_rays)):
            points.append(point)
    return tuple(points)


def _is_redundant(
    generators: list[tuple[int, ...]],
    target_index: int,
    dimension: int,
    deadline: float,
) -> bool:
    """Decide whether one candidate is a nonnegative integer combination of others."""

    target = generators[target_index]
    others = [
        generator for index, generator in enumerate(generators) if index != target_index
    ]
    if not others:
        return False
    import z3

    solver = z3.Solver()
    solver.set(rlimit=50_000_000, max_memory=256)
    bound = max(abs(value) for value in target)
    coefficients = [
        z3.Int(f"hilbert_coefficient_{index}") for index in range(len(others))
    ]
    for coefficient in coefficients:
        solver.add(coefficient >= 0, coefficient <= bound)
    for coordinate in range(dimension):
        solver.add(
            z3.Sum(
                [
                    coefficients[index] * int(others[index][coordinate])
                    for index in range(len(others))
                ]
            )
            == int(target[coordinate])
        )
    if monotonic() >= deadline:
        _reject_budget("affine-chart Hilbert reduction deadline expired")
    outcome = solver.check()
    if outcome == z3.sat:
        return True
    if outcome == z3.unsat:
        return False
    _reject_budget("affine-chart Hilbert reduction could not be decided exactly")


def _minimal_hilbert_generators(
    candidates: tuple[tuple[int, ...], ...],
    dimension: int,
    deadline: float,
) -> tuple[tuple[int, ...], ...]:
    """Reduce a generating candidate set to the unique minimal Hilbert basis."""

    generators = list(candidates)
    changed = True
    while changed:
        changed = False
        index = 0
        while index < len(generators):
            if _is_redundant(generators, index, dimension, deadline):
                del generators[index]
                changed = True
            else:
                index += 1
    return tuple(generators)


def _semigroup_relations(
    generators: tuple[tuple[int, ...], ...], dimension: int
) -> tuple[tuple[int, ...], ...]:
    """Canonical integer relation lattice ``ker_Z(G)`` of the Hilbert basis ``G``.

    The maintained affine-semigroup Smith/Hermite kernel computes the lattice;
    every returned relation is replayed against the generators here so the
    chart reports only relations that evaluate to zero.
    """

    from jacobian.math.affine_semigroups._kernel import compute_relation_lattice_data
    from jacobian.math.matrices.values import IntegerMatrix

    configuration = IntegerMatrix(
        row_count=dimension,
        column_count=len(generators),
        entries=tuple(
            tuple(generator[coordinate] for generator in generators)
            for coordinate in range(dimension)
        ),
    )
    data = compute_relation_lattice_data(configuration)
    relations = tuple(
        tuple(int(value) for value in row) for row in data.relation_basis.entries
    )
    for relation in relations:
        for coordinate in range(dimension):
            total = sum(
                relation[index] * generators[index][coordinate]
                for index in range(len(generators))
            )
            if total != 0:
                raise ArithmeticError(
                    "relation-lattice basis does not annihilate the Hilbert basis"
                )
    return relations


@dataclass(frozen=True, slots=True)
class AffineChartData:
    """Exact values produced by the admitted affine-chart kernel."""

    dual_cone_rays: tuple[tuple[int, ...], ...]
    hilbert_basis: tuple[tuple[int, ...], ...]
    relations: tuple[tuple[int, ...], ...]


def _max_component_digits(*families: tuple[tuple[int, ...], ...]) -> int:
    digits = 0
    for family in families:
        for vector in family:
            for value in vector:
                width = len(str(abs(value)))
                if width > digits:
                    digits = width
    return digits


def compute_affine_chart_data(
    rays: tuple[tuple[int, ...], ...],
    dimension: int,
    deadline: float,
) -> AffineChartData:
    """Compute the complete affine-monoid presentation of one full-dimensional cone.

    The dual cone, the fundamental-parallelepiped lattice points, the Hilbert
    basis, and the relation lattice are each checked against their admitted
    envelope. A cone whose derived work or output leaves the envelope is
    refused rather than returned with an incomplete basis.
    """

    dual_rays = _dual_cone_extreme_rays(rays, dimension)
    if not dual_rays:
        _reject_budget("affine charts require a full-dimensional cone")
    if len(dual_rays) > MAX_TORIC_CHART_DUAL_RAYS:
        _reject_budget(
            f"affine-chart dual cone exceeds {MAX_TORIC_CHART_DUAL_RAYS} rays"
        )
    points = _parallelepiped_points(dual_rays, rays, dimension)
    if points is None:
        _reject_budget(
            "affine-chart fundamental parallelepiped exceeds the "
            f"{MAX_TORIC_CHART_BOX}-point admission bound"
        )
    seen: set[tuple[int, ...]] = set()
    candidates: list[tuple[int, ...]] = []
    for vector in (*dual_rays, *points):
        if all(value == 0 for value in vector):
            continue
        if vector not in seen:
            seen.add(vector)
            candidates.append(vector)
    if len(candidates) > MAX_TORIC_CHART_CANDIDATES:
        _reject_budget(
            f"affine-chart candidate set exceeds {MAX_TORIC_CHART_CANDIDATES}"
        )
    candidates.sort()
    generators = _minimal_hilbert_generators(tuple(candidates), dimension, deadline)
    if len(generators) > MAX_TORIC_CHART_GENERATORS:
        _reject_budget(
            f"affine-chart Hilbert basis exceeds {MAX_TORIC_CHART_GENERATORS}"
        )
    relations = _semigroup_relations(generators, dimension)
    if len(relations) > MAX_TORIC_CHART_RELATIONS:
        _reject_budget(
            f"affine-chart relation lattice exceeds {MAX_TORIC_CHART_RELATIONS}"
        )
    digits = _max_component_digits(generators, relations)
    if digits > MAX_TORIC_CHART_COMPONENT_DIGITS:
        _reject_budget(
            "affine-chart exact components exceed "
            f"{MAX_TORIC_CHART_COMPONENT_DIGITS} decimal digits"
        )
    return AffineChartData(
        dual_cone_rays=tuple(sorted(dual_rays)),
        hilbert_basis=generators,
        relations=relations,
    )


def facet_localizing_character(
    cone_rays: tuple[tuple[int, ...], ...],
    facet_rays: tuple[tuple[int, ...], ...],
    dimension: int,
) -> tuple[int, ...] | None:
    """Primitive character cutting out one facet of a full-dimensional cone."""

    candidate = _one_dimensional_kernel(list(facet_rays), dimension)
    if candidate is None:
        return None
    if all(_dot(ray, candidate) >= 0 for ray in cone_rays):
        oriented = candidate
    else:
        negated = tuple(-value for value in candidate)
        if not all(_dot(ray, negated) >= 0 for ray in cone_rays):
            return None
        oriented = negated
    return _primitive(oriented)


@dataclass(frozen=True, slots=True)
class MorphismObstruction:
    """The first source cone whose image leaves every target cone."""

    source_cone_id: int
    source_ray_indices: tuple[int, ...]
    image_vectors: tuple[tuple[int, ...], ...]
    candidate_target_cone_ids: tuple[tuple[int, ...], ...]
    failing_ray_index: int | None
    failing_image: tuple[int, ...] | None
    reason: str


@dataclass(frozen=True, slots=True)
class MorphismData:
    """Exact values produced by the admitted toric-morphism kernel."""

    is_toric_morphism: bool
    ray_images: tuple[tuple[int, ...], ...]
    assignments: tuple[tuple[int, int], ...]
    obstruction: MorphismObstruction | None


def compute_toric_morphism_data(
    source_rays: tuple[tuple[int, ...], ...],
    source_cones: tuple[tuple[int, ...], ...],
    target_rays: tuple[tuple[int, ...], ...],
    target_cones: tuple[tuple[int, ...], ...],
    matrix: tuple[tuple[int, ...], ...],
) -> MorphismData:
    """Decide the fan map condition and retain the first exact obstruction.

    A lattice map is a toric morphism exactly when every source cone maps into
    a single target cone. Each assigned target cone is replayed by exact
    cone-membership of every source-ray image before it is returned.
    """

    source_rank = len(source_rays[0]) if source_rays else 0
    target_rank = len(target_rays[0]) if target_rays else 0
    ray_images = tuple(
        tuple(
            sum(matrix[row][column] * ray[column] for column in range(source_rank))
            for row in range(target_rank)
        )
        for ray in source_rays
    )
    assignments: list[tuple[int, int]] = []
    for cone_id, cone in enumerate(source_cones):
        images = tuple(ray_images[index] for index in cone)
        candidate_lists = tuple(
            tuple(
                target_id
                for target_id, target_cone in enumerate(target_cones)
                if _ray_in_cone(
                    image, tuple(target_rays[index] for index in target_cone)
                )
            )
            for image in images
        )
        common: int | None
        if candidate_lists:
            shared = set(candidate_lists[0])
            for candidates in candidate_lists[1:]:
                shared &= set(candidates)
            common = min(shared) if shared else None
        else:
            common = 0 if target_cones else None
        if common is not None:
            assignments.append((cone_id, common))
            continue
        failing_ray_index: int | None = None
        failing_image: tuple[int, ...] | None = None
        for position, candidates in enumerate(candidate_lists):
            if not candidates:
                failing_ray_index = cone[position]
                failing_image = images[position]
                break
        reason = (
            "RAY_IMAGE_OUTSIDE_ALL_TARGET_CONES"
            if failing_ray_index is not None
            else "NO_COMMON_TARGET_CONE"
        )
        return MorphismData(
            is_toric_morphism=False,
            ray_images=ray_images,
            assignments=(),
            obstruction=MorphismObstruction(
                source_cone_id=cone_id,
                source_ray_indices=cone,
                image_vectors=images,
                candidate_target_cone_ids=candidate_lists,
                failing_ray_index=failing_ray_index,
                failing_image=failing_image,
                reason=reason,
            ),
        )
    return MorphismData(
        is_toric_morphism=True,
        ray_images=ray_images,
        assignments=tuple(assignments),
        obstruction=None,
    )


__all__ = [
    "MAX_TORIC_FM_TABLEAU_ROWS",
    "AffineChartData",
    "FanObstruction",
    "MorphismData",
    "MorphismObstruction",
    "RecognizedCone",
    "RecognizedFan",
    "compute_affine_chart_data",
    "compute_toric_morphism_data",
    "facet_localizing_character",
    "recognize_fan",
]
