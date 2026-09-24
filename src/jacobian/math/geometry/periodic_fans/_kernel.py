"""Private exact kernels for periodic rational cell-complex validation.

Every decision is made in exact integer or rational arithmetic.  Translation
stabilizers are solved as integer linear systems, unimodularity claims go
through the integer Smith normal form, and cell intersections are decided by a
compact exact Fourier--Motzkin feasibility routine over ``Fraction``.  No
floating-point value participates in any accepted or rejected conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from math import factorial, floor

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.lattices._lattice_ops import hermite_basis
from jacobian.math.matrices._flint import integer_smith_normal_form

# One exact row ``coefficients . x (rel) rhs`` with ``rel`` in ``eq``/``ge``/``gt``.
_LpRow = tuple[tuple[Fraction, ...], str, Fraction]

MAX_PERIODIC_FM_ROWS = 4_096
MAX_PERIODIC_FM_GENERATED_ROWS = 65_536
MAX_PERIODIC_TRANSLATION_ENUMERATION = 200_000


def _reject_budget(message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("fan",),
        code="geometry.periodic_fan.resource_budget_exceeded",
        message=message,
    )


def _admit_fm_expansion(positive: int, negative: int, neutral: int) -> None:
    generated = positive * negative + neutral
    if generated > MAX_PERIODIC_FM_GENERATED_ROWS:
        _reject_budget(
            "exact periodic fan feasibility tableau expansion exceeds "
            f"{MAX_PERIODIC_FM_GENERATED_ROWS} generated rows"
        )


# ---------------------------------------------------------------------------
# Exact rational feasibility
# ---------------------------------------------------------------------------


def _fm_feasible(rows: list[_LpRow], variables: int) -> bool:
    """Decide exact rational feasibility by Fourier--Motzkin elimination.

    Equalities are substituted first; remaining variables are eliminated by
    pairing positive and negative coefficients.  The tableau is deduplicated
    after every step and bounded by ``MAX_PERIODIC_FM_ROWS``.
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
        _admit_fm_expansion(len(positive), len(negative), len(neutral))
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
        if len(rows) > MAX_PERIODIC_FM_ROWS:
            _reject_budget(
                "exact periodic fan feasibility tableau exceeds "
                f"{MAX_PERIODIC_FM_ROWS} rows"
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
    unique: dict[_LpRow, None] = {}
    for coefficients, kind, rhs in rows:
        scale: Fraction | None = None
        for value in coefficients:
            if value != 0:
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
    equalities: list[tuple[tuple[Fraction, ...], Fraction]],
    strict_rows: list[tuple[Fraction, ...]],
    nonnegative_variables: int,
    variables: int,
) -> bool:
    rows: list[_LpRow] = [(coefficients, "eq", rhs) for coefficients, rhs in equalities]
    rows.extend((coefficients, "gt", Fraction(0)) for coefficients in strict_rows)
    for variable in range(nonnegative_variables):
        unit = [Fraction(0)] * variables
        unit[variable] = Fraction(1)
        rows.append((tuple(unit), "ge", Fraction(0)))
    return _fm_feasible(rows, variables)


# ---------------------------------------------------------------------------
# Small exact linear algebra helpers
# ---------------------------------------------------------------------------


def _determinant(matrix: list[list[int]]) -> int:
    if not matrix:
        return 1
    from flint import fmpz_mat

    return int(fmpz_mat(matrix).det())


def _rank(matrix: list[list[int]]) -> int:
    if not matrix:
        return 0
    from flint import fmpz_mat

    return int(fmpz_mat(matrix).rank())


def _inverse(matrix: list[list[int]]) -> list[list[Fraction]]:
    size = len(matrix)
    working = [
        [Fraction(value) for value in row]
        + [Fraction(int(row_index == column)) for column in range(size)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next(row for row in range(column, size) if working[row][column] != 0)
        working[column], working[pivot] = working[pivot], working[column]
        inverse = Fraction(1, 1) / working[column][column]
        working[column] = [value * inverse for value in working[column]]
        for row in range(size):
            if row == column:
                continue
            factor = working[row][column]
            if factor == 0:
                continue
            working[row] = [
                a - factor * b
                for a, b in zip(working[row], working[column], strict=True)
            ]
    return [row[size:] for row in working]


def _lattice_coordinates(
    point: tuple[int, ...], basis_inverse: list[list[Fraction]]
) -> list[Fraction]:
    """Return rational coordinates ``a`` with ``point = a @ period_basis``."""

    return [
        sum(
            (point[j] * basis_inverse[j][i] for j in range(len(point))),
            Fraction(0),
        )
        for i in range(len(point))
    ]


def _reduce_mod_lattice(
    point: tuple[int, ...], basis_inverse: list[list[Fraction]]
) -> tuple[Fraction, ...]:
    coordinates = _lattice_coordinates(point, basis_inverse)
    return tuple(value - floor(value) for value in coordinates)


def _face_key(
    coordinates: tuple[tuple[int, ...], ...],
    basis_inverse: list[list[Fraction]],
) -> tuple[tuple[Fraction, ...], tuple[tuple[int, ...], ...]]:
    anchor = min(coordinates)
    differences = tuple(
        sorted(
            tuple(point[index] - anchor[index] for index in range(len(anchor)))
            for point in coordinates
        )
    )
    return (_reduce_mod_lattice(anchor, basis_inverse), differences)


# ---------------------------------------------------------------------------
# Simplex geometry
# ---------------------------------------------------------------------------


def _simplex_volume_numerator(coordinates: tuple[tuple[int, ...], ...]) -> int:
    if len(coordinates) > 3 and len(coordinates[0]) == 2:
        return abs(
            sum(
                coordinates[index][0] * coordinates[(index + 1) % len(coordinates)][1]
                - coordinates[(index + 1) % len(coordinates)][0] * coordinates[index][1]
                for index in range(len(coordinates))
            )
        )
    origin = coordinates[0]
    edges = [
        [point[index] - origin[index] for index in range(len(origin))]
        for point in coordinates[1:]
    ]
    return abs(_determinant(edges))


def _affinely_independent(coordinates: tuple[tuple[int, ...], ...]) -> bool:
    if len(coordinates) < 2:
        return True
    origin = coordinates[0]
    edges = [
        [point[index] - origin[index] for index in range(len(origin))]
        for point in coordinates[1:]
    ]
    return _rank(edges) == len(coordinates[0])


def _strictly_convex_ccw_polygon(
    coordinates: tuple[tuple[int, ...], ...],
) -> bool:
    if len(coordinates) < 4 or len(coordinates[0]) != 2:
        return False
    size = len(coordinates)
    turns = tuple(
        (coordinates[(index + 1) % size][0] - coordinates[index][0])
        * (coordinates[(index + 2) % size][1] - coordinates[(index + 1) % size][1])
        - (coordinates[(index + 1) % size][1] - coordinates[index][1])
        * (coordinates[(index + 2) % size][0] - coordinates[(index + 1) % size][0])
        for index in range(size)
    )
    return all(turn > 0 for turn in turns)


def _cell_face_positions(
    cell: tuple[int, ...], rank: int
) -> tuple[tuple[int, ...], ...]:
    if len(cell) == rank + 1:
        return tuple(
            positions
            for size in range(1, len(cell) + 1)
            for positions in combinations(range(len(cell)), size)
        )
    if rank == 2 and len(cell) > 3:
        size = len(cell)
        vertices = tuple((index,) for index in range(size))
        edges = tuple((index, (index + 1) % size) for index in range(size))
        return vertices + edges + (tuple(range(size)),)
    raise ArithmeticError("admitted periodic cell has no supported face lattice")


def _barycentric_feasible(
    point: tuple[int, ...], simplex: tuple[tuple[int, ...], ...]
) -> bool:
    variables = len(simplex)
    equalities: list[tuple[tuple[Fraction, ...], Fraction]] = []
    for coordinate in range(len(point)):
        equalities.append(
            (
                tuple(Fraction(simplex[i][coordinate]) for i in range(variables)),
                Fraction(point[coordinate]),
            )
        )
    equalities.append((tuple(Fraction(1) for _ in range(variables)), Fraction(1)))
    return _feasible(equalities, [], variables, variables)


def _point_in_simplex(
    point: tuple[int, ...], simplex: tuple[tuple[int, ...], ...]
) -> bool:
    return _barycentric_feasible(point, simplex)


_Point2 = tuple[Fraction, Fraction]


def _oriented_polygon(points: tuple[tuple[int, ...], ...]) -> tuple[_Point2, ...]:
    result = tuple((Fraction(x), Fraction(y)) for x, y in points)
    area2 = sum(
        result[index][0] * result[(index + 1) % len(result)][1]
        - result[(index + 1) % len(result)][0] * result[index][1]
        for index in range(len(result))
    )
    return result if area2 > 0 else tuple(reversed(result))


def _cross2(origin: _Point2, first: _Point2, second: _Point2) -> Fraction:
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
        second[0] - origin[0]
    )


def _convex_intersection(
    first: tuple[tuple[int, ...], ...], second: tuple[tuple[int, ...], ...]
) -> tuple[_Point2, ...]:
    """Clip one exact convex polygon by another, retaining degenerate results."""

    subject = list(_oriented_polygon(first))
    clip = _oriented_polygon(second)
    for index, start in enumerate(clip):
        end = clip[(index + 1) % len(clip)]
        if not subject:
            break
        output: list[_Point2] = []
        previous = subject[-1]
        previous_side = _cross2(start, end, previous)
        for current in subject:
            current_side = _cross2(start, end, current)
            if (current_side >= 0) != (previous_side >= 0):
                ratio = previous_side / (previous_side - current_side)
                output.append(
                    (
                        previous[0] + ratio * (current[0] - previous[0]),
                        previous[1] + ratio * (current[1] - previous[1]),
                    )
                )
            if current_side >= 0:
                output.append(current)
            previous, previous_side = current, current_side
        subject = []
        for point in output:
            if point not in subject:
                subject.append(point)
    return tuple(subject)


def _polygon_edges(
    points: tuple[tuple[int, ...], ...],
) -> set[frozenset[tuple[int, ...]]]:
    return {
        frozenset((points[index], points[(index + 1) % len(points)]))
        for index in range(len(points))
    }


def _intersection_nonempty(
    first: tuple[tuple[int, ...], ...],
    second: tuple[tuple[int, ...], ...],
) -> bool:
    if len(first[0]) == 2:
        return bool(_convex_intersection(first, second))
    variables = len(first) + len(second)
    equalities: list[tuple[tuple[Fraction, ...], Fraction]] = []
    equalities.append(
        (
            tuple(
                Fraction(1) if i < len(first) else Fraction(0) for i in range(variables)
            ),
            Fraction(1),
        )
    )
    equalities.append(
        (
            tuple(
                Fraction(0) if i < len(first) else Fraction(1) for i in range(variables)
            ),
            Fraction(1),
        )
    )
    for coordinate in range(len(first[0])):
        equalities.append(
            (
                tuple(
                    [Fraction(first[i][coordinate]) for i in range(len(first))]
                    + [-Fraction(second[j][coordinate]) for j in range(len(second))]
                ),
                Fraction(0),
            )
        )
    return _feasible(equalities, [], variables, variables)


def _strict_lambda_feasible(
    first: tuple[tuple[int, ...], ...],
    second: tuple[tuple[int, ...], ...],
    position: int,
) -> bool:
    variables = len(first) + len(second)
    equalities: list[tuple[tuple[Fraction, ...], Fraction]] = []
    equalities.append(
        (
            tuple(
                Fraction(1) if i < len(first) else Fraction(0) for i in range(variables)
            ),
            Fraction(1),
        )
    )
    equalities.append(
        (
            tuple(
                Fraction(0) if i < len(first) else Fraction(1) for i in range(variables)
            ),
            Fraction(1),
        )
    )
    for coordinate in range(len(first[0])):
        equalities.append(
            (
                tuple(
                    [Fraction(first[i][coordinate]) for i in range(len(first))]
                    + [-Fraction(second[j][coordinate]) for j in range(len(second))]
                ),
                Fraction(0),
            )
        )
    strict = [Fraction(0)] * variables
    strict[position] = Fraction(1)
    return _feasible(equalities, [tuple(strict)], variables, variables)


def _intersection_is_common_face(
    first: tuple[tuple[int, ...], ...],
    second: tuple[tuple[int, ...], ...],
) -> bool:
    """Decide whether ``conv(first) cap conv(second)`` is a common face.

    Every supported maximal cell is a convex polytope with all listed vertices
    extreme. Its intersection with another such polytope is a common face iff
    both contribute the same face vertices and the intersection contains no
    point outside their convex hull. Exact strict feasibility checks the latter.
    """

    if len(first[0]) == 2:
        if set(first) == set(second):
            return True
        intersection = _convex_intersection(first, second)
        if not intersection:
            return True
        if len(intersection) >= 3:
            area2 = sum(
                intersection[index][0]
                * intersection[(index + 1) % len(intersection)][1]
                - intersection[(index + 1) % len(intersection)][0]
                * intersection[index][1]
                for index in range(len(intersection))
            )
            if area2 != 0:
                return False
        unique = tuple(dict.fromkeys(intersection))
        integer_points = {
            point
            for point in unique
            if point[0].denominator == 1 and point[1].denominator == 1
        }
        first_vertices = set(first)
        second_vertices = set(second)
        if len(unique) == 1:
            point = next(iter(integer_points), None)
            return point in first_vertices and point in second_vertices
        if len(integer_points) != len(unique):
            return False
        endpoint1 = min(unique)
        endpoint2 = max(unique)
        edge = frozenset((tuple(map(int, endpoint1)), tuple(map(int, endpoint2))))
        return edge in _polygon_edges(first) and edge in _polygon_edges(second)

    first_inside = [
        index for index, point in enumerate(first) if _point_in_simplex(point, second)
    ]
    second_inside = [
        index for index, point in enumerate(second) if _point_in_simplex(point, first)
    ]
    first_points = sorted(first[index] for index in first_inside)
    second_points = sorted(second[index] for index in second_inside)
    if first_points != second_points:
        return False
    for index in range(len(first)):
        if index in first_inside:
            continue
        if _strict_lambda_feasible(first, second, index):
            return False
    return True


# ---------------------------------------------------------------------------
# Translation stabilizers
# ---------------------------------------------------------------------------


def _translation_stabilizer(
    coordinates: tuple[tuple[int, ...], ...],
    period_basis: list[list[int]],
    basis_inverse: list[list[Fraction]],
) -> tuple[tuple[tuple[int, ...], ...], int, int | None]:
    """Return the ambient-coordinate basis, rank, and index of the stabilizer.

    Only finitely many translations can map a bounded cell to itself. Every
    such translation maps the first vertex to another vertex, so checking those
    candidates against the vertex set is complete and quadratic in cell size.
    """

    dimension = len(coordinates[0])
    ambient_candidates: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    base = coordinates[0]
    coordinate_set = set(coordinates)
    for candidate in coordinates:
        translation = tuple(
            candidate[index] - base[index] for index in range(dimension)
        )
        translated = {
            tuple(point[index] + translation[index] for index in range(dimension))
            for point in coordinates
        }
        if translated != coordinate_set:
            continue
        if translation in seen:
            continue
        seen.add(translation)
        coordinates_in_period = _lattice_coordinates(translation, basis_inverse)
        if all(value.denominator == 1 for value in coordinates_in_period):
            ambient_candidates.append(translation)
    nonzero = [translation for translation in ambient_candidates if any(translation)]
    if not nonzero:
        return ((), 0, None)
    lattice_coordinates = [
        [int(value) for value in _lattice_coordinates(translation, basis_inverse)]
        for translation in nonzero
    ]
    reduced, _ = hermite_basis(lattice_coordinates)
    basis_rows = tuple(
        tuple(
            sum(
                reduced[row][column] * period_basis[column][index]
                for column in range(dimension)
            )
            for index in range(dimension)
        )
        for row in range(len(reduced))
        if any(reduced[row])
    )
    rank = len(basis_rows)
    index = _determinant(reduced) if rank == dimension else None
    return (basis_rows, rank, None if index is None else abs(index))


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PeriodicObstruction:
    """The first exact validation failure of a proposed periodic fan."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class QuotientCellData:
    """One orbit of faces with its exact stabilizer sublattice."""

    cell_id: int
    dimension: int
    representative_cell: int
    representative_vertices: tuple[int, ...]
    member_count: int
    stabilizer_basis: tuple[tuple[int, ...], ...]
    stabilizer_rank: int
    stabilizer_index: int | None


@dataclass(frozen=True, slots=True)
class OrbitRowData:
    """One maximal cell face and the quotient cell it maps to."""

    cell_id: int
    face_positions: tuple[int, ...]
    quotient_cell_id: int


@dataclass(frozen=True, slots=True)
class RecognizedPeriodicFan:
    """Every validated fact reused by the published periodic operations."""

    period_index: int
    quotient_cells: tuple[QuotientCellData, ...]
    orbit_rows: tuple[OrbitRowData, ...]
    face_relations: tuple[tuple[int, int], ...]
    covers_fundamental_domain: bool
    obstruction: PeriodicObstruction | None


@dataclass(slots=True)
class _QuotientAccumulator:
    cell_id: int
    dimension: int
    representative_cell: int
    representative_vertices: tuple[int, ...]
    member_count: int
    stabilizer_basis: tuple[tuple[int, ...], ...]
    stabilizer_rank: int
    stabilizer_index: int | None


def _bounding_box(
    coordinates: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    dimension = len(coordinates[0])
    minimum = tuple(
        min(point[index] for point in coordinates) for index in range(dimension)
    )
    maximum = tuple(
        max(point[index] for point in coordinates) for index in range(dimension)
    )
    return minimum, maximum


def _translations_between(
    first: tuple[tuple[int, ...], ...],
    second: tuple[tuple[int, ...], ...],
) -> tuple[range, ...]:
    first_min, first_max = _bounding_box(first)
    second_min, second_max = _bounding_box(second)
    return tuple(
        range(
            first_min[index] - second_max[index],
            first_max[index] - second_min[index] + 1,
        )
        for index in range(len(first_min))
    )


def _translation_count(ranges: tuple[range, ...]) -> int:
    count = 1
    for axis in ranges:
        count *= len(axis)
    return count


def recognize_periodic_fan(  # noqa: C901
    *,
    lattice_rank: int,
    period_basis: tuple[tuple[Fraction, ...], ...],
    vertices: tuple[tuple[int, ...], ...],
    cells: tuple[tuple[int, ...], ...],
    unimodular_cells: tuple[int, ...],
    overlap_candidates: tuple[tuple[int, int, tuple[int, ...]], ...],
) -> RecognizedPeriodicFan:
    """Run the complete exact periodic-fan recognition, retaining the first obstruction."""

    def obstruct(code: str, message: str) -> RecognizedPeriodicFan:
        return RecognizedPeriodicFan(
            period_index=0,
            quotient_cells=(),
            orbit_rows=(),
            face_relations=(),
            covers_fundamental_domain=False,
            obstruction=PeriodicObstruction(code=code, message=message),
        )

    if any(value.denominator != 1 for row in period_basis for value in row):
        return obstruct(
            "geometry.periodic_fan.period_not_integral",
            "the period lattice must be a sublattice of the ambient integer lattice",
        )
    matrix = [[int(value) for value in row] for row in period_basis]
    determinant = _determinant(matrix)
    if determinant == 0:
        return obstruct(
            "geometry.periodic_fan.period_rank_deficient",
            "the period lattice must be full rank in the ambient lattice",
        )
    period_index = abs(determinant)
    basis_inverse = _inverse(matrix)

    for vertex in vertices:
        coordinates = _lattice_coordinates(vertex, basis_inverse)
        if any(value < 0 or value > 1 for value in coordinates):
            return obstruct(
                "geometry.periodic_fan.cell_outside_fundamental_domain",
                f"vertex {vertex} does not lie in the closed fundamental parallelotope",
            )

    for cell_id, cell in enumerate(cells):
        cell_points = tuple(vertices[index] for index in cell)
        if not _affinely_independent(cell_points):
            return obstruct(
                "geometry.periodic_fan.cell_degenerate",
                f"cell {cell_id} is not full-dimensional",
            )
        if len(cell) > lattice_rank + 1 and not _strictly_convex_ccw_polygon(
            cell_points
        ):
            return obstruct(
                "geometry.periodic_fan.polygon_not_strictly_convex",
                f"cell {cell_id} is not a strictly convex counterclockwise polygon",
            )

    for cell_id in unimodular_cells:
        cell = cells[cell_id]
        if len(cell) != lattice_rank + 1:
            return obstruct(
                "geometry.periodic_fan.unimodular_cell_not_simplex",
                f"cell {cell_id} is a nonsimplicial polygon and cannot carry a simplex unimodularity claim",
            )
        origin = vertices[cell[0]]
        edges = tuple(
            tuple(
                vertices[index][coordinate] - origin[coordinate]
                for coordinate in range(lattice_rank)
            )
            for index in cell[1:]
        )
        diagonal = integer_smith_normal_form(edges)
        if any(diagonal[row][row] != 1 for row in range(lattice_rank)):
            return obstruct(
                "geometry.periodic_fan.claimed_cell_not_unimodular",
                f"cell {cell_id} is claimed unimodular but its edge lattice "
                "is not saturated",
            )

    declared = {
        _canonical_overlap_key(first, second, translation)
        for first, second, translation in overlap_candidates
    }

    cell_coordinates = [tuple(vertices[index] for index in cell) for cell in cells]
    for first, second, translation in overlap_candidates:
        if any(
            value.denominator != 1
            for value in _lattice_coordinates(translation, basis_inverse)
        ):
            continue
        shifted = tuple(
            tuple(point[index] + translation[index] for index in range(lattice_rank))
            for point in cell_coordinates[second]
        )
        if not _intersection_nonempty(cell_coordinates[first], shifted):
            continue
        if not _intersection_is_common_face(cell_coordinates[first], shifted):
            return obstruct(
                "geometry.periodic_fan.overlap_not_face_to_face",
                f"cells {first} and {second} intersect at translation {translation} "
                "in a set that is not a common face of both",
            )

    enumeration = 0
    for first in range(len(cells)):
        for second in range(first, len(cells)):
            # _translations_between returns precisely the inclusive ranges in
            # which the translated bounding boxes overlap on every axis.
            ranges = _translations_between(
                cell_coordinates[first], cell_coordinates[second]
            )
            enumeration += _translation_count(ranges)
            if enumeration > MAX_PERIODIC_TRANSLATION_ENUMERATION:
                _reject_budget(
                    "periodic overlap candidate enumeration exceeds "
                    f"{MAX_PERIODIC_TRANSLATION_ENUMERATION} translations"
                )
            for translation in product(*ranges):
                if any(
                    value.denominator != 1
                    for value in _lattice_coordinates(translation, basis_inverse)
                ):
                    continue
                shifted = tuple(
                    tuple(
                        point[index] + translation[index]
                        for index in range(lattice_rank)
                    )
                    for point in cell_coordinates[second]
                )
                if not _intersection_nonempty(cell_coordinates[first], shifted):
                    continue
                key = _canonical_overlap_key(first, second, translation)
                if key not in declared:
                    return obstruct(
                        "geometry.periodic_fan.undeclared_overlap",
                        f"cells {first} and {second} overlap at translation "
                        f"{translation} but no overlap candidate declares it",
                    )

    total_volume_numerator = sum(
        _simplex_volume_numerator(coordinates) for coordinates in cell_coordinates
    )
    if total_volume_numerator != factorial(lattice_rank) * period_index:
        return obstruct(
            "geometry.periodic_fan.cells_do_not_cover_fundamental_domain",
            "the maximal cells do not cover the fundamental parallelotope with "
            "total normalized volume equal to the period index",
        )

    quotient_cells, orbit_rows, face_relations = _build_quotient(
        cells=cells,
        cell_coordinates=cell_coordinates,
        period_basis=matrix,
        basis_inverse=basis_inverse,
        lattice_rank=lattice_rank,
    )
    return RecognizedPeriodicFan(
        period_index=period_index,
        quotient_cells=quotient_cells,
        orbit_rows=orbit_rows,
        face_relations=face_relations,
        covers_fundamental_domain=True,
        obstruction=None,
    )


def _canonical_overlap_key(
    first: int, second: int, translation: tuple[int, ...]
) -> tuple[int, int, tuple[int, ...]]:
    if first <= second:
        return (first, second, translation)
    return (second, first, tuple(-value for value in translation))


def _build_quotient(
    *,
    cells: tuple[tuple[int, ...], ...],
    cell_coordinates: list[tuple[tuple[int, ...], ...]],
    period_basis: list[list[int]],
    basis_inverse: list[list[Fraction]],
    lattice_rank: int,
) -> tuple[
    tuple[QuotientCellData, ...],
    tuple[OrbitRowData, ...],
    tuple[tuple[int, int], ...],
]:
    face_owner: dict[tuple[object, ...], int] = {}
    accumulator: list[_QuotientAccumulator] = []
    orbit_rows: list[OrbitRowData] = []
    local_face_quotient: dict[tuple[int, tuple[int, ...]], int] = {}
    for cell_id, cell in enumerate(cells):
        coordinates = cell_coordinates[cell_id]
        for positions in _cell_face_positions(cell, lattice_rank):
            face_coordinates = tuple(coordinates[position] for position in positions)
            key = _face_key(face_coordinates, basis_inverse)
            quotient_id = face_owner.get(key)
            if quotient_id is None:
                quotient_id = len(accumulator)
                face_owner[key] = quotient_id
                basis, stabilizer_rank, index = _translation_stabilizer(
                    face_coordinates, period_basis, basis_inverse
                )
                accumulator.append(
                    _QuotientAccumulator(
                        cell_id=quotient_id,
                        dimension=(
                            lattice_rank
                            if len(positions) == len(cell)
                            else len(positions) - 1
                        ),
                        representative_cell=cell_id,
                        representative_vertices=tuple(
                            cell[position] for position in positions
                        ),
                        member_count=1,
                        stabilizer_basis=basis,
                        stabilizer_rank=stabilizer_rank,
                        stabilizer_index=index,
                    )
                )
            else:
                accumulator[quotient_id].member_count += 1
            local_face_quotient[(cell_id, positions)] = quotient_id
            orbit_rows.append(
                OrbitRowData(
                    cell_id=cell_id,
                    face_positions=positions,
                    quotient_cell_id=quotient_id,
                )
            )

    relations: set[tuple[int, int]] = set()
    for cell_id, cell in enumerate(cells):
        faces = _cell_face_positions(cell, lattice_rank)
        for sigma_positions in faces:
            sigma = local_face_quotient[(cell_id, sigma_positions)]
            sigma_set = set(sigma_positions)
            for tau_positions in faces:
                if set(tau_positions) <= sigma_set:
                    tau = local_face_quotient[(cell_id, tau_positions)]
                    relations.add((tau, sigma))

    quotient_cells = tuple(
        QuotientCellData(
            cell_id=entry.cell_id,
            dimension=entry.dimension,
            representative_cell=entry.representative_cell,
            representative_vertices=entry.representative_vertices,
            member_count=entry.member_count,
            stabilizer_basis=entry.stabilizer_basis,
            stabilizer_rank=entry.stabilizer_rank,
            stabilizer_index=entry.stabilizer_index,
        )
        for entry in accumulator
    )
    return quotient_cells, tuple(orbit_rows), tuple(sorted(relations))


__all__ = [
    "MAX_PERIODIC_FM_ROWS",
    "MAX_PERIODIC_TRANSLATION_ENUMERATION",
    "OrbitRowData",
    "PeriodicObstruction",
    "QuotientCellData",
    "RecognizedPeriodicFan",
    "recognize_periodic_fan",
]
