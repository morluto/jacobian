"""Private exact conversion kernel for bounded rational polyhedra.

The kernel uses homogeneous primitive-integer rows throughout.  Conversion
from inequalities to generators is an incremental double-description pass;
conversion from generators to inequalities applies the same pass to the dual
cone.  Python owns orchestration and incidence bitsets, while FLINT owns exact
rank and inverse arithmetic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import gcd, lcm

IntegerVector = tuple[int, ...]

MAX_DD_RAY_BOUND = 20_000
"""Largest statically bounded intermediate ray family admitted to exact DD."""

MAX_DD_PAIR_BOUND = 20_000_000
"""Largest statically bounded positive/negative candidate-pair family."""

MAX_DD_COEFFICIENT_DIGITS = 500_000
"""Largest determinant-height bound admitted to one exact conversion."""

MAX_DD_WEIGHTED_HEIGHT_WORK = 100_000_000_000_000
"""Ceiling coupling DD candidate pairs with exact coefficient height."""

MAX_PULLING_SIMPLEX_BOUND = 10_000_000
"""Largest statically bounded simplex family admitted to pulling."""


class PolyhedralConversionError(RuntimeError):
    """An exact backend failure prevented a polyhedral conclusion."""


@dataclass(frozen=True)
class QuotientMap:
    """Exact lineality quotient coordinates and a chosen reconstruction.

    ``quotient_columns`` selects coordinates on which all inequality rows have
    full column rank.  ``lineality_basis`` spans the common kernel.  A quotient
    ray is reconstructed by placing its coordinates in the selected ambient
    columns and zeros elsewhere; adding the lineality basis recovers every
    ambient representative with the same inequality evaluations.
    """

    ambient_dimension: int
    quotient_columns: tuple[int, ...]
    lineality_basis: tuple[IntegerVector, ...]

    def reconstruct(self, vector: Sequence[int]) -> IntegerVector:
        if len(vector) != len(self.quotient_columns):
            raise ValueError("quotient vector has the wrong dimension")
        result = [0] * self.ambient_dimension
        for column, value in zip(self.quotient_columns, vector, strict=True):
            result[column] = int(value)
        return tuple(result)


@dataclass(frozen=True)
class ConeRay:
    """One primitive generator and its exact active-inequality bitset."""

    vector: IntegerVector
    active: int


@dataclass(frozen=True)
class ConeConversion:
    """The exact generator description of one homogeneous cone."""

    rays: tuple[ConeRay, ...]
    quotient: QuotientMap
    processed_rows: int
    candidate_pairs: int
    rank_tests: int


@dataclass(frozen=True)
class PolyhedronConversion:
    """Affine classification decoded from a homogeneous cone."""

    vertices: tuple[tuple[Fraction, ...], ...]
    vertex_incidence: tuple[int, ...]
    recession_rays: tuple[IntegerVector, ...]
    lineality_basis: tuple[IntegerVector, ...]
    empty: bool
    bounded: bool
    affine_dimension: int
    cone: ConeConversion


@dataclass(frozen=True)
class HullConversion:
    """Facet description and incidence of a finite point hull."""

    facets: tuple[tuple[IntegerVector, int], ...]
    facet_incidence: tuple[int, ...]
    affine_equalities: tuple[tuple[IntegerVector, int], ...]
    cone: ConeConversion


def upper_bound_facets(vertex_count: int, dimension: int) -> int:
    """McMullen upper bound for facets of a d-polytope with n vertices."""

    if vertex_count <= dimension:
        return 0
    if dimension <= 1:
        return min(vertex_count, 2)
    half = dimension // 2
    from math import comb

    if dimension % 2 == 0:
        return comb(vertex_count - half, half) + comb(vertex_count - half - 1, half - 1)
    return 2 * comb(vertex_count - half - 1, half)


def dd_work_bound(row_count: int, cone_dimension: int) -> tuple[int, int]:
    """Return sound ray and candidate-pair bounds for incremental DD."""

    if cone_dimension <= 1:
        return 1, 0
    section_dimension = cone_dimension - 1
    maximum_rays = 0
    pairs = 0
    for processed in range(cone_dimension, row_count + 1):
        rays = upper_bound_facets(processed, section_dimension)
        maximum_rays = max(maximum_rays, rays)
        pairs += (rays * rays) // 4
    return maximum_rays, pairs


def require_dd_work_admissible(row_count: int, cone_dimension: int) -> None:
    """Reject a conversion whose theorem-backed worst case exceeds its budget."""

    rays, pairs = dd_work_bound(row_count, cone_dimension)
    if rays > MAX_DD_RAY_BOUND or pairs > MAX_DD_PAIR_BOUND:
        raise ValueError(
            "exact double-description conversion exceeds the output-sensitive "
            f"work bound (at most {rays} rays and {pairs} candidate pairs; "
            f"limits are {MAX_DD_RAY_BOUND} and {MAX_DD_PAIR_BOUND})"
        )


def require_dd_weighted_work_admissible(
    row_count: int, cone_dimension: int, minor_digits: int
) -> None:
    """Couple theorem-backed pair work to conservative integer height."""

    _rays, pairs = dd_work_bound(row_count, cone_dimension)
    weighted = max(1, pairs) * minor_digits**2
    if weighted > MAX_DD_WEIGHTED_HEIGHT_WORK:
        raise ValueError(
            "exact double-description conversion exceeds the height-weighted "
            f"work bound ({weighted} > {MAX_DD_WEIGHTED_HEIGHT_WORK})"
        )


def pulling_triangulation(
    point_count: int,
    dimension: int,
    facet_incidence: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """Triangulate a full-dimensional hull from its exact face incidences.

    Every recursive face is represented by its vertex bitset.  Its facets are
    the maximal proper intersections with the hull's facets, so no supporting
    hyperplanes or vertex combinations are recomputed during triangulation.
    """

    if point_count < dimension + 1:
        return ()
    full_face = (1 << point_count) - 1

    def members(bits: int) -> tuple[int, ...]:
        return tuple(index for index in range(point_count) if bits & (1 << index))

    def boundary_faces(face: int) -> tuple[int, ...]:
        intersections = {
            face & facet for facet in facet_incidence if 0 < (face & facet) < face
        }
        return maximal_incidence_faces(intersections)

    def triangulate(face: int, face_dimension: int) -> tuple[tuple[int, ...], ...]:
        face_members = members(face)
        if face_dimension == 0:
            return ((face_members[0],),) if face_members else ()
        if face_dimension == 1:
            if len(face_members) < 2:
                return ()
            return ((face_members[0], face_members[-1]),)
        apex = face_members[0]
        simplices: list[tuple[int, ...]] = []
        for boundary in boundary_faces(face):
            if boundary & (1 << apex):
                continue
            for simplex in triangulate(boundary, face_dimension - 1):
                simplices.append((*simplex, apex))
        return tuple(dict.fromkeys(simplices))

    return triangulate(full_face, dimension)


def maximal_incidence_faces(faces: Sequence[int] | set[int]) -> tuple[int, ...]:
    """Return the inclusion-maximal nonempty bitsets in deterministic order."""

    unique = {face for face in faces if face}
    return tuple(
        sorted(
            candidate
            for candidate in unique
            if not any(
                candidate != other and candidate & other == candidate
                for other in unique
            )
        )
    )


def require_pulling_work_admissible(facet_count: int, dimension: int) -> None:
    """Bound recursive incidence intersections before triangulation."""

    simplex_bound = max(1, facet_count) ** max(1, dimension)
    if simplex_bound > MAX_PULLING_SIMPLEX_BOUND:
        raise ValueError(
            "incidence-driven pulling triangulation exceeds the work bound "
            f"({simplex_bound} possible simplices > {MAX_PULLING_SIMPLEX_BOUND})"
        )


def _fraction(value: object) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    numerator = getattr(
        value, "numerator", getattr(value, "p", getattr(value, "num", None))
    )
    denominator = getattr(
        value, "denominator", getattr(value, "q", getattr(value, "den", None))
    )
    if numerator is None or denominator is None:
        raise TypeError("polyhedral coordinates must be exact rationals")
    return Fraction(int(numerator), int(denominator))


def _integer_digit_bound(value: int) -> int:
    """Upper-bound decimal digits without converting a huge integer to text."""

    bits = abs(value).bit_length()
    return max(1, (bits * 30_103 + 99_999) // 100_000)


def _component_digit_bound(rows: Sequence[Sequence[object]]) -> int:
    bound = 1
    for row in rows:
        for raw in row:
            value = _fraction(raw)
            bound = max(
                bound,
                _integer_digit_bound(value.numerator),
                _integer_digit_bound(value.denominator),
            )
    return bound


def require_dd_height_admissible(
    component_digits: int, dimension: int, *, affine_halfspaces: bool
) -> int:
    """Bound primitive homogeneous rows and their maximal minors."""

    denominator_terms = dimension + (1 if affine_halfspaces else 0)
    row_digits = (denominator_terms + 1) * component_digits + 2
    minor_digits = max(1, dimension) * row_digits + dimension + 2
    if minor_digits > MAX_DD_COEFFICIENT_DIGITS:
        raise ValueError(
            "exact double-description conversion exceeds the coefficient-growth "
            f"bound ({minor_digits} digits > {MAX_DD_COEFFICIENT_DIGITS})"
        )
    return minor_digits


def primitive_integer_vector(values: Sequence[object]) -> IntegerVector:
    """Clear denominators and common content without changing ray orientation."""

    fractions = tuple(_fraction(value) for value in values)
    scale = 1
    for value in fractions:
        scale = lcm(scale, value.denominator)
    integers = [value.numerator * (scale // value.denominator) for value in fractions]
    content = 0
    for integer in integers:
        content = gcd(content, abs(integer))
    if content == 0:
        raise ValueError("the zero vector has no primitive ray normalization")
    return tuple(value // content for value in integers)


def _canonical_equation_vector(values: Sequence[object]) -> IntegerVector:
    primitive = primitive_integer_vector(values)
    first = next(value for value in primitive if value)
    return primitive if first > 0 else tuple(-value for value in primitive)


def primitive_homogeneous_halfspace(
    coefficients: Sequence[object], offset: object
) -> IntegerVector:
    """Encode ``a*x <= b`` as the primitive row ``(b, -a)``."""

    return primitive_integer_vector(
        (_fraction(offset), *(-_fraction(c) for c in coefficients))
    )


def primitive_homogeneous_point(point: Sequence[object]) -> IntegerVector:
    """Encode an affine point as the primitive ray ``(1, x)``."""

    return primitive_integer_vector((Fraction(1), *map(_fraction, point)))


def _dot(left: Sequence[int], right: Sequence[int]) -> int:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _rank(rows: Sequence[Sequence[int]], column_count: int) -> int:
    if not rows or column_count == 0:
        return 0
    try:
        from flint import fmpz_mat

        return int(fmpz_mat([list(row) for row in rows]).rank())
    except Exception as exc:  # pragma: no cover - backend boundary
        raise PolyhedralConversionError(
            "exact polyhedral rank computation failed"
        ) from exc


def _independent_indices(
    rows: Sequence[Sequence[int]], column_count: int
) -> tuple[int, ...]:
    chosen: list[int] = []
    rank = 0
    for index, row in enumerate(rows):
        next_rank = _rank([rows[i] for i in chosen] + [row], column_count)
        if next_rank > rank:
            chosen.append(index)
            rank = next_rank
            if rank == column_count:
                break
    return tuple(chosen)


def _independent_columns(
    rows: Sequence[Sequence[int]], column_count: int
) -> tuple[int, ...]:
    columns = [[row[column] for row in rows] for column in range(column_count)]
    return _independent_indices(columns, len(rows))


def _lineality_basis(
    rows: Sequence[Sequence[int]], column_count: int
) -> tuple[IntegerVector, ...]:
    if not rows:
        return tuple(
            tuple(1 if i == j else 0 for i in range(column_count))
            for j in range(column_count)
        )
    try:
        from flint import fmpz_mat

        matrix, nullity = fmpz_mat([list(row) for row in rows]).nullspace()
        return tuple(
            _canonical_equation_vector(
                tuple(int(matrix[row, column]) for row in range(column_count))
            )
            for column in range(int(nullity))
        )
    except Exception as exc:  # pragma: no cover - backend boundary
        raise PolyhedralConversionError(
            "exact polyhedral nullspace computation failed"
        ) from exc


def _inverse_columns(matrix: Sequence[Sequence[int]]) -> tuple[IntegerVector, ...]:
    try:
        from flint import fmpq_mat

        inverse = fmpq_mat([list(row) for row in matrix]).inv()
        size = len(matrix)
        return tuple(
            primitive_integer_vector(
                tuple(
                    Fraction(
                        int(inverse[row, column].numerator),
                        int(inverse[row, column].denominator),
                    )
                    for row in range(size)
                )
            )
            for column in range(size)
        )
    except Exception as exc:  # pragma: no cover - backend boundary
        raise PolyhedralConversionError(
            "exact polyhedral basis inversion failed"
        ) from exc


def _active_bits(rows: Sequence[Sequence[int]], vector: Sequence[int]) -> int:
    result = 0
    for index, row in enumerate(rows):
        if _dot(row, vector) == 0:
            result |= 1 << index
    return result


def _rows_from_bits(rows: Sequence[Sequence[int]], bits: int) -> list[Sequence[int]]:
    return [row for index, row in enumerate(rows) if bits & (1 << index)]


def cone_generators(rows: Sequence[Sequence[int]]) -> ConeConversion:
    """Convert ``{y: row*y >= 0}`` to rays plus exact lineality.

    The input is canonicalized by callers.  This function deliberately has no
    mid-computation budget: admission must bound all candidate pairs and output
    before invoking it.
    """

    canonical_rows = tuple(tuple(int(value) for value in row) for row in rows)
    if not canonical_rows:
        raise ValueError("a cone conversion requires at least one inequality")
    ambient = len(canonical_rows[0])
    if ambient == 0 or any(len(row) != ambient for row in canonical_rows):
        raise ValueError("cone rows must share one positive dimension")
    if any(not any(row) for row in canonical_rows):
        raise ValueError("zero cone inequalities must be removed before conversion")

    quotient_columns = _independent_columns(canonical_rows, ambient)
    quotient_dimension = len(quotient_columns)
    quotient = QuotientMap(
        ambient_dimension=ambient,
        quotient_columns=quotient_columns,
        lineality_basis=_lineality_basis(canonical_rows, ambient),
    )
    if quotient_dimension == 0:
        return ConeConversion((), quotient, len(canonical_rows), 0, 0)

    projected_rows = tuple(
        tuple(row[column] for column in quotient_columns) for row in canonical_rows
    )
    basis_indices = _independent_indices(projected_rows, quotient_dimension)
    if len(basis_indices) != quotient_dimension:
        raise PolyhedralConversionError("failed to select a full-rank quotient basis")
    remaining = tuple(
        index for index in range(len(projected_rows)) if index not in basis_indices
    )
    order = (*basis_indices, *remaining)
    ordered_rows = tuple(projected_rows[index] for index in order)

    rays = list(_inverse_columns(ordered_rows[:quotient_dimension]))
    processed = quotient_dimension
    candidate_pairs = 0
    rank_tests = 0
    active = [_active_bits(ordered_rows[:processed], ray) for ray in rays]

    for row_index in range(processed, len(ordered_rows)):
        row = ordered_rows[row_index]
        evaluations = [_dot(row, ray) for ray in rays]
        positive = [i for i, value in enumerate(evaluations) if value > 0]
        zero = [i for i, value in enumerate(evaluations) if value == 0]
        negative = [i for i, value in enumerate(evaluations) if value < 0]
        next_rays = [rays[i] for i in (*positive, *zero)]
        next_active = [
            active[i] | ((1 << row_index) if i in zero else 0)
            for i in (*positive, *zero)
        ]

        for p_index in positive:
            for n_index in negative:
                candidate_pairs += 1
                common = active[p_index] & active[n_index]
                if common.bit_count() < max(0, quotient_dimension - 2):
                    continue
                rank_tests += 1
                if _rank(
                    _rows_from_bits(ordered_rows[:row_index], common),
                    quotient_dimension,
                ) != max(0, quotient_dimension - 2):
                    continue
                p_value = evaluations[p_index]
                n_value = evaluations[n_index]
                candidate = primitive_integer_vector(
                    tuple(
                        (-n_value) * rays[p_index][axis] + p_value * rays[n_index][axis]
                        for axis in range(quotient_dimension)
                    )
                )
                if candidate in next_rays:
                    continue
                next_rays.append(candidate)
                next_active.append(
                    _active_bits(ordered_rows[: row_index + 1], candidate)
                )
        rays, active = next_rays, next_active
        processed += 1
        if not rays:
            break

    # Translate active bits back from processing order to caller row order.
    caller_rays: list[ConeRay] = []
    for ray in rays:
        ambient_ray = quotient.reconstruct(ray)
        caller_rays.append(
            ConeRay(ambient_ray, _active_bits(canonical_rows, ambient_ray))
        )
    caller_rays.sort(key=lambda item: item.vector)
    return ConeConversion(
        tuple(caller_rays), quotient, processed, candidate_pairs, rank_tests
    )


def halfspaces_to_generators(
    rows: Sequence[tuple[Sequence[object], object]], dimension: int
) -> PolyhedronConversion:
    """Convert affine inequalities ``a*x <= b`` to exact affine generators."""

    require_dd_work_admissible(len(rows) + 1, dimension + 1)
    minor_digits = require_dd_height_admissible(
        _component_digit_bound(
            [(*coefficients, offset) for coefficients, offset in rows]
        ),
        dimension,
        affine_halfspaces=True,
    )
    require_dd_weighted_work_admissible(len(rows) + 1, dimension + 1, minor_digits)
    homogeneous = [
        primitive_homogeneous_halfspace(coefficients, offset)
        for coefficients, offset in rows
    ]
    # The affine chart is t >= 0.  It distinguishes vertices from recession
    # directions and prevents the opposite representative of a point ray.
    homogeneous.append((1, *([0] * dimension)))
    cone = cone_generators(homogeneous)
    vertices: list[tuple[Fraction, ...]] = []
    incidences: list[int] = []
    recession: list[IntegerVector] = []
    for ray in cone.rays:
        t = ray.vector[0]
        if t > 0:
            point = tuple(Fraction(value, t) for value in ray.vector[1:])
            if point not in vertices:
                vertices.append(point)
                incidences.append(ray.active & ((1 << len(rows)) - 1))
        elif t == 0:
            recession.append(tuple(ray.vector[1:]))
    affine_lineality = tuple(
        tuple(vector[1:]) for vector in cone.quotient.lineality_basis
    )
    direction_rows: list[Sequence[int]] = [*recession, *affine_lineality]
    if vertices:
        base = vertices[0]
        direction_rows.extend(
            primitive_integer_vector(
                tuple(point[axis] - base[axis] for axis in range(dimension))
            )
            for point in vertices[1:]
            if point != base
        )
    affine_dimension = _rank(direction_rows, dimension) if vertices else -1
    return PolyhedronConversion(
        tuple(vertices),
        tuple(incidences),
        tuple(recession),
        affine_lineality,
        not vertices,
        bool(vertices) and not recession and not affine_lineality,
        affine_dimension,
        cone,
    )


def points_to_facets(
    points: Sequence[Sequence[object]],
    dimension: int,
    *,
    max_facets: int | None = None,
) -> HullConversion:
    """Convert affine generators to primitive facets through homogeneous duality."""

    minor_digits = require_dd_height_admissible(
        _component_digit_bound(points), dimension, affine_halfspaces=False
    )
    homogeneous = tuple(primitive_homogeneous_point(point) for point in points)
    box = _axis_aligned_box_hull(points, dimension, homogeneous)
    if box is not None:
        return box
    simplex = _containing_simplex_hull(points, dimension, homogeneous)
    if simplex is not None:
        return simplex
    facet_bound = upper_bound_facets(len(homogeneous), dimension)
    if max_facets is not None and facet_bound > max_facets:
        raise ValueError(
            "facet conversion exceeds the output bound "
            f"({facet_bound} possible facets > {max_facets})"
        )
    require_dd_weighted_work_admissible(len(homogeneous), dimension + 1, minor_digits)
    require_dd_work_admissible(len(homogeneous), dimension + 1)
    cone = cone_generators(homogeneous)
    facets: list[tuple[IntegerVector, int]] = []
    incidences: list[int] = []
    for ray in cone.rays:
        # h=(b,-a), with h*(1,x)>=0, decodes to a*x<=b.
        normal = tuple(-value for value in ray.vector[1:])
        offset = ray.vector[0]
        facets.append((normal, offset))
        incidences.append(ray.active)
    equalities: list[tuple[IntegerVector, int]] = []
    for row in cone.quotient.lineality_basis:
        equation = _canonical_equation_vector((*(-value for value in row[1:]), row[0]))
        equalities.append((equation[:-1], equation[-1]))
    return HullConversion(tuple(facets), tuple(incidences), tuple(equalities), cone)


def _axis_aligned_box_hull(
    points: Sequence[Sequence[object]],
    dimension: int,
    homogeneous: Sequence[IntegerVector],
) -> HullConversion | None:
    """Return the exact product hull for a complete two-level Cartesian box."""

    if len(points) != 1 << dimension:
        return None
    rational_points = tuple(
        tuple(_fraction(value) for value in point) for point in points
    )
    levels = tuple(
        tuple(sorted({point[axis] for point in rational_points}))
        for axis in range(dimension)
    )
    if any(len(axis_levels) != 2 for axis_levels in levels):
        return None
    expected = {tuple(choice) for choice in product(*levels)}
    if set(rational_points) != expected:
        return None

    facets: list[tuple[IntegerVector, int]] = []
    incidences: list[int] = []
    dual_rays: list[ConeRay] = []
    for axis, (low, high) in enumerate(levels):
        for coefficient, offset in ((-1, -low), (1, high)):
            normal = [0] * dimension
            normal[axis] = coefficient
            primitive = primitive_integer_vector((*normal, offset))
            facet_normal = primitive[:-1]
            facet_offset = primitive[-1]
            bits = 0
            for index, point in enumerate(rational_points):
                if (
                    sum(a * x for a, x in zip(facet_normal, point, strict=True))
                    == facet_offset
                ):
                    bits |= 1 << index
            facets.append((facet_normal, facet_offset))
            incidences.append(bits)
            dual = primitive_homogeneous_halfspace(facet_normal, facet_offset)
            dual_rays.append(ConeRay(dual, _active_bits(homogeneous, dual)))
    quotient = QuotientMap(dimension + 1, tuple(range(dimension + 1)), ())
    cone = ConeConversion(tuple(dual_rays), quotient, len(points), 0, 0)
    return HullConversion(tuple(facets), tuple(incidences), (), cone)


def _containing_simplex_hull(
    points: Sequence[Sequence[object]],
    dimension: int,
    homogeneous: Sequence[IntegerVector],
) -> HullConversion | None:
    """Recognize an affine basis whose simplex contains all other rows."""

    basis_indices = _independent_indices(homogeneous, dimension + 1)
    if len(basis_indices) != dimension + 1:
        return None
    basis_rows = tuple(homogeneous[index] for index in basis_indices)
    dual = cone_generators(basis_rows)
    decoded: list[tuple[IntegerVector, int]] = []
    incidences: list[int] = []
    dual_rays: list[ConeRay] = []
    for ray in dual.rays:
        normal = tuple(-value for value in ray.vector[1:])
        offset = ray.vector[0]
        if any(_dot(ray.vector, source) < 0 for source in homogeneous):
            return None
        bits = _active_bits(homogeneous, ray.vector)
        decoded.append((normal, offset))
        incidences.append(bits)
        dual_rays.append(ConeRay(ray.vector, bits))
    if len(decoded) != dimension + 1:
        return None
    cone = ConeConversion(
        tuple(dual_rays),
        dual.quotient,
        len(homogeneous),
        dual.candidate_pairs,
        dual.rank_tests,
    )
    return HullConversion(tuple(decoded), tuple(incidences), (), cone)


__all__ = [
    "MAX_DD_COEFFICIENT_DIGITS",
    "MAX_DD_PAIR_BOUND",
    "MAX_DD_RAY_BOUND",
    "MAX_DD_WEIGHTED_HEIGHT_WORK",
    "MAX_PULLING_SIMPLEX_BOUND",
    "ConeConversion",
    "ConeRay",
    "HullConversion",
    "PolyhedralConversionError",
    "PolyhedronConversion",
    "QuotientMap",
    "cone_generators",
    "dd_work_bound",
    "halfspaces_to_generators",
    "maximal_incidence_faces",
    "points_to_facets",
    "primitive_homogeneous_halfspace",
    "primitive_homogeneous_point",
    "primitive_integer_vector",
    "pulling_triangulation",
    "require_dd_height_admissible",
    "require_dd_weighted_work_admissible",
    "require_dd_work_admissible",
    "require_pulling_work_admissible",
    "upper_bound_facets",
]
