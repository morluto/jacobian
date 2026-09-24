"""Domain-owned exact rational polytope operations.

H/V conversion uses the private homogeneous primitive-integer
double-description kernel.  Its exact incidence bitsets drive pulling
triangulation, and FLINT supplies rank, nullspace, inverse, and determinant
arithmetic.  Cartesian boxes and containing simplices have exact structured
presolves; all other conversions use theorem-backed output-sensitive work
bounds before expansion.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations
from typing import Literal

from pydantic_core import PydanticCustomError
from sympy import Matrix, Rational

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.polytopes._models import (
    COORDINATE_DIGITS,
    MAX_COMPUTED_FACETS,
    MAX_COORDINATE_LABEL_LENGTH,
    MAX_DIMENSION,
    MAX_EXTREMALITY_HEIGHT_WORK,
    MAX_FACET_COORDINATE_DIGITS,
    MAX_FACET_DIMENSION,
    MAX_FACET_INCIDENCES,
    MAX_POLYTOPE_FACE_LATTICE_COVERS,
    MAX_POLYTOPE_FACE_LATTICE_DIMENSION,
    MAX_POLYTOPE_FACE_LATTICE_FACES,
    MAX_POLYTOPE_FACE_LATTICE_RESULT_CHARS,
    MAX_POLYTOPE_FACE_LATTICE_WORK,
    MAX_VERTICES,
    EdgeProfileResult,
    FacetIncidenceResult,
    JoinResult,
    JoinVertexMap,
    PolytopeAdmissionError,
    PolytopeEdge,
    PolytopeFace,
    PolytopeFaceCover,
    PolytopeFaceLatticeResult,
    PolytopeSupportResult,
    PolytopeVolumeResult,
    PrimitiveFacet,
    PrismResult,
    PrismVertexMap,
    PyramidBaseVertexMap,
    PyramidResult,
    RationalCoordinateSpace,
    RationalCovector,
    RationalPolytopeVertex,
    RationalVPolytope,
    VertexFigurePolytope,
    VertexFigureResult,
    VertexFigureVertexMap,
    _canonical_v_polytope_vertices,
    _prepare_volume_components,
    _validate_halfspaces,
    _validate_vertices,
)
from jacobian.math.geometry.polytopes._polyhedral_conversion import (
    PolyhedralConversionAdmissionError,
    dd_work_bound,
    points_to_facets,
    rational_rank,
)
from jacobian.math.geometry.polytopes._rational_geometry import (
    determinant_sign,
    recession_cone_is_trivial,
    vertices_from_halfspaces,
)
from jacobian.math.geometry.polytopes.values import (
    MAX_RATIONAL_POLYTOPE_DIMENSION,
    Halfspace,
    Vertex,
)


def _deduplicate_source_rows(points: list[list[Rational]]) -> list[list[Rational]]:
    """Drop repeated source rows, preserving first-seen order."""

    seen: set[tuple[Rational, ...]] = set()
    unique: list[list[Rational]] = []
    for point in points:
        key = tuple(point)
        if key not in seen:
            seen.add(key)
            unique.append(point)
    return unique


def _require_facet_preflight(vertices: tuple[Vertex, ...], dim: int) -> None:
    """Prove the exact V-to-facet enumeration is admitted before it starts."""

    points = [
        [Rational(*coordinate.as_integer_ratio()) for coordinate in vertex.coordinates]
        for vertex in vertices
    ]
    if dim == 1:
        if len({point[0] for point in points}) < 2:
            raise ValueError(
                "V-representation is not full-dimensional; lower-dimensional hulls "
                "require intrinsic affine coordinates"
            )
    else:
        differences = [
            [points[index][axis] - points[0][axis] for axis in range(dim)]
            for index in range(1, len(points))
        ]
        if rational_rank(differences, dim) < dim:
            raise ValueError(
                "V-representation is not full-dimensional; lower-dimensional hulls "
                "require intrinsic affine coordinates"
            )


def _primitive_facet_key(
    normal: Matrix, offset: Rational, dim: int
) -> tuple[tuple[int, ...], int]:
    """Normalize an oriented rational supporting inequality to primitive integers."""

    values = [Rational(normal[index]) for index in range(dim)] + [Rational(offset)]
    scale = 1
    for value in values:
        scale = math.lcm(scale, int(value.q))
    integers = [int(value * scale) for value in values]
    divisor = 0
    for value in integers:
        divisor = math.gcd(divisor, abs(value))
    if divisor == 0:
        raise ValueError("facet normal must not be zero")
    coefficients = tuple(value // divisor for value in integers[:-1])
    return coefficients, integers[-1] // divisor


def _primitive_halfspace(coefficients: tuple[int, ...], rhs: int) -> Halfspace:
    """Wrap a primitive integer supporting inequality in the shared value."""

    return Halfspace(
        coefficients=tuple(
            CanonicalRational.from_integer_ratio(value, 1) for value in coefficients
        ),
        offset=CanonicalRational.from_integer_ratio(rhs, 1),
    )


def _computed_facets_from_vertices(
    vertices: tuple[Vertex, ...], dim: int
) -> tuple[PrimitiveFacet, ...]:
    """Return every canonical supporting facet and complete source incidence.

    The pinned SymPy backend supplies exact nullspaces and rational arithmetic.
    This owner adapter enumerates the finite candidate family over the
    distinct source rows -- duplicates create no candidate hyperplanes --
    canonicalizes every oriented supporting row, and binds it to all equal
    source rows in the original ordered V-representation.
    """

    _require_facet_preflight(vertices, dim)
    points = [
        [Rational(*coordinate.as_integer_ratio()) for coordinate in vertex.coordinates]
        for vertex in vertices
    ]
    distinct_points = _deduplicate_source_rows(points)
    conversion = points_to_facets(distinct_points, dim, max_facets=MAX_COMPUTED_FACETS)
    canonical: dict[tuple[tuple[int, ...], int], PrimitiveFacet] = {}
    for coefficients, rhs in conversion.facets:
        incidence = tuple(
            index
            for index, point in enumerate(points)
            if sum(
                Rational(coefficient) * point[axis]
                for axis, coefficient in enumerate(coefficients)
            )
            == rhs
        )
        key = coefficients, rhs
        canonical[key] = PrimitiveFacet(
            halfspace=_primitive_halfspace(coefficients, rhs),
            source_vertex_indices=incidence,
        )
        if len(canonical) > MAX_COMPUTED_FACETS:
            raise ValueError(
                f"facet profile exceeds the {MAX_COMPUTED_FACETS}-facet result bound"
            )
    facets = tuple(canonical[key] for key in sorted(canonical))
    incidence_count = sum(len(facet.source_vertex_indices) for facet in facets)
    if incidence_count > MAX_FACET_INCIDENCES:
        raise ValueError(
            f"facet profile exceeds the {MAX_FACET_INCIDENCES}-incidence result bound"
        )
    return facets


def facet_incidence(
    vertices: tuple[Vertex, ...], dimension_bound: int
) -> FacetIncidenceResult:
    """Compute the complete canonical facet-incidence profile of ``conv(V)``."""

    dimension = len(vertices[0].coordinates)
    try:
        if dimension > dimension_bound:
            raise ValueError(
                f"dimension {dimension} exceeds the dimension bound {dimension_bound}"
            )
        if any(len(vertex.coordinates) != dimension for vertex in vertices):
            raise ValueError("all vertices must share one dimension")
        for vertex in vertices:
            for coordinate in vertex.coordinates:
                require_bounded_rational(
                    coordinate,
                    max_digits=MAX_FACET_COORDINATE_DIGITS,
                    label="facet-profile vertex coordinate",
                )
        facets = _computed_facets_from_vertices(vertices, dimension)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("vertices",),
            code="polytope.facet_profile_not_admitted",
            message=str(exc),
        ) from exc
    return FacetIncidenceResult._from_kernel(
        vertices=vertices,
        dimension=dimension,
        facets=facets,
    )


def _deduplicate_halfspaces(
    halfspaces: tuple[Halfspace, ...],
) -> tuple[Halfspace, ...]:
    """Drop rows that repeat an earlier half-space up to a positive scale.

    Two inequalities ``<a, x> <= b`` and ``<a', x> <= b'`` impose the
    identical constraint exactly when ``(a', b') = lambda * (a, b)`` for
    some ``lambda > 0``. Each row is normalized to that primitive form --
    coefficients cleared to coprime integers by the positive factor
    ``lcm(denominators)/gcd(numerators)`` with the sign kept -- and the
    offset scaled by the same factor is compared as an exact rational, so
    repeated or positively rescaled copies collapse onto their first
    occurrence without reordering the remaining rows. Sign-flipped rows
    impose a different inequality and are never merged.
    """

    seen: set[tuple[tuple[int, ...], tuple[int, int]]] = set()
    unique: list[Halfspace] = []
    for hs in halfspaces:
        fracs = [Fraction(*c.as_integer_ratio()) for c in hs.coefficients]
        offset = Fraction(*hs.offset.as_integer_ratio())
        lcm = 1
        for frac in (*fracs, offset):
            lcm = lcm * frac.denominator // math.gcd(lcm, frac.denominator)
        ints = [int(frac * lcm) for frac in fracs]
        rhs = int(offset * lcm)
        g = 0
        for value in (*ints, rhs):
            g = math.gcd(g, abs(value))
        key = (
            tuple(value // g for value in ints),
            (rhs // g, 1),
        )
        if key not in seen:
            seen.add(key)
            unique.append(hs)
    return tuple(unique)


def _is_bounded_h(halfspaces: tuple[Halfspace, ...]) -> bool:
    """Decide whether ``{x : A x <= b}`` is bounded.

    The polytope is bounded iff its recession cone ``{y : A y <= 0}`` is
    ``{0}``, which holds iff the origin lies strictly in the interior of
    the convex hull of the rows of ``A``. That interior test is an exact
    facet enumeration of the row normals.  The rows must positively span
    ``R^d``; equivalently their convex hull must be full-dimensional.
    Redundant rows -- exact copies or positive rescalings of an earlier
    half-space -- are removed first, so duplicated constraints neither
    change the decision nor inflate the combinatorial work estimate.
    """

    dim = len(halfspaces[0].coefficients)
    halfspaces = _deduplicate_halfspaces(halfspaces)
    normals: list[list[Rational]] = [
        [Rational(*c.as_integer_ratio()) for c in hs.coefficients] for hs in halfspaces
    ]
    return recession_cone_is_trivial(normals, dim)


def _canonical_rational(value: Rational) -> CanonicalRational:
    """Convert a SymPy ``Rational`` to the canonical value type."""

    return CanonicalRational.from_integer_ratio(int(value.p), int(value.q))


def _simplex_abs_det(simplex_points: list[list[Rational]]) -> Rational:
    """Absolute determinant of ``[p_1 - p_0, ..., p_d - p_0]``.

    The simplex volume is this determinant divided by ``d!``.
    """
    d = len(simplex_points) - 1
    if d == 0:
        return Rational(1)
    from flint import fmpq, fmpq_mat

    v0 = simplex_points[0]
    matrix = fmpq_mat(
        [
            [
                fmpq(
                    int((simplex_points[row + 1][axis] - v0[axis]).p),
                    int((simplex_points[row + 1][axis] - v0[axis]).q),
                )
                for axis in range(d)
            ]
            for row in range(d)
        ]
    )
    determinant = abs(matrix.det())
    return Rational(int(determinant.numerator), int(determinant.denominator))


def _rank_of_diffs(points: list[list[Rational]], dim: int) -> int:
    """Rank of the matrix of ``point - point[0]`` differences in ``dim`` dims."""
    if len(points) <= 1:
        return 0
    v0 = points[0]
    differences = [
        [points[i][k] - v0[k] for k in range(dim)] for i in range(1, len(points))
    ]
    return rational_rank(differences, dim)


def _hull_subfacets(points: list[list[Rational]], dim: int) -> list[tuple[int, ...]]:
    """Enumerate the dim-subsets of points on the convex hull boundary.

    A dim-subset is a (d-1)-subfacet if all remaining points lie on one
    side (or on) the hyperplane it spans. Subfacets of a coplanar larger
    facet are returned individually; merge with ``_max_facets``.
    """
    n = len(points)
    subfacets: list[tuple[int, ...]] = []
    for subset in combinations(range(n), dim):
        signs: set[int] = set()
        ok = True
        for p in range(n):
            if p in subset:
                continue
            sign = determinant_sign(
                [[points[i][k] for k in range(dim)] + [1] for i in subset]
                + [[points[p][k] for k in range(dim)] + [1]]
            )
            if sign > 0:
                signs.add(1)
            elif sign < 0:
                signs.add(-1)
            if len(signs) > 1:
                ok = False
                break
        if ok and signs:
            subfacets.append(tuple(subset))
    return subfacets


def _plane_signature(
    subfacet: tuple[int, ...], points: list[list[Rational]]
) -> tuple[int, ...] | None:
    """Canonical signature of the hyperplane through the subfacet points.

    Returns a reduced integer tuple ``(a_1, ..., a_d, b)`` (up to positive
    scaling) so that coplanar subfacets share one signature.
    """
    dim = len(subfacet)
    mat = Matrix([[points[i][k] for k in range(dim)] + [1] for i in subfacet])
    nullspace = mat.nullspace()
    if not nullspace:
        return None
    vec = [Rational(nullspace[0][j]) for j in range(dim + 1)]
    first_nonzero = next(j for j in range(dim + 1) if vec[j] != 0)
    sign = 1 if vec[first_nonzero] > 0 else -1
    # Clear denominators before integer reduction so fractional
    # coefficients are not truncated to zero.
    denominators = [v.denominator for v in vec]
    lcm = 1
    for d in denominators:
        lcm = lcm * d // math.gcd(lcm, d)
    scaled = [int(v * sign * lcm) for v in vec]
    g = 0
    for x in scaled:
        g = math.gcd(g, abs(x))
    if g == 0:
        g = 1
    return tuple(x // g for x in scaled)


def _max_facets(points: list[list[Rational]], dim: int) -> list[list[int]]:
    """Return the maximal (d-1)-facets, each a sorted list of point indices."""
    subfacets = _hull_subfacets(points, dim)
    groups: dict[tuple[int, ...], set[int]] = {}
    for subfacet in subfacets:
        sig = _plane_signature(subfacet, points)
        if sig is None:
            continue
        groups.setdefault(sig, set()).update(subfacet)
    return [sorted(members) for members in groups.values()]


def _extreme_point_indices(
    groups: dict[tuple[int, ...], set[int]],
    point_count: int,
    dim: int,
) -> tuple[list[int], list[int]]:
    """Return (extreme indices, boundary counts) from grouped maximal facets.

    A point is extreme when the normals of its containing facets span the
    ambient space; every group member lies exactly on that facet's plane.
    """
    counts = [0] * point_count
    active_normals: list[list[list[Rational]]] = [[] for _ in range(point_count)]
    for sig, members in groups.items():
        normal = list(sig[:-1])
        for idx in members:
            if 0 <= idx < point_count:
                counts[idx] += 1
                active_normals[idx].append(normal)
    kept = [
        i
        for i in range(point_count)
        if active_normals[i] and rational_rank(active_normals[i], dim) == dim
    ]
    return kept, counts


def _filter_redundant_vertices(
    points: list[list[Rational]], dim: int
) -> list[list[Rational]]:
    """Return the extreme hull vertices, dropping redundant boundary points.

    A point of the polytope is a vertex exactly when the normals of the
    maximal facets containing it span the ambient space (the active-
    constraint rank test).  Counting incident facets is not enough: a
    non-extreme point on a lower-dimensional face of a nonsimple polytope
    can lie on many facets whose normals are rank-deficient -- e.g. the
    midpoint of a vertical edge of ``conv(+/-e1,+/-e2,+/-e3)x[0,1]`` lies
    on four facets yet spans only rank 3 in dimension 4.  This prevents
    the 2-D adjacency graph from becoming non-simple (e.g., a 3x3 square
    with all 12 boundary integer points would otherwise give every node
    degree >2 and cause triangulation to fail).
    """

    if len(points) <= dim:
        return points
    hull = points_to_facets(points, dim)
    active_normals: list[list[list[Rational]]] = [[] for _ in points]
    for (normal, _offset), incidence in zip(
        hull.facets, hull.facet_incidence, strict=True
    ):
        for index in range(len(points)):
            if incidence & (1 << index):
                active_normals[index].append([Rational(value) for value in normal])
    keep_indices = [
        index
        for index, normals in enumerate(active_normals)
        if normals and rational_rank(normals, dim) == dim
    ]
    if len(keep_indices) < dim + 1:
        return points
    keep_set = set(keep_indices)
    return [pt for i, pt in enumerate(points) if i in keep_set]


def _support_sympy_points(polytope: RationalVPolytope) -> list[list[Rational]]:
    """Encode canonical V-vertices for the existing exact hull primitives."""

    return [
        [Rational(*coordinate.as_integer_ratio()) for coordinate in vertex.coordinates]
        for vertex in polytope.vertices
    ]


def require_full_dimensional_extreme_vertices(polytope: RationalVPolytope) -> None:
    """Prove the V-representation is full-dimensional and irredundant.

    The support operation uses a direct finite maximum only after the value
    has established that its labelled generators are exactly the polytope's
    vertices. The shared DD facet conversion gives the latter proof by the
    active-normal rank characterization of extreme vertices. Admission couples
    the theorem-backed candidate-pair bound, ambient dimension, and reduced
    component height before conversion; structured hull presolves then retain
    exact incidences for the rank test.
    """

    dimension = len(polytope.space.axes)
    vertex_count = len(polytope.vertices)
    _ray_bound, pair_bound = dd_work_bound(vertex_count, dimension + 1)
    extremality_tests = pair_bound + vertex_count + 1
    component_digits = max(
        max(
            len(format_canonical_integer(abs(coordinate.num))),
            len(format_canonical_integer(abs(coordinate.den))),
        )
        for vertex in polytope.vertices
        for coordinate in vertex.coordinates
    )
    height_work = extremality_tests * (dimension + 1) * component_digits**2
    if height_work > MAX_EXTREMALITY_HEIGHT_WORK:
        raise ValueError(
            "V-polytope extremality proof exceeds the height-work bound "
            f"({height_work} > {MAX_EXTREMALITY_HEIGHT_WORK})"
        )
    points = _support_sympy_points(polytope)
    differences = [
        [point[coordinate] - points[0][coordinate] for coordinate in range(dimension)]
        for point in points[1:]
    ]
    if rational_rank(differences, dimension) != dimension:
        raise ValueError("V-polytope vertices must affinely span the coordinate space")
    if len(_filter_redundant_vertices(points, dimension)) != len(points):
        raise ValueError("V-polytope vertices must all be exact extreme vertices")


def support_data(
    polytope: RationalVPolytope,
    covector: RationalCovector,
) -> tuple[Fraction, tuple[RationalPolytopeVertex, ...]]:
    """Return the exact support value and complete maximizing vertex family."""

    values = tuple(
        sum(
            (
                coordinate.as_fraction() * component.as_fraction()
                for coordinate, component in zip(
                    vertex.coordinates, covector.components, strict=True
                )
            ),
            Fraction(0),
        )
        for vertex in polytope.vertices
    )
    maximum = max(values)
    return maximum, tuple(
        vertex
        for vertex, value in zip(polytope.vertices, values, strict=True)
        if value == maximum
    )


def _project_facet(
    facet_points: list[list[Rational]], dim: int
) -> list[list[Rational]]:
    """Project a coplanar dim-dim facet into (dim-1)-dim coordinates.

    Drops the first axis whose projection keeps the facet full
    (dim-1)-dimensional rank.
    """
    for axis in range(dim):
        projected = [[pt[k] for k in range(dim) if k != axis] for pt in facet_points]
        if _rank_of_diffs(projected, dim - 1) == dim - 1:
            return projected
    return [[pt[k] for k in range(dim - 1)] for pt in facet_points]


def _triangulate_2d(points: list[list[Rational]]) -> list[tuple[int, ...]]:
    """Triangulate a 2D convex polygon by a fan from its first corner."""
    subfacets = _hull_subfacets(points, 2)
    adjacency: dict[int, set[int]] = {}
    for edge in subfacets:
        adjacency.setdefault(edge[0], set()).add(edge[1])
        adjacency.setdefault(edge[1], set()).add(edge[0])
    corners = [i for i in adjacency if len(adjacency[i]) == 2]
    if not corners:
        return []
    start = corners[0]
    order = [start]
    prev = -1
    cur = start
    while True:
        neighbors = [x for x in adjacency[cur] if x != prev]
        if not neighbors:
            break
        nxt = neighbors[0]
        if nxt == start:
            break
        order.append(nxt)
        prev, cur = cur, nxt
        if len(order) > len(corners) + 1:
            break
    return [(order[0], order[i], order[i + 1]) for i in range(1, len(order) - 1)]


def _triangulate(points: list[list[Rational]], dim: int) -> list[tuple[int, ...]]:
    """Return a triangulation of the convex hull as (dim+1)-tuples of indices.

    Recursive fan from a fixed apex: pick an extreme vertex ``apex`` (an
    extreme point of the hull), enumerate the maximal (d-1)-facets that do
    NOT contain the apex (the ``opposite`` facets), project and triangulate
    each such facet recursively, and cone each (d-1)-simplex to the apex.
    The cones from one apex tile the convex hull exactly because a convex
    polytope is star-shaped from any of its extreme vertices.
    """
    n = len(points)
    if n < dim + 1:
        return []
    if dim == 1:
        coords = sorted({p[0] for p in points})
        if len(coords) < 2:
            return []
        mn = min(range(n), key=lambda i: points[i][0])
        mx = max(range(n), key=lambda i: points[i][0])
        return [(mn, mx)]
    if dim == 2:
        return _triangulate_2d(points)
    # Pick an extreme apex: a vertex that is NOT in the convex hull of the
    # others (i.e., not interior). The lowest-indexed extreme vertex works.
    apex = _extreme_vertex(points, dim)
    if apex is None:
        return []
    facets = _max_facets(points, dim)
    triangulation: list[tuple[int, ...]] = []
    for members in facets:
        if apex in members:
            continue  # only the opposite facets define the fan from apex
        facet_points = [points[i] for i in members]
        projected = _project_facet(facet_points, dim)
        facet_triangulation = _triangulate(projected, dim - 1)
        for tri in facet_triangulation:
            triangulation.append((*tuple(members[i] for i in tri), apex))
    return triangulation


def _extreme_vertex(points: list[list[Rational]], dim: int) -> int | None:
    """Return the index of an extreme vertex of the convex hull.

    A vertex is extreme if it is not a convex combination of the others;
    equivalently, removing it changes the affine hull dimension. We pick
    the lowest-indexed vertex that lies on a hull (d-1)-facet.
    """
    subfacets = _hull_subfacets(points, dim)
    if not subfacets:
        return None
    on_hull: set[int] = set()
    for subfacet in subfacets:
        on_hull.update(subfacet)
    for i in range(len(points)):
        if i in on_hull:
            return i
    return None


def _polytope_volume(points: list[list[Rational]], dim: int) -> Rational:
    """Exact volume of the convex hull of ``points`` in ``dim`` dimensions."""
    prepared, triangulation = _prepare_volume_components(points, dim)
    return _polytope_volume_from_prepared(prepared, dim, triangulation)


def _polytope_volume_from_prepared(
    points: list[list[Rational]],
    dim: int,
    triangulation: list[tuple[int, ...]],
) -> Rational:
    """Compute volume from the hull data retained by request admission."""

    if len(points) < dim + 1:
        return Rational(0)
    if dim == 1:
        coordinates = sorted({point[0] for point in points})
        return (
            coordinates[-1] - coordinates[0] if len(coordinates) >= 2 else Rational(0)
        )
    if not triangulation:
        return Rational(0)
    volume = Rational(0)
    for simplex in triangulation:
        volume += _simplex_abs_det([points[i] for i in simplex])
    return volume / math.factorial(dim)


def _vertices_from_v_representation(
    vertices: tuple[Vertex, ...],
) -> tuple[tuple[Rational, ...], int]:
    """Return the ambient dimension and rational coordinates from a V-rep."""
    dim = len(vertices[0].coordinates)
    points: tuple[tuple[Rational, ...], ...] = ()
    for vertex in vertices:
        coord_fracs = tuple(c.as_fraction() for c in vertex.coordinates)
        points += (tuple(Rational(f.numerator, f.denominator) for f in coord_fracs),)
    return points, dim


def _halfspace_rows(
    halfspaces: tuple[Halfspace, ...],
) -> list[tuple[list[Rational], Rational]]:
    """Convert half-spaces to a list of (coefficients, offset) rational rows."""

    # Use ``as_integer_ratio`` to avoid CPython's 4_300-digit string-to-int
    # limit for canonical rationals up to 32_768 digits (the operation's
    # admitted domain).  Passing ints to ``Rational`` bypasses SymPy's
    # string parsing path.
    return [
        (
            [Rational(*c.as_integer_ratio()) for c in hs.coefficients],
            Rational(*hs.offset.as_integer_ratio()),
        )
        for hs in halfspaces
    ]


def _vertices_from_h_representation(
    halfspaces: tuple[Halfspace, ...],
) -> tuple[list[tuple[Rational, ...]], int]:
    """Enumerate the vertices of an H-representation exactly.

    Each half-space ``<a_i, x> <= b_i`` contributes one row to the
    inequality system ``A x <= b``. A vertex is the unique intersection
    of ``dim`` affinely independent half-space boundaries (the hyperplanes
    ``<a_i, x> = b_i``), provided it satisfies every remaining
    half-space. Solving every ``C(m, dim)`` subsystem of ``dim``
    half-spaces is a bounded, exact vertex enumeration for the small
    dimensions this operation admits; duplicate rows (identical up to a
    common positive factor) are removed first because they only add
    singular subsystems, so the enumeration counts each distinct
    constraint once.
    """
    dim = len(halfspaces[0].coefficients)
    halfspaces = _deduplicate_halfspaces(halfspaces)
    rows = _halfspace_rows(halfspaces)
    result: tuple[list[tuple[Rational, ...]], int] = (
        vertices_from_halfspaces(rows, dim),
        dim,
    )
    return result


def polytope_support(
    polytope: RationalVPolytope,
    covector: RationalCovector,
) -> PolytopeSupportResult:
    """Domain kernel for one exact support value and exposed vertex face."""

    if polytope.space != covector.space:
        raise ValueError("polytope and covector must use the same coordinate space")
    # Native callers bypass wire-request validation, so apply the same
    # operation-local component admission before exact hull work.
    from jacobian.math.geometry.polytopes._models import (
        require_support_components_within_envelope,
    )

    require_support_components_within_envelope(polytope, covector)
    require_full_dimensional_extreme_vertices(polytope)
    value, vertices = support_data(polytope, covector)
    from jacobian.math.geometry.polytopes._models import RationalExposedFace

    return PolytopeSupportResult._from_kernel(
        polytope=polytope,
        covector=covector,
        support_value=CanonicalRational.from_fraction(value),
        exposed_face=RationalExposedFace(
            space=polytope.space,
            vertices=vertices,
        ),
    )


def polytope_volume(
    vertices: tuple[Vertex, ...] | None,
    halfspaces: tuple[Halfspace, ...] | None,
    dimension_bound: int,
) -> PolytopeVolumeResult:
    """Compute the exact rational volume of a bounded rational polytope.

    The input is a polytope in exactly one representation:

    * ``vertices``: the V-representation. The convex hull facets are
      enumerated exactly, each facet is projected and triangulated
      recursively, and the simplex volumes (SymPy exact determinants)
      are summed.
    * ``halfspaces``: the H-representation. The vertices are enumerated
      exactly by solving every ``C(m, dim)`` subsystem of half-spaces,
      retaining the feasible intersections, and then the volume is
      computed as for the V-representation.

    The volume is exact rational; no floating-point approximation is used.
    """
    representation: Literal["vertices", "halfspaces"]
    location: tuple[str, ...]
    if (vertices is None) == (halfspaces is None):
        raise OperationDomainValidationError(
            location=("vertices", "halfspaces"),
            code="polytope.volume.exactly_one_representation",
            message="provide exactly one of vertices or halfspaces",
        )
    try:
        if vertices is not None:
            representation = "vertices"
            location = ("vertices",)
            prepared, dim, triangulation = _validate_vertices(vertices, dimension_bound)
        else:
            if halfspaces is None:
                raise RuntimeError("polytope volume has no source representation")
            representation = "halfspaces"
            location = ("halfspaces",)
            prepared, dim, triangulation = _validate_halfspaces(
                halfspaces, dimension_bound
            )
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=location,
            code=exc.type,
            message=str(exc),
        ) from exc
    except (PolytopeAdmissionError, ValueError) as exc:
        reason = exc.reason if isinstance(exc, PolytopeAdmissionError) else "admission"
        raise OperationDomainValidationError(
            location=location,
            code=f"polytope.volume.{reason}",
            message=str(exc),
        ) from exc
    volume = _polytope_volume_from_prepared(prepared, dim, triangulation)
    value = _canonical_rational(volume)
    return PolytopeVolumeResult(
        volume=value,
        dimension=dim,
        representation=representation,
    )


def convex_hull_volume(
    vertices: RationalVPolytope | tuple[Sequence[Fraction], ...],
) -> CanonicalRational:
    """Return the exact rational volume of the convex hull of rational points.

    This is the native domain kernel: it accepts mathematical values — a
    tuple of rational coordinate tuples in a consistent ambient dimension
    (at least one point) — and returns the canonical exact volume.  Degenerate
    inputs of fewer than ``dim + 1``
    distinct affinely independent points have exact volume zero.  Raises
    ``ValueError`` when the hull enumeration exceeds the combinatorial work
    bound or the input does not describe one consistent dimension.
    """

    if isinstance(vertices, RationalVPolytope):
        normalized = tuple(
            tuple(
                Fraction(*coordinate.as_integer_ratio())
                for coordinate in vertex.coordinates
            )
            for vertex in _canonical_v_polytope_vertices(vertices)
        )
    else:
        if not vertices:
            raise ValueError("`vertices` must be non-empty")
        if any(len(vertex) != len(vertices[0]) for vertex in vertices):
            raise ValueError("all vertices must share one dimension")
        normalized = tuple(
            tuple(Fraction(coordinate) for coordinate in vertex) for vertex in vertices
        )

    dim = len(normalized[0])
    if not 1 <= dim <= MAX_DIMENSION:
        raise ValueError(
            f"ambient dimension {dim} exceeds the {MAX_DIMENSION}-dimension bound"
        )
    if len(normalized) > MAX_VERTICES:
        raise ValueError(f"`vertices` exceeds the {MAX_VERTICES}-vertex bound")
    for vertex in normalized:
        for coordinate in vertex:
            numerator_digits = len(format_canonical_integer(abs(coordinate.numerator)))
            denominator_digits = len(format_canonical_integer(coordinate.denominator))
            if max(numerator_digits, denominator_digits) > COORDINATE_DIGITS:
                raise ValueError(
                    f"vertex coordinate exceeds the {COORDINATE_DIGITS}-digit bound"
                )

    prepared, triangulation = _prepare_volume_components(normalized, dim)
    exact_points = [
        [Rational(value.numerator, value.denominator) for value in point]
        for point in prepared
    ]
    return _canonical_rational(
        _polytope_volume_from_prepared(exact_points, dim, triangulation)
    )


def verify_primitive_facet(claim: PrimitiveFacet) -> bool:
    """Check nonzero primitive integer normalization of at most eight scalars."""
    if all(value.num == 0 for value in claim.halfspace.coefficients):
        return False
    entries = (*claim.halfspace.coefficients, claim.halfspace.offset)
    if any(value.den != 1 for value in entries):
        return False
    divisor = 0
    for value in entries:
        divisor = math.gcd(divisor, abs(value.num))
    return divisor == 1


def verify_facet_incidence(claim: FacetIncidenceResult) -> bool:
    """Check the complete normalized facet relation against its bounded source."""
    return facet_incidence(claim.vertices, claim.dimension) == claim


PYRAMID_APEX_VERTEX_ID = "apex"
"""Reserved vertex ID of the pyramid apex.

Base vertices retain their source IDs unchanged, so a source polytope
already using this label is outside the admitted domain; the caller
relabels that source vertex first.
"""


def _affine_dimension(points: list[list[Rational]]) -> int:
    """Exact affine dimension of a finite rational point family."""

    if len(points) <= 1:
        return 0
    reference = points[0]
    dim = len(reference)
    differences = [
        [point[axis] - reference[axis] for axis in range(dim)] for point in points[1:]
    ]
    return rational_rank(differences, dim)


def _admit_pyramid(polytope: RationalVPolytope, height_axis: object) -> int:
    """Enforce the pyramid execution envelope shared by native and catalog calls.

    Returns the source ambient dimension. Structural label conflicts raise
    ``OperationDomainValidationError``; envelope overflows raise
    ``OperationResourceAdmissionError``.
    """

    if not isinstance(height_axis, str) or not height_axis:
        raise OperationDomainValidationError(
            location=("height_axis",),
            code="polytope.pyramid.height_axis_not_a_label",
            message="pyramid height axis must be a nonempty label",
        )
    source_axes = tuple(polytope.space.axes)
    if height_axis in source_axes:
        raise OperationDomainValidationError(
            location=("height_axis",),
            code="polytope.pyramid.height_axis_not_fresh",
            message="pyramid height axis must not occur among the source axes",
        )
    source_ids = [vertex.vertex_id for vertex in polytope.vertices]
    if PYRAMID_APEX_VERTEX_ID in source_ids:
        raise OperationDomainValidationError(
            location=("polytope",),
            code="polytope.pyramid.apex_label_collision",
            message="source vertex IDs must not use the reserved apex label 'apex'",
        )
    ambient = len(source_axes)
    if ambient + 1 > MAX_RATIONAL_POLYTOPE_DIMENSION:
        raise OperationResourceAdmissionError(
            location=("polytope",),
            code="polytope.pyramid.ambient_dimension_over_envelope",
            message=(
                "pyramid ambient dimension "
                f"{ambient + 1} exceeds the {MAX_RATIONAL_POLYTOPE_DIMENSION} "
                "axis envelope"
            ),
        )
    if len(source_ids) + 1 > MAX_VERTICES:
        raise OperationResourceAdmissionError(
            location=("polytope",),
            code="polytope.pyramid.vertex_count_over_envelope",
            message=(f"pyramid vertex rows exceed the {MAX_VERTICES}-vertex envelope"),
        )
    return ambient


def polytope_pyramid(polytope: RationalVPolytope, height_axis: str) -> PyramidResult:
    """Compute the exact pyramid ``conv(base(P) union {apex})``.

    Each base vertex ``p`` embeds as ``(p, 0)`` on the fresh height axis and
    the apex is ``(0, ..., 0, 1)``. Before return the kernel replays the
    tagged realization (base rows at height 0, apex alone at height 1), the
    base-face exposure (height is the unique minimizer functional), and the
    dimension identity ``dim(pyramid) = dim(P) + 1``.
    """

    if not isinstance(polytope, RationalVPolytope):
        raise OperationDomainValidationError(
            location=("polytope",),
            code="polytope.pyramid.source_not_a_v_polytope",
            message="pyramid source must be a labelled rational V-polytope value",
        )
    ambient = _admit_pyramid(polytope, height_axis)
    zero = CanonicalRational.from_integer_ratio(0, 1)
    one = CanonicalRational.from_integer_ratio(1, 1)
    base_vertices = tuple(
        RationalPolytopeVertex(
            vertex_id=vertex.vertex_id,
            coordinates=(*vertex.coordinates, zero),
        )
        for vertex in polytope.vertices
    )
    apex_coordinates = (*tuple(zero for _ in range(ambient)), one)
    apex = RationalPolytopeVertex(
        vertex_id=PYRAMID_APEX_VERTEX_ID, coordinates=apex_coordinates
    )
    ordered = tuple(sorted((*base_vertices, apex), key=lambda v: v.vertex_id))
    pyramid = RationalVPolytope(
        space=RationalCoordinateSpace(axes=(*polytope.space.axes, height_axis)),
        vertices=ordered,
    )
    # Replay the tagged realization exactly.
    for vertex in base_vertices:
        if vertex.coordinates[-1].as_fraction() != 0:
            raise OperationDomainValidationError(
                location=("pyramid",),
                code="polytope.pyramid.base_height_replay_failed",
                message="every base vertex must carry height coordinate 0",
            )
    if apex_coordinates[-1].as_fraction() != 1 or any(
        c.as_fraction() != 0 for c in apex_coordinates[:-1]
    ):
        raise OperationDomainValidationError(
            location=("pyramid",),
            code="polytope.pyramid.apex_replay_failed",
            message="the apex must be (0, ..., 0, 1)",
        )
    source_points = [
        [Rational(*c.as_integer_ratio()) for c in vertex.coordinates]
        for vertex in polytope.vertices
    ]
    pyramid_points = [
        [Rational(*c.as_integer_ratio()) for c in vertex.coordinates]
        for vertex in ordered
    ]
    source_dim = _affine_dimension(source_points)
    result_dim = _affine_dimension(pyramid_points)
    if result_dim != source_dim + 1:
        raise OperationDomainValidationError(
            location=("pyramid",),
            code="polytope.pyramid.dimension_identity_failed",
            message=(
                "pyramid affine dimension "
                f"{result_dim} is not source dimension {source_dim} plus one"
            ),
        )
    transport = tuple(
        PyramidBaseVertexMap(
            source_vertex_id=vertex.vertex_id,
            pyramid_vertex_id=vertex.vertex_id,
        )
        for vertex in sorted(polytope.vertices, key=lambda v: v.vertex_id)
    )
    return PyramidResult._from_kernel(
        pyramid=pyramid,
        base_vertex_map=transport,
        source_affine_dimension=source_dim,
        pyramid_affine_dimension=result_dim,
    )


def _admit_prism(polytope: RationalVPolytope, height_axis: object) -> int:
    """Enforce the prism execution envelope shared by native and catalog calls.

    Returns the source ambient dimension. Structural label conflicts raise
    ``OperationDomainValidationError``; envelope overflows raise
    ``OperationResourceAdmissionError``.
    """

    if not isinstance(height_axis, str) or not height_axis:
        raise OperationDomainValidationError(
            location=("height_axis",),
            code="polytope.prism.height_axis_not_a_label",
            message="prism height axis must be a nonempty label",
        )
    source_axes = tuple(polytope.space.axes)
    if height_axis in source_axes:
        raise OperationDomainValidationError(
            location=("height_axis",),
            code="polytope.prism.height_axis_not_fresh",
            message="prism height axis must not occur among the source axes",
        )
    ambient = len(source_axes)
    if ambient + 1 > MAX_RATIONAL_POLYTOPE_DIMENSION:
        raise OperationResourceAdmissionError(
            location=("polytope",),
            code="polytope.prism.ambient_dimension_over_envelope",
            message=(
                "prism ambient dimension "
                f"{ambient + 1} exceeds the {MAX_RATIONAL_POLYTOPE_DIMENSION} "
                "axis envelope"
            ),
        )
    if 2 * len(polytope.vertices) > MAX_VERTICES:
        raise OperationResourceAdmissionError(
            location=("polytope",),
            code="polytope.prism.vertex_count_over_envelope",
            message=(f"prism vertex rows exceed the {MAX_VERTICES}-vertex envelope"),
        )
    return ambient


def polytope_prism(polytope: RationalVPolytope, height_axis: str) -> PrismResult:
    """Compute the exact prism ``P x [0, 1]``.

    Each source vertex ``p`` yields a bottom vertex ``(p, 0)`` and a top
    vertex ``(p, 1)`` with suffixed transport IDs. Before return the kernel
    replays the tagged realization (bottom rows at height 0, top rows at
    height 1, prefixes recovering the source coordinates) and the dimension
    identity ``dim(prism) = dim(P) + 1``.
    """

    if not isinstance(polytope, RationalVPolytope):
        raise OperationDomainValidationError(
            location=("polytope",),
            code="polytope.prism.source_not_a_v_polytope",
            message="prism source must be a labelled rational V-polytope value",
        )
    _admit_prism(polytope, height_axis)
    zero = CanonicalRational.from_integer_ratio(0, 1)
    one = CanonicalRational.from_integer_ratio(1, 1)
    bottom_ids: list[str] = []
    top_ids: list[str] = []
    for vertex in polytope.vertices:
        for side in ("bottom", "top"):
            candidate = f"{vertex.vertex_id}_{side}"
            if not 1 <= len(candidate) <= 64:
                raise OperationDomainValidationError(
                    location=("polytope",),
                    code="polytope.prism.transport_label_too_long",
                    message="prism transport vertex ID exceeds the label bound",
                )
        bottom_ids.append(f"{vertex.vertex_id}_bottom")
        top_ids.append(f"{vertex.vertex_id}_top")
    if len(set(bottom_ids)) != len(bottom_ids) or len(set(top_ids)) != len(top_ids):
        raise OperationDomainValidationError(
            location=("polytope",),
            code="polytope.prism.transport_label_collision",
            message="suffixed prism vertex IDs must be distinct within each side",
        )
    if set(bottom_ids) & set(top_ids):
        raise OperationDomainValidationError(
            location=("polytope",),
            code="polytope.prism.transport_label_collision",
            message="bottom and top prism vertex IDs must be disjoint",
        )
    bottom_vertices = tuple(
        RationalPolytopeVertex(
            vertex_id=f"{vertex.vertex_id}_bottom",
            coordinates=(*vertex.coordinates, zero),
        )
        for vertex in polytope.vertices
    )
    top_vertices = tuple(
        RationalPolytopeVertex(
            vertex_id=f"{vertex.vertex_id}_top",
            coordinates=(*vertex.coordinates, one),
        )
        for vertex in polytope.vertices
    )
    ordered = tuple(
        sorted((*bottom_vertices, *top_vertices), key=lambda v: v.vertex_id)
    )
    prism = RationalVPolytope(
        space=RationalCoordinateSpace(axes=(*polytope.space.axes, height_axis)),
        vertices=ordered,
    )
    # Replay the tagged realization exactly.
    source_by_id = {vertex.vertex_id: vertex for vertex in polytope.vertices}
    for vertex in bottom_vertices:
        if vertex.coordinates[-1].as_fraction() != 0:
            raise OperationDomainValidationError(
                location=("prism",),
                code="polytope.prism.bottom_height_replay_failed",
                message="every bottom vertex must carry height coordinate 0",
            )
        source = source_by_id[vertex.vertex_id[: -len("_bottom")]]
        if tuple(vertex.coordinates[:-1]) != source.coordinates:
            raise OperationDomainValidationError(
                location=("prism",),
                code="polytope.prism.bottom_prefix_replay_failed",
                message="every bottom vertex must project to its source coordinates",
            )
    for vertex in top_vertices:
        if vertex.coordinates[-1].as_fraction() != 1:
            raise OperationDomainValidationError(
                location=("prism",),
                code="polytope.prism.top_height_replay_failed",
                message="every top vertex must carry height coordinate 1",
            )
        source = source_by_id[vertex.vertex_id[: -len("_top")]]
        if tuple(vertex.coordinates[:-1]) != source.coordinates:
            raise OperationDomainValidationError(
                location=("prism",),
                code="polytope.prism.top_prefix_replay_failed",
                message="every top vertex must project to its source coordinates",
            )
    source_points = [
        [Rational(*c.as_integer_ratio()) for c in vertex.coordinates]
        for vertex in polytope.vertices
    ]
    prism_points = [
        [Rational(*c.as_integer_ratio()) for c in vertex.coordinates]
        for vertex in ordered
    ]
    source_dim = _affine_dimension(source_points)
    result_dim = _affine_dimension(prism_points)
    if result_dim != source_dim + 1:
        raise OperationDomainValidationError(
            location=("prism",),
            code="polytope.prism.dimension_identity_failed",
            message=(
                "prism affine dimension "
                f"{result_dim} is not source dimension {source_dim} plus one"
            ),
        )
    ordered_sources = sorted(polytope.vertices, key=lambda v: v.vertex_id)
    bottom_map = tuple(
        PrismVertexMap(
            source_vertex_id=vertex.vertex_id,
            prism_vertex_id=f"{vertex.vertex_id}_bottom",
            side="bottom",
        )
        for vertex in ordered_sources
    )
    top_map = tuple(
        PrismVertexMap(
            source_vertex_id=vertex.vertex_id,
            prism_vertex_id=f"{vertex.vertex_id}_top",
            side="top",
        )
        for vertex in ordered_sources
    )
    return PrismResult._from_kernel(
        prism=prism,
        bottom_vertex_map=bottom_map,
        top_vertex_map=top_map,
        source_affine_dimension=source_dim,
        prism_affine_dimension=result_dim,
    )


def _admit_join(
    left: RationalVPolytope, right: RationalVPolytope, height_axis: object
) -> None:
    """Enforce the join execution envelope shared by native and catalog calls."""

    if not isinstance(height_axis, str) or not height_axis:
        raise OperationDomainValidationError(
            location=("height_axis",),
            code="polytope.join.height_axis_not_a_label",
            message="join height axis must be a nonempty label",
        )
    left_axes = tuple(left.space.axes)
    right_axes = tuple(right.space.axes)
    if set(left_axes) & set(right_axes):
        raise OperationDomainValidationError(
            location=("right",),
            code="polytope.join.factor_axes_overlap",
            message="join factors must live on disjoint axis labels",
        )
    if height_axis in set(left_axes) | set(right_axes):
        raise OperationDomainValidationError(
            location=("height_axis",),
            code="polytope.join.height_axis_not_fresh",
            message="join height axis must not occur among either factor's axes",
        )
    left_ids = [vertex.vertex_id for vertex in left.vertices]
    right_ids = [vertex.vertex_id for vertex in right.vertices]
    if set(left_ids) & set(right_ids):
        raise OperationDomainValidationError(
            location=("right",),
            code="polytope.join.factor_vertex_ids_overlap",
            message="join factors must carry disjoint vertex IDs",
        )
    total_axes = len(left_axes) + len(right_axes) + 1
    if total_axes > MAX_RATIONAL_POLYTOPE_DIMENSION:
        raise OperationResourceAdmissionError(
            location=("left",),
            code="polytope.join.ambient_dimension_over_envelope",
            message=(
                "join ambient dimension "
                f"{total_axes} exceeds the {MAX_RATIONAL_POLYTOPE_DIMENSION} "
                "axis envelope"
            ),
        )
    if len(left_ids) + len(right_ids) > MAX_VERTICES:
        raise OperationResourceAdmissionError(
            location=("left",),
            code="polytope.join.vertex_count_over_envelope",
            message=(f"join vertex rows exceed the {MAX_VERTICES}-vertex envelope"),
        )


def polytope_join(
    left: RationalVPolytope, right: RationalVPolytope, height_axis: str
) -> JoinResult:
    """Compute the exact join ``P * Q``.

    The left factor embeds as ``(p, 0, 0)`` and the right factor as
    ``(0, q, 1)`` on ``(*left.axes, *right.axes, height_axis)``, keeping
    source vertex IDs unchanged. Before return the kernel replays the
    tagged realization (zero blocks and heights, projections recovering
    each factor) and the dimension identity
    ``dim(join) = dim(P) + dim(Q) + 1``.
    """

    if not isinstance(left, RationalVPolytope) or not isinstance(
        right, RationalVPolytope
    ):
        raise OperationDomainValidationError(
            location=("left",),
            code="polytope.join.source_not_a_v_polytope",
            message="join factors must be labelled rational V-polytope values",
        )
    _admit_join(left, right, height_axis)
    left_axes = tuple(left.space.axes)
    right_axes = tuple(right.space.axes)
    zero = CanonicalRational.from_integer_ratio(0, 1)
    one = CanonicalRational.from_integer_ratio(1, 1)
    left_zeros = tuple(zero for _ in right_axes)
    right_zeros = tuple(zero for _ in left_axes)
    left_vertices = tuple(
        RationalPolytopeVertex(
            vertex_id=vertex.vertex_id,
            coordinates=(*vertex.coordinates, *left_zeros, zero),
        )
        for vertex in left.vertices
    )
    right_vertices = tuple(
        RationalPolytopeVertex(
            vertex_id=vertex.vertex_id,
            coordinates=(*right_zeros, *vertex.coordinates, one),
        )
        for vertex in right.vertices
    )
    ordered = tuple(
        sorted((*left_vertices, *right_vertices), key=lambda v: v.vertex_id)
    )
    join = RationalVPolytope(
        space=RationalCoordinateSpace(axes=(*left_axes, *right_axes, height_axis)),
        vertices=ordered,
    )
    # Replay the tagged realization exactly.
    left_by_id = {vertex.vertex_id: vertex for vertex in left.vertices}
    right_by_id = {vertex.vertex_id: vertex for vertex in right.vertices}
    left_width = len(left_axes)
    for vertex in left_vertices:
        if vertex.coordinates[left_width:-1] != left_zeros:
            raise OperationDomainValidationError(
                location=("join",),
                code="polytope.join.left_zero_block_replay_failed",
                message="every left join vertex must carry zeros on the right block",
            )
        if vertex.coordinates[-1].as_fraction() != 0:
            raise OperationDomainValidationError(
                location=("join",),
                code="polytope.join.left_height_replay_failed",
                message="every left join vertex must carry height coordinate 0",
            )
        if (
            tuple(vertex.coordinates[:left_width])
            != left_by_id[vertex.vertex_id].coordinates
        ):
            raise OperationDomainValidationError(
                location=("join",),
                code="polytope.join.left_prefix_replay_failed",
                message="every left join vertex must project to its source coordinates",
            )
    for vertex in right_vertices:
        if vertex.coordinates[:left_width] != right_zeros:
            raise OperationDomainValidationError(
                location=("join",),
                code="polytope.join.right_zero_block_replay_failed",
                message="every right join vertex must carry zeros on the left block",
            )
        if vertex.coordinates[-1].as_fraction() != 1:
            raise OperationDomainValidationError(
                location=("join",),
                code="polytope.join.right_height_replay_failed",
                message="every right join vertex must carry height coordinate 1",
            )
        if (
            tuple(vertex.coordinates[left_width:-1])
            != right_by_id[vertex.vertex_id].coordinates
        ):
            raise OperationDomainValidationError(
                location=("join",),
                code="polytope.join.right_prefix_replay_failed",
                message="every right join vertex must project to its source coordinates",
            )
    left_points = [
        [Rational(*c.as_integer_ratio()) for c in vertex.coordinates]
        for vertex in left.vertices
    ]
    right_points = [
        [Rational(*c.as_integer_ratio()) for c in vertex.coordinates]
        for vertex in right.vertices
    ]
    join_points = [
        [Rational(*c.as_integer_ratio()) for c in vertex.coordinates]
        for vertex in ordered
    ]
    left_dim = _affine_dimension(left_points)
    right_dim = _affine_dimension(right_points)
    result_dim = _affine_dimension(join_points)
    if result_dim != left_dim + right_dim + 1:
        raise OperationDomainValidationError(
            location=("join",),
            code="polytope.join.dimension_identity_failed",
            message=(
                "join affine dimension "
                f"{result_dim} is not {left_dim} + {right_dim} + 1"
            ),
        )
    left_map = tuple(
        JoinVertexMap(
            source_vertex_id=vertex.vertex_id,
            join_vertex_id=vertex.vertex_id,
            side="left",
        )
        for vertex in sorted(left.vertices, key=lambda v: v.vertex_id)
    )
    right_map = tuple(
        JoinVertexMap(
            source_vertex_id=vertex.vertex_id,
            join_vertex_id=vertex.vertex_id,
            side="right",
        )
        for vertex in sorted(right.vertices, key=lambda v: v.vertex_id)
    )
    return JoinResult._from_kernel(
        join=join,
        left_vertex_map=left_map,
        right_vertex_map=right_map,
        left_affine_dimension=left_dim,
        right_affine_dimension=right_dim,
        join_affine_dimension=result_dim,
    )


def _admit_edge_profile(polytope: RationalVPolytope, dimension_bound: object) -> int:
    """Enforce the edge-profile execution envelope shared by native calls.

    Returns the source ambient dimension. Structural violations raise
    ``OperationDomainValidationError``; envelope overflows raise
    ``OperationResourceAdmissionError``. Facet conversion owns the shared
    output-sensitive ray, pair, coefficient-growth, and result bounds.
    """

    if not isinstance(polytope, RationalVPolytope):
        raise OperationDomainValidationError(
            location=("polytope",),
            code="polytope.edge_profile.source_not_a_v_polytope",
            message="edge-profile source must be a labelled rational V-polytope value",
        )
    if (
        not isinstance(dimension_bound, int)
        or isinstance(dimension_bound, bool)
        or not 1 <= dimension_bound <= MAX_FACET_DIMENSION
    ):
        raise OperationDomainValidationError(
            location=("dimension_bound",),
            code="polytope.edge_profile.dimension_bound_not_admitted",
            message=(
                "edge-profile dimension bound must be an integer between 1 and "
                f"{MAX_FACET_DIMENSION}"
            ),
        )
    ambient = len(polytope.space.axes)
    if ambient > dimension_bound:
        raise OperationDomainValidationError(
            location=("polytope",),
            code="polytope.edge_profile.dimension_over_bound",
            message=(
                f"dimension {ambient} exceeds the dimension bound {dimension_bound}"
            ),
        )
    for vertex in polytope.vertices:
        for coordinate in vertex.coordinates:
            try:
                require_bounded_rational(
                    coordinate,
                    max_digits=MAX_FACET_COORDINATE_DIGITS,
                    label="edge-profile vertex coordinate",
                )
            except ValueError as exc:
                raise OperationResourceAdmissionError(
                    location=("polytope",),
                    code="polytope.edge_profile.coordinate_digits_over_envelope",
                    message=str(exc),
                ) from exc
    points = [
        [Rational(*coordinate.as_integer_ratio()) for coordinate in vertex.coordinates]
        for vertex in polytope.vertices
    ]
    if ambient == 1:
        if len({point[0] for point in points}) < 2:
            raise OperationDomainValidationError(
                location=("polytope",),
                code="polytope.edge_profile.not_full_dimensional",
                message=(
                    "V-representation is not full-dimensional; lower-dimensional "
                    "hulls require intrinsic affine coordinates"
                ),
            )
    elif (
        rational_rank(
            [
                [points[index][axis] - points[0][axis] for axis in range(ambient)]
                for index in range(1, len(points))
            ],
            ambient,
        )
        < ambient
    ):
        raise OperationDomainValidationError(
            location=("polytope",),
            code="polytope.edge_profile.not_full_dimensional",
            message=(
                "V-representation is not full-dimensional; lower-dimensional "
                "hulls require intrinsic affine coordinates"
            ),
        )
    return ambient


def _extreme_positions_from_facets(
    facets: tuple[PrimitiveFacet, ...], point_count: int, dim: int
) -> list[int]:
    """Return the source positions that are exact extreme vertices.

    A source row is an extreme vertex exactly when the normals of the
    facets containing it span the ambient space (the active-constraint
    rank test). Redundant interior and boundary rows are rank-deficient
    and carry no polytope edges.
    """

    active_normals: list[list[list[Rational]]] = [[] for _ in range(point_count)]
    for facet in facets:
        normal = [
            Rational(*coefficient.as_integer_ratio())
            for coefficient in facet.halfspace.coefficients
        ]
        for index in facet.source_vertex_indices:
            if 0 <= index < point_count:
                active_normals[index].append(normal)
    return [
        index
        for index in range(point_count)
        if active_normals[index] and rational_rank(active_normals[index], dim) == dim
    ]


def _minimal_face_dimension(
    containing: list[set[int]],
    points: list[list[Rational]],
    first: int,
    second: int,
) -> int:
    """Affine dimension of the minimal face containing two source rows.

    Faces of a polytope are exactly the intersections of facet families,
    so the minimal face containing ``{first, second}`` is the
    intersection of all facets containing both, and its vertex set is
    every source row lying on each of those common facets (with the
    empty intersection ranging over the whole polytope). The dimension
    is the exact rank of that vertex set's coordinate differences.
    """

    common = containing[first] & containing[second]
    members = [
        point for index, point in enumerate(points) if common <= containing[index]
    ]
    return _affine_dimension(members)


def _compute_edge_data(
    polytope: RationalVPolytope, ambient: int
) -> tuple[tuple[tuple[str, str], ...], int]:
    """Shared exact edge enumeration used by the edge and figure kernels.

    Returns the sorted endpoint-ID pairs and the replayed affine dimension.
    Raises ``OperationDomainValidationError`` for structural failures and
    ``OperationResourceAdmissionError`` when the materialized profile
    overflows its result envelope.
    """

    bare = tuple(Vertex(coordinates=vertex.coordinates) for vertex in polytope.vertices)
    try:
        facets = _computed_facets_from_vertices(bare, ambient)
    except PolyhedralConversionAdmissionError as exc:
        raise OperationResourceAdmissionError(
            location=("polytope",),
            code="polytope.edge_profile.enumeration_over_envelope",
            message=str(exc),
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("polytope",),
            code="polytope.edge_profile.facet_profile_not_admitted",
            message=str(exc),
        ) from exc
    points = [
        [Rational(*coordinate.as_integer_ratio()) for coordinate in vertex.coordinates]
        for vertex in polytope.vertices
    ]
    point_count = len(points)
    containing: list[set[int]] = [set() for _ in range(point_count)]
    for facet_index, facet in enumerate(facets):
        for index in facet.source_vertex_indices:
            if 0 <= index < point_count:
                containing[index].add(facet_index)
    extreme = _extreme_positions_from_facets(facets, point_count, ambient)
    extreme_set = set(extreme)
    vertex_ids = [vertex.vertex_id for vertex in polytope.vertices]
    pairs: list[tuple[str, str]] = []
    for position_a in range(len(extreme)):
        for position_b in range(position_a + 1, len(extreme)):
            first = extreme[position_a]
            second = extreme[position_b]
            if _minimal_face_dimension(containing, points, first, second) != 1:
                continue
            first_id = vertex_ids[first]
            second_id = vertex_ids[second]
            # Replay that no third extreme vertex lies in the minimal face:
            # the face members restricted to extreme rows are exactly the
            # pair, up to redundant collinear source rows on the segment.
            common = containing[first] & containing[second]
            face_extreme = sorted(
                index for index in extreme_set if common <= containing[index]
            )
            face_points = [points[index] for index in face_extreme]
            if len(face_extreme) != 2 or _affine_dimension(face_points) != 1:
                continue
            pairs.append(
                (first_id, second_id) if first_id < second_id else (second_id, first_id)
            )
    pairs.sort()
    source_dimension = _affine_dimension(points)
    if source_dimension != ambient:
        raise OperationDomainValidationError(
            location=("polytope",),
            code="polytope.edge_profile.dimension_replay_failed",
            message=(
                "replayed affine dimension "
                f"{source_dimension} does not match the ambient dimension {ambient}"
            ),
        )
    return tuple(pairs), source_dimension


def _face_lattice_work_bound(vertex_count: int) -> int:
    """Bound all rank-three face and Hasse-cover postprocessing scans."""

    maximum_facets = min(MAX_COMPUTED_FACETS, 2 * vertex_count - 4)
    maximum_edges = 3 * vertex_count - 6
    maximum_faces = 6 * vertex_count - 8
    maximum_covers = 15 * vertex_count - 28
    return (
        math.comb(maximum_facets, 2) * vertex_count
        + 2 * maximum_edges * maximum_facets
        + maximum_faces * vertex_count
        + 4 * maximum_covers
        + 64 * vertex_count
    )


def _admit_face_lattice_source(polytope_value: RationalVPolytope) -> RationalVPolytope:
    """Revalidate and admit a canonical source before facet enumeration."""
    if not isinstance(polytope_value, RationalVPolytope):
        raise OperationDomainValidationError(
            location=("polytope",),
            code="polytope.face_lattice.source_not_a_v_polytope",
            message="face-lattice source must be a labelled rational V-polytope",
        )
    try:
        polytope = RationalVPolytope.model_validate(
            polytope_value.model_dump(mode="python", warnings=False), strict=True
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("polytope",),
            code="polytope.face_lattice.source_structure_invalid",
            message="face-lattice source must satisfy the canonical V-polytope contract",
        ) from exc

    if len(polytope.space.axes) != MAX_POLYTOPE_FACE_LATTICE_DIMENSION:
        raise OperationDomainValidationError(
            location=("polytope", "space", "axes"),
            code="polytope.face_lattice.dimension_not_three",
            message="polytope face-lattice construction currently supports dimension three",
        )
    vertex_count = len(polytope.vertices)
    if vertex_count > MAX_VERTICES:
        raise OperationResourceAdmissionError(
            location=("polytope", "vertices"),
            code="polytope.face_lattice.vertex_bound_exceeded",
            message=f"face-lattice input exceeds {MAX_VERTICES} source vertices",
        )
    for vertex in polytope.vertices:
        for coordinate in vertex.coordinates:
            try:
                require_bounded_rational(
                    coordinate,
                    max_digits=MAX_FACET_COORDINATE_DIGITS,
                    label="face-lattice source coordinate",
                )
            except ValueError as exc:
                raise OperationResourceAdmissionError(
                    location=("polytope", "vertices"),
                    code="polytope.face_lattice.coordinate_height_exceeded",
                    message=str(exc),
                ) from exc

    maximum_faces = 6 * vertex_count - 8
    maximum_covers = 15 * vertex_count - 28
    if (
        maximum_faces > MAX_POLYTOPE_FACE_LATTICE_FACES
        or maximum_covers > MAX_POLYTOPE_FACE_LATTICE_COVERS
    ):
        raise OperationResourceAdmissionError(
            location=("polytope",),
            code="polytope.face_lattice.output_bound_exceeded",
            message="the derived rank-three face-lattice output exceeds its published bound",
        )
    estimated_work = _face_lattice_work_bound(vertex_count)
    if estimated_work > MAX_POLYTOPE_FACE_LATTICE_WORK:
        raise OperationResourceAdmissionError(
            location=("polytope",),
            code="polytope.face_lattice.work_budget_exceeded",
            message=(
                f"face-lattice postprocessing is estimated at {estimated_work} units; "
                f"limit is {MAX_POLYTOPE_FACE_LATTICE_WORK}"
            ),
        )
    source_chars = len(polytope.model_dump_json(warnings=False))
    output_chars_bound = (
        source_chars
        + MAX_POLYTOPE_FACE_LATTICE_FACES * 256
        + MAX_POLYTOPE_FACE_LATTICE_COVERS * 64
        + 64_000
    )
    if output_chars_bound > MAX_POLYTOPE_FACE_LATTICE_RESULT_CHARS:
        raise OperationResourceAdmissionError(
            location=("result",),
            code="polytope.face_lattice.result_size_exceeded",
            message=(
                f"face-lattice output may require {output_chars_bound} characters; "
                f"limit is {MAX_POLYTOPE_FACE_LATTICE_RESULT_CHARS}"
            ),
        )

    return polytope


def _compute_face_lattice_incidence(
    polytope: RationalVPolytope,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], tuple[tuple[int, int], ...]]:
    """Derive extremes, facets, and edges from internally computed incidence."""

    vertex_count = len(polytope.vertices)
    bare_vertices = _canonical_v_polytope_vertices(polytope)
    try:
        facets = _computed_facets_from_vertices(
            bare_vertices, MAX_POLYTOPE_FACE_LATTICE_DIMENSION
        )
    except PolyhedralConversionAdmissionError as exc:
        raise OperationResourceAdmissionError(
            location=("polytope",),
            code="polytope.face_lattice.facet_enumeration_exceeded",
            message=str(exc),
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("polytope",),
            code="polytope.face_lattice.facet_profile_invalid",
            message=str(exc),
        ) from exc

    extreme = tuple(_extreme_positions_from_facets(facets, vertex_count, 3))
    if len(extreme) < 4:
        raise OperationDomainValidationError(
            location=("polytope", "vertices"),
            code="polytope.face_lattice.extreme_vertices_invalid",
            message="a full-dimensional 3-polytope must have at least four extreme vertices",
        )
    extreme_set = set(extreme)
    facet_vertices = tuple(
        tuple(index for index in facet.source_vertex_indices if index in extreme_set)
        for facet in facets
    )
    if any(len(face_vertices) < 3 for face_vertices in facet_vertices):
        raise OperationDomainValidationError(
            location=("polytope",),
            code="polytope.face_lattice.facet_extremes_invalid",
            message="each supporting facet must contain at least three extreme vertices",
        )

    facet_sets = tuple(set(vertices) for vertices in facet_vertices)
    edge_occurrences: dict[tuple[int, int], int] = {}
    for left, right in combinations(range(len(facets)), 2):
        common = tuple(
            index
            for index in extreme
            if index in facet_sets[left] and index in facet_sets[right]
        )
        if len(common) == 2:
            edge = (common[0], common[1])
            edge_occurrences[edge] = edge_occurrences.get(edge, 0) + 1
    if any(count != 1 for count in edge_occurrences.values()):
        raise OperationDomainValidationError(
            location=("polytope",),
            code="polytope.face_lattice.edge_incidence_invalid",
            message="each polytope edge must be the intersection of exactly two facets",
        )
    edges = tuple(sorted(edge_occurrences))
    expected_edge_count = len(extreme) + len(facets) - 2
    if len(edges) != expected_edge_count:
        raise OperationDomainValidationError(
            location=("polytope",),
            code="polytope.face_lattice.euler_identity_failed",
            message="the derived vertices, edges, and facets do not satisfy Euler's identity",
        )

    return extreme, facet_vertices, edges


def _assemble_face_lattice(
    polytope: RationalVPolytope,
    extreme: tuple[int, ...],
    facet_vertices: tuple[tuple[int, ...], ...],
    edges: tuple[tuple[int, int], ...],
) -> PolytopeFaceLatticeResult:
    """Build canonical face labels and Hasse covers from exact incidences."""

    face_specs: list[tuple[int, tuple[int, ...]]] = [(-1, ())]
    face_specs.extend((0, (index,)) for index in extreme)
    face_specs.extend((1, edge) for edge in edges)
    face_specs.extend((2, vertices) for vertices in facet_vertices)
    face_specs.append((3, extreme))
    face_specs.sort()
    faces = tuple(
        PolytopeFace(dimension=dimension, source_vertex_indices=indices)
        for dimension, indices in face_specs
    )
    face_indices = {
        (face.dimension, face.source_vertex_indices): index
        for index, face in enumerate(faces)
    }
    bottom = face_indices[(-1, ())]
    top = face_indices[(3, extreme)]
    cover_pairs: set[tuple[int, int]] = {
        (bottom, face_indices[(0, (vertex,))]) for vertex in extreme
    }
    for edge in edges:
        edge_index = face_indices[(1, edge)]
        cover_pairs.update(
            (face_indices[(0, (vertex,))], edge_index) for vertex in edge
        )
    for facet_index, vertices in enumerate(facet_vertices):
        facet_key = (2, vertices)
        face_index = face_indices[facet_key]
        for edge in edges:
            if edge[0] in vertices and edge[1] in vertices:
                cover_pairs.add((face_indices[(1, edge)], face_index))
        if not any(
            (face_indices[(1, edge)], face_index) in cover_pairs for edge in edges
        ):
            raise OperationDomainValidationError(
                location=("polytope", "facets", facet_index),
                code="polytope.face_lattice.facet_boundary_missing",
                message="every facet must contain at least one derived boundary edge",
            )
        cover_pairs.add((face_index, top))

    covers = tuple(
        PolytopeFaceCover(lower_face_index=lower, upper_face_index=upper)
        for lower, upper in sorted(cover_pairs)
    )
    if (
        len(faces) > MAX_POLYTOPE_FACE_LATTICE_FACES
        or len(covers) > MAX_POLYTOPE_FACE_LATTICE_COVERS
    ):
        raise OperationResourceAdmissionError(
            location=("result",),
            code="polytope.face_lattice.output_bound_exceeded",
            message="the exact face-lattice result exceeds its published face or cover bound",
        )
    return PolytopeFaceLatticeResult._from_kernel(
        polytope=polytope,
        extreme_vertex_indices=extreme,
        faces=faces,
        covers=covers,
    )


def polytope_face_lattice(
    polytope_value: RationalVPolytope,
) -> PolytopeFaceLatticeResult:
    """Compute the complete face lattice of an exact three-dimensional hull.

    Facets are recomputed from the retained V-representation; caller-supplied
    incidence claims are never treated as geometry.
    """

    polytope = _admit_face_lattice_source(polytope_value)
    extreme, facet_vertices, edges = _compute_face_lattice_incidence(polytope)
    return _assemble_face_lattice(polytope, extreme, facet_vertices, edges)


def polytope_edge_profile(
    polytope: RationalVPolytope, dimension_bound: int = MAX_FACET_DIMENSION
) -> EdgeProfileResult:
    """Compute the exact vertex-adjacency (edge) graph of a V-polytope.

    A pair ``(i, j)`` of distinct source rows is an edge exactly when the
    minimal face containing both is one-dimensional. Faces of a polytope
    are precisely the intersections of facet families, so with ``F(k)``
    the set of facets containing row ``k`` the minimal face containing
    ``{i, j}`` is ``intersection{f : f in F(i) cap F(j)}`` and its vertex
    set is ``{k : F(i) cap F(j) subset F(k)}``; the pair is an edge iff
    that vertex set has affine rank one. Equivalently the pair is
    contained in a common facet and adjacent within every common facet's
    restricted profile: if the minimal face had dimension two or more it
    would contain a third extreme vertex of some common facet, and if the
    pair shared no facet the minimal face would be the whole polytope of
    dimension at least two. The kernel therefore enumerates the complete
    facet profile with the existing bounded machinery, restricts candidate
    pairs to exact extreme vertices (the active-normal rank test, so
    redundant rows carry no edges), and keeps exactly the pairs whose
    minimal face has dimension one with no third extreme vertex inside.
    Endpoints are returned as sorted source-ID pairs with the replayed
    affine dimension.
    """

    ambient = _admit_edge_profile(polytope, dimension_bound)
    pairs, source_dimension = _compute_edge_data(polytope, ambient)
    edges = tuple(
        PolytopeEdge(endpoint_a=first, endpoint_b=second) for first, second in pairs
    )
    return EdgeProfileResult._from_kernel(
        polytope=polytope,
        edges=edges,
        affine_dimension=source_dimension,
    )


def polytope_vertex_figure(
    polytope: RationalVPolytope, vertex_id: str
) -> VertexFigureResult:
    """Compute the exact vertex figure of one polytope vertex.

    The vertex figure at the extreme vertex ``v`` is the convex hull of
    the edge-midpoints ``(v + u) / 2`` over the edge neighbors ``u`` read
    from the shared edge kernel above (not from the public operation).
    Figure vertex IDs are derived deterministically as
    ``sec_<neighbor_id>``. Before return the kernel replays the midpoint
    identity ``2 * sec(u) - v = u`` for every figure vertex and the
    dimension identity ``dim(figure) = dim(P) - 1``; a center with no
    incident edges (an unknown ID is rejected earlier, a redundant row
    here) and a dimension-zero source are outside the admitted domain.
    """

    if not isinstance(polytope, RationalVPolytope):
        raise OperationDomainValidationError(
            location=("polytope",),
            code="polytope.vertex_figure.source_not_a_v_polytope",
            message="vertex-figure source must be a labelled rational V-polytope value",
        )
    if not isinstance(vertex_id, str) or not vertex_id:
        raise OperationDomainValidationError(
            location=("vertex_id",),
            code="polytope.vertex_figure.vertex_id_not_a_label",
            message="vertex-figure center must be a nonempty vertex ID",
        )
    by_id = {vertex.vertex_id: vertex for vertex in polytope.vertices}
    center = by_id.get(vertex_id)
    if center is None:
        raise OperationDomainValidationError(
            location=("vertex_id",),
            code="polytope.vertex_figure.unknown_vertex_id",
            message=f"vertex ID {vertex_id!r} does not occur among the source vertices",
        )
    ambient = _admit_edge_profile(polytope, MAX_FACET_DIMENSION)
    pairs, source_dimension = _compute_edge_data(polytope, ambient)
    if source_dimension < 1:
        raise OperationDomainValidationError(
            location=("polytope",),
            code="polytope.vertex_figure.source_dimension_too_small",
            message="vertex figures require a source of affine dimension at least one",
        )
    neighbors = sorted(
        second if first == vertex_id else first
        for first, second in pairs
        if first == vertex_id or second == vertex_id
    )
    if not neighbors:
        raise OperationDomainValidationError(
            location=("vertex_id",),
            code="polytope.vertex_figure.center_not_a_vertex",
            message="vertex-figure center must be an exact extreme vertex",
        )
    for neighbor_id in neighbors:
        candidate = f"sec_{neighbor_id}"
        if not 1 <= len(candidate) <= MAX_COORDINATE_LABEL_LENGTH:
            raise OperationDomainValidationError(
                location=("polytope",),
                code="polytope.vertex_figure.transport_label_too_long",
                message="vertex-figure vertex ID exceeds the label bound",
            )
    if len({f"sec_{neighbor_id}" for neighbor_id in neighbors}) != len(neighbors):
        raise OperationDomainValidationError(
            location=("polytope",),
            code="polytope.vertex_figure.transport_label_collision",
            message="derived vertex-figure vertex IDs must be distinct",
        )
    center_coordinates = tuple(
        coordinate.as_fraction() for coordinate in center.coordinates
    )
    neighbor_coordinates = {
        neighbor_id: tuple(
            coordinate.as_fraction() for coordinate in by_id[neighbor_id].coordinates
        )
        for neighbor_id in neighbors
    }
    figure_vertices = tuple(
        RationalPolytopeVertex(
            vertex_id=f"sec_{neighbor_id}",
            coordinates=tuple(
                CanonicalRational.from_fraction((center_value + neighbor_value) / 2)
                for center_value, neighbor_value in zip(
                    center_coordinates,
                    neighbor_coordinates[neighbor_id],
                    strict=True,
                )
            ),
        )
        for neighbor_id in neighbors
    )
    ordered = tuple(sorted(figure_vertices, key=lambda vertex: vertex.vertex_id))
    figure = VertexFigurePolytope(
        space=RationalCoordinateSpace(axes=polytope.space.axes),
        vertices=ordered,
    )
    # Replay the midpoint identity exactly: 2 * sec(u) - v recovers u.
    figure_by_id = {vertex.vertex_id: vertex for vertex in ordered}
    for neighbor_id in neighbors:
        midpoint = tuple(
            coordinate.as_fraction()
            for coordinate in figure_by_id[f"sec_{neighbor_id}"].coordinates
        )
        recovered = tuple(
            2 * middle - center_value
            for middle, center_value in zip(midpoint, center_coordinates, strict=True)
        )
        if recovered != neighbor_coordinates[neighbor_id]:
            raise OperationDomainValidationError(
                location=("figure",),
                code="polytope.vertex_figure.midpoint_replay_failed",
                message="every figure vertex must be the edge-midpoint (v + u) / 2",
            )
    figure_points = [
        [Rational(*coordinate.as_integer_ratio()) for coordinate in vertex.coordinates]
        for vertex in ordered
    ]
    figure_dimension = _affine_dimension(figure_points)
    if figure_dimension != source_dimension - 1:
        raise OperationDomainValidationError(
            location=("figure",),
            code="polytope.vertex_figure.dimension_identity_failed",
            message=(
                "vertex-figure affine dimension "
                f"{figure_dimension} is not source dimension {source_dimension} "
                "minus one"
            ),
        )
    transport = tuple(
        VertexFigureVertexMap(
            source_vertex_id=neighbor_id,
            figure_vertex_id=f"sec_{neighbor_id}",
        )
        for neighbor_id in neighbors
    )
    return VertexFigureResult._from_kernel(
        figure=figure,
        center_vertex_id=vertex_id,
        vertex_map=transport,
        source_affine_dimension=source_dimension,
        figure_affine_dimension=figure_dimension,
    )


__all__ = [
    "convex_hull_volume",
    "facet_incidence",
    "polytope_edge_profile",
    "polytope_face_lattice",
    "polytope_join",
    "polytope_prism",
    "polytope_pyramid",
    "polytope_support",
    "polytope_vertex_figure",
    "polytope_volume",
    "verify_facet_incidence",
    "verify_primitive_facet",
]
