"""Domain-owned exact rational polytope volume operations.

The volume of a bounded rational polytope is computed exactly by
recursively triangulating its boundary and coning each boundary
simplex to an interior reference point.

For a V-representation the caller-supplied vertices are used directly.
For an H-representation the vertices are first enumerated by solving
every ``C(m, d)`` subsystem of half-spaces and retaining the feasible
intersections.

The convex hull (d-1)-facets are enumerated by a bounded brute-force
test (every d-subset of points is a subfacet if all remaining points
lie on one side of the hyperplane it spans), merged into maximal
coplanar facets by a canonical hyperplane signature, and triangulated
recursively: each (d-1)-facet is projected to (d-1)-dimensional
coordinates, triangulated, and each boundary simplex is coned to an
interior point to form a d-simplex whose exact SymPy determinant is
summed. The recursion bottoms out at d=1 (an interval) and d=2 (a
polygon fan). This is exact and bounded for the small dimensions
(``d <= 6``) and vertex counts (``<= 128``) this operation admits.
"""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import combinations
from typing import Literal

from sympy import Matrix, Rational

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.polytope._models import (
    MAX_BOUNDEDNESS_COMBINATIONS,
    MAX_COMPUTED_FACETS,
    MAX_EXTREMALITY_HEIGHT_WORK,
    MAX_FACET_INCIDENCES,
    MAX_FACET_SIGN_TESTS,
    MAX_HULL_SUBFACETS,
    MAX_SUPPORT_ORIENTATION_TESTS,
    MAX_SUPPORT_VERTEX_SUBSETS,
    FacetIncidenceRequest,
    FacetIncidenceResult,
    PolytopeSupportRequest,
    PolytopeSupportResult,
    PolytopeVolumeRequest,
    PolytopeVolumeResult,
    PrimitiveFacet,
    RationalCovector,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.polytope._rational_geometry import (
    facets_from_points as _facets_from_points,
)
from jacobian.math.polytope._rational_geometry import (
    recession_cone_is_trivial,
    vertices_from_halfspaces,
)
from jacobian.math.polytope.values import Halfspace, Vertex

# Absolute combinatorial ceiling: reject vertex enumeration that would
# attempt to solve more subsystems than this. With ``MAX_FACETS = 64``
# and ``d = 6`` the worst case is ``C(64, 6) = 74,974,368``, so the bound
# is the practical gate on budget exhaustion.
MAX_SUBSYSTEM_SOLVES = 5_000_000

# ``MAX_HULL_SUBFACETS`` (the C(n, d) hull-enumeration ceiling) is owned
# by ``_models`` beside the other published request bounds, which quote it
# in their schema-visible descriptions.


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
        differences = Matrix(
            [
                [points[index][axis] - points[0][axis] for axis in range(dim)]
                for index in range(1, len(points))
            ]
        )
        if differences.rank() < dim:
            raise ValueError(
                "V-representation is not full-dimensional; lower-dimensional hulls "
                "require intrinsic affine coordinates"
            )
    # Repeated source rows create neither candidate hyperplanes nor
    # candidate side tests: ``_facets_from_points`` receives exactly the
    # distinct rows below, so each of the ``C(m, d)`` candidates is
    # side-tested against those ``m`` distinct rows and the sign-test
    # budget is charged per distinct row actually tested. The final-facet
    # incidence pass ranges over all source positions and is accounted
    # separately -- its work is bounded by the facet and incidence result
    # limits enforced exactly on the materialized profile this bounded
    # enumeration produces (see ``_computed_facets_from_vertices``); row
    # counts alone cannot derive those bounds because interior and other
    # non-extreme rows inflate every vertex-count upper bound while
    # contributing no facets.
    distinct_points = _deduplicate_source_rows(points)
    candidate_count = math.comb(len(distinct_points), dim)
    side_tests = len(distinct_points) * candidate_count
    if side_tests > MAX_FACET_SIGN_TESTS:
        raise ValueError(
            "facet enumeration exceeds the "
            f"{MAX_FACET_SIGN_TESTS}-side-test bound "
            f"({side_tests} > {MAX_FACET_SIGN_TESTS})"
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
    candidates = _facets_from_points(_deduplicate_source_rows(points), dim)
    canonical: dict[tuple[tuple[int, ...], int], PrimitiveFacet] = {}
    for normal, offset in candidates:
        coefficients, rhs = _primitive_facet_key(normal, offset, dim)
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
    facets = tuple(canonical[key] for key in sorted(canonical))
    if len(facets) > MAX_COMPUTED_FACETS:
        raise ValueError(
            f"facet profile exceeds the {MAX_COMPUTED_FACETS}-facet result bound"
        )
    incidence_count = sum(len(facet.source_vertex_indices) for facet in facets)
    if incidence_count > MAX_FACET_INCIDENCES:
        raise ValueError(
            f"facet profile exceeds the {MAX_FACET_INCIDENCES}-incidence result bound"
        )
    return facets


def compute_facet_incidence(
    request: FacetIncidenceRequest,
) -> FacetIncidenceResult:
    """Compute the complete canonical facet-incidence profile of ``conv(V)``."""

    vertices = request.vertices
    assert isinstance(vertices, tuple)  # projected by the request validator
    dimension = len(vertices[0].coordinates)
    facets = _computed_facets_from_vertices(vertices, dimension)
    return FacetIncidenceResult(
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
        for frac in fracs:
            lcm = lcm * frac.denominator // math.gcd(lcm, frac.denominator)
        ints = [int(frac * lcm) for frac in fracs]
        g = 0
        for value in ints:
            g = math.gcd(g, abs(value))
        if g == 0:
            # All-zero normals cannot occur: request validation rejects
            # them before admission; keep the row untouched as a fallback.
            g = 1
        key = (
            tuple(value // g for value in ints),
            ((offset * lcm / g).numerator, (offset * lcm / g).denominator),
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
    if dim == 1:
        positive = any(
            Rational(*hs.coefficients[0].as_integer_ratio()) > 0 for hs in halfspaces
        )
        negative = any(
            Rational(*hs.coefficients[0].as_integer_ratio()) < 0 for hs in halfspaces
        )
        return positive and negative
    normals: list[list[Rational]] = [
        [Rational(*c.as_integer_ratio()) for c in hs.coefficients] for hs in halfspaces
    ]
    # Positive spanning requires the normals' convex hull to be full-
    # dimensional; otherwise the polyhedron is unbounded.
    if len(normals) < dim + 1:
        return False
    # Guard the exact facet enumeration of the row normals: the budget
    # applies to the distinct rows after duplicate removal, exactly as the
    # enumeration below counts its own work.
    try:
        combo_count = math.comb(len(normals), dim)
    except ValueError:
        combo_count = 10**18
    if combo_count > MAX_BOUNDEDNESS_COMBINATIONS:
        raise ValueError(
            "H-representation boundedness precheck exceeds the "
            f"{MAX_BOUNDEDNESS_COMBINATIONS}-combination budget "
            f"({combo_count} > {MAX_BOUNDEDNESS_COMBINATIONS})"
        )
    return recession_cone_is_trivial(normals, dim)


def _format_rational(value: Rational) -> str:
    """Render a SymPy ``Rational`` as a canonical ``num/den`` string."""

    if value.denominator == 1:
        return format_canonical_integer(int(value.numerator))
    return (
        f"{format_canonical_integer(int(value.numerator))}/"
        f"{format_canonical_integer(int(value.denominator))}"
    )


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
    v0 = simplex_points[0]
    cols = [
        Matrix([[simplex_points[i][k] - v0[k]] for k in range(d)])
        for i in range(1, d + 1)
    ]
    return abs(Matrix.hstack(*cols).det())


def _rank_of_diffs(points: list[list[Rational]], dim: int) -> int:
    """Rank of the matrix of ``point - point[0]`` differences in ``dim`` dims."""
    if len(points) <= 1:
        return 0
    v0 = points[0]
    cols = [
        Matrix([[points[i][k] - v0[k]] for k in range(dim)])
        for i in range(1, len(points))
    ]
    return Matrix.hstack(*cols).rank() if cols else 0


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
            mat = Matrix(
                [[points[i][k] for k in range(dim)] + [1] for i in subset]
                + [[points[p][k] for k in range(dim)] + [1]]
            )
            det = mat.det()
            if det > 0:
                signs.add(1)
            elif det < 0:
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
        if active_normals[i] and Matrix(active_normals[i]).rank() == dim
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

    if len(points) <= dim + 1:
        return points
    # Guard the facet enumeration that the filter itself requires.
    subfacet_count = math.comb(len(points), dim)
    if subfacet_count > MAX_HULL_SUBFACETS:
        # Too many to enumerate exactly; fall back to no filtering and let
        # the caller raise the budget error. Filtering is an optimization
        # for the small, non-budget-exhausting cases that the operation
        # admits.
        return points
    groups: dict[tuple[int, ...], set[int]] = {}
    for subfacet in _hull_subfacets(points, dim):
        sig = _plane_signature(subfacet, points)
        if sig is None:
            continue
        groups.setdefault(sig, set()).update(subfacet)
    if not groups:
        return points
    keep_indices, counts = _extreme_point_indices(groups, len(points), dim)
    # If filtering would discard too much (e.g., degenerate or numeric
    # failure), keep the hull boundary instead of collapsing.
    if len(keep_indices) < dim + 1:
        hull_indices = [i for i, c in enumerate(counts) if c > 0]
        if len(hull_indices) >= dim + 1:
            keep_indices = hull_indices
        else:
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
    vertices.  The existing hull-facet code gives the latter proof by the
    active-normal rank characterization of extreme vertices.  The published
    budgets are enforced from typed counts and reduced-component digit
    lengths before any canonical coordinate is converted or any exact linear
    algebra runs: the subfacet and orientation-test bounds depend only on
    the vertex count and dimension, and the height-work bound couples those
    test counts with the operand heights because one orientation determinant
    costs ``Theta(D^2)`` limb operations at reduced-component height ``D``
    (see ``MAX_EXTREMALITY_HEIGHT_WORK``), so all three together bound the
    proof's exact work across the whole canonical coordinate domain.
    """

    dimension = len(polytope.space.axes)
    vertex_count = len(polytope.vertices)
    subset_count = math.comb(vertex_count, dimension)
    if subset_count > MAX_SUPPORT_VERTEX_SUBSETS:
        raise ValueError(
            "V-polytope extremality proof exceeds the subfacet bound "
            f"({subset_count} > {MAX_SUPPORT_VERTEX_SUBSETS})"
        )
    orientation_tests = subset_count * (vertex_count - dimension)
    if orientation_tests > MAX_SUPPORT_ORIENTATION_TESTS:
        raise ValueError(
            "V-polytope extremality proof exceeds the orientation-test bound "
            f"({orientation_tests} > {MAX_SUPPORT_ORIENTATION_TESTS})"
        )
    component_digits = max(
        max(len(coordinate.num.lstrip("-")), len(coordinate.den.lstrip("-")))
        for vertex in polytope.vertices
        for coordinate in vertex.coordinates
    )
    height_work = orientation_tests * component_digits**2
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
    if Matrix(differences).rank() != dimension:
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
    # Remove redundant boundary points before triangulating (P2): keeps
    # only extreme hull vertices, avoiding double-counting and degenerate
    # adjacency (e.g., 3x3 square with collinear edge points).
    points = _filter_redundant_vertices(points, dim)
    n = len(points)
    if n < dim + 1:
        return Rational(0)
    # Guard the brute-force hull enumeration against combinatorial blow-up.
    subfacet_count = math.comb(n, dim)
    if subfacet_count > MAX_HULL_SUBFACETS:
        raise ValueError(
            "polytope hull enumeration exceeds the combinatorial bound "
            f"({subfacet_count} > {MAX_HULL_SUBFACETS} d-subsets)"
        )
    if dim == 1:
        coords = sorted({p[0] for p in points})
        return coords[-1] - coords[0] if len(coords) >= 2 else Rational(0)
    triangulation = _triangulate(points, dim)
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
    n = len(rows)
    # Guard against combinatorial blow-up before materialising combinations.
    subsystem_count = math.comb(n, dim)
    if subsystem_count > MAX_SUBSYSTEM_SOLVES:
        raise ValueError(
            "polytope vertex enumeration exceeds the combinatorial bound "
            f"({subsystem_count} > {MAX_SUBSYSTEM_SOLVES} subsystems)"
        )

    result: tuple[list[tuple[Rational, ...]], int] = (
        vertices_from_halfspaces(rows, dim),
        dim,
    )
    return result


def compute_polytope_support(
    request: PolytopeSupportRequest,
) -> PolytopeSupportResult:
    """Compute one exact support value and the complete exposed vertex face."""

    return polytope_support(request.polytope, request.covector)


def polytope_support(
    polytope: RationalVPolytope,
    covector: RationalCovector,
) -> PolytopeSupportResult:
    """Domain kernel for one exact support value and exposed vertex face."""

    if polytope.space != covector.space:
        raise ValueError("polytope and covector must use the same coordinate space")
    value, vertices = support_data(polytope, covector)
    from jacobian.math.polytope._models import RationalExposedFace

    return PolytopeSupportResult(
        polytope=polytope,
        covector=covector,
        support_value=CanonicalRational.from_fraction(value),
        exposed_face=RationalExposedFace(
            space=polytope.space,
            vertices=vertices,
        ),
    )


def compute_polytope_volume(
    request: PolytopeVolumeRequest,
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
    if request.vertices is not None:
        representation = "vertices"
        vertices = request.vertices
        assert isinstance(vertices, tuple)  # projected by the request validator
        raw_vertices = tuple(
            tuple(c.as_fraction() for c in vertex.coordinates) for vertex in vertices
        )
    else:
        assert request.halfspaces is not None
        representation = "halfspaces"
        derived, _dim = _vertices_from_h_representation(request.halfspaces)
        raw_vertices = tuple(
            tuple(Fraction(int(c.p), int(c.q)) for c in point) for point in derived
        )
    value, dim = convex_hull_volume(raw_vertices)
    return PolytopeVolumeResult(
        volume=value,
        dimension=dim,
        representation=representation,
    )


def convex_hull_volume(
    vertices: tuple[tuple[Fraction, ...], ...],
) -> tuple[CanonicalRational, int]:
    """Return the exact rational volume of the convex hull of rational points.

    This is the native domain kernel: it accepts mathematical values — a
    tuple of rational coordinate tuples in a consistent ambient dimension
    (at least one point) — and returns the canonical exact volume together
    with the ambient dimension.  Degenerate inputs of fewer than ``dim + 1``
    distinct affinely independent points have exact volume zero.  Raises
    ``ValueError`` when the hull enumeration exceeds the combinatorial work
    bound or the input does not describe one consistent dimension.
    """

    if not vertices:
        raise ValueError("`vertices` must be non-empty")
    dim = len(vertices[0])
    if any(len(vertex) != dim for vertex in vertices):
        raise ValueError("all vertices must share one dimension")
    points: list[list[Rational]] = [
        [Rational(fraction.numerator, fraction.denominator) for fraction in vertex]
        for vertex in vertices
    ]
    n = len(points)
    if n < dim + 1:
        return CanonicalRational.from_integer_ratio(0, 1), dim

    # Deduplicate coincident vertices.
    seen: set[tuple[Rational, ...]] = set()
    unique_vertices: list[list[Rational]] = []
    for vertex in points:
        key = tuple(vertex)
        if key not in seen:
            seen.add(key)
            unique_vertices.append(vertex)

    if len(unique_vertices) < dim + 1:
        return CanonicalRational.from_integer_ratio(0, 1), dim

    # Delegate redundant-vertex filtering to ``_polytope_volume``; the outer
    # call was previously duplicated and is removed to avoid double exact
    # work near the ``MAX_HULL_SUBFACETS`` ceiling.
    volume = _polytope_volume(unique_vertices, dim)
    return _canonical_rational(volume), dim


__all__ = [
    "compute_facet_incidence",
    "compute_polytope_volume",
    "convex_hull_volume",
]
