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
from sympy.matrices.exceptions import NonInvertibleMatrixError

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.polytope._models import (
    MAX_BOUNDEDNESS_COMBINATIONS,
    MAX_HULL_SUBFACETS,
    Halfspace,
    PolytopeVolumeRequest,
    PolytopeVolumeResult,
    Vertex,
)

# Absolute combinatorial ceiling: reject vertex enumeration that would
# attempt to solve more subsystems than this. With ``MAX_FACETS = 64``
# and ``d = 6`` the worst case is ``C(64, 6) = 74,974,368``, so the bound
# is the practical gate on budget exhaustion.
MAX_SUBSYSTEM_SOLVES = 5_000_000

# ``MAX_HULL_SUBFACETS`` (the C(n, d) hull-enumeration ceiling) is owned
# by ``_models`` beside the other published request bounds, which quote it
# in their schema-visible descriptions.


def _hyperplane_normal(points: list[list[Rational]]) -> Matrix | None:
    """Return a normal vector to the hyperplane through ``points``.

    ``points`` holds exactly ``d`` points in ``d``-dimensional space.
    The normal is any non-zero vector in the null space of the matrix
    whose rows are ``points[i] - points[0]`` for ``i >= 1``. Returns
    ``None`` when the points are affinely dependent (no unique
    hyperplane).
    """

    d = len(points)
    if d == 1:
        return Matrix([Rational(1)])
    diffs = Matrix(
        [[points[i][k] - points[0][k] for k in range(d)] for i in range(1, d)]
    )
    basis = diffs.nullspace()
    if not basis:
        return None
    return basis[0]


def _facets_from_points(
    verts: list[list[Rational]], dim: int
) -> list[tuple[Matrix, Rational]]:
    """Enumerate the facet half-spaces of the convex hull of ``verts``.

    Each facet is returned as ``(normal, offset)`` with the orientation
    ``normal . x <= offset`` satisfied by every vertex. Facets are
    deduplicated by their (normal, offset) signature.
    """

    facets: list[tuple[Matrix, Rational]] = []
    for combo in combinations(range(len(verts)), dim):
        pts = [verts[i] for i in combo]
        normal = _hyperplane_normal(pts)
        if normal is None:
            continue
        offset = sum(normal[k] * pts[0][k] for k in range(dim))
        residuals = [sum(normal[k] * v[k] for k in range(dim)) - offset for v in verts]
        if all(v <= 0 for v in residuals):
            facets.append((normal, offset))
        elif all(v >= 0 for v in residuals):
            facets.append((Matrix([-x for x in normal]), -offset))
    seen: set[tuple[tuple[Rational, ...], Rational]] = set()
    out: list[tuple[Matrix, Rational]] = []
    for normal, offset in facets:
        key = (tuple(Rational(x) for x in normal), offset)
        if key not in seen:
            seen.add(key)
            out.append((normal, offset))
    return out


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
    diff_rows = [
        [normals[i][k] - normals[0][k] for k in range(dim)]
        for i in range(1, len(normals))
    ]
    try:
        if Matrix(diff_rows).rank() < dim:
            return False
    except Exception:
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
    hull_facets = _facets_from_points(normals, dim)
    if not hull_facets:
        return False
    return all(Rational(0) < offset for _, offset in hull_facets)


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


def _solve_subsystem_vertex(
    rows: list[tuple[list[Rational], Rational]],
    indices: tuple[int, ...],
    dim: int,
) -> tuple[Rational, ...] | None:
    """Solve the dim-subsystem of hyperplanes, or None if singular."""
    mat = Matrix([rows[i][0] for i in indices])
    rhs = Matrix([[rows[i][1]] for i in indices])
    try:
        det = mat.det()
    except Exception:
        return None
    if det == 0:
        return None
    try:
        solution = mat.solve(rhs)
    except (NonInvertibleMatrixError, ValueError):
        return None
    return tuple(Rational(solution[i, 0]) for i in range(dim))


def _is_feasible(
    point: tuple[Rational, ...],
    rows: list[tuple[list[Rational], Rational]],
) -> bool:
    """Return True if ``point`` satisfies every half-space ``<a, x> <= b``."""
    for coeffs, offset in rows:
        lhs = sum(c * p for c, p in zip(coeffs, point, strict=True))
        if lhs > offset:
            return False
    return True


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

    found: list[tuple[Rational, ...]] = []
    for indices in combinations(range(n), dim):
        point = _solve_subsystem_vertex(rows, indices, dim)
        if point is None or not _is_feasible(point, rows):
            continue
        found.append(point)

    # Deduplicate coincident vertices found from different subsystems.
    unique: list[tuple[Rational, ...]] = []
    unique_seen: set[tuple[Rational, ...]] = set()
    for point in found:
        if point not in unique_seen:
            unique_seen.add(point)
            unique.append(point)
    result: tuple[list[tuple[Rational, ...]], int] = (unique, dim)
    return result


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
        raw_vertices = tuple(
            tuple(c.as_fraction() for c in vertex.coordinates)
            for vertex in request.vertices
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


__all__ = ["compute_polytope_volume", "convex_hull_volume"]
