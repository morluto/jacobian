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
from itertools import combinations
from typing import Literal

from sympy import Matrix, Rational
from sympy.matrices.exceptions import NonInvertibleMatrixError

from jacobian.canonical import format_canonical_integer
from jacobian.math.polytope._models import (
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

# Absolute ceiling on the number of d-subsets the brute-force hull
# enumeration may consider (``C(n, d)``). Polytopes exceeding this bound
# are rejected as budget exhaustion rather than enumerated, keeping the
# exact computation bounded. This admits the unit cube through d=4 and
# the standard simplex at every supported dimension.
MAX_HULL_SUBFACETS = 200_000


def _format_rational(value: Rational) -> str:
    """Render a SymPy ``Rational`` as a canonical ``num/den`` string."""
    if value.denominator == 1:
        return format_canonical_integer(int(value.numerator))
    return (
        f"{format_canonical_integer(int(value.numerator))}/"
        f"{format_canonical_integer(int(value.denominator))}"
    )


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
    return [
        (
            [Rational(c.num, c.den) for c in hs.coefficients],
            Rational(hs.offset.num, hs.offset.den),
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
    dimensions this operation admits.
    """
    dim = len(halfspaces[0].coefficients)
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
    return unique, dim


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
        vertices, dim = _vertices_from_v_representation(request.vertices)
    else:
        assert request.halfspaces is not None
        representation = "halfspaces"
        vertices_list, dim = _vertices_from_h_representation(request.halfspaces)
        vertices = tuple(vertices_list)

    if dim == 0:
        return PolytopeVolumeResult(
            volume="1",
            dimension=0,
            representation=representation,
        )

    n = len(vertices)
    if n < dim + 1:
        raise ValueError(
            "polytope is lower-dimensional or empty; volume is zero, "
            "but the input vertices do not span the ambient dimension"
        )

    # Deduplicate coincident vertices.
    seen: set[tuple[Rational, ...]] = set()
    unique_vertices: list[tuple[Rational, ...]] = []
    for vertex in vertices:
        if vertex not in seen:
            seen.add(vertex)
            unique_vertices.append(vertex)

    if len(unique_vertices) < dim + 1:
        raise ValueError(
            "polytope is lower-dimensional or empty; volume is zero, "
            "but the input vertices do not span the ambient dimension"
        )

    points: list[list[Rational]] = [list(v) for v in unique_vertices]
    volume = _polytope_volume(points, dim)
    if volume == 0:
        raise ValueError(
            "polytope is lower-dimensional or empty; volume is zero, "
            "but the input vertices do not span the ambient dimension"
        )
    return PolytopeVolumeResult(
        volume=_format_rational(volume),
        dimension=dim,
        representation=representation,
    )


__all__ = ["compute_polytope_volume"]
