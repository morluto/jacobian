"""Domain-owned exact lattice-point enumeration and counting operations.

Two operations are exposed over a bounded rational polytope given in
either V-representation (vertices) or H-representation (half-spaces):

* ``enumerate`` returns every lattice (integer) point inside the polytope.
* ``count`` returns the number of lattice points without listing them.

Both are exact.  The implementation never uses floating point: it builds
the facet half-spaces of the convex hull with SymPy's exact rational
linear algebra, derives a finite integer bounding box, and tests each
candidate integer point against the exact half-space inequalities.

For a V-representation the facets are enumerated exactly: every
``d``-subset of vertices defines a candidate hyperplane whose normal is
the null space of the vertex differences (SymPy ``Matrix.nullspace``);
the hyperplane is a facet when all vertices lie on one closed side.
The convex hull of finitely many points is always bounded, so the
bounding box is the per-axis min/max of the vertices.

For an H-representation ``{x : A x <= b}`` the polytope is bounded iff
its recession cone ``{d : A d <= 0}`` is ``{0}``, which holds iff the
origin lies strictly in the interior of the convex hull of the rows of
``A``.  That interior test is itself an exact facet enumeration of the
row normals.  Once boundedness is established the bounding box is the
per-axis min/max of the enumerated vertices (every ``C(m, d)``
subsystem of half-space boundaries that satisfies all half-spaces).
When that enumeration finds no vertex, the bounded polyhedron is
empty and both operations return their exact empty value (count zero,
no points).
"""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import combinations, product
from typing import Literal

from sympy import Matrix, Rational
from sympy.matrices.exceptions import NonInvertibleMatrixError

from jacobian.canonical import format_canonical_integer
from jacobian.math.lattice_polytopes._models import (
    MAX_BOUND_SPAN,
    MAX_LATTICE_POINTS,
    CountLatticePointsResult,
    EnumerateLatticePointsResult,
    LatticePoint,
    LatticePolytopeRequest,
    Vertex,
)

__all__ = ["count_lattice_points", "enumerate_lattice_points"]

AdmittedGeometry = tuple[
    list[tuple[tuple[int, ...], int]],
    list[int],
    list[int],
    int,
]
"""The integer facet inequalities, bounding box, and ambient dimension."""


class LatticePointBudgetError(ValueError):
    """Raised when the enumeration would exceed a fail-closed budget bound."""


def _hyperplane_normal(points: list[list[Rational]]) -> Matrix | None:
    """Return a normal vector to the hyperplane through ``points``.

    ``points`` holds exactly ``d`` points in ``d``-dimensional space.
    The normal is any non-zero vector in the null space of the matrix
    whose rows are ``points[i] - points[0]`` for ``i >= 1``.  Returns
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
    verts: list[list[Rational]], d: int
) -> list[tuple[Matrix, Rational]]:
    """Enumerate the facet half-spaces of the convex hull of ``verts``.

    Each facet is returned as ``(normal, offset)`` with the orientation
    ``normal . x <= offset`` satisfied by every vertex.  Facets are
    deduplicated by their (normal, offset) signature.
    """
    facets: list[tuple[Matrix, Rational]] = []
    for combo in combinations(range(len(verts)), d):
        pts = [verts[i] for i in combo]
        normal = _hyperplane_normal(pts)
        if normal is None:
            continue
        offset = sum(normal[k] * pts[0][k] for k in range(d))
        residuals = [sum(normal[k] * v[k] for k in range(d)) - offset for v in verts]
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


def _is_bounded_h(halfspaces: list[tuple[list[Rational], Rational]], d: int) -> bool:
    """Decide whether ``{x : A x <= b}`` is bounded.

    The polytope is bounded iff its recession cone ``{d : A d <= 0}`` is
    ``{0}``, which holds iff the origin lies strictly in the interior of
    the convex hull of the rows of ``A``.  That interior test is an exact
    facet enumeration of the row normals.  The rows must positively span
    ``R^d``; equivalently their convex hull must be full-dimensional and
    contain the origin strictly interior.  If the rows' hull is not
    full-dimensional (e.g. normals ``(1,0)`` and ``(1,1)`` in 2D) the
    polyhedron is unbounded even though the hull's single facet has
    positive offset.
    """
    normals = [coeffs for coeffs, _ in halfspaces]
    if d == 1:
        positive = any(n[0] > 0 for n in normals)
        negative = any(n[0] < 0 for n in normals)
        return positive and negative
    rows = [list(n) for n in normals]
    # Positive spanning requires the normals' convex hull to be full-
    # dimensional.  Check affine rank of the point set.
    if len(rows) < d + 1:
        return False
    # Affine rank via differences from first point
    diff_rows = [
        [rows[i][k] - rows[0][k] for k in range(d)] for i in range(1, len(rows))
    ]
    # Use Rational matrix rank for exactness
    try:
        if Matrix(diff_rows).rank() < d:
            return False
    except Exception:
        return False
    hull_facets = _facets_from_points([Matrix(r) for r in rows], d)
    if not hull_facets:
        return False
    return all(Rational(0) < offset for _, offset in hull_facets)


def _vertices_from_h_representation(
    halfspaces: list[tuple[list[Rational], Rational]],
) -> tuple[list[list[Rational]], int]:
    """Enumerate the vertices of ``{x : A x <= b}`` exactly.

    Each vertex is the unique intersection of ``d`` affinely independent
    half-space boundaries ``<a_i, x> = b_i``.  Every ``C(m, d)`` subsystem
    is solved exactly with SymPy and retained when it satisfies all
    half-spaces.  Bounded, exact vertex enumeration for the small
    dimensions this operation admits.
    """
    dim = len(halfspaces[0][0])
    n = len(halfspaces)

    found: list[list[Rational]] = []
    for indices in combinations(range(n), dim):
        rows = [halfspaces[i] for i in indices]
        mat = Matrix([rows[i][0] for i in range(dim)])
        rhs = Matrix([[rows[i][1]] for i in range(dim)])
        try:
            det = mat.det()
        except Exception:
            continue
        if det == 0:
            continue
        try:
            solution = mat.solve(rhs)
        except (NonInvertibleMatrixError, ValueError):
            continue
        point = [Rational(solution[i, 0]) for i in range(dim)]
        feasible = True
        for coeffs, offset in halfspaces:
            lhs = sum(c * p for c, p in zip(coeffs, point, strict=True))
            if lhs > offset:
                feasible = False
                break
        if not feasible:
            continue
        found.append(point)

    unique: list[list[Rational]] = []
    seen: set[tuple[Rational, ...]] = set()
    for point in found:
        key = tuple(point)
        if key not in seen:
            seen.add(key)
            unique.append(point)
    return unique, dim


def _rational_vertices(
    request: LatticePolytopeRequest,
) -> tuple[list[list[Rational]], int, Literal["vertices", "halfspaces"]]:
    """Return the rational vertices and ambient dimension of the polytope."""
    if request.vertices is not None:
        dim = request.dimension()
        verts = [[c.as_fraction() for c in v.coordinates] for v in request.vertices]
        return verts, dim, "vertices"
    assert request.halfspaces is not None
    halfspaces = [
        (
            [c.as_fraction() for c in hs.coefficients],
            hs.offset.as_fraction(),
        )
        for hs in request.halfspaces
    ]
    verts, _dim = _vertices_from_h_representation(halfspaces)
    return verts, _dim, "halfspaces"


def _floor(value: Rational) -> int:
    """Exact integer floor of a rational (Fraction or SymPy ``Rational``)."""
    frac = Fraction(value)
    return frac.numerator // frac.denominator


def _ceil(value: Rational) -> int:
    """Exact integer ceiling of a rational (Fraction or SymPy ``Rational``)."""
    frac = Fraction(value)
    return -((-frac.numerator) // frac.denominator)


def _bounding_box(verts: list[list[Rational]], d: int) -> tuple[list[int], list[int]]:
    """Return the tight integer per-axis bounds containing every lattice point.

    Integer points of the polytope lie in ``[ceil(min), floor(max)]`` per
    axis; rounding outwards would admit candidates that cannot lie in the
    polytope and inflate the scanned span past admission budgets.
    """
    lo = [_ceil(min(v[k] for v in verts)) for k in range(d)]
    hi = [_floor(max(v[k] for v in verts)) for k in range(d)]
    return lo, hi


def _dedupe_normalized_halfspaces(
    halfspaces: list[tuple[list[Fraction], Fraction]],
) -> list[tuple[tuple[int, ...], int]]:
    """Return the distinct half-spaces in primitive integer normal form.

    Each inequality ``<a, x> <= b`` is scaled by the positive LCM of its
    denominators, divided by the GCD of the resulting integers, and kept
    as ``(A, C)`` with ``sum(A_k * x_k) <= C``.  Positive scaling
    preserves the inequality exactly, so two half-spaces share a normal
    form iff they define the same constraint; duplicates collapse and
    repeated inequalities cannot multiply the scan's membership work.
    """
    unique: dict[tuple[tuple[int, ...], int], tuple[tuple[int, ...], int]] = {}
    for coeffs, offset in halfspaces:
        scale = 1
        for value in (*coeffs, offset):
            scale = scale * value.denominator // math.gcd(scale, value.denominator)
        ints = [int(value * scale) for value in coeffs]
        rhs = int(offset * scale)
        g = 0
        for item in (*ints, rhs):
            g = math.gcd(g, abs(item))
        if g > 1:
            ints = [value // g for value in ints]
            rhs //= g
        key = (tuple(ints), rhs)
        if key not in unique:
            unique[key] = key
    return list(unique.values())


def _to_integer_facet(
    normal: Matrix, offset: Rational, d: int
) -> tuple[tuple[int, ...], int]:
    """Scale a rational facet ``<n, x> <= b`` to integer coefficients.

    Multiplying the inequality by the LCM of the component denominators
    yields integer ``A`` and integer ``C`` so that membership of an
    integer point is the exact integer test ``sum(A_k * x_k) <= C``.
    """
    fracs = [Fraction(int(normal[k].p), int(normal[k].q)) for k in range(d)]
    bound = Fraction(offset)
    dens = [f.denominator for f in fracs] + [bound.denominator]
    scale = 1
    for den in dens:
        scale = scale * den // math.gcd(scale, den)
    coeffs = tuple(int(f * scale) for f in fracs)
    rhs = int(bound * scale)
    return coeffs, rhs


def _h_system_feasible(
    halfspaces: list[tuple[list[Rational], Rational]],
) -> bool:
    """Decide exactly whether ``{x : A x <= b}`` contains any rational point.

    Uses SymPy's exact simplex with a zero objective; infeasibility is the
    distinguishing test between a bounded-but-empty H-polytope (admitted as
    the canonical empty geometry) and an unbounded polyhedron without
    vertices (rejected).
    """
    from sympy import Rational as _SRational
    from sympy.solvers.simplex import InfeasibleLPError, linprog

    matrix = [[_SRational(c) for c in coeffs] for coeffs, _ in halfspaces]
    rhs = [_SRational(offset) for _, offset in halfspaces]
    objective = [0] * len(matrix[0])
    # SymPy's simplex constrains every variable nonnegative unless bounds
    # are given; lattice coordinates are unrestricted integers, so the probe
    # must pass explicit (None, None) bounds or systems like x <= -1 would
    # be misread as infeasible and an unbounded polyhedron admitted as the
    # empty geometry.
    bounds = [(None, None)] * len(matrix[0])
    try:
        linprog(objective, matrix, rhs, bounds=bounds)
        return True
    except InfeasibleLPError:
        return False


def _facets_and_box(  # noqa: C901
    request: LatticePolytopeRequest,
) -> AdmittedGeometry:
    """Build the integer facet inequalities and the integer bounding box.

    Raises ``ValueError`` when the H-representation is unbounded, and
    ``LatticePointBudgetError`` when the bounding box spans more than
    ``MAX_BOUND_SPAN`` integer points in any axis or more than the total
    scan budget.  A bounded but empty H-polytope admits no vertex, so it
    returns the canonical empty geometry (no facets, per-axis ``[0, -1]``
    boxes) whose scan is exactly empty.
    """
    facets: list[tuple[tuple[int, ...], int]] = []
    if request.halfspaces is not None:
        halfspaces = [
            (
                [c.as_fraction() for c in hs.coefficients],
                hs.offset.as_fraction(),
            )
            for hs in request.halfspaces
        ]
        d = request.dimension()
        # Normalize and deduplicate before any geometry routine: vertex
        # enumeration and the recession-cone test each cost combinations
        # over the row count, so repeated or positively rescaled
        # inequalities must collapse onto their primitive form before
        # either runs.  Positive scaling preserves every inequality, so
        # this cannot change the polyhedron or any derived value.
        facets = _dedupe_normalized_halfspaces(halfspaces)
        deduped = [
            ([Fraction(a) for a in coeffs], Fraction(rhs)) for coeffs, rhs in facets
        ]
        verts, _ = _vertices_from_h_representation(deduped)
        # Solving the H-system can derive coordinates taller than every
        # input component (e.g. x <= 1/N with -x <= -1/N pins x = N);
        # each derived vertex coordinate must stay inside the canonical
        # rational representable bound or LatticePoint construction would
        # fail after acceptance.
        from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
        from jacobian.canonical import format_canonical_integer as _fmt

        for point in verts:
            for value in point:
                # The canonical digit bound measures magnitude: canonical
                # integers carry the sign outside the digit count, exactly
                # as the CanonicalRational validator admits them.
                if (
                    len(_fmt(value.numerator).lstrip("-"))
                    > MAX_CANONICAL_RATIONAL_DIGITS
                    or len(_fmt(value.denominator).lstrip("-"))
                    > MAX_CANONICAL_RATIONAL_DIGITS
                ):
                    raise ValueError(
                        "the H-system derives vertex coordinates beyond the "
                        f"canonical {MAX_CANONICAL_RATIONAL_DIGITS}-digit "
                        "representable bound; tighten the half-space heights"
                    )
        bounded = _is_bounded_h(deduped, d)
        # Infeasibility is checked BEFORE the recession-cone rejection: an
        # infeasible system defines the empty - therefore bounded - polytope
        # even when its normals do not positively span the ambient space.
        if not bounded and (verts or _h_system_feasible(deduped)):
            raise ValueError(
                "the H-representation is unbounded whenever non-empty "
                "(its recession cone is nontrivial); lattice-point "
                "enumeration requires a bounded polytope"
            )
        if not verts:
            # Empty: its lattice-point set is empty, and the canonical
            # empty box scans no candidate at all.
            return [], [0] * d, [-1] * d, d
    else:
        d = request.dimension()
        vertex_models: tuple[Vertex, ...] = request.vertices  # type: ignore[assignment]
        verts = [[c.as_fraction() for c in v.coordinates] for v in vertex_models]
        # Facet-combination budget: C(n,d) subsets of vertices define candidate
        # hyperplanes. For n=64,d=4 this is 635k; larger would be unbounded work.
        if d > 1:
            from math import comb as _comb

            try:
                facet_combinations = _comb(len(verts), d)
            except ValueError:
                facet_combinations = 10**18
            if facet_combinations > 700_000:
                raise LatticePointBudgetError(
                    "vertex facet enumeration exceeds the 700k-combination budget"
                )
            # Lower-dimensional hulls: if vertices do not span full dimension,
            # the facet enumeration would be empty and the scan would be wrong.
            # Detect affine rank regardless of vertex count and reject; this
            # is exact: a lower-dimensional polytope's lattice points are
            # still well-defined, but our facet method assumes full
            # dimension. Rejecting is fail-closed.
            diffs = Matrix(
                [
                    [verts[i][k] - verts[0][k] for k in range(d)]
                    for i in range(1, len(verts))
                ]
            )
            if diffs.rank() < d:
                raise LatticePointBudgetError(
                    "V-representation is not full-dimensional; lower-dimensional hulls require exact handling"
                )
        if d == 1:
            # The single facet pair is the interval endpoints.
            low = min(v[0] for v in verts)
            high = max(v[0] for v in verts)
            rational_facets = [
                (Matrix([Rational(1)]), high),
                (Matrix([Rational(-1)]), -low),
            ]
        else:
            rational_facets = _facets_from_points(verts, d)
        facets = [
            _to_integer_facet(normal, offset, d) for normal, offset in rational_facets
        ]
    lo, hi = _bounding_box(verts, d)
    spans = [hi[k] - lo[k] + 1 for k in range(d)]
    if any(span <= 0 for span in spans):
        # An axis whose ceil(min) > floor(max) holds no integer candidate:
        # the full Cartesian scan is exactly empty regardless of the other
        # axes, so return the canonical empty geometry BEFORE the budget
        # products reject an exact empty result.
        return [], [0] * d, [-1] * d, d
    for span in spans:
        if span > MAX_BOUND_SPAN:
            raise LatticePointBudgetError(
                "the integer bounding box exceeds the "
                f"{MAX_BOUND_SPAN}-point per-axis span bound"
            )
    # Total scan bound: product of per-axis spans, not just per-axis.
    total_scan = 1
    for span in spans:
        total_scan *= span
        if total_scan > 10_000_000:
            raise LatticePointBudgetError(
                "integer bounding box total scan exceeds the 10M-point budget"
            )
    return facets, lo, hi, d


def _is_inside_int(
    coord: tuple[int, ...], facets: list[tuple[tuple[int, ...], int]]
) -> bool:
    """Exact integer half-space membership test for one integer point."""
    for coeffs, rhs in facets:
        if sum(a * c for a, c in zip(coeffs, coord, strict=True)) > rhs:
            return False
    return True


def _scan_box(
    facets: list[tuple[tuple[int, ...], int]],
    lo: list[int],
    hi: list[int],
    d: int,
    *,
    collect: bool,
) -> tuple[list[LatticePoint], int]:
    """Scan the integer bounding box, returning collected points and a count.

    When ``collect`` is ``True`` every lattice point is materialised as a
    ``LatticePoint``; otherwise only the count is tracked.  The
    ``MAX_LATTICE_POINTS`` bound is a materialisation cap: enumeration
    aborts with ``LatticePointBudgetError`` once more points than the cap
    would be listed, while counting continues to the admitted scan limit
    so its small exact integer answer can still be returned.
    """
    points: list[LatticePoint] = []
    count = 0
    for coord in product(*(range(lo[k], hi[k] + 1) for k in range(d))):
        if not _is_inside_int(coord, facets):
            continue
        count += 1
        if collect and count > MAX_LATTICE_POINTS:
            raise LatticePointBudgetError(
                "lattice-point enumeration exceeds the "
                f"{MAX_LATTICE_POINTS}-point budget bound"
            )
        if collect:
            points.append(
                LatticePoint(
                    coordinates=tuple(format_canonical_integer(c) for c in coord)
                )
            )
    return points, count


_MAX_OUTPUT_BYTES = 10 * 1024 * 1024


def enumeration_output_admission(
    request: LatticePolytopeRequest,
) -> None:
    """Reject enumerations whose serialized artifact cannot fit the output limits.

    Runs the exact count pass (bounded by the admitted scan budget) and
    checks both the materialization cap and the conservative canonical
    JSON size estimate.  Reuses the geometry computed once during request
    validation instead of repeating the facet enumeration.  Raises
    ``ValueError`` so the enumerate-specific request boundary turns
    oversize artifacts into invalid requests instead of internal
    operation failures.
    """

    facets, lo, hi, d = request.admitted_geometry()
    _, count = _scan_box(facets, lo, hi, d, collect=False)
    if count > MAX_LATTICE_POINTS:
        raise ValueError(
            "lattice-point enumeration exceeds the "
            f"{MAX_LATTICE_POINTS}-point budget bound"
        )
    # Per-point JSON is roughly {"coordinates":["x",...]} with d strings.
    # Max coordinate string length is bounded by the bounding box; the
    # canonical formatter is digit-limit-safe for 32,768-digit coordinates.
    max_coord_len = max(
        max(
            len(format_canonical_integer(lo[k])),
            len(format_canonical_integer(hi[k])),
        )
        for k in range(d)
    )
    # Conservative: 20 bytes overhead + per-coordinate (digits+quotes+comma) + brackets
    per_point = 20 + d * (max_coord_len + 4)
    base_overhead = 80
    estimated = base_overhead + count * per_point
    if estimated > _MAX_OUTPUT_BYTES:
        raise ValueError(
            "lattice-point enumeration would exceed the 10 MiB canonical JSON output limit"
        )


def enumerate_lattice_points(
    request: LatticePolytopeRequest,
) -> EnumerateLatticePointsResult:
    """Enumerate every lattice point inside a bounded rational polytope.

    The admitted geometry (exact facet inequalities and integer bounding
    box) was computed once during request validation and is reused here,
    so execution performs only the collecting scan.
    """
    representation: Literal["vertices", "halfspaces"] = (
        "vertices" if request.vertices is not None else "halfspaces"
    )
    facets, lo, hi, d = request.admitted_geometry()
    points, _count = _scan_box(facets, lo, hi, d, collect=True)
    return EnumerateLatticePointsResult(
        dimension=d,
        point_count=len(points),
        points=tuple(points),
        representation=representation,
    )


def count_lattice_points(
    request: LatticePolytopeRequest,
) -> CountLatticePointsResult:
    """Count the lattice points inside a bounded rational polytope."""
    representation: Literal["vertices", "halfspaces"] = (
        "vertices" if request.vertices is not None else "halfspaces"
    )
    facets, lo, hi, d = request.admitted_geometry()
    _points, count = _scan_box(facets, lo, hi, d, collect=False)
    return CountLatticePointsResult(
        dimension=d,
        point_count=count,
        representation=representation,
    )
