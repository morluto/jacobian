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
from itertools import product
from operator import mul
from typing import Literal

from sympy import Matrix, Rational

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.polytopes import _rational_geometry
from jacobian.math.geometry.polytopes._rational_geometry import (
    recession_cone_is_trivial,
    vertices_from_halfspaces,
)
from jacobian.math.geometry.polytopes.lattice._models import (
    MAX_BOUND_SPAN,
    MAX_DIMENSION,
    MAX_FACET_COMBINATIONS,
    MAX_FACET_TESTS,
    MAX_LATTICE_POINTS,
    MAX_TOTAL_SCAN,
    CountLatticePointsResult,
    EhrhartResult,
    EnumerateLatticePointsResult,
    LatticePoint,
    require_ehrhart_source,
)
from jacobian.math.geometry.polytopes.values import Halfspace, Vertex
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

__all__ = ["count_lattice_points", "ehrhart_polynomial", "enumerate_lattice_points"]

AdmittedGeometry = tuple[
    list[tuple[tuple[int, ...], int]],
    list[int],
    list[int],
    int,
]
"""The integer facet inequalities, bounding box, and ambient dimension."""


class LatticePolytopeAdmissionError(ValueError):
    """Native admission failure for lattice-polytope operations."""


class LatticePointBudgetError(LatticePolytopeAdmissionError):
    """Raised when the enumeration would exceed a fail-closed budget bound."""


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
    return recession_cone_is_trivial(
        [coefficients for coefficients, _offset in halfspaces], d
    )


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

    return [list(point) for point in vertices_from_halfspaces(halfspaces, dim)], dim


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
    vertices: tuple[Vertex, ...] | None,
    halfspaces: tuple[Halfspace, ...] | None,
    dimension_bound: int,
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
    if halfspaces is not None:
        if any(
            all(coefficient.num == 0 for coefficient in halfspace.coefficients)
            for halfspace in halfspaces
        ):
            raise OperationDomainValidationError(
                location=("halfspaces",),
                code="polytope.halfspace_normal_zero",
                message="half-space coefficients must not all be zero",
            )
        normalized_input = [
            (
                [c.as_fraction() for c in hs.coefficients],
                hs.offset.as_fraction(),
            )
            for hs in halfspaces
        ]
        d = len(normalized_input[0][0])
        if d > dimension_bound:
            raise LatticePolytopeAdmissionError(
                f"dimension {d} exceeds the dimension bound {dimension_bound}"
            )
        # Normalize and deduplicate before any geometry routine: vertex
        # enumeration and the recession-cone test each cost combinations
        # over the row count, so repeated or positively rescaled
        # inequalities must collapse onto their primitive form before
        # either runs.  Positive scaling preserves every inequality, so
        # this cannot change the polyhedron or any derived value.
        facets = _dedupe_normalized_halfspaces(normalized_input)
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
                    raise LatticePolytopeAdmissionError(
                        "the H-system derives vertex coordinates beyond the "
                        f"canonical {MAX_CANONICAL_RATIONAL_DIGITS}-digit "
                        "representable bound; tighten the half-space heights"
                    )
        bounded = _is_bounded_h(deduped, d)
        # Infeasibility is checked BEFORE the recession-cone rejection: an
        # infeasible system defines the empty - therefore bounded - polytope
        # even when its normals do not positively span the ambient space.
        if not bounded and (verts or _h_system_feasible(deduped)):
            raise LatticePolytopeAdmissionError(
                "the H-representation is unbounded whenever non-empty "
                "(its recession cone is nontrivial); lattice-point "
                "enumeration requires a bounded polytope"
            )
        if not verts:
            # Empty: its lattice-point set is empty, and the canonical
            # empty box scans no candidate at all.
            return [], [0] * d, [-1] * d, d
    else:
        assert vertices is not None
        vertex_models = vertices
        d = len(vertex_models[0].coordinates)
        if d > dimension_bound:
            raise LatticePolytopeAdmissionError(
                f"dimension {d} exceeds the dimension bound {dimension_bound}"
            )
        verts = [[c.as_fraction() for c in v.coordinates] for v in vertex_models]
        # Facet-combination budget: C(n,d) subsets of vertices define candidate
        # hyperplanes. For n=64,d=4 this is 635k; larger would be unbounded work.
        if d > 1:
            from math import comb as _comb

            try:
                facet_combinations = _comb(len(verts), d)
            except ValueError:
                facet_combinations = 10**18
            if facet_combinations > MAX_FACET_COMBINATIONS:
                raise LatticePointBudgetError(
                    "vertex facet enumeration exceeds the "
                    f"{MAX_FACET_COMBINATIONS}-combination budget"
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
            rational_facets = _rational_geometry.facets_from_points(verts, d)
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
        if total_scan > MAX_TOTAL_SCAN:
            raise LatticePointBudgetError(
                "integer bounding box total scan exceeds the "
                f"{MAX_TOTAL_SCAN}-point budget"
            )
    if total_scan * len(facets) > MAX_FACET_TESTS:
        raise LatticePointBudgetError(
            "the lattice-point scan exceeds the exact facet-membership work "
            f"budget of {MAX_FACET_TESTS} tests"
        )
    return facets, lo, hi, d


def _is_inside_int(
    coord: tuple[int, ...], facets: list[tuple[tuple[int, ...], int]]
) -> bool:
    """Exact integer half-space membership test for one integer point."""
    for coeffs, rhs in facets:  # noqa: SIM110 - measured faster than all(...)
        if sum(map(mul, coeffs, coord)) > rhs:
            return False
    return True


def _scan_box(
    facets: list[tuple[tuple[int, ...], int]],
    lo: list[int],
    hi: list[int],
    d: int,
    *,
    collect: bool,
) -> tuple[list[tuple[int, ...]], int]:
    """Scan the integer bounding box, returning collected points and a count.

    When ``collect`` is ``True`` every lattice point is materialised as a
    ``LatticePoint``; otherwise only the count is tracked.  The
    ``MAX_LATTICE_POINTS`` bound is a materialisation cap: enumeration
    aborts with ``LatticePointBudgetError`` once more points than the cap
    would be listed, while counting continues to the admitted scan limit
    so its small exact integer answer can still be returned.
    """
    points: list[tuple[int, ...]] = []
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
            points.append(coord)
    return points, count


def enumerate_lattice_points(
    vertices: tuple[Vertex, ...] | None,
    halfspaces: tuple[Halfspace, ...] | None,
    dimension_bound: int,
) -> EnumerateLatticePointsResult:
    """Enumerate every lattice point inside a bounded rational polytope.

    Geometry admission and result-envelope checks happen at this execution
    boundary.  The collecting scan is the only mathematical pass used to
    produce the enumeration; request parsing remains structural.
    """
    representation: Literal["vertices", "halfspaces"] = (
        "vertices" if vertices is not None else "halfspaces"
    )
    facets, lo, hi, d = _facets_and_box(vertices, halfspaces, dimension_bound)
    points, _count = _scan_box(facets, lo, hi, d, collect=True)
    return EnumerateLatticePointsResult._from_kernel(
        dimension=d,
        points=tuple(LatticePoint._from_kernel(tuple(point)) for point in points),
        representation=representation,
    )


def count_lattice_points(
    vertices: tuple[Vertex, ...] | None,
    halfspaces: tuple[Halfspace, ...] | None,
    dimension_bound: int,
) -> CountLatticePointsResult:
    """Count the lattice points inside a bounded rational polytope."""
    representation: Literal["vertices", "halfspaces"] = (
        "vertices" if vertices is not None else "halfspaces"
    )
    try:
        facets, lo, hi, d = _facets_and_box(vertices, halfspaces, dimension_bound)
    except LatticePolytopeAdmissionError as exc:
        raise OperationDomainValidationError(
            location=("vertices", "halfspaces"),
            code="polytope.lattice_points.admission",
            message=str(exc),
        ) from exc
    _points, count = _scan_box(facets, lo, hi, d, collect=False)
    return CountLatticePointsResult._from_kernel(
        dimension=d,
        point_count=count,
        representation=representation,
    )


def _ehrhart_aggregate_scan_bound(
    vertices: tuple[Vertex, ...], max_dilation: int
) -> int:
    """Bound all dilation boxes from the unscaled integral source."""

    spans = [
        max(vertex.coordinates[axis].num for vertex in vertices)
        - min(vertex.coordinates[axis].num for vertex in vertices)
        for axis in range(len(vertices[0].coordinates))
    ]
    for span in spans:
        if max_dilation * span + 1 > MAX_BOUND_SPAN:
            raise LatticePointBudgetError(
                "the Ehrhart dilation range exceeds the "
                f"{MAX_BOUND_SPAN}-point per-axis span bound"
            )
    total_scan = 0
    for dilation in range(1, max_dilation + 1):
        scan = 1
        for span in spans:
            scan *= dilation * span + 1
            if scan > MAX_TOTAL_SCAN:
                raise LatticePointBudgetError(
                    "the Ehrhart dilation range exceeds the aggregate "
                    f"{MAX_TOTAL_SCAN}-candidate scan budget"
                )
        total_scan += scan
        if total_scan > MAX_TOTAL_SCAN:
            raise LatticePointBudgetError(
                "the Ehrhart dilation range exceeds the aggregate "
                f"{MAX_TOTAL_SCAN}-candidate scan budget"
            )
    return total_scan


def _ehrhart_scan_plans(
    vertices: tuple[Vertex, ...], max_dilation: int
) -> list[AdmittedGeometry]:
    """Admit every scaled geometry and the aggregate scan before enumeration."""
    dimension = len(vertices[0].coordinates)
    _ehrhart_aggregate_scan_bound(vertices, max_dilation)
    if dimension > 1:
        facet_work = math.comb(len(vertices), dimension) * max_dilation
        if facet_work > MAX_FACET_COMBINATIONS:
            raise LatticePointBudgetError(
                "the Ehrhart dilation range exceeds the aggregate "
                f"{MAX_FACET_COMBINATIONS}-combination facet budget"
            )
    plans: list[AdmittedGeometry] = []
    total_scan = 0
    total_facet_tests = 0
    for dilation in range(1, max_dilation + 1):
        scaled = tuple(
            Vertex(
                coordinates=tuple(
                    CanonicalRational.from_integer_ratio(
                        coordinate.num * dilation, coordinate.den
                    )
                    for coordinate in vertex.coordinates
                )
            )
            for vertex in vertices
        )
        plan = _facets_and_box(scaled, None, MAX_DIMENSION)
        plans.append(plan)
        _facets, lo, hi, _dimension = plan
        scan = 1
        for lower, upper in zip(lo, hi, strict=True):
            scan *= upper - lower + 1
        total_scan += scan
        total_facet_tests += scan * len(_facets)
    if total_scan > MAX_TOTAL_SCAN:
        raise LatticePointBudgetError(
            "the Ehrhart dilation range exceeds the aggregate "
            f"{MAX_TOTAL_SCAN}-candidate scan budget"
        )
    if total_facet_tests > MAX_FACET_TESTS:
        raise LatticePointBudgetError(
            "the Ehrhart dilation range exceeds the aggregate exact "
            f"facet-membership budget of {MAX_FACET_TESTS} tests"
        )
    return plans


def _interpolate_ehrhart(values: list[int], degree: int) -> list[Fraction]:
    """Interpolate and return ascending-power exact coefficients."""
    coefficients = [Fraction(0) for _ in range(degree + 1)]
    for sample in range(degree + 1):
        basis = [Fraction(1)]
        denominator = 1
        for other in range(degree + 1):
            if other == sample:
                continue
            denominator *= sample - other
            updated = [Fraction(0)] * (len(basis) + 1)
            for power, coefficient in enumerate(basis):
                updated[power] -= coefficient * other
                updated[power + 1] += coefficient
            basis = updated
        for power, coefficient in enumerate(basis):
            coefficients[power] += values[sample] * coefficient / denominator
    return coefficients


def ehrhart_polynomial(
    vertices: tuple[Vertex, ...], degree_bound: int, max_dilation: int
) -> EhrhartResult:
    """Count integral dilates and recover their exact Ehrhart polynomial."""
    try:
        require_ehrhart_source(vertices, degree_bound, max_dilation)
        plans = _ehrhart_scan_plans(vertices, max_dilation)
    except LatticePolytopeAdmissionError as exc:
        raise OperationDomainValidationError(
            location=("vertices", "max_dilation"),
            code="polytope.ehrhart.admission",
            message=str(exc),
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("vertices", "degree_bound", "max_dilation"),
            code="polytope.ehrhart.invalid_source",
            message=str(exc),
        ) from exc

    values: list[int] = [1]
    for facets, lo, hi, dimension in plans:
        _points, count = _scan_box(facets, lo, hi, dimension, collect=False)
        values.append(count)

    degree = degree_bound
    coefficients = _interpolate_ehrhart(values, degree)

    def evaluate(dilation: int) -> int:
        result = Fraction(0)
        for coefficient in reversed(coefficients):
            result = result * dilation + coefficient
        if result.denominator != 1:
            raise ValueError("interpolated Ehrhart value is not integral")
        return result.numerator

    if any(evaluate(dilation) != values[dilation] for dilation in range(len(values))):
        raise OperationDomainValidationError(
            location=("degree_bound",),
            code="polytope.ehrhart.degree_insufficient",
            message="degree_bound does not reproduce every requested dilation count",
        )
    polynomial = RationalPolynomial(
        variables=("t",),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(value),
                    exponents=(power,),
                )
                for power, value in reversed(tuple(enumerate(coefficients)))
                if value
            )
        ),
    )
    return EhrhartResult._from_kernel(
        vertices=vertices,
        dimension=len(vertices[0].coordinates),
        degree_bound=degree,
        max_dilation=max_dilation,
        counts=tuple((dilation, values[dilation]) for dilation in range(len(values))),
        polynomial=polynomial,
    )
