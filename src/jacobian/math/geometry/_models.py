"""Exact rational planar-geometry wire contracts."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Literal, Self

from pydantic import ConfigDict, Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer

MAX_CONFIGURATION_POINTS = 32
MAX_COORDINATE_DIGITS = 256
# Serialized-output budget for the circumradius profile, kept below the 10 MiB
# transport envelope to leave room for request and JSON overhead.
_MAX_PROFILE_OUTPUT_CHARS = 8_000_000
# Joint work bound for the exhaustive general-position search.  The sweep
# performs one exact 4x4 determinant per point quadruple, so the determinant
# count grows as C(n,4) while every Fraction multiplication grows
# quadratically in coordinate digit count; the admitted work proxy is
# ``C(n,4) * max_digits**2`` (measured reference: 32 points x 32 digits
# costs about 36M proxy units and roughly 16s, so a 1M-unit budget keeps an
# accepted request well under a second for both the search and its
# source-binding replay).
_MAX_GENERAL_POSITION_DETERMINANT_WORK = 1_000_000


def _require_bounded_point(point: RationalPoint2D) -> None:
    require_bounded_rational(
        point.x, max_digits=MAX_COORDINATE_DIGITS, label="point x-coordinate"
    )
    require_bounded_rational(
        point.y, max_digits=MAX_COORDINATE_DIGITS, label="point y-coordinate"
    )


def _require_bounded_configuration(points: tuple[RationalPoint2D, ...]) -> None:
    for point in points:
        _require_bounded_point(point)


def _max_coordinate_digits(points: tuple[RationalPoint2D, ...]) -> int:
    return max(
        max(
            len(p.x.num.lstrip("-")),
            len(p.x.den),
            len(p.y.num.lstrip("-")),
            len(p.y.den),
        )
        for p in points
    )


def _require_general_position_work_bound(
    points: tuple[RationalPoint2D, ...],
) -> None:
    n = len(points)
    if n == 0:
        return
    max_digits = _max_coordinate_digits(points)
    quadruples = n * (n - 1) * (n - 2) * (n - 3) // 24
    work = quadruples * max_digits * max_digits
    if work > _MAX_GENERAL_POSITION_DETERMINANT_WORK:
        raise ValueError(
            f"general-position search with {n} points and {max_digits}-digit "
            f"coordinates exceeds the exhaustive work bound "
            f"(C(n,4)*digits^2={work} > "
            f"{_MAX_GENERAL_POSITION_DETERMINANT_WORK}); reduce point count "
            "or coordinate size"
        )


def _require_circumradius_output_bound(points: tuple[RationalPoint2D, ...]) -> None:
    """Bound the serialized profile size before execution.

    Derivation from exact rational growth with ``d`` = max coordinate digits:
    a coordinate difference has numerator and denominator of at most ``2d``
    digits; a squared length reaches ``8d`` digits on each side; the squared
    circumradius ``|AB||BC||CA| / (2*cross)^2`` therefore carries at most
    ``40d`` digits in its numerator and ``40d`` in its denominator. Allowing
    80 characters of sign/slash/JSON overhead per entry gives the conservative
    per-entry estimate ``80*d + 80`` characters.
    """

    n = len(points)
    if n == 0:
        return
    max_digits = _max_coordinate_digits(points)
    triples = n * (n - 1) * (n - 2) // 6
    estimated_chars = triples * (80 * max_digits + 80)
    if estimated_chars > _MAX_PROFILE_OUTPUT_CHARS:
        raise ValueError(
            f"circumradius profile for {n} points with {max_digits}-digit "
            f"coordinates can serialize up to {estimated_chars} characters "
            f"(worst-case rational growth), exceeding the "
            f"{_MAX_PROFILE_OUTPUT_CHARS}-character output budget; reduce "
            "point count or coordinate size"
        )


def _is_collinear_frac(
    a: tuple[Fraction, Fraction],
    b: tuple[Fraction, Fraction],
    c: tuple[Fraction, Fraction],
) -> bool:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) == 0


def _det3_frac(m: tuple[tuple[Fraction, ...], ...]) -> Fraction:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _det4_frac(m: tuple[tuple[Fraction, ...], ...]) -> Fraction:
    result = Fraction(0)
    for col in range(4):
        sub = tuple(
            tuple(row[col2] for col2 in range(4) if col2 != col) for row in m[1:]
        )
        cofactor = _det3_frac(sub)
        sign = 1 if col % 2 == 0 else -1
        result += sign * m[0][col] * cofactor
    return result


def _expected_collinear_indices(
    pts: list[tuple[Fraction, Fraction]],
) -> set[tuple[int, int, int]]:
    result: set[tuple[int, int, int]] = set()
    for i, j, k in combinations(range(len(pts)), 3):
        if _is_collinear_frac(pts[i], pts[j], pts[k]):
            result.add((i, j, k))
    return result


def _expected_concyclic_indices(
    pts: list[tuple[Fraction, Fraction]],
    collinear_set: set[tuple[int, int, int]],
) -> set[tuple[int, int, int, int]]:
    result: set[tuple[int, int, int, int]] = set()
    n = len(pts)
    for i, j, k, m in combinations(range(n), 4):
        if (
            (i, j, k) in collinear_set
            or (i, j, m) in collinear_set
            or (i, k, m) in collinear_set
            or (j, k, m) in collinear_set
        ):
            continue
        rows = tuple(
            (px * px + py * py, px, py, Fraction(1))
            for px, py in (pts[i], pts[j], pts[k], pts[m])
        )
        if _det4_frac(rows) == 0:
            result.add((i, j, k, m))
    return result


def _check_witness_sorted_distinct(
    indices: tuple[int, ...], n: int, expected: int, label: str
) -> None:
    if len(indices) != expected or len(set(indices)) != expected:
        raise ValueError(f"{label} indices must be {expected} distinct values")
    if indices != tuple(sorted(indices)):
        raise ValueError(f"{label} indices must be sorted")
    if any(i >= n for i in indices):
        raise ValueError("index out of range")


def _validate_general_position_points(
    points: tuple[RationalPoint2D, ...], num_points: int
) -> None:
    keys = tuple((p.x.num, p.x.den, p.y.num, p.y.den) for p in points)
    if len(keys) != len(set(keys)):
        raise ValueError("point-set coordinates must be unique")
    if num_points != len(points):
        raise ValueError("num_points must equal len(points)")


def _validate_general_position_witnesses(
    collinear: tuple[CollinearTripleWitness, ...],
    concyclic: tuple[ConcyclicQuadrupleWitness, ...],
    has_collinear: bool,
    has_concyclic: bool,
    n: int,
) -> None:
    for triple in collinear:
        _check_witness_sorted_distinct(triple.indices, n, 3, "collinear triple")
    for quad in concyclic:
        _check_witness_sorted_distinct(quad.indices, n, 4, "concyclic quadruple")
    if has_collinear != bool(collinear):
        raise ValueError("has_collinear_triple must match collinear_triples")
    if has_concyclic != bool(concyclic):
        raise ValueError("has_concyclic_quadruple must match concyclic_quadruples")
    if tuple(sorted(collinear, key=lambda w: w.indices)) != collinear:
        raise ValueError("collinear_triples must be sorted lexicographically")
    if len({w.indices for w in collinear}) != len(collinear):
        raise ValueError("collinear_triples must be unique")
    if tuple(sorted(concyclic, key=lambda w: w.indices)) != concyclic:
        raise ValueError("concyclic_quadruples must be sorted lexicographically")
    if len({w.indices for w in concyclic}) != len(concyclic):
        raise ValueError("concyclic_quadruples must be unique")


def _validate_general_position_binding(
    points: tuple[RationalPoint2D, ...],
    collinear: tuple[CollinearTripleWitness, ...],
    concyclic: tuple[ConcyclicQuadrupleWitness, ...],
) -> None:
    pts = [(p.x.as_fraction(), p.y.as_fraction()) for p in points]
    expected_collinear = _expected_collinear_indices(pts)
    actual_collinear = {w.indices for w in collinear}
    if actual_collinear != expected_collinear:
        raise ValueError(
            "collinear_triples must exactly match the configuration's collinear triples"
        )
    expected_concyclic = _expected_concyclic_indices(pts, expected_collinear)
    actual_concyclic = {w.indices for w in concyclic}
    if actual_concyclic != expected_concyclic:
        raise ValueError(
            "concyclic_quadruples must exactly match the configuration's concyclic quadruples "
            "(excluding collinear quadruples)"
        )


def _validate_circumradius_entries_basic(
    entries: tuple[CircumradiusTripleEntry, ...], n: int
) -> set[tuple[int, int, int]]:
    seen: set[tuple[int, int, int]] = set()
    for e in entries:
        if e.indices in seen:
            raise ValueError("duplicate triple in circumradius profile")
        seen.add(e.indices)
        if any(i >= n for i in e.indices):
            raise ValueError("index out of range")
    if tuple(sorted(entries, key=lambda e: e.indices)) != entries:
        raise ValueError("entries must be sorted lexicographically")
    expected = set(combinations(range(n), 3))
    if seen != expected:
        raise ValueError("entries must cover exactly C(n,3) triples")
    return seen


def _validate_circumradius_binding(
    points: tuple[RationalPoint2D, ...],
    entries: tuple[CircumradiusTripleEntry, ...],
) -> None:
    pts = [(p.x.as_fraction(), p.y.as_fraction()) for p in points]
    for e in entries:
        i, j, k = e.indices
        ax, ay = pts[i]
        bx, by = pts[j]
        cx, cy = pts[k]
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        is_deg = cross == 0
        if e.is_degenerate != is_deg:
            raise ValueError("is_degenerate must match collinearity")
        if is_deg:
            if e.radius_squared is not None:
                raise ValueError("degenerate triple cannot have a radius")
        else:
            if e.radius_squared is None:
                raise ValueError("non-degenerate triple must have a radius")
            ab_sq = (bx - ax) ** 2 + (by - ay) ** 2
            bc_sq = (cx - bx) ** 2 + (cy - by) ** 2
            ca_sq = (ax - cx) ** 2 + (ay - cy) ** 2
            r_sq = Fraction(ab_sq * bc_sq * ca_sq) / Fraction(4 * cross * cross)
            expected = CanonicalRational.from_fraction(r_sq)
            require_bounded_rational(
                expected, max_digits=MAX_COORDINATE_DIGITS * 40, label="circumradius"
            )
            if e.radius_squared != expected:
                raise ValueError("radius_squared must match exact circumradius")


class RationalPoint2D(StrictModel):
    x: CanonicalRational
    y: CanonicalRational


def _point_key(point: RationalPoint2D) -> tuple[Fraction, Fraction]:
    return point.x.as_fraction(), point.y.as_fraction()


def _cross(
    left: tuple[Fraction, Fraction],
    right: tuple[Fraction, Fraction],
) -> Fraction:
    return left[0] * right[1] - left[1] * right[0]


def _subtract(
    left: tuple[Fraction, Fraction],
    right: tuple[Fraction, Fraction],
) -> tuple[Fraction, Fraction]:
    return left[0] - right[0], left[1] - right[1]


def _on_segment(
    point: tuple[Fraction, Fraction],
    start: tuple[Fraction, Fraction],
    end: tuple[Fraction, Fraction],
) -> bool:
    return _cross(_subtract(point, start), _subtract(end, start)) == 0 and all(
        min(left, right) <= value <= max(left, right)
        for value, left, right in zip(point, start, end, strict=True)
    )


def _segment_intersection_status(
    first: ClosedSegment2D,
    second: ClosedSegment2D,
) -> Literal["DISJOINT", "POINT", "OVERLAP"]:
    first_start, first_end = _point_key(first.start), _point_key(first.end)
    second_start, second_end = _point_key(second.start), _point_key(second.end)
    if first_start == first_end:
        return (
            "POINT"
            if _on_segment(first_start, second_start, second_end)
            else "DISJOINT"
        )
    if second_start == second_end:
        return (
            "POINT" if _on_segment(second_start, first_start, first_end) else "DISJOINT"
        )
    first_direction = _subtract(first_end, first_start)
    second_direction = _subtract(second_end, second_start)
    denominator = _cross(first_direction, second_direction)
    offset = _subtract(second_start, first_start)
    if denominator != 0:
        first_parameter = _cross(offset, second_direction) / denominator
        second_parameter = _cross(offset, first_direction) / denominator
        return (
            "POINT"
            if 0 <= first_parameter <= 1 and 0 <= second_parameter <= 1
            else "DISJOINT"
        )
    if _cross(offset, first_direction) != 0:
        return "DISJOINT"
    common = {
        point
        for point in (first_start, first_end, second_start, second_end)
        if _on_segment(point, first_start, first_end)
        and _on_segment(point, second_start, second_end)
    }
    if not common:
        return "DISJOINT"
    return "POINT" if len(common) == 1 else "OVERLAP"


def _edge(points: tuple[RationalPoint2D, ...], index: int) -> ClosedSegment2D:
    return ClosedSegment2D(
        start=points[index],
        end=points[(index + 1) % len(points)],
    )


def _adjacent_edges(first: int, second: int, order: int) -> bool:
    return (first - second) % order in {1, order - 1}


def _is_simple_ring(points: tuple[RationalPoint2D, ...]) -> bool:
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            status = _segment_intersection_status(
                _edge(points, first),
                _edge(points, second),
            )
            if _adjacent_edges(first, second, len(points)):
                if status != "POINT":
                    return False
            elif status != "DISJOINT":
                return False
    return True


class PointPairRequest(StrictModel):
    first: RationalPoint2D
    second: RationalPoint2D


class LineRequest(StrictModel):
    first: RationalPoint2D
    second: RationalPoint2D

    @model_validator(mode="after")
    def require_distinct_points(self) -> Self:
        if self.first == self.second:
            raise ValueError("a line requires two distinct points")
        return self


class LinePairRequest(StrictModel):
    first_line: LineRequest
    second_line: LineRequest


class PointLineRequest(StrictModel):
    point: RationalPoint2D
    line: LineRequest


INVERSION_ADMISSION_DIGITS = 2048
"""Per-component digit bound on both sides of an admitted inversion.

Exact inversion ``I(p) = c + s(p - c)/||p - c||^2`` builds, from
``D``-digit components, intermediates of at most about ``12*D + O(1)``
digits: the coordinate differences roughly double each height, the
squared norm doubles again, and scale, product, and sum add the power
and center heights on top.  With ``D = 2048`` every admission
intermediate stays far below the 32,768-digit canonical limit, so one
validation performs bounded work.  Requiring the exact inverted point to
satisfy the same bound keeps the accepted domain closed under the
advertised involution: a re-fed result passes both checks identically.
"""


def _inverted_components_within_bound(
    center: RationalPoint2D,
    power: CanonicalRational,
    point: RationalPoint2D,
    max_digits: int,
) -> bool:
    """Whether every canonical component of ``I(p)`` fits ``max_digits``.

    ``I(p) = c + s(p - c)/||p - c||^2`` is computed exactly over the
    rationals; reduction is canonical, so these are the precise component
    digit counts of the returned point, not estimates.
    """

    dx = point.x.as_fraction() - center.x.as_fraction()
    dy = point.y.as_fraction() - center.y.as_fraction()
    norm_squared = dx * dx + dy * dy
    scale = power.as_fraction() / norm_squared
    return all(
        len(format_canonical_integer(component.numerator).lstrip("-")) <= max_digits
        and len(format_canonical_integer(component.denominator)) <= max_digits
        for component in (
            center.x.as_fraction() + scale * dx,
            center.y.as_fraction() + scale * dy,
        )
    )


def _require_inversion_admission_bound(value: CanonicalRational, label: str) -> None:
    if (
        len(value.num.lstrip("-")) > INVERSION_ADMISSION_DIGITS
        or len(value.den.lstrip("-")) > INVERSION_ADMISSION_DIGITS
    ):
        raise ValueError(
            f"{label} exceeds the {INVERSION_ADMISSION_DIGITS}-digit "
            "circle-inversion admission bound"
        )


class CircleInversionRequest(StrictModel):
    """Invert a rational planar point ``p`` in a circle.

    The circle has center ``c`` and positive squared radius ``s``. Requires
    ``p != c``. See the published schema metadata for the effective numeric
    admission bounds on both sides of the transform.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Invert a rational planar point p in a circle with center c "
                "and positive squared radius s (the inversion power). The "
                "point must satisfy p != c; inverting the center is "
                "undefined. Every canonical component of c, s, and p - and "
                "of the exact inverted point I(p) = c + (s / ||p - c||^2) * "
                f"(p - c) - carries at most {INVERSION_ADMISSION_DIGITS} "
                "decimal digits, so an accepted result can be re-fed as a "
                "request: the admitted domain is closed under the involution."
            ),
            "inversion_admission_digit_bound": INVERSION_ADMISSION_DIGITS,
        },
    )

    center: RationalPoint2D = Field(
        description=(
            "Rational planar inversion center c. The point to invert must "
            f"satisfy p != c; every component is bounded to "
            f"{INVERSION_ADMISSION_DIGITS} decimal digits."
        ),
    )
    power: CanonicalRational = Field(
        description=(
            "Positive rational inversion power, interpreted as the squared "
            "inversion radius. Must be strictly positive and bounded to "
            f"{INVERSION_ADMISSION_DIGITS} decimal digits per component."
        ),
    )
    point: RationalPoint2D = Field(
        description=(
            "Rational planar point p to invert. Must satisfy p != c; every "
            f"component is bounded to {INVERSION_ADMISSION_DIGITS} decimal "
            "digits."
        ),
    )

    @model_validator(mode="after")
    def require_positive_power_and_distinct_point(self) -> Self:
        if self.power.as_fraction() <= 0:
            raise ValueError("inversion power must be a positive rational")
        if self.point == self.center:
            raise ValueError("the point to invert must differ from the center")

        # Input-side static bound first: with every admitted component at
        # most INVERSION_ADMISSION_DIGITS digits, the exact inversion below
        # forms only bounded intermediates (~12x the input height, far
        # inside the canonical limit), so validation work is bounded.
        _require_inversion_admission_bound(self.center.x, "inversion center x")
        _require_inversion_admission_bound(self.center.y, "inversion center y")
        _require_inversion_admission_bound(self.power, "inversion power")
        _require_inversion_admission_bound(self.point.x, "point x")
        _require_inversion_admission_bound(self.point.y, "point y")

        # Inversion-stable admission: accept exactly when the exact inverted
        # point satisfies the same bound.  Inputs already carry that bound,
        # so re-feeding an accepted result I(p) re-derives the original
        # admitted p and passes identically: the domain is symmetric under
        # the advertised involution and every accepted result can be fed back.
        if not _inverted_components_within_bound(
            self.center, self.power, self.point, INVERSION_ADMISSION_DIGITS
        ):
            raise ValueError(
                "circle inversion result exceeds the "
                f"{INVERSION_ADMISSION_DIGITS}-digit circle-inversion "
                "admission bound"
            )
        return self


class PointTripleRequest(StrictModel):
    first: RationalPoint2D
    second: RationalPoint2D
    third: RationalPoint2D


class CircumcircleRequest(PointTripleRequest):
    """Three distinct non-collinear points defining a circumcircle."""

    @model_validator(mode="after")
    def require_noncollinear_distinct(self) -> Self:
        points = [self.first, self.second, self.third]
        keys = tuple(_point_key(point) for point in points)
        if len(set(keys)) != len(keys):
            raise ValueError("circumcircle requires three distinct points")
        p0, p1, p2 = points
        x0, y0 = _point_key(p0)
        dx1 = p1.x.as_fraction() - x0
        dy1 = p1.y.as_fraction() - y0
        dx2 = p2.x.as_fraction() - x0
        dy2 = p2.y.as_fraction() - y0
        if dx1 * dy2 - dy1 * dx2 == 0:
            raise ValueError("circumcircle requires three noncollinear points")
        return self


class PointQuadrupleRequest(PointTripleRequest):
    fourth: RationalPoint2D


class PointSetRequest(StrictModel):
    points: tuple[RationalPoint2D, ...] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def require_unique_points(self) -> Self:
        keys = tuple(
            (point.x.num, point.x.den, point.y.num, point.y.den)
            for point in self.points
        )
        if len(keys) != len(set(keys)):
            raise ValueError("point-set coordinates must be unique")
        return self


class PolygonRequest(PointSetRequest):
    points: tuple[RationalPoint2D, ...] = Field(min_length=3, max_length=128)


class ClosedSegment2D(StrictModel):
    """One closed segment; equal endpoints explicitly denote a point segment."""

    start: RationalPoint2D
    end: RationalPoint2D


class SegmentIntersectionRequest(StrictModel):
    first: ClosedSegment2D
    second: ClosedSegment2D


class SegmentIntersectionResult(StrictModel):
    status: Literal["DISJOINT", "POINT", "OVERLAP"]
    point: RationalPoint2D | None = None
    contact_kind: Literal["PROPER", "ENDPOINT_TOUCH", "DEGENERATE_TOUCH"] | None = None
    overlap: ClosedSegment2D | None = None

    @model_validator(mode="after")
    def bind_discriminated_intersection(self) -> Self:
        if self.status == "DISJOINT":
            if (
                self.point is not None
                or self.contact_kind is not None
                or self.overlap is not None
            ):
                raise ValueError("a disjoint segment result carries no intersection")
            return self
        if self.status == "POINT":
            if (
                self.point is None
                or self.contact_kind is None
                or self.overlap is not None
            ):
                raise ValueError(
                    "a point segment intersection requires one contact classification"
                )
            return self
        if (
            self.point is not None
            or self.contact_kind is not None
            or self.overlap is None
        ):
            raise ValueError("an overlap result carries only one maximal segment")
        if _point_key(self.overlap.start) >= _point_key(self.overlap.end):
            raise ValueError("an overlap segment requires canonical distinct endpoints")
        return self


class PolygonIntersectionWitness(StrictModel):
    first_edge_index: StrictInt = Field(ge=0, le=127)
    second_edge_index: StrictInt = Field(ge=0, le=127)
    intersection: SegmentIntersectionResult

    @model_validator(mode="after")
    def require_ordered_intersecting_pair(self) -> Self:
        if self.first_edge_index >= self.second_edge_index:
            raise ValueError("polygon witness edge indices must be strictly ordered")
        if self.intersection.status == "DISJOINT":
            raise ValueError("polygon witness edges must intersect")
        return self


class SimplePolygonDecisionResult(StrictModel):
    vertex_count: StrictInt = Field(ge=3, le=128)
    is_simple: bool
    checked_edge_pairs: StrictInt = Field(ge=0, le=8128)
    witness: PolygonIntersectionWitness | None = None

    @model_validator(mode="after")
    def bind_decision_to_witness(self) -> Self:
        if self.is_simple is (self.witness is not None):
            raise ValueError("exactly a non-simple polygon carries one witness")
        total_pairs = self.vertex_count * (self.vertex_count - 1) // 2
        if self.is_simple and self.checked_edge_pairs != total_pairs:
            raise ValueError("a simple decision must exhaust every edge pair")
        if self.witness is not None:
            if self.witness.second_edge_index >= self.vertex_count:
                raise ValueError("polygon witness edge index exceeds the ring")
            expected_checked = (
                self.witness.first_edge_index
                * (2 * self.vertex_count - self.witness.first_edge_index - 1)
                // 2
                + self.witness.second_edge_index
                - self.witness.first_edge_index
            )
            if self.checked_edge_pairs != expected_checked:
                raise ValueError(
                    "polygon witness does not match the checked pair prefix"
                )
        return self


class SimplePolygonPointRequest(StrictModel):
    polygon: PolygonRequest
    point: RationalPoint2D

    @model_validator(mode="after")
    def require_simple_polygon(self) -> Self:
        if not _is_simple_ring(self.polygon.points):
            raise ValueError("point classification requires a simple polygon")
        return self


class PolygonPointClassificationResult(StrictModel):
    polygon_vertex_count: StrictInt = Field(ge=3, le=128)
    classification: Literal["INSIDE", "BOUNDARY", "OUTSIDE"]
    boundary_edge_index: StrictInt | None = Field(default=None, ge=0, le=127)

    @model_validator(mode="after")
    def bind_boundary_witness(self) -> Self:
        if (self.classification == "BOUNDARY") is (self.boundary_edge_index is None):
            raise ValueError("only a boundary classification carries an edge index")
        if (
            self.boundary_edge_index is not None
            and self.boundary_edge_index >= self.polygon_vertex_count
        ):
            raise ValueError("boundary edge index exceeds the polygon ring")
        return self


class GeometryBooleanResult(StrictModel):
    holds: bool


class GeometryRationalResult(StrictModel):
    value: CanonicalRational


class GeometryPointResult(StrictModel):
    point: RationalPoint2D


class GeometryOrientationResult(StrictModel):
    orientation: Literal[-1, 0, 1]


class GeometryLineIntersectionResult(StrictModel):
    status: Literal["POINT", "PARALLEL", "COINCIDENT"]
    point: RationalPoint2D | None = None

    @model_validator(mode="after")
    def bind_point_status(self) -> Self:
        if (self.status == "POINT") is (self.point is None):
            raise ValueError("only POINT intersections carry one point")
        return self


class GeometryConvexHullResult(StrictModel):
    points: tuple[RationalPoint2D, ...] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def require_canonical_strict_convex_boundary(self) -> Self:
        keys = tuple(_point_key(point) for point in self.points)
        if len(keys) != len(set(keys)):
            raise ValueError("convex-hull vertices must be unique")
        if len(keys) <= 2:
            if keys != tuple(sorted(keys)):
                raise ValueError("a zero-dimensional hull must be canonical")
            return self
        if keys[0] != min(keys):
            raise ValueError("a polygon hull must begin at its least vertex")
        turns = tuple(
            _cross(
                _subtract(keys[(index + 1) % len(keys)], keys[index]),
                _subtract(keys[(index + 2) % len(keys)], keys[index]),
            )
            for index in range(len(keys))
        )
        if any(turn <= 0 for turn in turns):
            raise ValueError("a polygon hull must be strictly counterclockwise")
        return self


class GeometryCircleResult(StrictModel):
    center: RationalPoint2D
    radius_squared: CanonicalRational


class WeightedPolygonDiagonal(StrictModel):
    first: StrictInt = Field(ge=0, le=31)
    second: StrictInt = Field(ge=0, le=31)
    weight: CanonicalRational

    @model_validator(mode="after")
    def require_canonical_positive_pair(self) -> Self:
        if self.first >= self.second:
            raise ValueError("weighted diagonal endpoints must be strictly increasing")
        if self.weight.as_fraction() < 0:
            raise ValueError("weighted diagonal cost must be nonnegative")
        return self


class ConvexPolygonTriangulationRequest(StrictModel):
    polygon: PolygonRequest
    diagonal_weights: tuple[WeightedPolygonDiagonal, ...] = Field(
        min_length=1, max_length=464
    )
    objective: Literal["NON_HULL_DIAGONAL_WEIGHT_SUM"] = "NON_HULL_DIAGONAL_WEIGHT_SUM"

    @model_validator(mode="after")
    def require_strict_convexity_and_complete_weights(self) -> Self:
        points = tuple(_point_key(point) for point in self.polygon.points)
        if not 4 <= len(points) <= 32:
            raise ValueError("weighted triangulation supports 4 to 32 vertices")
        turns = tuple(
            _cross(
                _subtract(points[(index + 1) % len(points)], points[index]),
                _subtract(points[(index + 2) % len(points)], points[index]),
            )
            for index in range(len(points))
        )
        if any(turn <= 0 for turn in turns):
            raise ValueError("weighted triangulation requires strict CCW convexity")
        expected = {
            (first, second)
            for first in range(len(points))
            for second in range(first + 1, len(points))
            if second != first + 1 and (first, second) != (0, len(points) - 1)
        }
        actual = {(item.first, item.second) for item in self.diagonal_weights}
        if len(actual) != len(self.diagonal_weights) or actual != expected:
            raise ValueError("diagonal weights must cover every non-hull pair exactly")
        pairs = tuple((item.first, item.second) for item in self.diagonal_weights)
        if pairs != tuple(sorted(pairs)):
            raise ValueError("diagonal weights must use lexicographic pair order")
        return self


class PolygonTriangle(StrictModel):
    vertices: tuple[StrictInt, StrictInt, StrictInt]


class TriangulationSplitEntry(StrictModel):
    start: StrictInt = Field(ge=0, le=31)
    end: StrictInt = Field(ge=0, le=31)
    split: StrictInt = Field(ge=0, le=31)
    optimum: CanonicalRational


class ConvexPolygonTriangulationResult(StrictModel):
    vertex_count: StrictInt = Field(ge=4, le=32)
    diagonals: tuple[WeightedPolygonDiagonal, ...] = Field(min_length=1, max_length=29)
    triangles: tuple[PolygonTriangle, ...] = Field(min_length=2, max_length=30)
    split_table: tuple[TriangulationSplitEntry, ...] = Field(
        min_length=3, max_length=496
    )
    optimum: CanonicalRational
    objective: Literal["NON_HULL_DIAGONAL_WEIGHT_SUM"] = "NON_HULL_DIAGONAL_WEIGHT_SUM"
    tie_break: Literal["LOWEST_SPLIT_INDEX"] = "LOWEST_SPLIT_INDEX"
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"


# ---------------------------------------------------------------------------
# Configuration-level operations (issues #2107, #2106)
# ---------------------------------------------------------------------------


class GeneralPositionRequest(StrictModel):
    """Search a bounded point configuration for collinear triples and concyclic quadruples.

    Each rational coordinate is bounded to at most 256 digits in numerator and
    denominator (operation-specific, stricter than the global 32,768-digit
    CanonicalRational limit). The exhaustive determinant work is coupled to
    both the combinatorial point count and rational complexity:
    ``C(n,4) * max_digits**2 <= 1,000,000`` keeps both the search and its
    source-binding replay within the bounded-work envelope.
    """

    points: tuple[RationalPoint2D, ...] = Field(
        min_length=3,
        max_length=MAX_CONFIGURATION_POINTS,
        description=(
            "Bounded point configuration with 3..32 points; each rational coordinate "
            f"(numerator/denominator) is bounded to at most {MAX_COORDINATE_DIGITS} digits "
            "(operation-specific limit, stricter than CanonicalRational's 32768-digit "
            "global limit); additionally C(n,4)*max_digits^2 <= 1000000 bounds the "
            "exhaustive determinant work"
        ),
    )

    @model_validator(mode="after")
    def require_unique(self) -> Self:
        _require_bounded_configuration(self.points)
        _require_general_position_work_bound(self.points)
        keys = tuple((p.x.num, p.x.den, p.y.num, p.y.den) for p in self.points)
        if len(keys) != len(set(keys)):
            raise ValueError("point-set coordinates must be unique")
        return self


class CollinearTripleWitness(StrictModel):
    indices: tuple[int, int, int]


class ConcyclicQuadrupleWitness(StrictModel):
    indices: tuple[int, int, int, int]


class GeneralPositionResult(StrictModel):
    """Complete search result for collinear triples and concyclic quadruples, bound to its source points."""

    points: tuple[RationalPoint2D, ...] = Field(
        min_length=3, max_length=MAX_CONFIGURATION_POINTS
    )
    num_points: int = Field(ge=0)
    has_collinear_triple: bool
    has_concyclic_quadruple: bool
    collinear_triples: tuple[CollinearTripleWitness, ...] = Field(default=())
    concyclic_quadruples: tuple[ConcyclicQuadrupleWitness, ...] = Field(default=())

    @model_validator(mode="after")
    def require_canonical(self) -> Self:
        _require_bounded_configuration(self.points)
        _require_general_position_work_bound(self.points)
        _validate_general_position_points(self.points, self.num_points)
        _validate_general_position_witnesses(
            self.collinear_triples,
            self.concyclic_quadruples,
            self.has_collinear_triple,
            self.has_concyclic_quadruple,
            len(self.points),
        )
        _validate_general_position_binding(
            self.points, self.collinear_triples, self.concyclic_quadruples
        )
        return self


class CircumradiusProfileRequest(StrictModel):
    """Compute circumradius data for every unordered triple in a point configuration.

    Each rational coordinate is bounded to at most 256 digits. The serialized
    profile is bounded before execution by worst-case rational growth: with
    ``d`` = max coordinate digits, each squared circumradius can carry ``40d``
    digits in numerator and denominator, so the request is admitted only when
    ``C(n,3) * (80*d + 80)`` characters stay within the 8,000,000-character
    output budget (under the 10 MiB transport envelope).
    """

    points: tuple[RationalPoint2D, ...] = Field(
        min_length=3,
        max_length=MAX_CONFIGURATION_POINTS,
        description=(
            "Bounded point configuration with 3..32 points; each rational coordinate "
            f"is bounded to at most {MAX_COORDINATE_DIGITS} digits (operation-specific, "
            "stricter than CanonicalRational's 32768-digit limit); additionally "
            "C(n,3)*(80*max_digits+80) characters of worst-case profile size must "
            "stay within the 8,000,000-character output budget"
        ),
    )

    @model_validator(mode="after")
    def require_unique(self) -> Self:
        _require_bounded_configuration(self.points)
        _require_circumradius_output_bound(self.points)
        keys = tuple((p.x.num, p.x.den, p.y.num, p.y.den) for p in self.points)
        if len(keys) != len(set(keys)):
            raise ValueError("point-set coordinates must be unique")
        return self


class CircumradiusTripleEntry(StrictModel):
    """One triple and its circumradius disposition."""

    indices: tuple[int, int, int]
    is_degenerate: bool
    radius_squared: CanonicalRational | None = None

    @model_validator(mode="after")
    def require_canonical(self) -> Self:
        idx = self.indices
        if len(idx) != 3 or len(set(idx)) != 3:
            raise ValueError("triple indices must be 3 distinct values")
        if idx != tuple(sorted(idx)):
            raise ValueError("triple indices must be sorted")
        if self.is_degenerate and self.radius_squared is not None:
            raise ValueError("degenerate triple cannot have a radius")
        if not self.is_degenerate and self.radius_squared is None:
            raise ValueError("non-degenerate triple must have a radius")
        return self


class CircumradiusProfileResult(StrictModel):
    """Complete circumradius profile for every unordered triple, bound to its source points."""

    points: tuple[RationalPoint2D, ...] = Field(
        min_length=3, max_length=MAX_CONFIGURATION_POINTS
    )
    num_points: int = Field(ge=0)
    entries: tuple[CircumradiusTripleEntry, ...] = Field(default=())

    @model_validator(mode="after")
    def require_canonical(self) -> Self:
        _require_bounded_configuration(self.points)
        _require_circumradius_output_bound(self.points)
        keys = tuple((p.x.num, p.x.den, p.y.num, p.y.den) for p in self.points)
        if len(keys) != len(set(keys)):
            raise ValueError("point-set coordinates must be unique")
        n = len(self.points)
        if self.num_points != n:
            raise ValueError("num_points must equal len(points)")
        if len(self.entries) != n * (n - 1) * (n - 2) // 6:
            raise ValueError("entries must cover exactly C(n,3) triples")
        _validate_circumradius_entries_basic(self.entries, n)
        _validate_circumradius_binding(self.points, self.entries)
        return self
