"""Exact rational planar-geometry wire contracts."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel

# Worst-case squared-circumradius component size for coordinates with b-digit
# components: differences have numerators < 2*10^2b over denominators <= 10^2b,
# so R^2 = dab*dbc*dac / (4*cross^2) has numerator < 10^(40b+3) and denominator
# < 10^(40b+4) before cancellation; both must fit the canonical 32,768-digit
# limit, which admits b <= 819.
_MAX_CIRCUMRADIUS_COORDINATE_DIGITS = 819


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


class LabelledPoint2D(StrictModel):
    """A labelled rational point in the plane."""

    label: str = Field(min_length=1, max_length=64)
    point: RationalPoint2D


class CircumradiusProfileRequest(StrictModel):
    """A bounded labelled rational planar point configuration."""

    points: tuple[LabelledPoint2D, ...] = Field(min_length=3, max_length=64)

    @model_validator(mode="after")
    def require_unique_labels_and_coordinates(self) -> Self:
        labels = tuple(item.label for item in self.points)
        if len(labels) != len(set(labels)):
            raise ValueError("point labels must be unique")
        keys = tuple(
            (
                item.point.x.num,
                item.point.x.den,
                item.point.y.num,
                item.point.y.den,
            )
            for item in self.points
        )
        if len(keys) != len(set(keys)):
            raise ValueError("point coordinates must be unique")
        # Bound coordinate digit length so exact circumradius stays within
        # the canonical 32,768-digit limit (see the constant's derivation).
        for item in self.points:
            require_bounded_rational(
                item.point.x, max_digits=_MAX_CIRCUMRADIUS_COORDINATE_DIGITS, label="point.x"
            )
            require_bounded_rational(
                item.point.y, max_digits=_MAX_CIRCUMRADIUS_COORDINATE_DIGITS, label="point.y"
            )
        return self


class CircumradiusTripleEntry(StrictModel):
    """Circumradius data for one unordered triple of points."""

    labels: tuple[str, str, str]
    indices: tuple[StrictInt, StrictInt, StrictInt]
    collinear: bool
    squared_circumradius: CanonicalRational | None = None

    @model_validator(mode="after")
    def bind_collinear_to_value(self) -> Self:
        if self.collinear is (self.squared_circumradius is not None):
            raise ValueError(
                "exactly a collinear triple has no squared circumradius"
            )
        if (
            self.squared_circumradius is not None
            and self.squared_circumradius.as_fraction() <= 0
        ):
            raise ValueError("squared circumradius must be positive")
        return self


def _replay_circumradius_entry(
    entry: CircumradiusTripleEntry,
    points: tuple[LabelledPoint2D, ...],
    coords: list[tuple[Fraction, Fraction]],
    indices: tuple[int, int, int],
) -> None:
    first, second, third = indices
    exp_labels = (points[first].label, points[second].label, points[third].label)
    if entry.labels != exp_labels:
        raise ValueError("circumradius entry labels must follow the retained configuration order")
    if entry.indices != indices:
        raise ValueError("circumradius entry indices must follow the retained configuration order")
    ax, ay = coords[first]
    bx, by = coords[second]
    cx, cy = coords[third]
    cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    is_collinear = cross == 0
    if entry.collinear != is_collinear:
        raise ValueError("collinear flag must match the replayed cross product")
    if is_collinear:
        if entry.squared_circumradius is not None:
            raise ValueError("collinear triple must not carry a squared radius")
        return
    if entry.squared_circumradius is None:
        raise ValueError("noncollinear triple must carry a squared radius")
    dab = (ax - bx) ** 2 + (ay - by) ** 2
    dbc = (bx - cx) ** 2 + (by - cy) ** 2
    dac = (ax - cx) ** 2 + (ay - cy) ** 2
    expected = (dab * dbc * dac) / (4 * cross * cross)
    if entry.squared_circumradius.as_fraction() != expected:
        raise ValueError(
            "squared_circumradius must be the exact circumradius of the triple"
        )


class CircumradiusProfileResult(StrictModel):
    points: tuple[LabelledPoint2D, ...] = Field(min_length=3, max_length=64)
    point_count: StrictInt = Field(ge=3, le=64)
    triple_count: StrictInt = Field(ge=1, le=41664)
    entries: tuple[CircumradiusTripleEntry, ...] = Field(min_length=1)
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"

    @model_validator(mode="after")
    def require_complete_profile(self) -> Self:
        if len(self.points) != self.point_count:
            raise ValueError("point_count must match the retained point configuration")
        if len(self.entries) != self.triple_count:
            raise ValueError("circumradius profile must be complete")
        # Replay every triple's exact circumradius relation.
        from itertools import combinations

        expected_count = len(self.points) * (len(self.points) - 1) * (len(self.points) - 2) // 6
        if self.triple_count != expected_count:
            raise ValueError("triple_count must equal C(n,3) of the retained configuration")
        if len(self.entries) != expected_count:
            raise ValueError("entries must be complete for the retained configuration")
        # Build coordinate map and expected entries.
        coords = [(p.point.x.as_fraction(), p.point.y.as_fraction()) for p in self.points]
        for entry, indices in zip(self.entries, combinations(range(len(self.points)), 3), strict=True):
            _replay_circumradius_entry(entry, self.points, coords, indices)
        return self


class ForbiddenLabelledPoint(StrictModel):
    """A labelled rational point in the affine plane."""

    label: str = Field(min_length=1, max_length=64)
    point: RationalPoint2D


# The forbidden-pattern screen enumerates every triple and quadruple and the
# result validation replays it, so the point bound is derived from a
# practical enumeration budget: C(32,3) + C(32,4) = 40,920 exact predicate
# evaluations for one request.
MAX_FORBIDDEN_PATTERN_POINTS = 32


class ForbiddenConfiguration(StrictModel):
    """A finite set of labelled rational planar points."""

    points: tuple[ForbiddenLabelledPoint, ...] = Field(
        min_length=1, max_length=MAX_FORBIDDEN_PATTERN_POINTS
    )

    @model_validator(mode="after")
    def require_unique_labels_and_coords(self) -> Self:
        labels = tuple(item.label for item in self.points)
        if len(labels) != len(set(labels)):
            raise ValueError("configuration point labels must be unique")
        keys = tuple(_point_key(item.point) for item in self.points)
        if len(keys) != len(set(keys)):
            raise ValueError("configuration point coordinates must be unique")
        for item in self.points:
            require_bounded_rational(
                item.point.x, max_digits=_MAX_CIRCUMRADIUS_COORDINATE_DIGITS, label="point.x"
            )
            require_bounded_rational(
                item.point.y, max_digits=_MAX_CIRCUMRADIUS_COORDINATE_DIGITS, label="point.y"
            )
        return self


class ForbiddenPatternsRequest(StrictModel):
    """A labelled rational planar point configuration to screen."""

    configuration: ForbiddenConfiguration


class CollinearTriple(StrictModel):
    """A triple of configuration point indices lying on one line."""

    first: StrictInt = Field(ge=0, le=127)
    second: StrictInt = Field(ge=0, le=127)
    third: StrictInt = Field(ge=0, le=127)

    @model_validator(mode="after")
    def require_strictly_ascending(self) -> Self:
        if not (self.first < self.second < self.third):
            raise ValueError("collinear triple indices must be strictly ascending")
        return self


class ConcyclicQuadruple(StrictModel):
    """A quadruple of configuration point indices lying on one circle."""

    first: StrictInt = Field(ge=0, le=127)
    second: StrictInt = Field(ge=0, le=127)
    third: StrictInt = Field(ge=0, le=127)
    fourth: StrictInt = Field(ge=0, le=127)

    @model_validator(mode="after")
    def require_strictly_ascending(self) -> Self:
        if not (self.first < self.second < self.third < self.fourth):
            raise ValueError("concyclic quadruple indices must be strictly ascending")
        return self


def _replay_collinear_triples(
    xy: list[tuple[Fraction, Fraction]],
) -> tuple[bool, CollinearTriple | None, int]:
    """First collinear triple in lexicographic order plus the checked prefix."""
    from itertools import combinations

    n = len(xy)
    for checked, (i, j, k) in enumerate(combinations(range(n), 3), start=1):
        xi, yi = xy[i]
        xj, yj = xy[j]
        xk, yk = xy[k]
        if (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi) == 0:
            return True, CollinearTriple(first=i, second=j, third=k), checked
    total = n * (n - 1) * (n - 2) // 6 if n >= 3 else 0
    return False, None, total


def _is_collinear_triple(
    xy: list[tuple[Fraction, Fraction]],
    first: int,
    second: int,
    third: int,
) -> bool:
    xa, ya = xy[first]
    xb, yb = xy[second]
    xc, yc = xy[third]
    return (xb - xa) * (yc - ya) - (yb - ya) * (xc - xa) == 0


def _triple_circle_minor(
    first: tuple[Fraction, Fraction],
    second: tuple[Fraction, Fraction],
    third: tuple[Fraction, Fraction],
) -> Fraction:
    """Minor with lifted-coordinate leading column for the concyclic test."""
    fx, fy = first
    sx, sy = second
    tx, ty = third
    lifted_first = fx * fx + fy * fy
    lifted_second = sx * sx + sy * sy
    lifted_third = tx * tx + ty * ty
    return (
        lifted_first * (sx * ty - sy * tx)
        - fx * (lifted_second * ty - sy * lifted_third)
        + fy * (lifted_second * tx - sx * lifted_third)
    )


def _concyclic_determinant(
    first: tuple[Fraction, Fraction],
    second: tuple[Fraction, Fraction],
    third: tuple[Fraction, Fraction],
    fourth: tuple[Fraction, Fraction],
) -> Fraction:
    """Determinant of [[x^2+y^2, x, y, 1]] over four points; zero iff concyclic
    or collinear."""
    return (
        -_triple_circle_minor(second, third, fourth)
        + _triple_circle_minor(first, third, fourth)
        - _triple_circle_minor(first, second, fourth)
        + _triple_circle_minor(first, second, third)
    )


def _replay_concyclic_quadruples(
    xy: list[tuple[Fraction, Fraction]],
) -> tuple[bool, ConcyclicQuadruple | None, int]:
    """First noncollinear concyclic quadruple plus the checked prefix length."""
    from itertools import combinations

    n = len(xy)
    for checked, (i, j, k, ell) in enumerate(combinations(range(n), 4), start=1):
        if _concyclic_determinant(xy[i], xy[j], xy[k], xy[ell]) != 0:
            continue
        if all(
            _is_collinear_triple(xy, a, b, c)
            for a, b, c in ((i, j, k), (i, j, ell), (i, k, ell), (j, k, ell))
        ):
            continue
        witness = ConcyclicQuadruple(first=i, second=j, third=k, fourth=ell)
        return True, witness, checked
    total = n * (n - 1) * (n - 2) * (n - 3) // 24 if n >= 4 else 0
    return False, None, total


class ForbiddenPatternsResult(StrictModel):
    """Result of screening a configuration for forbidden patterns."""

    configuration: ForbiddenConfiguration
    point_count: StrictInt = Field(ge=1, le=MAX_FORBIDDEN_PATTERN_POINTS)
    has_collinear_triple: bool
    has_concyclic_quadruple: bool
    collinear_triple: CollinearTriple | None = None
    concyclic_quadruple: ConcyclicQuadruple | None = None
    checked_triples: StrictInt = Field(ge=0)
    checked_quadruples: StrictInt = Field(ge=0)

    def _require_witness_shape(self) -> None:
        if len(self.configuration.points) != self.point_count:
            raise ValueError("point_count must match the retained configuration length")
        if self.has_collinear_triple is (self.collinear_triple is None):
            raise ValueError("exactly a collinear triple carries one witness")
        if self.has_concyclic_quadruple is (self.concyclic_quadruple is None):
            raise ValueError("exactly a concyclic quadruple carries one witness")
        if (
            self.collinear_triple is not None
            and self.collinear_triple.third >= self.point_count
        ):
            raise ValueError("collinear triple index exceeds configuration")
        if (
            self.concyclic_quadruple is not None
            and self.concyclic_quadruple.fourth >= self.point_count
        ):
            raise ValueError("concyclic quadruple index exceeds configuration")

    @model_validator(mode="after")
    def bind_witnesses(self) -> Self:
        self._require_witness_shape()
        # Replay complete bounded enumeration to bind the decision.
        xy = [
            (point.point.x.as_fraction(), point.point.y.as_fraction())
            for point in self.configuration.points
        ]
        expected_has_collinear, expected_collinear, expected_checked_triples = (
            _replay_collinear_triples(xy)
        )
        if self.has_collinear_triple != expected_has_collinear:
            raise ValueError("has_collinear_triple must match the replayed enumeration")
        if self.collinear_triple != expected_collinear:
            raise ValueError("collinear witness must match the replayed enumeration")
        if self.checked_triples != expected_checked_triples:
            raise ValueError("checked_triples must match the replayed enumeration prefix")
        expected_has_concyclic, expected_concyclic, expected_checked_quadruples = (
            _replay_concyclic_quadruples(xy)
        )
        if self.has_concyclic_quadruple != expected_has_concyclic:
            raise ValueError("has_concyclic_quadruple must match the replayed enumeration")
        if self.concyclic_quadruple != expected_concyclic:
            raise ValueError("concyclic witness must match the replayed enumeration")
        if self.checked_quadruples != expected_checked_quadruples:
            raise ValueError("checked_quadruples must match the replayed enumeration prefix")
        return self
