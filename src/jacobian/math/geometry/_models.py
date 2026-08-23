"""Exact rational planar-geometry wire contracts."""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import combinations
from typing import Literal, NamedTuple, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian._models import StrictModel
from jacobian.math._rational_height import RationalHeight, sum_heights


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


def _minor3(
    m: tuple[list[int], ...],
    r0: int,
    r1: int,
    r2: int,
    c0: int,
    c1: int,
    c2: int,
) -> int:
    """3x3 determinant of selected rows and columns."""
    return (
        m[r0][c0] * (m[r1][c1] * m[r2][c2] - m[r1][c2] * m[r2][c1])
        - m[r0][c1] * (m[r1][c0] * m[r2][c2] - m[r1][c2] * m[r2][c0])
        + m[r0][c2] * (m[r1][c0] * m[r2][c1] - m[r1][c1] * m[r2][c0])
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


# The complete profile must serialize under the 10 MiB transport limit with
# margin, so admission budgets the aggregate at 9 MiB; every entry costs its
# exact numerator and denominator digits plus labels, indices, and syntax.
_MAX_CIRCUMRADIUS_PROFILE_BYTES = 9 * 1024 * 1024
_CIRCUMRADIUS_ENTRY_OVERHEAD_BYTES = 512


def _circumradius_squared_height(
    xs: tuple[RationalHeight, ...],
    ys: tuple[RationalHeight, ...],
    triple: tuple[int, int, int],
) -> RationalHeight:
    """Conservatively bound the decimal digits of one squared circumradius.

    For a triple with side-squared sums ``d`` and cross product ``c``, the
    exact value is ``R^2 = d_ab * d_bc * d_ac / (4 * c^2)``.  Every
    subtraction, square, sum, and product is bounded by rational-height
    propagation, so any accepted triple keeps its exact result within the
    canonical limit regardless of reduction.
    """
    first, second, third = triple
    dx_ab = sum_heights((xs[first], xs[second]))
    dy_ab = sum_heights((ys[first], ys[second]))
    dx_bc = sum_heights((xs[second], xs[third]))
    dy_bc = sum_heights((ys[second], ys[third]))
    dx_ac = sum_heights((xs[first], xs[third]))
    dy_ac = sum_heights((ys[first], ys[third]))

    def side(delta_x: RationalHeight, delta_y: RationalHeight) -> RationalHeight:
        return sum_heights((delta_x.product(delta_x), delta_y.product(delta_y)))

    cross = sum_heights((dx_ab.product(dy_ac), dy_ab.product(dx_ac)))
    return (
        side(dx_ab, dy_ab)
        .product(side(dx_bc, dy_bc))
        .product(side(dx_ac, dy_ac))
        .quotient(cross.product(cross))
        .quotient(RationalHeight(1, 1))
    )


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
        for item in self.points:
            require_bounded_rational(item.point.x, max_digits=4096, label="point x")
            require_bounded_rational(item.point.y, max_digits=4096, label="point y")
        return self

    @model_validator(mode="after")
    def require_bounded_circumradius_growth(self) -> Self:
        xs = tuple(RationalHeight.from_canonical(item.point.x) for item in self.points)
        ys = tuple(RationalHeight.from_canonical(item.point.y) for item in self.points)
        estimated_bytes = 0
        for triple in combinations(range(len(self.points)), 3):
            height = _circumradius_squared_height(xs, ys, triple)
            if height.exceeds(MAX_CANONICAL_RATIONAL_DIGITS):
                raise ValueError(
                    f"triple {triple} has a squared-circumradius height "
                    f"exceeding the canonical {MAX_CANONICAL_RATIONAL_DIGITS}-digit limit"
                )
            # Each individually bounded entry still contributes its exact
            # digits to the complete profile; a 64-point configuration can
            # hold 41,664 triples whose combined serialization exceeds the
            # 10 MiB transport limit even when no single entry does.
            estimated_bytes += (
                _CIRCUMRADIUS_ENTRY_OVERHEAD_BYTES
                + height.numerator_digits
                + height.denominator_digits
            )
        if estimated_bytes > _MAX_CIRCUMRADIUS_PROFILE_BYTES:
            raise ValueError(
                "point configuration would produce a circumradius profile "
                f"exceeding the {_MAX_CIRCUMRADIUS_PROFILE_BYTES}-byte "
                "aggregate transport budget"
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
            raise ValueError("exactly a collinear triple has no squared circumradius")
        if (
            self.squared_circumradius is not None
            and self.squared_circumradius.as_fraction() <= 0
        ):
            raise ValueError("squared circumradius must be positive")
        return self


def _circumradius_squared(
    coords: tuple[tuple[Fraction, Fraction], ...],
    triple: tuple[int, int, int],
) -> Fraction | None:
    """Replay one exact squared circumradius, or None for a collinear triple."""
    (ax, ay), (bx, by), (cx, cy) = (coords[index] for index in triple)
    dab = (ax - bx) ** 2 + (ay - by) ** 2
    dbc = (bx - cx) ** 2 + (by - cy) ** 2
    dac = (ax - cx) ** 2 + (ay - cy) ** 2
    cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if cross == 0:
        return None
    return (dab * dbc * dac) / (4 * cross * cross)


def _require_canonical_triple_coverage(
    entries: tuple[CircumradiusTripleEntry, ...],
    point_count: int,
) -> None:
    seen: set[tuple[int, int, int]] = set()
    for entry in entries:
        idx = entry.indices
        if idx != tuple(sorted(idx)):
            raise ValueError("circumradius entry indices must be sorted")
        if idx in seen:
            raise ValueError("circumradius entries must be unique")
        if not (0 <= idx[0] < idx[1] < idx[2] < point_count):
            raise ValueError("circumradius entry indices out of range")
        seen.add(idx)
    if tuple(entries) != tuple(sorted(entries, key=lambda e: e.indices)):
        raise ValueError("circumradius entries must be in lexicographic order")


def _require_source_bound_circumradii(
    points: tuple[LabelledPoint2D, ...],
    entries: tuple[CircumradiusTripleEntry, ...],
) -> None:
    """Replay labels, degeneracy, and exact radii from the retained sources."""
    labels = tuple(item.label for item in points)
    if len(labels) != len(set(labels)):
        raise ValueError("source point labels must be unique")
    coords = tuple(
        (item.point.x.as_fraction(), item.point.y.as_fraction()) for item in points
    )
    if len(coords) != len(set(coords)):
        raise ValueError("source point coordinates must be unique")
    for entry in entries:
        idx = entry.indices
        if entry.labels != tuple(labels[index] for index in idx):
            raise ValueError("entry labels must match the retained source points")
        replayed = _circumradius_squared(coords, idx)
        if entry.collinear is not (replayed is None):
            raise ValueError(
                "entry degeneracy flags must match the retained source points"
            )
        if replayed is None:
            continue
        if (
            entry.squared_circumradius is None
            or entry.squared_circumradius.as_fraction() != replayed
        ):
            raise ValueError(
                "squared circumradius must equal the value replayed from "
                "the retained source points"
            )


class CircumradiusProfileResult(StrictModel):
    """The complete circumradius profile bound to its retained source points."""

    points: tuple[LabelledPoint2D, ...] = Field(min_length=3, max_length=64)
    point_count: StrictInt = Field(ge=3, le=64)
    triple_count: StrictInt = Field(ge=1, le=41664)
    entries: tuple[CircumradiusTripleEntry, ...] = Field(min_length=1)
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"

    @model_validator(mode="after")
    def require_complete_source_bound_profile(self) -> Self:
        import math

        if self.point_count != len(self.points):
            raise ValueError("point_count must match the retained source points")
        expected = math.comb(self.point_count, 3)
        if self.triple_count != expected or len(self.entries) != self.triple_count:
            raise ValueError("circumradius profile must cover every unordered triple")
        _require_canonical_triple_coverage(self.entries, self.point_count)
        _require_source_bound_circumradii(self.points, self.entries)
        return self


class ForbiddenLabelledPoint(StrictModel):
    """A labelled rational point in the affine plane."""

    label: str = Field(min_length=1, max_length=64)
    point: RationalPoint2D


# Exhaustive screening enumerates every quadruple with exact arithmetic, so
# admission must bound the complete enumeration work, not only the input
# shape.  Clearing denominators once scales every coordinate by the lcm of
# the coordinate denominators, whose digit size is at most the summed
# nontrivial component denominators; each of the comb(n, 4) determinants
# then costs at least linearly more as that cleared size grows.  The product
# below is the named conservative work proxy: a pattern-free 64-point
# small-integer configuration (about 3.2M units) screens in roughly two
# seconds, and admitted configurations perform comparable or less
# determinant work per unit.
_MAX_FORBIDDEN_COORDINATE_DIGITS = 4_096
_MAX_FORBIDDEN_SCREENING_WORK_UNITS = 4_000_000


def _forbidden_screening_work_units(
    points: tuple[ForbiddenLabelledPoint, ...],
) -> int:
    """Conservative exact-enumeration work proxy for one configuration."""

    denominator_digit_sum = 0
    max_numerator_digits = 0
    for item in points:
        for value in (item.point.x, item.point.y):
            if value.den != "1":
                denominator_digit_sum += len(value.den)
            max_numerator_digits = max(max_numerator_digits, len(value.num.lstrip("-")))
    return math.comb(len(points), 4) * (
        denominator_digit_sum + max_numerator_digits + 1
    )


class ForbiddenConfiguration(StrictModel):
    """A bounded finite set of labelled rational planar points.

    The joint point-count and coordinate-height budget keeps the complete
    exhaustive collinear/concyclic screening (performed twice: once by the
    operation and once by source-bound result validation) inside the inline
    execution envelope.
    """

    points: tuple[ForbiddenLabelledPoint, ...] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def require_unique_labels_and_bounded_screening(self) -> Self:
        labels = tuple(item.label for item in self.points)
        if len(labels) != len(set(labels)):
            raise ValueError("configuration point labels must be unique")
        keys = tuple(_point_key(item.point) for item in self.points)
        if len(keys) != len(set(keys)):
            raise ValueError("configuration point coordinates must be unique")
        for item in self.points:
            require_bounded_rational(
                item.point.x,
                max_digits=_MAX_FORBIDDEN_COORDINATE_DIGITS,
                label="configuration point x",
            )
            require_bounded_rational(
                item.point.y,
                max_digits=_MAX_FORBIDDEN_COORDINATE_DIGITS,
                label="configuration point y",
            )
        if _forbidden_screening_work_units(self.points) > (
            _MAX_FORBIDDEN_SCREENING_WORK_UNITS
        ):
            raise ValueError(
                "point configuration exceeds the exhaustive screening work "
                f"budget of {_MAX_FORBIDDEN_SCREENING_WORK_UNITS} units "
                "(quadruple count times cleared coordinate digit size); "
                "reduce the point count or coordinate height"
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


class _ForbiddenScreening(NamedTuple):
    """Exhaustive enumeration outcome for one configuration."""

    has_collinear_triple: bool
    collinear_triple: tuple[int, int, int] | None
    has_concyclic_quadruple: bool
    concyclic_quadruple: tuple[int, int, int, int] | None
    checked_triples: int
    checked_quadruples: int


def _collinear_scaled(
    first: tuple[int, int],
    second: tuple[int, int],
    third: tuple[int, int],
) -> bool:
    (xi, yi), (xj, yj), (xk, yk) = first, second, third
    return (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi) == 0


def _concyclic_det4_scaled(
    first: tuple[int, int],
    second: tuple[int, int],
    third: tuple[int, int],
    fourth: tuple[int, int],
) -> int:
    def row(point: tuple[int, int]) -> list[int]:
        x, y = point
        return [x * x + y * y, x, y, 1]

    m = (row(first), row(second), row(third), row(fourth))
    return (
        m[0][0] * _minor3(m, 1, 2, 3, 1, 2, 3)
        - m[0][1] * _minor3(m, 1, 2, 3, 0, 2, 3)
        + m[0][2] * _minor3(m, 1, 2, 3, 0, 1, 3)
        - m[0][3] * _minor3(m, 1, 2, 3, 0, 1, 2)
    )


def _screen_forbidden_patterns(
    points: tuple[ForbiddenLabelledPoint, ...],
) -> _ForbiddenScreening:
    """Enumerate every triple and quadruple of one configuration exactly.

    Coordinates are cleared once by scaling with the positive lcm of all
    coordinate denominators.  Both determinants then become integer
    polynomials in the scaled coordinates whose vanishing is equivalent to
    the original rational test, so decisions, witnesses, and enumeration
    counts are identical to direct exact Fraction evaluation.
    """

    xy = [(item.point.x.as_fraction(), item.point.y.as_fraction()) for item in points]
    scale = math.lcm(
        *(d for point in xy for d in (point[0].denominator, point[1].denominator))
    )
    scaled = [(int(x * scale), int(y * scale)) for x, y in xy]

    collinear_triple = None
    checked_triples = 0
    for i, j, k in combinations(range(len(scaled)), 3):
        checked_triples += 1
        if _collinear_scaled(scaled[i], scaled[j], scaled[k]):
            collinear_triple = (i, j, k)
            break

    concyclic_quadruple = None
    checked_quadruples = 0
    for i, j, k, ell in combinations(range(len(scaled)), 4):
        checked_quadruples += 1
        if _concyclic_det4_scaled(scaled[i], scaled[j], scaled[k], scaled[ell]) != 0:
            continue
        # A quadruple containing a collinear triple is degenerate: three
        # distinct collinear points cannot lie on a finite circle.
        if any(
            _collinear_scaled(scaled[a], scaled[b], scaled[c])
            for a, b, c in ((i, j, k), (i, j, ell), (i, k, ell), (j, k, ell))
        ):
            continue
        concyclic_quadruple = (i, j, k, ell)
        break

    return _ForbiddenScreening(
        has_collinear_triple=collinear_triple is not None,
        collinear_triple=collinear_triple,
        has_concyclic_quadruple=concyclic_quadruple is not None,
        concyclic_quadruple=concyclic_quadruple,
        checked_triples=checked_triples,
        checked_quadruples=checked_quadruples,
    )


class ForbiddenPatternsResult(StrictModel):
    """Result of screening a configuration for forbidden patterns."""

    configuration: ForbiddenConfiguration
    point_count: StrictInt = Field(ge=1, le=128)
    has_collinear_triple: bool
    has_concyclic_quadruple: bool
    collinear_triple: CollinearTriple | None = None
    concyclic_quadruple: ConcyclicQuadruple | None = None
    checked_triples: StrictInt = Field(ge=0)
    checked_quadruples: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def bind_witnesses(self) -> Self:
        if self.point_count != len(self.configuration.points):
            raise ValueError("point_count must match the retained configuration")
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
        screening = _screen_forbidden_patterns(self.configuration.points)
        expected_collinear = (
            None
            if screening.collinear_triple is None
            else CollinearTriple(
                first=screening.collinear_triple[0],
                second=screening.collinear_triple[1],
                third=screening.collinear_triple[2],
            )
        )
        expected_concyclic = (
            None
            if screening.concyclic_quadruple is None
            else ConcyclicQuadruple(
                first=screening.concyclic_quadruple[0],
                second=screening.concyclic_quadruple[1],
                third=screening.concyclic_quadruple[2],
                fourth=screening.concyclic_quadruple[3],
            )
        )
        if (
            self.has_collinear_triple != screening.has_collinear_triple
            or self.has_concyclic_quadruple != screening.has_concyclic_quadruple
            or self.collinear_triple != expected_collinear
            or self.concyclic_quadruple != expected_concyclic
            or self.checked_triples != screening.checked_triples
            or self.checked_quadruples != screening.checked_quadruples
        ):
            raise ValueError(
                "decisions and witnesses must equal the exhaustive enumeration "
                "of the retained configuration"
            )
        return self
