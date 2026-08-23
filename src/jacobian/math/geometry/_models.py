"""Exact rational planar-geometry wire contracts."""

from __future__ import annotations

from fractions import Fraction
from math import lcm
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
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


def _circumradius_result_height(
    points: tuple[LabelledPoint2D, ...],
) -> RationalHeight:
    """Conservative height of R^2 = |a-b|^2|b-c|^2|c-a|^2 / (4*(2*area)^2).

    Every coordinate is bounded by the componentwise maximum input height.
    A displacement sums two coordinates, a squared distance sums two squared
    displacements, and the twice-area cross product subtracts two products of
    two displacements, so it shares the squared-distance bound shape; the
    quotient propagates both unreduced-fraction components.
    """
    heights = [
        RationalHeight.from_canonical(rational)
        for item in points
        for rational in (item.point.x, item.point.y)
    ]
    coordinate = RationalHeight(
        max(height.numerator_digits for height in heights),
        max(height.denominator_digits for height in heights),
    )
    displacement = sum_heights((coordinate, coordinate))
    displacement_squared = displacement.product(displacement)
    side = sum_heights((displacement_squared, displacement_squared))
    side_cubed = side.product(side).product(side)
    cross = sum_heights((displacement_squared, displacement_squared))
    four_cross_squared = cross.product(cross).product(RationalHeight(1, 1))
    return side_cubed.quotient(four_cross_squared)


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
        # The formula R^2 = |a-b|^2|b-c|^2|c-a|^2 / (4*(2*area)^2) multiplies
        # three squared distances and divides by the squared twice-area
        # cross product, so subtracting rationals, squaring, and multiplying
        # grow digit counts well past any per-coordinate cap.  Propagate a
        # conservative height bound through the complete formula instead and
        # admit only configurations whose every triple result stays within
        # the canonical limit.
        if _circumradius_result_height(self.points).exceeds(
            MAX_CANONICAL_RATIONAL_DIGITS
        ):
            raise ValueError(
                "point configuration would produce squared circumradii "
                f"exceeding the {MAX_CANONICAL_RATIONAL_DIGITS}-digit "
                "canonical result bound"
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


class CircumradiusProfileResult(StrictModel):
    points: tuple[LabelledPoint2D, ...] = Field(min_length=3, max_length=64)
    point_count: StrictInt = Field(ge=3, le=64)
    triple_count: StrictInt = Field(ge=1, le=41664)
    entries: tuple[CircumradiusTripleEntry, ...] = Field(min_length=1)
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"

    @model_validator(mode="after")
    def require_complete_profile(self) -> Self:
        import math

        if self.point_count != len(self.points):
            raise ValueError("point_count must match the retained points length")
        labels = tuple(item.label for item in self.points)
        if len(labels) != len(set(labels)):
            raise ValueError("source point labels must be unique")
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
            raise ValueError("source point coordinates must be unique")
        expected = math.comb(self.point_count, 3)
        if self.triple_count != expected:
            raise ValueError("triple_count must equal comb(point_count, 3)")
        if len(self.entries) != self.triple_count:
            raise ValueError("circumradius profile must be complete")
        _require_canonical_triple_coverage(self, expected)
        # Replay every collinearity flag and squared circumradius from the
        # retained source points so a serialized result cannot detach its
        # radii from their configuration.
        coords = [
            (item.point.x.as_fraction(), item.point.y.as_fraction())
            for item in self.points
        ]
        _require_replayed_circumradii(self, coords)
        return self


def _require_canonical_triple_coverage(
    result: CircumradiusProfileResult, expected: int
) -> None:
    """Every unordered triple appears exactly once in canonical order."""
    seen: set[tuple[int, int, int]] = set()
    for entry in result.entries:
        idx = entry.indices
        if idx != tuple(sorted(idx)):
            raise ValueError("circumradius entry indices must be sorted")
        if idx in seen:
            raise ValueError("circumradius entries must be unique")
        if not (0 <= idx[0] < idx[1] < idx[2] < result.point_count):
            raise ValueError("circumradius entry indices out of range")
        expected_labels = (
            result.points[idx[0]].label,
            result.points[idx[1]].label,
            result.points[idx[2]].label,
        )
        if entry.labels != expected_labels:
            raise ValueError("circumradius entry labels must match the source points")
        seen.add(idx)
    if tuple(result.entries) != tuple(sorted(result.entries, key=lambda e: e.indices)):
        raise ValueError("circumradius entries must be in lexicographic order")
    if len(seen) != expected:
        raise ValueError("circumradius profile must cover every unordered triple")


def _require_replayed_circumradii(
    result: CircumradiusProfileResult,
    coords: list[tuple[Fraction, Fraction]],
) -> None:
    """Replay each entry's collinearity and radius from the retained points."""
    for entry in result.entries:
        i, j, k = entry.indices
        (ax, ay), (bx, by), (cx, cy) = coords[i], coords[j], coords[k]
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        is_collinear = cross == 0
        if entry.collinear != is_collinear:
            raise ValueError(
                "circumradius entry collinear flag does not match the points"
            )
        if not is_collinear:
            dab = (ax - bx) ** 2 + (ay - by) ** 2
            dbc = (bx - cx) ** 2 + (by - cy) ** 2
            dac = (ax - cx) ** 2 + (ay - cy) ** 2
            expected_r2 = (dab * dbc * dac) / (4 * cross * cross)
            if (
                entry.squared_circumradius is None
                or entry.squared_circumradius.as_fraction() != expected_r2
            ):
                raise ValueError("squared circumradius does not match the exact value")
        elif entry.squared_circumradius is not None:
            raise ValueError("collinear entry must not have a circumradius")


class ForbiddenLabelledPoint(StrictModel):
    """A labelled rational point in the affine plane."""

    label: str = Field(min_length=1, max_length=64)
    point: RationalPoint2D


class ForbiddenConfiguration(StrictModel):
    """A finite set of labelled rational planar points."""

    points: tuple[ForbiddenLabelledPoint, ...] = Field(min_length=1, max_length=40)

    @model_validator(mode="after")
    def require_unique_labels_and_coords(self) -> Self:
        labels = tuple(item.label for item in self.points)
        if len(labels) != len(set(labels)):
            raise ValueError("configuration point labels must be unique")
        keys = tuple(_point_key(item.point) for item in self.points)
        if len(keys) != len(set(keys)):
            raise ValueError("configuration point coordinates must be unique")
        return self


class ForbiddenPatternsRequest(StrictModel):
    """A labelled rational planar point configuration to screen."""

    configuration: ForbiddenConfiguration


class CollinearTriple(StrictModel):
    """A triple of configuration point indices lying on one line."""

    first: StrictInt = Field(ge=0, le=39)
    second: StrictInt = Field(ge=0, le=39)
    third: StrictInt = Field(ge=0, le=39)

    @model_validator(mode="after")
    def require_strictly_ascending(self) -> Self:
        if not (self.first < self.second < self.third):
            raise ValueError("collinear triple indices must be strictly ascending")
        return self


class ConcyclicQuadruple(StrictModel):
    """A quadruple of configuration point indices lying on one circle."""

    first: StrictInt = Field(ge=0, le=39)
    second: StrictInt = Field(ge=0, le=39)
    third: StrictInt = Field(ge=0, le=39)
    fourth: StrictInt = Field(ge=0, le=39)

    @model_validator(mode="after")
    def require_strictly_ascending(self) -> Self:
        if not (self.first < self.second < self.third < self.fourth):
            raise ValueError("concyclic quadruple indices must be strictly ascending")
        return self


def _cleared_configuration_points(
    points: tuple[ForbiddenLabelledPoint, ...],
) -> tuple[tuple[int, int, int], ...]:
    """Exact homogeneous rows ``(X, Y, D)`` for each point ``(X/D, Y/D)``.

    Distinct points may carry distinct row scales, so every predicate replayed
    from these rows must use the full ``(X, Y, D)`` determinant rather than an
    affine shortcut on the first two columns.
    """
    cleared = []
    for item in points:
        fx = item.point.x.as_fraction()
        fy = item.point.y.as_fraction()
        scale = lcm(fx.denominator, fy.denominator)
        cleared.append(
            (
                fx.numerator * (scale // fx.denominator),
                fy.numerator * (scale // fy.denominator),
                scale,
            )
        )
    return tuple(cleared)


def _int_collinear(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    c: tuple[int, int, int],
) -> bool:
    return _int_det3([list(a), list(b), list(c)]) == 0


def _int_det3(m: list[list[int]]) -> int:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _int_det4(m: list[list[int]]) -> int:
    return sum(
        (-1) ** column
        * m[0][column]
        * _int_det3([row[:column] + row[column + 1 :] for row in m[1:]])
        for column in range(4)
    )


def _scan_collinear_triple(
    rows: tuple[tuple[int, int, int], ...],
) -> tuple[CollinearTriple | None, int]:
    """Mirror the operation's ascending triple scan; report work performed."""
    from itertools import combinations

    examined = 0
    for i, j, k in combinations(range(len(rows)), 3):
        examined += 1
        if _int_collinear(rows[i], rows[j], rows[k]):
            return CollinearTriple(first=i, second=j, third=k), examined
    return None, examined


def _scan_concyclic_quadruple(
    rows: tuple[tuple[int, int, int], ...],
) -> tuple[ConcyclicQuadruple | None, int]:
    """Mirror the operation's ascending quadruple scan of nondegenerate circles."""
    from itertools import combinations

    examined = 0
    for i, j, k, ell in combinations(range(len(rows)), 4):
        examined += 1
        circle_rows = [
            [x * x + y * y, x * d, y * d, d * d]
            for x, y, d in (rows[index] for index in (i, j, k, ell))
        ]
        if _int_det4(circle_rows) == 0 and not any(
            _int_collinear(rows[a], rows[b], rows[c])
            for a, b, c in combinations((i, j, k, ell), 3)
        ):
            return ConcyclicQuadruple(first=i, second=j, third=k, fourth=ell), examined
    return None, examined


class ForbiddenPatternsResult(StrictModel):
    """Result of screening a configuration for forbidden patterns."""

    configuration: ForbiddenConfiguration
    point_count: StrictInt = Field(ge=1, le=40)
    has_collinear_triple: bool
    has_concyclic_quadruple: bool
    collinear_triple: CollinearTriple | None = None
    concyclic_quadruple: ConcyclicQuadruple | None = None
    checked_triples: StrictInt = Field(ge=0)
    checked_quadruples: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def bind_witnesses(self) -> Self:
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
        return self

    @model_validator(mode="after")
    def require_source_bound_decision(self) -> Self:
        # Replay the enumeration against the retained configuration: every
        # negative conclusion must follow from examining every triple and
        # quadruple, each witness must be geometrically true, and the checked
        # counts must match the exact scan prefix that produced them.
        cleared = _cleared_configuration_points(self.configuration.points)
        first_collinear, examined_triples = _scan_collinear_triple(cleared)
        if self.has_collinear_triple != (first_collinear is not None):
            raise ValueError(
                "collinear decision does not match the retained configuration"
            )
        if self.checked_triples != examined_triples:
            raise ValueError(
                "checked_triples must equal the number of triples the "
                "enumeration examined"
            )
        if first_collinear is not None and self.collinear_triple != first_collinear:
            raise ValueError(
                "collinear witness must be the first collinear triple in enumeration order"
            )
        first_concyclic, examined_quadruples = _scan_concyclic_quadruple(cleared)
        if self.has_concyclic_quadruple != (first_concyclic is not None):
            raise ValueError(
                "concyclic decision does not match the retained configuration"
            )
        if self.checked_quadruples != examined_quadruples:
            raise ValueError(
                "checked_quadruples must equal the number of quadruples the "
                "enumeration examined"
            )
        if first_concyclic is not None and self.concyclic_quadruple != first_concyclic:
            raise ValueError(
                "concyclic witness must be the first nondegenerate quadruple "
                "in enumeration order"
            )
        return self
