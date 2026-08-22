"""Exact rational planar-geometry wire contracts."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math._rational_height import RationalHeight


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


def _circumradius_input_height_ok(point: RationalPoint2D) -> bool:
    # Conservative worst-case propagation for
    # R^2 = (|a-b|^2 |b-c|^2 |c-a|^2) / (4 * cross^2).
    # A coordinate difference of two H-digit rationals has numerator at most
    # 2H+1 digits and denominator at most 2H; each squared side reaches
    # ~4H+3 over 4H, the cross product ~4H+5 over 4H, so the reduced R^2
    # stays within roughly (12H+9) + (8H+10) + small slack = 20H + 25
    # digits. Requiring 20*1024 + 25 <= 32768 admits H = 1024; independent
    # denominators cannot exceed the canonical limit at execution.
    max_input = 1024
    for v in (point.x, point.y):
        if RationalHeight.from_canonical(v).exceeds(max_input):
            return False
    return True


class CircumradiusProfileRequest(StrictModel):
    """A bounded labelled rational planar point configuration."""

    points: tuple[LabelledPoint2D, ...] = Field(min_length=3, max_length=24)

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
            if not _circumradius_input_height_ok(item.point):
                raise ValueError(
                    "circumradius coordinates exceed the conservative 1024-digit input bound for exact output"
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
    point_count: StrictInt = Field(ge=3, le=24)
    triple_count: StrictInt = Field(ge=1, le=2024)
    entries: tuple[CircumradiusTripleEntry, ...] = Field(min_length=1)
    points: tuple[LabelledPoint2D, ...] = Field(
        min_length=3,
        max_length=24,
        description="Source labelled points; every entry must match the exact circumradius recomputation",
    )
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"

    @model_validator(mode="after")
    def require_complete_profile(self) -> Self:
        if len(self.entries) != self.triple_count:
            raise ValueError("circumradius profile must be complete")
        _require_triple_index_coverage(self.point_count, self.entries)
        if len(self.points) != self.point_count:
            raise ValueError("points length must match point_count")
        _require_circumradius_source_replay(self.points, self.entries)
        return self


def _require_triple_index_coverage(
    point_count: int,
    entries: tuple[CircumradiusTripleEntry, ...],
) -> None:
    """Entries must cover every unordered triple of configuration indices."""
    import math

    expected = math.comb(point_count, 3) if point_count >= 3 else 0
    seen: set[tuple[int, ...]] = set()
    for entry in entries:
        key = tuple(sorted(entry.indices))
        if (
            len(set(entry.indices)) != 3
            or list(entry.indices) != list(key)
            or key[0] < 0
            or key[2] >= point_count
        ):
            raise ValueError(
                "circumradius entry indices must be three distinct ascending positions in range"
            )
        if key in seen:
            raise ValueError("circumradius entries must cover each triple exactly once")
        seen.add(key)
    if len(seen) != expected:
        raise ValueError(
            f"triple_count {len(seen)} must equal C(point_count,3)={expected}"
        )


def _require_circumradius_source_replay(
    points: tuple[LabelledPoint2D, ...],
    entries: tuple[CircumradiusTripleEntry, ...],
) -> None:
    """Replay every circumradius entry against its source coordinates."""
    from jacobian._exact import CanonicalRational

    coords = [(pt.point.x.as_fraction(), pt.point.y.as_fraction()) for pt in points]
    for e in entries:
        i, j, k = e.indices
        # Bind entry identity to the source before any early return so
        # collinear entries cannot claim the wrong labels.
        if e.labels != (points[i].label, points[j].label, points[k].label):
            raise ValueError("circumradius labels must match source points")
        (ax, ay), (bx, by), (cx, cy) = coords[i], coords[j], coords[k]
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if cross == 0:
            if not e.collinear or e.squared_circumradius is not None:
                raise ValueError("collinear triple must have no radius")
            continue
        if e.collinear:
            raise ValueError("non-collinear triple must have a radius")
        dab = (ax - bx) ** 2 + (ay - by) ** 2
        dbc = (bx - cx) ** 2 + (by - cy) ** 2
        dac = (ax - cx) ** 2 + (ay - cy) ** 2
        expected_r2 = CanonicalRational.from_fraction(
            (dab * dbc * dac) / (4 * cross * cross)
        )
        if e.squared_circumradius != expected_r2:
            raise ValueError(
                "squared_circumradius does not match recomputed exact value"
            )


class ForbiddenLabelledPoint(StrictModel):
    """A labelled rational point in the affine plane."""

    label: str = Field(min_length=1, max_length=64)
    point: RationalPoint2D


MAX_FORBIDDEN_POINTS = 32
MAX_FORBIDDEN_COORDINATE_DIGITS = 64


class ForbiddenConfiguration(StrictModel):
    """A finite set of labelled rational planar points."""

    points: tuple[ForbiddenLabelledPoint, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def require_bounded_unique_labels_and_coords(self) -> Self:
        labels = tuple(item.label for item in self.points)
        if len(labels) != len(set(labels)):
            raise ValueError("configuration point labels must be unique")
        keys = tuple(_point_key(item.point) for item in self.points)
        if len(keys) != len(set(keys)):
            raise ValueError("configuration point coordinates must be unique")
        # Conservative worst-case work budget derived from the screening
        # enumeration: C(n,3) collinear determinants and C(n,4) concyclic
        # determinants. At n = 32 that is 4960 triple determinants and 35960
        # quadruple determinants; bounding every coordinate numerator and
        # denominator at 64 digits keeps squared sums near 130 digits, 3x3
        # cofactors near 200 digits, and each full determinant entry near 330
        # digits, so the complete exact enumeration - and the identical
        # replay performed by the result validator - stays within seconds.
        for item in self.points:
            for value in (item.point.x, item.point.y):
                if RationalHeight.from_canonical(value).exceeds(
                    MAX_FORBIDDEN_COORDINATE_DIGITS
                ):
                    raise ValueError(
                        "configuration coordinates exceed the conservative "
                        "64-digit forbidden-pattern enumeration bound"
                    )
        return self


class ForbiddenPatternsRequest(StrictModel):
    """A labelled rational planar point configuration to screen."""

    configuration: ForbiddenConfiguration


class CollinearTriple(StrictModel):
    """A triple of configuration point indices lying on one line."""

    first: StrictInt = Field(ge=0, le=31)
    second: StrictInt = Field(ge=0, le=31)
    third: StrictInt = Field(ge=0, le=31)

    @model_validator(mode="after")
    def require_strictly_ascending(self) -> Self:
        if not (self.first < self.second < self.third):
            raise ValueError("collinear triple indices must be strictly ascending")
        return self


class ConcyclicQuadruple(StrictModel):
    """A quadruple of configuration point indices lying on one circle."""

    first: StrictInt = Field(ge=0, le=31)
    second: StrictInt = Field(ge=0, le=31)
    third: StrictInt = Field(ge=0, le=31)
    fourth: StrictInt = Field(ge=0, le=31)

    @model_validator(mode="after")
    def require_strictly_ascending(self) -> Self:
        if not (self.first < self.second < self.third < self.fourth):
            raise ValueError("concyclic quadruple indices must be strictly ascending")
        return self


class ForbiddenPatternsResult(StrictModel):
    """Result of screening a configuration for forbidden patterns."""

    configuration: ForbiddenConfiguration
    point_count: StrictInt = Field(ge=1, le=32)
    has_collinear_triple: bool
    has_concyclic_quadruple: bool
    collinear_triple: CollinearTriple | None = None
    concyclic_quadruple: ConcyclicQuadruple | None = None
    checked_triples: StrictInt = Field(ge=0)
    checked_quadruples: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def bind_decision_to_configuration(self) -> Self:
        if self.point_count != len(self.configuration.points):
            raise ValueError("point count must match the screened configuration")
        if self.has_collinear_triple is (self.collinear_triple is None):
            raise ValueError("exactly a collinear triple carries one witness")
        if self.has_concyclic_quadruple is (self.concyclic_quadruple is None):
            raise ValueError("exactly a concyclic quadruple carries one witness")
        # Source-bound conclusion: the retained decision, witnesses, and
        # enumeration counts must be the exact screening of this result's own
        # configuration, so a relayed or forged payload cannot validate
        # against a different configuration.
        from jacobian.math.geometry._operations import _screen_configuration

        (
            has_collinear,
            has_concyclic,
            collinear_indices,
            concyclic_indices,
            checked_triples,
            checked_quadruples,
        ) = _screen_configuration(self.configuration.points)
        expected_collinear = (
            CollinearTriple(
                first=collinear_indices[0],
                second=collinear_indices[1],
                third=collinear_indices[2],
            )
            if collinear_indices is not None
            else None
        )
        expected_concyclic = (
            ConcyclicQuadruple(
                first=concyclic_indices[0],
                second=concyclic_indices[1],
                third=concyclic_indices[2],
                fourth=concyclic_indices[3],
            )
            if concyclic_indices is not None
            else None
        )
        if (
            self.has_collinear_triple != has_collinear
            or self.has_concyclic_quadruple != has_concyclic
            or self.collinear_triple != expected_collinear
            or self.concyclic_quadruple != expected_concyclic
            or self.checked_triples != checked_triples
            or self.checked_quadruples != checked_quadruples
        ):
            raise ValueError(
                "forbidden-pattern conclusion must be the exact screening "
                "of its retained configuration"
            )
        return self
