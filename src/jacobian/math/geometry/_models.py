"""Exact rational planar-geometry wire contracts."""

from __future__ import annotations

import unicodedata
from fractions import Fraction
from itertools import combinations
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import encode_strict_json, format_canonical_integer
from jacobian.math.geometry.exact._models import (
    LabelledRationalPoint,
    PointConfiguration,
)


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


CIRCUMRADIUS_COORDINATE_DIGITS = 256
"""Conservative per-coordinate digit bound.

``R^2 = (|a-b|^2 |b-c|^2 |c-a|^2) / (4 * cross^2)`` roughly squares each
input height, so bounding inputs at a fraction of the 32,768-digit canonical
limit keeps every intermediate and output representable.
"""

CIRCUMRADIUS_INPUT_HEIGHT = CIRCUMRADIUS_COORDINATE_DIGITS // 4
"""Schema-published per-coordinate component digit bound (numerator and denominator)."""

MAX_CIRCUMRADIUS_PROFILE_RESULT_BYTES = 10 * 1024 * 1024
"""Aggregate canonical-output budget for one complete circumradius profile."""

_CIRCUMRADIUS_RESULT_BOUND_PADDING_BYTES = 1_024

_CIRCUMRADIUS_ENTRY_SLACK_BYTES = 16
"""Per-entry slack over the exact skeleton, label, index, and digit bounds."""


def _bounded_circumradius_coordinate(value: CanonicalRational, label: str) -> None:
    from jacobian._exact import require_bounded_rational

    require_bounded_rational(
        value,
        max_digits=CIRCUMRADIUS_INPUT_HEIGHT,
        label=label,
    )


def _difference_digit_heights(
    left: LabelledRationalPoint,
    right: LabelledRationalPoint,
) -> tuple[int, int, int, int]:
    """Return reduced coordinate-difference component digit counts.

    The four counts are ``(digits(|dx numerator|), digits(dx denominator),
    digits(|dy numerator|), digits(dy denominator))`` for the planar
    difference ``left - right``.
    """

    delta_x = left.coordinates[0].as_fraction() - right.coordinates[0].as_fraction()
    delta_y = left.coordinates[1].as_fraction() - right.coordinates[1].as_fraction()
    return (
        len(format_canonical_integer(abs(delta_x.numerator))),
        len(format_canonical_integer(delta_x.denominator)),
        len(format_canonical_integer(abs(delta_y.numerator))),
        len(format_canonical_integer(delta_y.denominator)),
    )


def _nfc_normalized(value: object) -> object:
    """Apply the output encoder's NFC normalization step to a wire value.

    ``OperationResult.require_canonical_output`` canonicalizes every string
    with NFC, which can expand a label (for example U+0958 becomes two
    characters), so byte estimates must charge the normalized encoding.
    """

    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, dict):
        return {key: _nfc_normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_nfc_normalized(item) for item in value]
    return value


def _maximum_profile_wire_bytes(configuration: PointConfiguration) -> int:
    """Upper-bound the canonical wire encoding of the complete profile result.

    Every bound below is taken on unreduced forms, so canonical reduction at
    any step can only shrink the real encoding. For a reduced coordinate
    difference with numerator digit count ``n`` and denominator digit count
    ``d``, its square has numerator within ``2n`` and denominator within
    ``2d`` digits; a squared distance sums two such squares over the product
    denominator, gaining one sum-carry digit on the numerator. The triangle's
    squared-distance product multiplies those bounds. The cross of two edge
    differences scales each term to the other's denominator, so its numerator
    gains the opposite term's denominator digits plus a carry and its
    denominator is the summed denominator digits. Dividing by ``4 * cross^2``
    moves twice the cross bounds across the fraction with one multiplier
    carry digit. Each entry is costed as an encoding skeleton plus both
    radius component bounds (not their maximum), exact label and index
    encodings, and slack; the configuration echo and every label are
    charged at their NFC-normalized encoding, matching the canonical
    output encoder.
    """

    points = configuration.points
    pairs = {
        (first, second): _difference_digit_heights(
            points[first],
            points[second],
        )
        for first, second in combinations(range(len(points)), 2)
    }

    def squared_distance_digit_bounds(
        first: int,
        second: int,
    ) -> tuple[int, int]:
        num_x, den_x, num_y, den_y = pairs[(first, second)]
        return (
            max(2 * num_x + 2 * den_y, 2 * num_y + 2 * den_x) + 1,
            2 * den_x + 2 * den_y,
        )

    label_bytes = [
        len(encode_strict_json(_nfc_normalized(item.label))) for item in points
    ]
    index_bytes = [len(str(index)) for index in range(len(points))]
    skeleton = {
        "collinear": False,
        "indices": [0, 0, 0],
        "labels": ["", "", ""],
        "squared_circumradius": {"num": "", "den": ""},
    }
    entry_base = len(encode_strict_json(skeleton))
    total = len(
        encode_strict_json(_nfc_normalized(configuration.model_dump(mode="json")))
    )
    triple_count = 0
    radius_numerator_digits_max = 0
    radius_denominator_digits_max = 0
    for first, second, third in combinations(range(len(points)), 3):
        numerator_bound = 0
        denominator_bound = 0
        for left, right in ((first, second), (second, third), (first, third)):
            squared_numerator, squared_denominator = squared_distance_digit_bounds(
                left,
                right,
            )
            numerator_bound += squared_numerator
            denominator_bound += squared_denominator
        # Cross of the (first, second) and (first, third) differences: each
        # term scales to the other term's denominator before subtracting.
        nx_ab, bx_ab, ny_ab, by_ab = pairs[(first, second)]
        nx_ac, bx_ac, ny_ac, by_ac = pairs[(first, third)]
        first_numerator = nx_ab + ny_ac
        first_denominator = bx_ab + by_ac
        second_numerator = ny_ab + nx_ac
        second_denominator = by_ab + bx_ac
        cross_numerator_digits = (
            max(
                first_numerator + second_denominator,
                second_numerator + first_denominator,
            )
            + 1
        )
        cross_denominator_digits = first_denominator + second_denominator
        radius_numerator_digits = numerator_bound + 2 * cross_denominator_digits + 1
        radius_denominator_digits = denominator_bound + 2 * cross_numerator_digits + 1
        radius_numerator_digits_max = max(
            radius_numerator_digits_max, radius_numerator_digits
        )
        radius_denominator_digits_max = max(
            radius_denominator_digits_max, radius_denominator_digits
        )
        total += (
            entry_base
            + sum(label_bytes[index] - 2 for index in (first, second, third))
            + sum(index_bytes[index] - 1 for index in (first, third, second))
            + radius_numerator_digits
            + radius_denominator_digits
            + _CIRCUMRADIUS_ENTRY_SLACK_BYTES
        )
        triple_count += 1
    multiplicity_entry_skeleton = len(
        encode_strict_json(
            {"squared_circumradius": {"num": "", "den": ""}, "triple_count": 0}
        )
    )
    total += triple_count * (
        multiplicity_entry_skeleton
        + radius_numerator_digits_max
        + radius_denominator_digits_max
        + _CIRCUMRADIUS_ENTRY_SLACK_BYTES
    )
    total += (
        len(encode_strict_json({"degenerate_triple_count": 0}))
        + len(encode_strict_json({"nondegenerate_triple_count": triple_count}))
        + triple_count
        + 1
        + _CIRCUMRADIUS_RESULT_BOUND_PADDING_BYTES
    )
    return total


def _require_admissible_circumradius_configuration(
    configuration: PointConfiguration,
) -> None:
    """Apply the operation's complete admission to one retained configuration.

    Shared by the request boundary and the result replay validator so an
    authored or deserialized result can never retain a configuration outside
    ``CircumradiusProfileRequest``'s domain: at least three points, exactly
    two coordinates per point, unique coordinates, every component within the
    published digit bound, and a complete profile within the aggregate
    result budget.
    """

    points = configuration.points
    if len(points) < 3:
        raise ValueError("circumradius profile requires at least three points")
    if any(len(item.coordinates) != 2 for item in points):
        raise ValueError(
            "circumradius profile requires a planar configuration "
            "(exactly two coordinates per point)"
        )
    keys = tuple(
        tuple((component.num, component.den) for component in item.coordinates)
        for item in points
    )
    if len(keys) != len(set(keys)):
        raise ValueError("point coordinates must be unique")
    for index, item in enumerate(points):
        for axis, component in enumerate(item.coordinates):
            _bounded_circumradius_coordinate(
                component, f"point {index} coordinate {axis}"
            )
    estimated_bytes = _maximum_profile_wire_bytes(configuration)
    if estimated_bytes > MAX_CIRCUMRADIUS_PROFILE_RESULT_BYTES:
        raise ValueError(
            "the complete circumradius profile would exceed the "
            f"{MAX_CIRCUMRADIUS_PROFILE_RESULT_BYTES}-byte aggregate "
            "result budget; reduce the point count or coordinate heights"
        )


class CircumradiusProfileRequest(StrictModel):
    """A bounded labelled rational planar point configuration.

    Requires at least three points (a circumradius profile needs triples),
    unique labels, unique coordinates, exactly two coordinates per point,
    coordinates within the published ``coordinate_digit_bound`` schema
    metadata, and an aggregate profile that fits the published aggregate
    result budget, so every accepted request returns one typed exact result
    instead of an unencodable value.
    """

    configuration: PointConfiguration = Field(
        description=(
            "Canonical labelled rational point configuration (the same "
            "value accepted by distance_profile and distance_graph) with "
            "at least three points and exactly two coordinates per point. "
            "Every coordinate numerator and denominator carries at most "
            f"{CIRCUMRADIUS_INPUT_HEIGHT} canonical decimal digits, and "
            "configurations whose complete profile would exceed the "
            f"{MAX_CIRCUMRADIUS_PROFILE_RESULT_BYTES}-byte aggregate result "
            "budget are rejected before execution."
        ),
        json_schema_extra={
            "coordinate_digit_bound": CIRCUMRADIUS_INPUT_HEIGHT,
            "aggregate_result_budget_bytes": MAX_CIRCUMRADIUS_PROFILE_RESULT_BYTES,
            "min_point_count": 3,
            "dimension": 2,
        },
    )

    @model_validator(mode="after")
    def require_admissible_configuration(self) -> Self:
        _require_admissible_circumradius_configuration(self.configuration)
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
    """Complete exact circumradius profile bound to its source configuration."""

    configuration: PointConfiguration = Field(
        description=(
            "The canonical labelled rational point configuration retained "
            "for replay; the same value accepted by distance_profile and "
            "distance_graph, so it composes into those operations unchanged."
        )
    )
    point_count: StrictInt = Field(ge=3, le=64)
    triple_count: StrictInt = Field(ge=1, le=41664)
    entries: tuple[CircumradiusTripleEntry, ...] = Field(min_length=1, max_length=41664)
    radius_multiplicities: tuple[tuple[CanonicalRational, StrictInt], ...] = Field(
        default=(),
        max_length=41664,
        description=(
            "Distinct positive squared circumradii of the nondegenerate "
            "triples with their multiplicities, sorted ascending by value; "
            "empty when every triple is collinear."
        ),
    )
    degenerate_triple_count: StrictInt = Field(ge=0)
    nondegenerate_triple_count: StrictInt = Field(ge=0)
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"

    @model_validator(mode="after")
    def require_complete_profile(self) -> Self:  # noqa: C901
        import math

        # Cap point_count before enumerating expected triples.
        points = self.configuration.points
        if self.point_count > 64 or len(points) != self.point_count:
            raise ValueError(
                "point_count must match the retained configuration (at most 64)"
            )
        expected_count = math.comb(self.point_count, 3)
        if len(self.entries) != self.triple_count:
            raise ValueError("circumradius profile must be complete")
        if self.triple_count != expected_count:
            raise ValueError("triple_count must equal C(point_count, 3)")
        # The retained configuration must satisfy the same operation-specific
        # admission as a fresh request before any replay work.
        _require_admissible_circumradius_configuration(self.configuration)
        coords: list[tuple[Fraction, Fraction]] = [
            (item.coordinates[0].as_fraction(), item.coordinates[1].as_fraction())
            for item in points
        ]
        seen: set[tuple[int, int, int]] = set()
        histogram: dict[Fraction, int] = {}
        from itertools import combinations

        expected_triples = set(combinations(range(self.point_count), 3))
        for entry in self.entries:
            i, j, k = entry.indices
            triple = (i, j, k)
            if triple not in expected_triples or triple in seen:
                raise ValueError(
                    "entries must cover exactly the canonical triples once"
                )
            seen.add(triple)
            # Replay collinearity and radius against the source configuration.
            (ax, ay), (bx, by), (cx, cy) = coords[i], coords[j], coords[k]
            cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if entry.collinear != (cross == 0):
                raise ValueError(
                    "collinearity flag does not match the source configuration"
                )
            if entry.labels != (points[i].label, points[j].label, points[k].label):
                raise ValueError("entry does not match its recomputed labels")
            if entry.collinear:
                continue
            dab = (ax - bx) ** 2 + (ay - by) ** 2
            dbc = (bx - cx) ** 2 + (by - cy) ** 2
            dac = (ax - cx) ** 2 + (ay - cy) ** 2
            expected_radius = (dab * dbc * dac) / (4 * cross * cross)
            assert entry.squared_circumradius is not None
            if entry.squared_circumradius.as_fraction() != expected_radius:
                raise ValueError("entry does not match its recomputed radius")
            d = entry.squared_circumradius.as_fraction()
            histogram[d] = histogram.get(d, 0) + 1

        # The advertised multiplicity profile is a defining invariant of the
        # flat entries: reconstruct it from the replayed radii and compare.
        degenerate_count = sum(1 for entry in self.entries if entry.collinear)
        nondegenerate_count = len(self.entries) - degenerate_count
        if self.degenerate_triple_count != degenerate_count:
            raise ValueError("degenerate triple count does not match the entries")
        if self.nondegenerate_triple_count != nondegenerate_count:
            raise ValueError("nondegenerate triple count does not match the entries")
        reconstructed = tuple(
            (
                CanonicalRational.from_fraction(d),
                count,
            )
            for d, count in sorted(histogram.items())
        )
        if reconstructed != tuple(self.radius_multiplicities):
            raise ValueError(
                "radius multiplicities must partition the replayed "
                "nondegenerate radii and be sorted by value"
            )
        return self
