"""Exact rational planar-geometry wire contracts."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from itertools import combinations
from typing import Literal, Self

from pydantic import ConfigDict, Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian._models import StrictModel
from jacobian.canonical import encode_strict_json, format_canonical_integer


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable error owned by the geometry contracts."""

    return PydanticCustomError(f"geometry.{reason}", message)


MAX_CONFIGURATION_POINTS = 32
MAX_COORDINATE_DIGITS = 256
# Serialized-output budget for the circumradius profile, kept below the 10 MiB
# transport envelope to leave room for request and JSON overhead.
_MAX_PROFILE_OUTPUT_CHARS = 8_000_000
_CIRCUMRADIUS_DIGIT_GROWTH = 80
_CIRCUMRADIUS_ENTRY_OVERHEAD = 80
# Joint work bound for the exhaustive general-position search.  The sweep
# performs one exact 4x4 determinant per point quadruple, so the determinant
# count grows as C(n,4) while every Fraction multiplication grows
# quadratically in coordinate digit count; the admitted work proxy is
# ``C(n,4) * max_digits**2`` (measured reference: 32 points x 32 digits
# costs about 36M proxy units and roughly 16s, so a 1M-unit budget keeps an
# accepted request well under a second for the exhaustive search and canonical
# output construction).
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
        raise _validation_error(
            "general_position_search_n_points_max",
            f"general-position search with {n} points and {max_digits}-digit "
            f"coordinates exceeds the exhaustive work bound "
            f"(C(n,4)*digits^2={work} > "
            f"{_MAX_GENERAL_POSITION_DETERMINANT_WORK}); reduce point count "
            "or coordinate size",
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
    estimated_chars = triples * (
        _CIRCUMRADIUS_DIGIT_GROWTH * max_digits + _CIRCUMRADIUS_ENTRY_OVERHEAD
    )
    if estimated_chars > _MAX_PROFILE_OUTPUT_CHARS:
        raise _validation_error(
            "circumradius_profile_n_points_max_digits",
            f"circumradius profile for {n} points with {max_digits}-digit "
            f"coordinates can serialize up to {estimated_chars} characters "
            f"(worst-case rational growth), exceeding the "
            f"{_MAX_PROFILE_OUTPUT_CHARS}-character output budget; reduce "
            "point count or coordinate size",
        )


def _check_witness_sorted_distinct(
    indices: tuple[int, ...], n: int, expected: int, label: str
) -> None:
    if len(indices) != expected or len(set(indices)) != expected:
        raise _validation_error(
            "label_indices_expected_distinct_values",
            f"{label} indices must be {expected} distinct values",
        )
    if indices != tuple(sorted(indices)):
        raise _validation_error(
            "label_indices_sorted", f"{label} indices must be sorted"
        )
    if any(i >= n for i in indices):
        raise _validation_error("index_out_range", "index out of range")


def _validate_general_position_points(
    points: tuple[RationalPoint2D, ...], num_points: int
) -> None:
    keys = tuple((p.x.num, p.x.den, p.y.num, p.y.den) for p in points)
    if len(keys) != len(set(keys)):
        raise _validation_error(
            "point_set_coordinates_unique", "point-set coordinates must be unique"
        )
    if num_points != len(points):
        raise _validation_error(
            "num_points_len_points", "num_points must equal len(points)"
        )


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
        raise _validation_error(
            "has_collinear_triple_collinear_triples",
            "has_collinear_triple must match collinear_triples",
        )
    if has_concyclic != bool(concyclic):
        raise _validation_error(
            "has_concyclic_quadruple_concyclic_quadruples",
            "has_concyclic_quadruple must match concyclic_quadruples",
        )
    if tuple(sorted(collinear, key=lambda w: w.indices)) != collinear:
        raise _validation_error(
            "collinear_triples_sorted_lexicographically",
            "collinear_triples must be sorted lexicographically",
        )
    if len({w.indices for w in collinear}) != len(collinear):
        raise _validation_error(
            "collinear_triples_unique", "collinear_triples must be unique"
        )
    if tuple(sorted(concyclic, key=lambda w: w.indices)) != concyclic:
        raise _validation_error(
            "concyclic_quadruples_sorted_lexicographically",
            "concyclic_quadruples must be sorted lexicographically",
        )
    if len({w.indices for w in concyclic}) != len(concyclic):
        raise _validation_error(
            "concyclic_quadruples_unique", "concyclic_quadruples must be unique"
        )


def _validate_circumradius_entries_basic(
    entries: tuple[CircumradiusTripleEntry, ...], n: int
) -> set[tuple[int, int, int]]:
    seen: set[tuple[int, int, int]] = set()
    for e in entries:
        if e.indices in seen:
            raise _validation_error(
                "duplicate_triple_circumradius_profile",
                "duplicate triple in circumradius profile",
            )
        seen.add(e.indices)
        if any(i >= n for i in e.indices):
            raise _validation_error("index_out_range", "index out of range")
    if tuple(sorted(entries, key=lambda e: e.indices)) != entries:
        raise _validation_error(
            "entries_sorted_lexicographically",
            "entries must be sorted lexicographically",
        )
    expected = set(combinations(range(n), 3))
    if seen != expected:
        raise _validation_error(
            "entries_cover_c_n_triples", "entries must cover exactly C(n,3) triples"
        )
    return seen


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


class RationalLine2D(StrictModel):
    """Canonical exact line determined by two distinct planar points."""

    first: RationalPoint2D
    second: RationalPoint2D

    @model_validator(mode="after")
    def require_distinct_points(self) -> Self:
        if self.first == self.second:
            raise _validation_error(
                "a_line_requires_two_distinct_points",
                "a line requires two distinct points",
            )
        return self


class LineRequest(RationalLine2D):
    """Wire request carrying one canonical exact planar line."""


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
        raise _validation_error(
            "label_exceeds_inversion_admission_digits_digit",
            f"{label} exceeds the {INVERSION_ADMISSION_DIGITS}-digit "
            "circle-inversion admission bound",
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


class PointTripleRequest(StrictModel):
    first: RationalPoint2D
    second: RationalPoint2D
    third: RationalPoint2D


class CircumcircleRequest(PointTripleRequest):
    """Three distinct non-collinear points defining a circumcircle."""


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
            raise _validation_error(
                "point_set_coordinates_unique", "point-set coordinates must be unique"
            )
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
                raise _validation_error(
                    "a_disjoint_segment_result_carries_no",
                    "a disjoint segment result carries no intersection",
                )
            return self
        if self.status == "POINT":
            if (
                self.point is None
                or self.contact_kind is None
                or self.overlap is not None
            ):
                raise _validation_error(
                    "a_point_segment_intersection_requires_contact",
                    "a point segment intersection requires one contact classification",
                )
            return self
        if (
            self.point is not None
            or self.contact_kind is not None
            or self.overlap is None
        ):
            raise _validation_error(
                "an_overlap_result_carries_maximal_segment",
                "an overlap result carries only one maximal segment",
            )
        if _point_key(self.overlap.start) >= _point_key(self.overlap.end):
            raise _validation_error(
                "an_overlap_segment_requires_canonical_distinct",
                "an overlap segment requires canonical distinct endpoints",
            )
        return self


class PolygonIntersectionWitness(StrictModel):
    first_edge_index: StrictInt = Field(ge=0, le=127)
    second_edge_index: StrictInt = Field(ge=0, le=127)
    intersection: SegmentIntersectionResult

    @model_validator(mode="after")
    def require_ordered_intersecting_pair(self) -> Self:
        if self.first_edge_index >= self.second_edge_index:
            raise _validation_error(
                "polygon_witness_edge_indices_strictly_ordered",
                "polygon witness edge indices must be strictly ordered",
            )
        if self.intersection.status == "DISJOINT":
            raise _validation_error(
                "polygon_witness_edges_intersect",
                "polygon witness edges must intersect",
            )
        return self


class SimplePolygonDecisionResult(StrictModel):
    vertex_count: StrictInt = Field(ge=3, le=128)
    is_simple: bool
    checked_edge_pairs: StrictInt = Field(ge=0, le=8128)
    witness: PolygonIntersectionWitness | None = None

    @model_validator(mode="after")
    def bind_decision_to_witness(self) -> Self:
        if self.is_simple is (self.witness is not None):
            raise _validation_error(
                "a_non_simple_polygon_carries_witness",
                "exactly a non-simple polygon carries one witness",
            )
        total_pairs = self.vertex_count * (self.vertex_count - 1) // 2
        if self.is_simple and self.checked_edge_pairs != total_pairs:
            raise _validation_error(
                "a_simple_decision_exhaust_every_edge",
                "a simple decision must exhaust every edge pair",
            )
        if self.witness is not None:
            if self.witness.second_edge_index >= self.vertex_count:
                raise _validation_error(
                    "polygon_witness_edge_index_exceeds_ring",
                    "polygon witness edge index exceeds the ring",
                )
            expected_checked = (
                self.witness.first_edge_index
                * (2 * self.vertex_count - self.witness.first_edge_index - 1)
                // 2
                + self.witness.second_edge_index
                - self.witness.first_edge_index
            )
            if self.checked_edge_pairs != expected_checked:
                raise _validation_error(
                    "polygon_witness_checked_pair_prefix",
                    "polygon witness does not match the checked pair prefix",
                )
        return self


class SimplePolygonPointRequest(StrictModel):
    polygon: PolygonRequest
    point: RationalPoint2D


class PolygonPointClassificationResult(StrictModel):
    polygon_vertex_count: StrictInt = Field(ge=3, le=128)
    classification: Literal["INSIDE", "BOUNDARY", "OUTSIDE"]
    boundary_edge_index: StrictInt | None = Field(default=None, ge=0, le=127)

    @model_validator(mode="after")
    def bind_boundary_witness(self) -> Self:
        if (self.classification == "BOUNDARY") is (self.boundary_edge_index is None):
            raise _validation_error(
                "a_boundary_classification_carries_an_edge",
                "only a boundary classification carries an edge index",
            )
        if (
            self.boundary_edge_index is not None
            and self.boundary_edge_index >= self.polygon_vertex_count
        ):
            raise _validation_error(
                "boundary_edge_index_exceeds_polygon_ring",
                "boundary edge index exceeds the polygon ring",
            )
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
            raise _validation_error(
                "point_intersections_carry_point",
                "only POINT intersections carry one point",
            )
        return self


class GeometryConvexHullResult(StrictModel):
    points: tuple[RationalPoint2D, ...] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def require_canonical_strict_convex_boundary(self) -> Self:
        keys = tuple(_point_key(point) for point in self.points)
        if len(keys) != len(set(keys)):
            raise _validation_error(
                "convex_hull_vertices_unique", "convex-hull vertices must be unique"
            )
        if len(keys) <= 2:
            if keys != tuple(sorted(keys)):
                raise _validation_error(
                    "a_zero_dimensional_hull_canonical",
                    "a zero-dimensional hull must be canonical",
                )
            return self
        if keys[0] != min(keys):
            raise _validation_error(
                "a_polygon_hull_begin_least_vertex",
                "a polygon hull must begin at its least vertex",
            )
        turns = tuple(
            _cross(
                _subtract(keys[(index + 1) % len(keys)], keys[index]),
                _subtract(keys[(index + 2) % len(keys)], keys[index]),
            )
            for index in range(len(keys))
        )
        if any(turn <= 0 for turn in turns):
            raise _validation_error(
                "a_polygon_hull_strictly_counterclockwise",
                "a polygon hull must be strictly counterclockwise",
            )
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
            raise _validation_error(
                "weighted_diagonal_endpoints_strictly_increasing",
                "weighted diagonal endpoints must be strictly increasing",
            )
        if self.weight.as_fraction() < 0:
            raise _validation_error(
                "weighted_diagonal_cost_nonnegative",
                "weighted diagonal cost must be nonnegative",
            )
        return self


def _triangulation_subproblem_costs(
    count: int,
    weight: Callable[[int, int], Fraction],
) -> tuple[dict[tuple[int, int], Fraction], dict[tuple[int, int], int]]:
    """Derive every convex-subpolygon minimum and split for one polygon size.

    The recurrence charges ``weight(start, end)`` at state ``(start, end)``
    itself: every non-hull diagonal is the boundary of exactly one non-root
    subpolygon, so each selected diagonal is counted exactly once and the
    root optimum equals the minimum non-hull-diagonal weight sum. The admitted
    split table is returned for execution so the derivation runs only once.
    """

    optimum: dict[tuple[int, int], Fraction] = {
        (index, index + 1): Fraction() for index in range(count - 1)
    }
    split: dict[tuple[int, int], int] = {}
    for span in range(2, count):
        for start in range(count - span):
            end = start + span
            candidates = [
                (
                    optimum[start, pivot] + optimum[pivot, end] + weight(start, end),
                    pivot,
                )
                for pivot in range(start + 1, end)
            ]
            value, pivot = min(candidates)
            optimum[start, end] = value
            split[start, end] = pivot
    return optimum, split


def _reconstruct_split_triangulation(
    count: int,
    split: dict[tuple[int, int], int],
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int, int], ...]]:
    """Replay one stored split table into its deterministic triangulation.

    Both execution and admission reconstruct the selected triangles and
    non-hull diagonals through this single walk, so the echoed diagonal set
    can never diverge between computing a result and bounding its payload.
    """

    triangles: list[tuple[int, int, int]] = []
    diagonals: set[tuple[int, int]] = set()

    def walk(start: int, end: int) -> None:
        if end == start + 1:
            return
        pivot = split[start, end]
        triangles.append((start, pivot, end))
        for pair in ((start, pivot), (pivot, end)):
            ordered = pair if pair[0] < pair[1] else (pair[1], pair[0])
            if pair[1] != pair[0] + 1 and ordered != (0, count - 1):
                diagonals.add(ordered)
        walk(start, pivot)
        walk(pivot, end)

    walk(0, count - 1)
    return tuple(sorted(diagonals)), tuple(sorted(triangles))


# Serialized-output budget for the weighted triangulation result, kept below
# the 10 MiB canonical transport envelope to leave room for request and
# JSON envelope overhead.
MAX_TRIANGULATION_OUTPUT_CHARS = 8_000_000
_TRIANGULATION_ENTRY_OVERHEAD_CHARS = 96
_TRIANGULATION_TRIANGLE_ENTRY_CHARS = 32
_TRIANGULATION_RESULT_SLACK_CHARS = 512


def _bounded_split_table_rationals(
    count: int,
    diagonal_weights: tuple[WeightedPolygonDiagonal, ...],
) -> tuple[dict[tuple[int, int], Fraction], dict[tuple[int, int], int]]:
    """Compute and admit every derived split-table rational.

    Admission computes the bounded recurrence exactly and checks each retained
    ledger optimum - the values the result model serializes - against the
    shared canonical rational cap. Reduced sums of compatible weights need
    not grow: shared denominator factors cancel, so multiplying component
    heights would reject requests whose actual ledger values remain
    representable. A request is rejected only when a state the result must
    serialize genuinely exceeds the cap.

    Per-entry caps alone cannot bound the aggregate payload: every retained
    optimum may sit just under the cap while their combined serialization
    outgrows the transport envelope. The same exact computation therefore also
    sums each retained optimum's own serialized size - numerator plus
    denominator digits beside fixed punctuation - and charges every echoed
    weight at its own height. Admission reconstructs the deterministic
    selected-diagonal set from the shared split table, adds the duplicated
    top-level optimum, then fixed triangle and header slack. Execution consumes
    that admitted split table directly, so this estimate soundly bounds the
    complete serialized result without
    charging unselected or small entries at the largest component height,
    and a genuinely oversized aggregate is rejected at request validation
    instead of failing canonical output validation after computation.
    """

    weights = {
        (item.first, item.second): item.weight.as_fraction()
        for item in diagonal_weights
    }

    def diagonal_cost(first: int, second: int) -> Fraction:
        if second == first + 1 or (first, second) == (0, count - 1):
            return Fraction()
        return weights[first, second]

    optimum, split = _triangulation_subproblem_costs(count, diagonal_cost)
    ledger_chars = 0
    for span in range(2, count):
        for start in range(count - span):
            end = start + span
            value = optimum[start, end]
            digits = max(
                len(format_canonical_integer(value.numerator)),
                len(format_canonical_integer(value.denominator)),
            )
            if digits > MAX_CANONICAL_RATIONAL_DIGITS:
                raise _validation_error(
                    "split_table_state_start_end_carries",
                    f"split-table state ({start}, {end}) carries {digits}-digit "
                    f"rational components, exceeding the canonical "
                    f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit rational limit",
                )
            ledger_chars += 2 * digits + _TRIANGULATION_ENTRY_OVERHEAD_CHARS
    selected_diagonals, _ = _reconstruct_split_triangulation(count, split)
    selected_weight_chars = sum(
        2
        * max(
            len(format_canonical_integer(weights[pair].numerator)),
            len(format_canonical_integer(weights[pair].denominator)),
        )
        + _TRIANGULATION_ENTRY_OVERHEAD_CHARS
        for pair in selected_diagonals
    )
    root = optimum[0, count - 1]
    estimated_chars = (
        ledger_chars
        + selected_weight_chars
        + 2
        * max(
            len(format_canonical_integer(root.numerator)),
            len(format_canonical_integer(root.denominator)),
        )
        + _TRIANGULATION_ENTRY_OVERHEAD_CHARS
        + (count - 2) * _TRIANGULATION_TRIANGLE_ENTRY_CHARS
        + _TRIANGULATION_RESULT_SLACK_CHARS
    )
    if estimated_chars > MAX_TRIANGULATION_OUTPUT_CHARS:
        raise _validation_error(
            "weighted_triangulation_result_serialize_up_f",
            f"weighted triangulation result can serialize up to "
            f"{estimated_chars} characters, exceeding the "
            f"{MAX_TRIANGULATION_OUTPUT_CHARS}-character output bound",
        )
    return optimum, split


class ConvexPolygonTriangulationRequest(StrictModel):
    """One strict CCW convex rational polygon and its complete diagonal weights."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Minimum-weight triangulation of one strict CCW convex "
                "rational polygon under a complete nonnegative exact rational "
                "weight per non-hull diagonal. Each split-table state sums "
                "one feasible subpolygon triangulation - at most vertex_count "
                "- 3 pairwise noncrossing selected weights - so admission "
                "computes the bounded recurrence exactly and rejects requests "
                "whose derived split-table rationals exceed the canonical "
                "32,768-digit rational limit or whose complete serialized "
                "result exceeds the "
                f"{MAX_TRIANGULATION_OUTPUT_CHARS}-character output bound."
            ),
        },
    )

    polygon: PolygonRequest
    diagonal_weights: tuple[WeightedPolygonDiagonal, ...] = Field(
        min_length=1,
        max_length=464,
        description=(
            "Exactly one nonnegative exact rational weight per non-hull "
            "diagonal in lexicographic pair order; the exact derived "
            "split-table rationals must stay inside the canonical "
            "32,768-digit limit and the aggregate serialized result must "
            f"stay inside the {MAX_TRIANGULATION_OUTPUT_CHARS}-character "
            "output bound."
        ),
    )
    objective: Literal["NON_HULL_DIAGONAL_WEIGHT_SUM"] = "NON_HULL_DIAGONAL_WEIGHT_SUM"


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


# Euclidean convex-polygon triangulation
# ---------------------------------------------------------------------------

MAX_EUCLIDEAN_TRIANGULATION_OUTPUT_CHARS = 7_000_000
_MIN_EUCLIDEAN_SPLIT_TERM_CHARS = 2 * (4 * 1 + 1) + 128
_EUCLIDEAN_TRIANGLE_ENTRY_CHARS = 32
_EUCLIDEAN_RESULT_ENVELOPE_SLACK_CHARS = 512


def _span_term_occurrences(count: int) -> int:
    """Total retained expression term occurrences charged by a ``count``-vertex source.

    A non-root span-``s`` state carries at most ``s - 1`` terms - its
    subpolygon triangulates with ``s - 2`` diagonals plus one charged
    boundary - and spans ``1..count - 2`` each occur ``count - s`` times.
    The root span ``count - 1`` carries exactly ``count - 3`` terms, since
    its boundary is an uncharged hull edge, and it occurs once in the
    retained table plus once more as the duplicated top-level optimum, so
    the serialized envelope counts both copies rather than charging every
    state the root's term count.
    """

    return sum((count - span) * (span - 1) for span in range(1, count - 1)) + 2 * (
        count - 3
    )


def _echoed_result_envelope_chars(polygon: PolygonRequest) -> int:
    """Canonical characters charged for the echoed source and fixed envelope.

    Every certified or unresolved result repeats the complete source ring
    beside literal header fields, so admission measures that echo directly;
    the difference-based estimates below are translation-invariant and
    cannot see absolute positions. The returned charge adds slack for the
    status-specific fields the shared template cannot carry: an unresolved
    comparison swaps a longer status string and carries its comparison
    skeleton, while a certified result lists per-entry ledgers charged
    separately by the caller.
    """

    return (
        len(
            encode_strict_json(
                {
                    "comparison_basis": "ARB_OUTWARD_ROUNDED_INTERVAL",
                    "comparison_precision_bits": (
                        EUCLIDEAN_TRIANGULATION_COMPARISON_PRECISION_BITS
                    ),
                    "objective": "NON_HULL_EUCLIDEAN_LENGTH_SUM",
                    "polygon": polygon.model_dump(mode="json"),
                    "status": "CERTIFIED_OPTIMUM",
                    "vertex_count": 0,
                }
            )
        )
        + _EUCLIDEAN_RESULT_ENVELOPE_SLACK_CHARS
    )


def _euclidean_envelope_vertex_ceiling() -> int:
    """Largest vertex count that admission could ever accept.

    The split-table estimate multiplies ``_span_term_occurrences`` by
    ``term_chars``, which grows with the pairwise-difference digit count
    derived from the source and never drops below one digit, so that
    product evaluated at the one-digit floor is a necessary condition for
    every admitted source. The returned ceiling restates this closed-form
    consequence of ``MAX_EUCLIDEAN_TRIANGULATION_OUTPUT_CHARS`` for schemas
    that need static bounds; the echoed-source and metadata charges in the
    full envelope are strictly positive, so it can never reject a source
    whose estimate fits the budget and the derived envelope alone decides
    admission.
    """

    count = 4
    while True:
        candidate = count + 1
        if (
            _span_term_occurrences(candidate) * _MIN_EUCLIDEAN_SPLIT_TERM_CHARS
            > MAX_EUCLIDEAN_TRIANGULATION_OUTPUT_CHARS
        ):
            return count
        count = candidate


MAX_EUCLIDEAN_TRIANGULATION_VERTICES = _euclidean_envelope_vertex_ceiling()
EUCLIDEAN_TRIANGULATION_COMPARISON_PRECISION_BITS = 128


def _require_euclidean_triangulation_envelope(
    polygon: PolygonRequest,
) -> tuple[tuple[Fraction, Fraction], ...]:
    """Validate the bounded exact source and return its rational coordinates.

    Positive consecutive turns establish strict convexity only for a simple
    ring, so global simplicity is checked before the recurrence runs.  Raw
    positions control no kernel quantity: every turn, diagonal squared
    length, and serialized expression depends only on pairwise coordinate
    differences, so admission derives ``d``, the maximum decimal digits in
    any pairwise-difference component, from the source itself.  A squared
    length then has at most ``4d + 1`` digits in each component (each
    product doubles its side and the final sum adds one digit), and each
    split-table expression term is charged twice that plus fixed punctuation
    slack.  A non-root span-``s`` table state carries at most ``s - 1``
    terms, the root span carries ``count - 3``, and the top-level optimum
    duplicates the root expression, so summing those span-specific term
    counts over all retained expression serializations gives the conservative
    serialized-expression estimate below, bounding every retained exact sum
    before Arb is invoked; each raw input coordinate stays inside the shared
    canonical rational cap.

    The bound covers the complete deterministic result envelope, not just
    the split table.  Every result echoes the full source polygon beside
    fixed literal header fields, and a pure translation inflates that echo
    without touching any difference, so admission measures the canonical
    encoding of the echoed source directly.  Each certified diagonal's
    squared length stays within one term-scale serialization plus its entry
    skeleton, each triangle is charged a fixed skeleton cost, two further
    root-scale terms cover the largest possible unresolved comparison pair,
    and named slack absorbs the remaining status-specific fields.
    Every candidate diagonal's exact squared length is also checked against
    the canonical rational cap, because the aggregate serialized estimate
    alone admits sources whose derived values cannot be represented at all.
    The static ``MAX_EUCLIDEAN_TRIANGULATION_VERTICES`` bound is the
    closed-form consequence of this estimate at its one-digit difference
    floor, never an independent admission gate.
    """

    points = tuple(_point_key(point) for point in polygon.points)
    count = len(points)
    if not 4 <= count <= MAX_EUCLIDEAN_TRIANGULATION_VERTICES:
        raise _validation_error(
            "euclidean_triangulation_supports_f_max_euclidean",
            "Euclidean triangulation supports 4 to "
            f"{MAX_EUCLIDEAN_TRIANGULATION_VERTICES} vertices",
        )
    difference_digits = 1
    for first in range(count):
        for second in range(first + 1, count):
            for left, right in zip(points[first], points[second], strict=True):
                delta = right - left
                difference_digits = max(
                    difference_digits,
                    len(format_canonical_integer(abs(delta.numerator))),
                    len(format_canonical_integer(delta.denominator)),
                )
    turns = tuple(
        _cross(
            _subtract(points[(index + 1) % count], points[index]),
            _subtract(points[(index + 2) % count], points[index]),
        )
        for index in range(count)
    )
    if any(turn <= 0 for turn in turns):
        raise _validation_error(
            "euclidean_triangulation_requires_strict_ccw_convexity",
            "Euclidean triangulation requires strict CCW convexity",
        )
    if not _is_simple_ring(polygon.points):
        raise _validation_error(
            "euclidean_triangulation_requires_a_simple_ring",
            "Euclidean triangulation requires a simple ring",
        )
    for first in range(count - 2):
        for second in range(first + 2, count):
            if (first, second) == (0, count - 1):
                continue
            squared = _euclidean_squared_length(points, first, second)
            numerator = format_canonical_integer(squared.numerator)
            denominator = format_canonical_integer(squared.denominator)
            if (
                len(numerator) > MAX_CANONICAL_RATIONAL_DIGITS
                or len(denominator) > MAX_CANONICAL_RATIONAL_DIGITS
            ):
                digits = max(len(numerator), len(denominator))
                raise _validation_error(
                    "euclidean_triangulation_diagonal_squared_length_carries",
                    "Euclidean triangulation diagonal squared length carries "
                    f"{digits} digits, exceeding the canonical "
                    f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit rational limit",
                )
    term_chars = 2 * (4 * difference_digits + 1) + 128
    estimated_chars = (
        _span_term_occurrences(count) * term_chars
        + _echoed_result_envelope_chars(polygon)
        + (count - 3) * term_chars
        + (count - 2) * _EUCLIDEAN_TRIANGLE_ENTRY_CHARS
        + 2 * term_chars
    )
    if estimated_chars > MAX_EUCLIDEAN_TRIANGULATION_OUTPUT_CHARS:
        raise _validation_error(
            "euclidean_triangulation_result_serialize_up_f",
            "Euclidean triangulation result can serialize up to "
            f"{estimated_chars} characters, exceeding the "
            f"{MAX_EUCLIDEAN_TRIANGULATION_OUTPUT_CHARS}-character output bound",
        )
    return points


def _euclidean_squared_length(
    points: tuple[tuple[Fraction, Fraction], ...], first: int, second: int
) -> Fraction:
    dx = points[second][0] - points[first][0]
    dy = points[second][1] - points[first][1]
    return dx * dx + dy * dy


def _compare_euclidean_root_sums(
    left: tuple[Fraction, ...], right: tuple[Fraction, ...]
) -> int | None:
    """Return a rigorous root-sum order, or ``None`` for an overlapping ball."""

    if left == right:
        return 0
    from flint import arb, ctx, fmpq

    with ctx.workprec(EUCLIDEAN_TRIANGULATION_COMPARISON_PRECISION_BITS):
        difference = arb(0)
        for value in left:
            difference += arb(fmpq(value.numerator, value.denominator)).sqrt()
        for value in right:
            difference -= arb(fmpq(value.numerator, value.denominator)).sqrt()
        if difference.contains(0):
            return None
        return 1 if difference > 0 else -1


class EuclideanTriangulationPolygonRequest(PolygonRequest):
    """Strict CCW convex simple rational ring admitted by Euclidean triangulation."""

    points: tuple[RationalPoint2D, ...] = Field(
        min_length=4,
        max_length=MAX_EUCLIDEAN_TRIANGULATION_VERTICES,
        description=(
            "Ring vertices listed counterclockwise; the closed ring must be "
            "simple and strictly convex. Admission bounds the complete "
            "serialized result - the split table, the echoed source ring, "
            "and fixed result metadata - from the exact source, so absolute "
            "positions consume the published output budget even though the "
            "mathematical work depends only on pairwise differences."
        ),
        json_schema_extra={
            "maximum_serialized_result_characters": (
                MAX_EUCLIDEAN_TRIANGULATION_OUTPUT_CHARS
            ),
        },
    )


class EuclideanConvexPolygonTriangulationRequest(StrictModel):
    """One bounded strict convex rational polygon with Euclidean diagonal cost."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Minimum Euclidean triangulation of one strict CCW convex "
                "simple rational polygon. Admitted requests contain 4 to "
                f"{MAX_EUCLIDEAN_TRIANGULATION_VERTICES} vertices whose "
                "complete serialized result, including the echoed source "
                "ring, stays inside the published output bound; strict "
                "counterclockwise convexity and ring simplicity are enforced "
                "by the request validator after parsing."
            ),
        },
    )

    polygon: EuclideanTriangulationPolygonRequest
    objective: Literal["NON_HULL_EUCLIDEAN_LENGTH_SUM"] = (
        "NON_HULL_EUCLIDEAN_LENGTH_SUM"
    )


class EuclideanDiagonal(StrictModel):
    """One selected non-hull diagonal and its exact squared Euclidean length."""

    first: StrictInt = Field(ge=0, le=MAX_EUCLIDEAN_TRIANGULATION_VERTICES - 1)
    second: StrictInt = Field(ge=0, le=MAX_EUCLIDEAN_TRIANGULATION_VERTICES - 1)
    squared_length: CanonicalRational

    @model_validator(mode="after")
    def require_positive_canonical_pair(self) -> Self:
        if self.first >= self.second:
            raise _validation_error(
                "euclidean_diagonal_endpoints_strictly_increasing",
                "Euclidean diagonal endpoints must be strictly increasing",
            )
        if self.squared_length.as_fraction() <= 0:
            raise _validation_error(
                "euclidean_diagonal_squared_length_positive",
                "Euclidean diagonal squared length must be positive",
            )
        return self


class EuclideanLengthExpression(StrictModel):
    """An exact ordered sum of positive square roots of rational squared lengths.

    Its source-bound use in a triangulation result is exactly
    ``sum(sqrt(term) for term in squared_lengths)``. The ordered presentation
    retains the selected diagonal lengths without a decimal approximation or
    an unbounded expanded number-field representation.
    """

    squared_lengths: tuple[CanonicalRational, ...] = Field(
        default=(), max_length=MAX_EUCLIDEAN_TRIANGULATION_VERTICES - 3
    )

    @model_validator(mode="after")
    def require_sorted_positive_terms(self) -> Self:
        values = tuple(term.as_fraction() for term in self.squared_lengths)
        if any(value <= 0 for value in values):
            raise _validation_error(
                "euclidean_length_terms_positive",
                "Euclidean length terms must be positive",
            )
        if values != tuple(sorted(values)):
            raise _validation_error(
                "euclidean_length_terms_ordered_canonically",
                "Euclidean length terms must be ordered canonically",
            )
        return self


class EuclideanTriangulationSplitEntry(StrictModel):
    start: StrictInt = Field(ge=0, le=MAX_EUCLIDEAN_TRIANGULATION_VERTICES - 1)
    end: StrictInt = Field(ge=0, le=MAX_EUCLIDEAN_TRIANGULATION_VERTICES - 1)
    split: StrictInt = Field(ge=0, le=MAX_EUCLIDEAN_TRIANGULATION_VERTICES - 1)
    optimum: EuclideanLengthExpression

    @model_validator(mode="after")
    def require_proper_subproblem(self) -> Self:
        if not self.start < self.split < self.end:
            raise _validation_error(
                "triangulation_split_lie_strictly_inside_span",
                "triangulation split must lie strictly inside its span",
            )
        return self


class EuclideanComparisonUnresolved(StrictModel):
    """The first finite DP comparison whose rigorous Arb enclosure overlaps zero."""

    start: StrictInt = Field(ge=0, le=MAX_EUCLIDEAN_TRIANGULATION_VERTICES - 1)
    end: StrictInt = Field(ge=0, le=MAX_EUCLIDEAN_TRIANGULATION_VERTICES - 1)
    left_split: StrictInt = Field(ge=0, le=MAX_EUCLIDEAN_TRIANGULATION_VERTICES - 1)
    right_split: StrictInt = Field(ge=0, le=MAX_EUCLIDEAN_TRIANGULATION_VERTICES - 1)
    left: EuclideanLengthExpression
    right: EuclideanLengthExpression
    precision_bits: StrictInt = Field(
        ge=EUCLIDEAN_TRIANGULATION_COMPARISON_PRECISION_BITS,
        le=EUCLIDEAN_TRIANGULATION_COMPARISON_PRECISION_BITS,
    )


class EuclideanConvexPolygonTriangulationResult(StrictModel):
    """A certified optimum, or an explicit unresolved exact comparison."""

    status: Literal["CERTIFIED_OPTIMUM", "COMPARISON_UNRESOLVED"]
    polygon: PolygonRequest
    vertex_count: StrictInt = Field(ge=4, le=MAX_EUCLIDEAN_TRIANGULATION_VERTICES)
    objective: Literal["NON_HULL_EUCLIDEAN_LENGTH_SUM"] = (
        "NON_HULL_EUCLIDEAN_LENGTH_SUM"
    )
    comparison_basis: Literal["ARB_OUTWARD_ROUNDED_INTERVAL"] = (
        "ARB_OUTWARD_ROUNDED_INTERVAL"
    )
    comparison_precision_bits: StrictInt = Field(
        ge=EUCLIDEAN_TRIANGULATION_COMPARISON_PRECISION_BITS,
        le=EUCLIDEAN_TRIANGULATION_COMPARISON_PRECISION_BITS,
    )
    diagonals: tuple[EuclideanDiagonal, ...] = Field(
        default=(), max_length=MAX_EUCLIDEAN_TRIANGULATION_VERTICES - 3
    )
    triangles: tuple[PolygonTriangle, ...] = Field(
        default=(), max_length=MAX_EUCLIDEAN_TRIANGULATION_VERTICES - 2
    )
    split_table: tuple[EuclideanTriangulationSplitEntry, ...] = Field(
        default=(),
        max_length=(MAX_EUCLIDEAN_TRIANGULATION_VERTICES - 1)
        * (MAX_EUCLIDEAN_TRIANGULATION_VERTICES - 2)
        // 2,
    )
    optimum: EuclideanLengthExpression | None = None
    unresolved_comparison: EuclideanComparisonUnresolved | None = None

    @model_validator(mode="after")
    def bind_status_fields(self) -> Self:
        if self.vertex_count != len(self.polygon.points):
            raise _validation_error(
                "vertex_count_source_polygon_vertex_count",
                "vertex_count must equal the source polygon vertex count",
            )
        certified = self.status == "CERTIFIED_OPTIMUM"
        if certified != (self.optimum is not None):
            raise _validation_error(
                "a_certified_optimum_carries_an_exact",
                "only a certified optimum carries an exact cost expression",
            )
        if certified != (self.unresolved_comparison is None):
            raise _validation_error(
                "an_unresolved_result_carries_an_ambiguous",
                "only an unresolved result carries an ambiguous comparison",
            )
        if not certified and (self.diagonals or self.triangles or self.split_table):
            raise _validation_error(
                "an_unresolved_comparison_claim_a_triangulation",
                "an unresolved comparison must not claim a triangulation",
            )
        if certified:
            assert self.optimum is not None
            if len(self.diagonals) != self.vertex_count - 3:
                raise _validation_error(
                    "a_triangulation_contain_vertex_count_diagonals",
                    "a triangulation must contain vertex_count - 3 diagonals",
                )
            if len(self.triangles) != self.vertex_count - 2:
                raise _validation_error(
                    "a_triangulation_contain_vertex_count_triangles",
                    "a triangulation must contain vertex_count - 2 triangles",
                )
            expected_states = (self.vertex_count - 1) * (self.vertex_count - 2) // 2
            if len(self.split_table) != expected_states:
                raise _validation_error(
                    "split_table_contain_every_nontrivial_dp",
                    "split table must contain every nontrivial DP state",
                )
            if tuple(
                sorted(edge.squared_length.as_fraction() for edge in self.diagonals)
            ) != tuple(term.as_fraction() for term in self.optimum.squared_lengths):
                raise _validation_error(
                    "optimum_expression_list_selected_diagonal_lengths",
                    "optimum expression must list the selected diagonal lengths",
                )
        else:
            assert self.unresolved_comparison is not None
            comparison = self.unresolved_comparison
            if (
                comparison.end >= self.vertex_count
                or comparison.end - comparison.start < 2
            ):
                raise _validation_error(
                    "unresolved_comparison_name_a_nontrivial_dp",
                    "unresolved comparison must name a nontrivial DP subproblem span",
                )
            if not (
                comparison.start
                < comparison.right_split
                < comparison.left_split
                < comparison.end
            ):
                raise _validation_error(
                    "unresolved_comparison_splits_lie_strictly_inside",
                    "unresolved comparison splits must lie strictly inside its span "
                    "with the incumbent split first",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: EuclideanConvexPolygonTriangulationRequest,
        *,
        status: Literal["CERTIFIED_OPTIMUM", "COMPARISON_UNRESOLVED"],
        diagonals: tuple[EuclideanDiagonal, ...] = (),
        triangles: tuple[PolygonTriangle, ...] = (),
        split_table: tuple[EuclideanTriangulationSplitEntry, ...] = (),
        optimum: EuclideanLengthExpression | None = None,
        unresolved_comparison: EuclideanComparisonUnresolved | None = None,
    ) -> Self:
        """Build a result after the admitted triangulation kernel established it."""

        return cls.model_construct(
            polygon=request.polygon,
            vertex_count=len(request.polygon.points),
            comparison_precision_bits=EUCLIDEAN_TRIANGULATION_COMPARISON_PRECISION_BITS,
            status=status,
            diagonals=diagonals,
            triangles=triangles,
            split_table=split_table,
            optimum=optimum,
            unresolved_comparison=unresolved_comparison,
        )


# ---------------------------------------------------------------------------
# Configuration-level operations
# ---------------------------------------------------------------------------


class GeneralPositionRequest(StrictModel):
    """Search a bounded point configuration for collinear triples and concyclic quadruples.

    Admission couples the combinatorial point count to exact-coordinate
    complexity rather than imposing an independent convenience cap.
    """

    points: tuple[RationalPoint2D, ...] = Field(
        min_length=3,
        max_length=MAX_CONFIGURATION_POINTS,
        description=(
            f"Bounded point configuration with 3..{MAX_CONFIGURATION_POINTS} points; "
            "each rational coordinate (numerator/denominator) is bounded to at most "
            f"{MAX_COORDINATE_DIGITS} digits (operation-specific limit, stricter than "
            f"CanonicalRational's {MAX_CANONICAL_RATIONAL_DIGITS}-digit global limit); "
            f"additionally C(n,4)*max_digits^2 <= "
            f"{_MAX_GENERAL_POSITION_DETERMINANT_WORK} bounds the exhaustive "
            "determinant work"
        ),
    )

    @model_validator(mode="after")
    def require_unique(self) -> Self:
        keys = tuple((p.x.num, p.x.den, p.y.num, p.y.den) for p in self.points)
        if len(keys) != len(set(keys)):
            raise _validation_error(
                "point_set_coordinates_unique", "point-set coordinates must be unique"
            )
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
        _validate_general_position_points(self.points, self.num_points)
        _validate_general_position_witnesses(
            self.collinear_triples,
            self.concyclic_quadruples,
            self.has_collinear_triple,
            self.has_concyclic_quadruple,
            len(self.points),
        )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        points: tuple[RationalPoint2D, ...],
        collinear_triples: tuple[CollinearTripleWitness, ...],
        concyclic_quadruples: tuple[ConcyclicQuadrupleWitness, ...],
    ) -> Self:
        return cls.model_construct(
            points=points,
            num_points=len(points),
            has_collinear_triple=bool(collinear_triples),
            has_concyclic_quadruple=bool(concyclic_quadruples),
            collinear_triples=collinear_triples,
            concyclic_quadruples=concyclic_quadruples,
        )


class CircumradiusProfileRequest(StrictModel):
    """Compute circumradius data for every unordered triple in a point configuration.

    The serialized profile is bounded before execution using the exact-rational
    growth estimate owned by this operation.
    """

    points: tuple[RationalPoint2D, ...] = Field(
        min_length=3,
        max_length=MAX_CONFIGURATION_POINTS,
        description=(
            f"Bounded point configuration with 3..{MAX_CONFIGURATION_POINTS} points; "
            f"each rational coordinate is bounded to at most {MAX_COORDINATE_DIGITS} "
            "digits (operation-specific, stricter than CanonicalRational's "
            f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit limit); additionally "
            f"C(n,3)*({_CIRCUMRADIUS_DIGIT_GROWTH}*max_digits+"
            f"{_CIRCUMRADIUS_ENTRY_OVERHEAD}) characters of worst-case profile size "
            f"must stay within the {_MAX_PROFILE_OUTPUT_CHARS}-character output budget"
        ),
    )

    @model_validator(mode="after")
    def require_unique(self) -> Self:
        keys = tuple((p.x.num, p.x.den, p.y.num, p.y.den) for p in self.points)
        if len(keys) != len(set(keys)):
            raise _validation_error(
                "point_set_coordinates_unique", "point-set coordinates must be unique"
            )
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
            raise _validation_error(
                "triple_indices_distinct_values",
                "triple indices must be 3 distinct values",
            )
        if idx != tuple(sorted(idx)):
            raise _validation_error(
                "triple_indices_sorted", "triple indices must be sorted"
            )
        if self.is_degenerate and self.radius_squared is not None:
            raise _validation_error(
                "degenerate_triple_a_radius", "degenerate triple cannot have a radius"
            )
        if not self.is_degenerate and self.radius_squared is None:
            raise _validation_error(
                "non_degenerate_triple_a_radius",
                "non-degenerate triple must have a radius",
            )
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
        keys = tuple((p.x.num, p.x.den, p.y.num, p.y.den) for p in self.points)
        if len(keys) != len(set(keys)):
            raise _validation_error(
                "point_set_coordinates_unique", "point-set coordinates must be unique"
            )
        n = len(self.points)
        if self.num_points != n:
            raise _validation_error(
                "num_points_len_points", "num_points must equal len(points)"
            )
        if len(self.entries) != n * (n - 1) * (n - 2) // 6:
            raise _validation_error(
                "entries_cover_c_n_triples", "entries must cover exactly C(n,3) triples"
            )
        _validate_circumradius_entries_basic(self.entries, n)
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        points: tuple[RationalPoint2D, ...],
        entries: tuple[CircumradiusTripleEntry, ...],
    ) -> Self:
        return cls.model_construct(
            points=points, num_points=len(points), entries=entries
        )
