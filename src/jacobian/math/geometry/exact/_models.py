"""Typed wire contracts for exact geometry point-configuration operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Self

from pydantic import ConfigDict, Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import encode_strict_json, format_canonical_integer
from jacobian.math.geometry.exact._line_arithmetic import (
    canonical_line_coefficients,
    squared_point_line_distance,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable error owned by the geometry contracts."""

    return PydanticCustomError(f"geometry.{reason}", message)


MAX_POINTS = 64
MAX_DIMENSION = 20
MAX_PAIRS = MAX_POINTS * (MAX_POINTS - 1) // 2
"""Maximum distinct source pairs spanned by a bounded configuration: C(64, 2)."""
COORDINATE_DIGITS = 256
"""Per-coordinate digit bound for pinned line-distance profile so squared distances stay representable."""


def _require_bounded_point_configuration(
    configuration: PointConfiguration,
    anchor: tuple[CanonicalRational, ...] | None = None,
) -> None:
    """Enforce the 256-digit coordinate bound for pinned operations.

    The shared ``LabelledRationalPoint`` remains at the canonical 32,768-digit
    limit so ``geometry.points.distance_profile`` and ``distance_graph`` stay
    usable far beyond the pinned-line result budget. This helper narrows only
    the pinned-line admission.
    """

    from jacobian._exact import require_bounded_rational

    for pt in configuration.points:
        for coord in pt.coordinates:
            require_bounded_rational(
                coord, max_digits=COORDINATE_DIGITS, label="point coordinate"
            )
    if anchor is not None:
        for coord in anchor:
            require_bounded_rational(
                coord, max_digits=COORDINATE_DIGITS, label="anchor coordinate"
            )


class LabelledRationalPoint(StrictModel):
    """A labelled rational point in bounded dimension."""

    label: str = Field(min_length=1, max_length=64)
    coordinates: tuple[CanonicalRational, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_dimension(self) -> Self:
        if len(self.coordinates) > MAX_DIMENSION:
            raise _validation_error(
                "dimension_exceeds_bound", "dimension exceeds bound"
            )
        return self


class PointConfiguration(StrictModel):
    """A finite set of labelled rational points in a fixed dimension."""

    points: tuple[LabelledRationalPoint, ...] = Field(
        min_length=2,
        max_length=MAX_POINTS,
    )

    @model_validator(mode="after")
    def require_uniform_dimension(self) -> Self:
        if not self.points:
            return self
        dim = len(self.points[0].coordinates)
        for p in self.points[1:]:
            if len(p.coordinates) != dim:
                raise _validation_error(
                    "points_same_dimension", "all points must have the same dimension"
                )
        labels = [p.label for p in self.points]
        if len(labels) != len(set(labels)):
            raise _validation_error(
                "point_labels_unique", "point labels must be unique"
            )
        return self


class DistanceProfileRequest(StrictModel):
    """Compute exact pairwise squared distances."""

    configuration: PointConfiguration


class DistanceMultiplicityEntry(StrictModel):
    """One squared distance and how many pairs have it."""

    squared_distance: CanonicalRational
    pair_count: int = Field(gt=0)


class DistanceProfileResult(StrictModel):
    """Complete distance multiplicity profile of a point configuration."""

    dimension: int = Field(ge=1)
    point_count: int = Field(ge=2)
    entries: tuple[DistanceMultiplicityEntry, ...]


class DistanceGraphRequest(StrictModel):
    """Build the graph induced by a selected squared distance."""

    configuration: PointConfiguration
    target_squared_distance: CanonicalRational = Field(
        description="Nonnegative squared Euclidean distance to select.",
    )

    @model_validator(mode="after")
    def require_nonnegative_target(self) -> Self:
        if self.target_squared_distance.as_fraction() < 0:
            raise _validation_error(
                "squared_distance_target_nonnegative",
                "squared distance target must be nonnegative",
            )
        return self


__all__ = [
    "DistanceGraphRequest",
    "DistanceMultiplicityEntry",
    "DistanceProfileRequest",
    "DistanceProfileResult",
    "LabelledRationalPoint",
    "PinnedLineDistanceRequest",
    "PinnedLineDistanceResult",
    "PinnedLineEntry",
    "PointConfiguration",
]


# ---------------------------------------------------------------------------
# Pinned line-distance profile
# ---------------------------------------------------------------------------

MAX_PINNED_PROFILE_RESULT_BYTES = 10 * 1024 * 1024
"""Aggregate canonical-output budget for one complete pinned-line profile."""


PinnedBoundedInteger = Annotated[
    str,
    StringConstraints(
        pattern=rf"^(?:0|-?[1-9][0-9]{{0,{COORDINATE_DIGITS - 1}}})$",
        strict=True,
        max_length=COORDINATE_DIGITS + 1,
    ),
]
"""Canonical signed integer whose magnitude carries at most ``COORDINATE_DIGITS`` digits.

The bound is published as a standard JSON Schema ``pattern``/``maxLength`` so
schema-driven clients can pre-validate the pinned-line wire domain."""

PinnedPositiveInteger = Annotated[
    str,
    StringConstraints(
        pattern=rf"^[1-9][0-9]{{0,{COORDINATE_DIGITS - 1}}}$",
        strict=True,
        max_length=COORDINATE_DIGITS,
    ),
]
"""Canonical positive integer whose magnitude carries at most ``COORDINATE_DIGITS`` digits."""


class PinnedBoundedRational(CanonicalRational):
    """A canonical rational bounded to the pinned-line coordinate digit cap.

    The shared ``CanonicalRational`` keeps its global 32,768-digit limit;
    this operation-local type publishes the pinned-line cap as enforceable
    JSON Schema constraints on both components without narrowing the shared
    value type.  ``from_attributes`` lets callers supply existing canonical
    values unchanged; over-cap values are rejected while parsing.
    """

    model_config = ConfigDict(from_attributes=True)

    num: PinnedBoundedInteger = Field(
        description=(
            "Canonical decimal numerator; at most "
            f"{COORDINATE_DIGITS} digits for pinned-line admission."
        ),
        examples=["1"],
    )
    den: PinnedPositiveInteger = Field(
        description=(
            "Positive canonical decimal denominator, reduced, integers use "
            f"den='1'; at most {COORDINATE_DIGITS} digits."
        ),
        examples=["2"],
    )


class PinnedLinePoint(LabelledRationalPoint):
    """A labelled point whose coordinates carry the pinned-line digit cap.

    Operation-local view of the shared ``LabelledRationalPoint``: the
    shared type stays at the canonical limit for distance-profile and
    distance-graph callers, while this subclass publishes the 256-digit
    component cap in the pinned-line schema.  ``from_attributes`` lets
    callers supply existing shared-type values unchanged.
    """

    model_config = ConfigDict(from_attributes=True)

    coordinates: tuple[PinnedBoundedRational, ...] = Field(min_length=1)


class PinnedLineConfiguration(PointConfiguration):
    """A configuration whose points carry the pinned-line coordinate cap.

    Operation-local view of the shared ``PointConfiguration`` with identical
    wire shape; over-cap coordinates are rejected by standard JSON Schema
    constraints before any mathematical work.  ``from_attributes`` lets
    callers supply an existing shared configuration unchanged.
    """

    model_config = ConfigDict(from_attributes=True)

    points: tuple[PinnedLinePoint, ...] = Field(min_length=2, max_length=MAX_POINTS)


_PINNED_ENTRY_SLACK_BYTES = 16
"""Per-entry slack over the exact skeleton, coefficient, and pair bounds."""

_PINNED_RESULT_BOUND_PADDING_BYTES = 1_024


def _component_heights(value: CanonicalRational) -> tuple[int, int]:
    return len(value.num.lstrip("-")), len(value.den)


def _difference_digit_heights(
    left: LabelledRationalPoint,
    right: LabelledRationalPoint,
) -> tuple[int, int, int, int]:
    """Return reduced planar difference component digit counts.

    The four counts are ``(digits(|dx numerator|), digits(dx denominator),
    digits(|dy numerator|), digits(dy denominator))`` for ``left - right``.
    """

    delta_x = left.coordinates[0].as_fraction() - right.coordinates[0].as_fraction()
    delta_y = left.coordinates[1].as_fraction() - right.coordinates[1].as_fraction()
    return (
        len(format_canonical_integer(abs(delta_x.numerator))),
        len(format_canonical_integer(delta_x.denominator)),
        len(format_canonical_integer(abs(delta_y.numerator))),
        len(format_canonical_integer(delta_y.denominator)),
    )


def _maximum_pinned_profile_wire_bytes(
    configuration: PointConfiguration,
    anchor: tuple[CanonicalRational, ...],
) -> int:
    """Upper-bound the canonical wire encoding of the complete profile.

    Every bound below is taken on unreduced forms, so canonical reduction at
    any step can only shrink the real encoding. A canonical line's first two
    coefficients carry their reduced coordinate-difference heights; the
    constant coefficient scales those differences by the first point's
    coordinates and sums, gaining the coordinate heights plus a carry. The
    squared anchor-to-line distance squares a cross of two differences over a
    squared-norm sum, doubling the cross bounds. Each line entry is costed as
    its encoding skeleton plus all four rational components (numerators and
    denominators charged separately) and slack; the pair ledger, the
    multiplicity ledger, and the configuration and anchor echoes are counted
    exactly or by the same component bounds.
    """

    from itertools import combinations

    points = configuration.points
    n = len(points)
    if n < 2:
        return 0
    point_heights = [
        (
            _component_heights(item.coordinates[0]),
            _component_heights(item.coordinates[1]),
        )
        for item in points
    ]
    diffs = {
        (first, second): _difference_digit_heights(points[first], points[second])
        for first, second in combinations(range(n), 2)
    }

    def entry_digit_bounds(first: int, second: int) -> tuple[int, int]:
        """Bound ``(coefficient digits, squared-distance digits)`` totals."""

        nx, bx, ny, by = diffs[(first, second)]
        (hxp_n, hxp_d), (hyp_n, hyp_d) = point_heights[first]
        # c = -(a*x_p + b*y_p) over a common denominator: each term scales a
        # reduced difference by a reduced coordinate of the same point.
        c_numerator = max(ny + hxp_n, nx + hyp_n) + 1
        c_denominator = by + hxp_d + bx + hyp_d
        # Cross of (p - q) with (anchor - p) differences.
        day = _component_heights_pair(anchor[1], points[first].coordinates[1])
        dax = _component_heights_pair(anchor[0], points[first].coordinates[0])
        first_term = (nx + day[0], bx + day[1])
        second_term = (ny + dax[0], by + dax[1])
        cross_numerator = (
            max(first_term[0] + second_term[1], second_term[0] + first_term[1]) + 1
        )
        cross_denominator = first_term[1] + second_term[1]
        squared_numerator = max(2 * nx + 2 * by, 2 * ny + 2 * bx) + 1
        squared_denominator = 2 * bx + 2 * by
        coefficient_digits = (ny + by) + (nx + bx) + (c_numerator + c_denominator)
        distance_digits = (
            2 * cross_denominator
            + squared_numerator
            + 2 * cross_numerator
            + squared_denominator
            + 2
        )
        return coefficient_digits, distance_digits

    coefficient_digits_max = 0
    distance_digits_max = 0
    for first, second in combinations(range(n), 2):
        coefficient, distance = entry_digit_bounds(first, second)
        coefficient_digits_max = max(coefficient_digits_max, coefficient)
        distance_digits_max = max(distance_digits_max, distance)

    line_entry_skeleton = len(
        encode_strict_json(
            {
                "line_coefficients": [
                    {"num": "", "den": ""},
                    {"num": "", "den": ""},
                    {"num": "", "den": ""},
                ],
                "squared_distance": {"num": "", "den": ""},
                "pairs": [[0, 1]],
            }
        )
    )
    multiplicity_entry_skeleton = len(
        encode_strict_json(
            {"pair_count": 0, "squared_distance": {"num": "", "den": ""}}
        )
    )
    echo_exact = len(
        encode_strict_json(
            {
                "configuration": configuration.model_dump(mode="json"),
                "anchor": [item.model_dump(mode="json") for item in anchor],
                "dimension": 2,
                "point_count": n,
                "exactness": "",
            }
        )
    )
    line_count_bound = n * (n - 1) // 2
    pairs_total_bytes = sum(
        len(str(first)) + len(str(second)) + 5
        for first, second in combinations(range(n), 2)
    )
    total = (
        echo_exact
        + line_count_bound
        * (
            line_entry_skeleton
            + coefficient_digits_max
            + distance_digits_max
            + _PINNED_ENTRY_SLACK_BYTES
        )
        + pairs_total_bytes
        + line_count_bound
        * (
            multiplicity_entry_skeleton
            + distance_digits_max
            + len(str(line_count_bound))
            + _PINNED_ENTRY_SLACK_BYTES
        )
        + _PINNED_RESULT_BOUND_PADDING_BYTES
    )
    return total


def _component_heights_pair(
    left: CanonicalRational,
    right: CanonicalRational,
) -> tuple[int, int]:

    delta = left.as_fraction() - right.as_fraction()
    return (
        len(format_canonical_integer(abs(delta.numerator))),
        len(format_canonical_integer(delta.denominator)),
    )


class PinnedLineDistanceRequest(StrictModel):
    """Compute distances from an anchor to all pair-spanned lines.

    The configuration must be planar (dimension 2) with distinct point
    coordinates; two identically-located points do not span a line. Both
    configuration coordinates and anchor coordinates are bounded to at most
    256 decimal digits per component so all derived squared distances remain
    representable as ``CanonicalRational`` (canonical limit 32,768 digits).
    """

    configuration: PinnedLineConfiguration = Field(
        description=(
            "Planar point configuration (dimension 2) with distinct coordinates; "
            "all points must have distinct locations and at most 64 points, "
            "each coordinate at most 256 digits for pinned-line admission. "
            "The point count is further coupled to the coordinate heights by "
            f"an aggregate result budget ({MAX_PINNED_PROFILE_RESULT_BYTES} "
            "bytes for the complete pair-spanned-line profile)."
        ),
        json_schema_extra={
            "coordinate_digit_bound": COORDINATE_DIGITS,
            "aggregate_result_budget_bytes": MAX_PINNED_PROFILE_RESULT_BYTES,
        },
    )
    anchor: tuple[PinnedBoundedRational, ...] = Field(
        min_length=2,
        max_length=2,
        description=(
            "Planar rational anchor point (exactly two coordinates); both at most "
            "256 digits so derived squared distances remain representable."
        ),
    )

    @model_validator(mode="after")
    def require_planar_and_matching_anchor(self) -> Self:
        if not self.configuration.points:
            return self
        _require_bounded_point_configuration(self.configuration, self.anchor)
        if len(self.configuration.points[0].coordinates) != 2:
            raise _validation_error(
                "pinned_line_distance_profile_requires_a",
                "pinned line-distance profile requires a planar configuration",
            )
        # A pair of coincident points does not span a line; require distinct
        # coordinates so every pair defines a geometric line.
        coords = {
            tuple(c.as_fraction() for c in pt.coordinates)
            for pt in self.configuration.points
        }
        if len(coords) != len(self.configuration.points):
            raise _validation_error(
                "pinned_line_distance_profile_requires_distinct",
                "pinned line-distance profile requires distinct point coordinates",
            )
        # Couple the point count to the coordinate heights through the
        # aggregate output budget: C(n,2) lines with height-proportional
        # rational components must stay canonically encodable.
        estimated_bytes = _maximum_pinned_profile_wire_bytes(
            self.configuration, self.anchor
        )
        if estimated_bytes > MAX_PINNED_PROFILE_RESULT_BYTES:
            raise _validation_error(
                "complete_pinned_line_distance_profile_would",
                "the complete pinned line-distance profile would exceed the "
                f"{MAX_PINNED_PROFILE_RESULT_BYTES}-byte aggregate result "
                "budget; reduce the point count or coordinate heights",
            )
        return self


class PinnedLineEntry(StrictModel):
    """One pair-spanned line with its canonical equation and source pairs."""

    line_coefficients: tuple[CanonicalRational, ...] = Field(min_length=3, max_length=3)
    squared_distance: CanonicalRational
    pairs: tuple[tuple[int, int], ...] = Field(min_length=1, max_length=MAX_PAIRS)

    @model_validator(mode="after")
    def require_sorted_pairs(self) -> Self:
        for i, j in self.pairs:
            if not i < j:
                raise _validation_error(
                    "source_pairs_ordered_i_j", "source pairs must be ordered (i < j)"
                )
        if len(set(self.pairs)) != len(self.pairs):
            raise _validation_error(
                "source_pairs_unique", "source pairs must be unique"
            )
        if self.pairs != tuple(sorted(self.pairs)):
            raise _validation_error(
                "source_pairs_sorted_so_profile_has",
                "source pairs must be sorted so each profile has exactly "
                "one canonical serialization",
            )
        if self.squared_distance.as_fraction() < 0:
            raise _validation_error(
                "squared_distance_nonnegative", "squared distance must be nonnegative"
            )
        return self


def _rational_component_bytes(value: object) -> int:
    """Raw numerator/denominator character count of one authored rational."""
    if isinstance(value, CanonicalRational):
        return len(value.num) + len(value.den)
    if isinstance(value, dict):
        return sum(len(str(value.get(key, ""))) for key in ("num", "den"))
    return 0


def _count_line_ledger(line: object) -> tuple[int, int]:
    """One entry's source-pair count plus its authored rational bytes.

    The byte count covers EVERY authored rational on the entry: the three
    line coefficients and the squared distance, so no field can carry
    globally permitted oversized components past the pre-parse bound.
    """
    if isinstance(line, PinnedLineEntry):
        rational_bytes = _rational_component_bytes(line.squared_distance)
        for coefficient in line.line_coefficients:
            rational_bytes += _rational_component_bytes(coefficient)
        return len(line.pairs), rational_bytes
    if isinstance(line, dict):
        pairs = line.get("pairs")
        pair_count = len(pairs) if isinstance(pairs, (list, tuple)) else 0
        rational_bytes = _rational_component_bytes(line.get("squared_distance"))
        coefficients = line.get("line_coefficients")
        if isinstance(coefficients, (list, tuple)):
            for coefficient in coefficients:
                rational_bytes += _rational_component_bytes(coefficient)
        return pair_count, rational_bytes
    return 0, 0


def _pinned_profile_source(
    configuration: PinnedLineConfiguration,
    anchor: tuple[PinnedBoundedRational, ...],
    point_count: int,
) -> tuple[list[tuple[Fraction, ...]], tuple[Fraction, ...]]:
    _require_bounded_point_configuration(configuration, anchor)
    if any(len(pt.coordinates) != 2 for pt in configuration.points):
        raise _validation_error(
            "retained_configuration_a_planar_configuration_two",
            "retained configuration must be a planar configuration "
            "(exactly two coordinates per point)",
        )
    if (
        _maximum_pinned_profile_wire_bytes(configuration, anchor)
        > MAX_PINNED_PROFILE_RESULT_BYTES
    ):
        raise _validation_error(
            "complete_pinned_line_distance_profile_would",
            "the complete pinned line-distance profile would exceed the "
            f"{MAX_PINNED_PROFILE_RESULT_BYTES}-byte aggregate result "
            "budget; reduce the point count or coordinate heights",
        )
    if len(configuration.points) != point_count:
        raise _validation_error(
            "point_count_retained_configuration",
            "point_count must match the retained configuration",
        )
    coords = {
        tuple(c.as_fraction() for c in pt.coordinates) for pt in configuration.points
    }
    if len(coords) != len(configuration.points):
        raise _validation_error(
            "retained_configuration_points_distinct_coordinates",
            "retained configuration points must have distinct coordinates",
        )
    if point_count > MAX_POINTS:
        raise _validation_error(
            "point_count_exceeds_configuration_bound",
            "point_count exceeds the configuration bound",
        )
    points = [
        tuple(c.as_fraction() for c in pt.coordinates) for pt in configuration.points
    ]
    return points, tuple(c.as_fraction() for c in anchor)


def _expected_pinned_profile_geometry(
    points: list[tuple[Fraction, ...]],
    anchor: tuple[Fraction, ...],
    point_count: int,
) -> tuple[
    dict[tuple[Fraction, Fraction, Fraction], list[tuple[int, int]]],
    dict[tuple[Fraction, Fraction, Fraction], Fraction],
]:
    from itertools import combinations

    expected_lines: dict[
        tuple[Fraction, Fraction, Fraction], list[tuple[int, int]]
    ] = {}
    expected_distances: dict[tuple[Fraction, Fraction, Fraction], Fraction] = {}
    for i, j in combinations(range(point_count), 2):
        coeffs = canonical_line_coefficients(points[i], points[j])
        expected_lines.setdefault(coeffs, []).append((i, j))
        if coeffs not in expected_distances:
            expected_distances[coeffs] = squared_point_line_distance(
                anchor, points[i], points[j]
            )
    return expected_lines, expected_distances


def _validate_pinned_profile_entries(
    lines: tuple[PinnedLineEntry, ...],
    point_count: int,
    expected_lines: dict[tuple[Fraction, Fraction, Fraction], list[tuple[int, int]]],
    expected_distances: dict[tuple[Fraction, Fraction, Fraction], Fraction],
) -> tuple[list[tuple[int, int]], dict[Fraction, int]]:
    seen_pairs: list[tuple[int, int]] = []
    seen_lines: set[tuple[Fraction, ...]] = set()
    multiplicities: dict[Fraction, int] = {}
    for entry in lines:
        entry_coeffs = tuple(c.as_fraction() for c in entry.line_coefficients)
        if entry_coeffs in seen_lines:
            raise _validation_error(
                "duplicate_lines_collapsed_entry",
                "duplicate lines must be collapsed into one entry",
            )
        seen_lines.add(entry_coeffs)
        if entry_coeffs not in expected_lines:
            raise _validation_error(
                "line_coefficients_do_any_source_pair",
                "line coefficients do not match any source pair line",
            )
        if tuple(sorted(entry.pairs)) != tuple(sorted(expected_lines[entry_coeffs])):
            raise _validation_error(
                "source_pairs_do_line_s_geometry",
                "source pairs do not match the line's geometry",
            )
        if entry.squared_distance.as_fraction() != expected_distances[entry_coeffs]:
            raise _validation_error(
                "squared_distance_source_geometry",
                "squared distance does not match the source geometry",
            )
        for i, j in entry.pairs:
            if not 0 <= i < j < point_count:
                raise _validation_error(
                    "source_pairs_reference_valid_point_indices",
                    "source pairs must reference valid point indices",
                )
            seen_pairs.append((i, j))
        distance = entry.squared_distance.as_fraction()
        multiplicities[distance] = multiplicities.get(distance, 0) + 1
    return seen_pairs, multiplicities


def _validate_pinned_profile_order(
    lines: tuple[PinnedLineEntry, ...],
    expected_lines: dict[tuple[Fraction, Fraction, Fraction], list[tuple[int, int]]],
    expected_distances: dict[tuple[Fraction, Fraction, Fraction], Fraction],
) -> None:
    ordered_coeffs = sorted(
        expected_lines, key=lambda coeffs: (expected_distances[coeffs], coeffs)
    )
    actual_coeffs = [
        tuple(c.as_fraction() for c in entry.line_coefficients) for entry in lines
    ]
    if actual_coeffs != ordered_coeffs:
        raise _validation_error(
            "lines_sorted_squared_distance_coefficients",
            "lines must be sorted by (squared_distance, coefficients)",
        )


class PinnedLineDistanceResult(StrictModel):
    """Complete pinned line-distance profile for a point configuration.

    The result retains its source ``configuration`` and ``anchor`` so validation
    can replay the defining geometry: every pair-spanned line is recomputed
    canonically from the retained points and its squared distance from the
    retained anchor is verified.
    """

    configuration: PinnedLineConfiguration = Field(
        description=(
            "Source planar point configuration with distinct coordinates; "
            "retained for result binding and replay."
        ),
        json_schema_extra={"coordinate_digit_bound": COORDINATE_DIGITS},
    )
    anchor: tuple[PinnedBoundedRational, ...] = Field(
        min_length=2,
        max_length=2,
        description=(
            "Retained planar anchor point (exactly two coordinates); both at "
            "most 256 digits."
        ),
    )
    dimension: int = Field(ge=2, le=2)
    point_count: int = Field(ge=2, le=MAX_POINTS)
    lines: tuple[PinnedLineEntry, ...] = Field(max_length=MAX_PAIRS)
    distance_multiplicities: tuple[tuple[CanonicalRational, int], ...] = Field(
        max_length=MAX_PAIRS
    )

    @model_validator(mode="before")
    @classmethod
    def require_aggregate_pair_ledger_bound(cls, data: object) -> object:
        """Cap the aggregate source-pair ledger before any parsing.

        Each ``pairs`` dimension is capped separately, so an authored result
        could still carry ``MAX_PAIRS`` entries times ``MAX_PAIRS`` pairs.
        A valid profile contains only ``MAX_PAIRS`` source pairs in total,
        so the raw aggregate count is checked here — before Pydantic
        constructs every nested entry — to keep accepted-parse memory tied
        to the mathematical bound. Already-parsed ``PinnedLineEntry``
        instances are counted as well so native callers cannot bypass the
        declared aggregate work and intermediate-memory bound through the
        typed Python boundary.
        """

        data = canonicalize_json_containers(data)

        if not isinstance(data, dict):
            return data
        lines = data.get("lines")
        if not isinstance(lines, (list, tuple)):
            return data
        total = 0
        authored_rational_bytes = 0
        for line in lines:
            pair_total, rational_bytes = _count_line_ledger(line)
            total += pair_total
            authored_rational_bytes += rational_bytes
            if total > MAX_PAIRS:
                raise _validation_error(
                    "aggregate_source_pair_ledger_exceeds_f",
                    "the aggregate source-pair ledger exceeds the "
                    f"{MAX_PAIRS}-pair profile bound",
                )
        # Distance multiplicities carry authored rationals too; count them
        # so no field can bypass the pre-parse aggregate bound.
        multiplicities = data.get("distance_multiplicities")
        if isinstance(multiplicities, (list, tuple)):
            for entry in multiplicities:
                if isinstance(entry, (list, tuple)) and entry:
                    authored_rational_bytes += _rational_component_bytes(entry[0])
        # Authored rational components are bounded by the same aggregate
        # result budget as the parsed profile: every valid entry needs at
        # least two characters per canonical rational, so any payload whose
        # raw numerator/denominator characters alone approach the budget is
        # forged padding that must be rejected BEFORE nested parsing.
        if authored_rational_bytes > MAX_PINNED_PROFILE_RESULT_BYTES:
            raise _validation_error(
                "authored_rational_components_exceed_f_max",
                "authored rational components exceed the "
                f"{MAX_PINNED_PROFILE_RESULT_BYTES}-byte aggregate result "
                "budget before parsing",
            )
        return data

    @model_validator(mode="after")
    def require_consistent_profile(self) -> Self:
        from itertools import combinations

        points, anchor = _pinned_profile_source(
            self.configuration, self.anchor, self.point_count
        )
        expected_lines, expected_distances = _expected_pinned_profile_geometry(
            points, anchor, self.point_count
        )
        expected_pairs = sorted(combinations(range(self.point_count), 2))
        seen_pairs, multiplicities = _validate_pinned_profile_entries(
            self.lines, self.point_count, expected_lines, expected_distances
        )
        if sorted(seen_pairs) != expected_pairs or len(seen_pairs) != len(
            set(seen_pairs)
        ):
            raise _validation_error(
                "lines_cover_set_source_pairs_once",
                "lines must cover exactly the set of source pairs once",
            )
        if len(self.lines) != len(expected_lines):
            raise _validation_error(
                "lines_correspond_distinct_geometric_lines",
                "lines must correspond to distinct geometric lines",
            )
        _validate_pinned_profile_order(self.lines, expected_lines, expected_distances)

        reconstructed = tuple(
            (
                CanonicalRational.from_fraction(d),
                count,
            )
            for d, count in sorted(multiplicities.items())
        )
        if reconstructed != self.distance_multiplicities:
            raise _validation_error(
                "distance_multiplicities_partition_lines_sorted",
                "distance multiplicities must partition the lines and be sorted",
            )
        return self
