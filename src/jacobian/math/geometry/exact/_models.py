"""Typed wire contracts for exact geometry point-configuration operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Any, Literal, Self

from pydantic import ConfigDict, Field, StringConstraints, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import encode_strict_json, format_canonical_integer

MAX_POINTS = 64
MAX_DIMENSION = 20
MAX_PAIRS = MAX_POINTS * (MAX_POINTS - 1) // 2
"""Maximum distinct source pairs spanned by a bounded configuration: C(64, 2)."""
COORDINATE_DIGITS = 256
"""Per-coordinate digit bound for pinned line-distance profile so squared distances stay representable."""

MAX_QUADRUPLE_SEARCH_POINTS = 18
"""Cap on configuration size for C(n,4) quadruple enumeration (3060 subsets)."""

# A collinear search over n points must be able to return every C(n,3)
# witness triple in one bounded math.run response.  The complete witness set
# for an all-collinear configuration is the worst case, so the point cap is
# derived from a fixed witness budget: each index triple serializes to at
# most 12 bytes of compact JSON, and C(40,3) = 9,880 triples stay under
# 120 KB.  Larger collinear searches are rejected at request admission.
MAX_COLLINEAR_SEARCH_POINTS = 40
"""Collinear-search point cap derived from the fixed C(40,3) witness budget."""

COLLINEAR_WITNESS_BUDGET = 9880
"""Maximum complete collinear witness count: C(40,3) index triples."""

INCIDENCE_COORDINATE_DIGITS = 256
"""Conservative per-coordinate digit bound; keeps 10k-quadruple Fractions bounded."""
_INCIDENCE_INPUT_HEIGHT = INCIDENCE_COORDINATE_DIGITS // 4

CONCYCLIC_WORK_BUDGET = 65536
"""Joint admission budget C(n,4)*h for concyclic search.

h is the largest decimal digit length over all coordinate numerators and
denominators.  Measured accepted-call cost (enumeration plus the mandatory
completeness replay in IncidenceSearchResult) is near-linear in C(n,4)*h;
the budget admits 64-digit coordinates up to 14 points, 36-digit
coordinates at 16 points, and 21-digit coordinates at the 18-point cap,
holding every admitted call to roughly two seconds.
"""


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


def _bounded_incidence_coordinate(value: CanonicalRational, label: str) -> None:
    from jacobian._exact import require_bounded_rational

    require_bounded_rational(
        value,
        max_digits=_INCIDENCE_INPUT_HEIGHT,
        label=label,
    )


def _coordinate_height(value: CanonicalRational) -> int:
    return max(len(value.num.lstrip("-")), len(value.den.lstrip("-")))


def _require_concyclic_work_bound(points: Any) -> None:
    """Reject configurations whose joint enumeration work exceeds budget.

    The concyclic search performs several exact determinant checks per
    C(n,4) quadruple over h-digit rationals, and result validation replays
    the same complete search; neither per-coordinate nor point-count caps
    alone bound their product.
    """

    from math import comb

    height = max(_coordinate_height(c) for point in points for c in point.coordinates)
    subsets = comb(len(points), 4)
    if subsets * height > CONCYCLIC_WORK_BUDGET:
        raise ValueError(
            "concyclic-quadruple search exceeds the joint work budget "
            f"C({len(points)},4)*{height} = {subsets * height} > "
            f"{CONCYCLIC_WORK_BUDGET}; reduce the point count or the "
            "coordinate digit length"
        )


def _require_distinct_incidence_coordinates(points: Any) -> None:
    """Reject coordinate-coincident labelled entries.

    Coincident points make collinearity and concyclicity degenerate: every
    triple containing a repeated point has zero cross product, so the
    concyclicity guard would silently skip such quadruples and report a
    false negative.  Labels alone do not establish distinct points.
    """

    coords = {tuple(c.as_fraction() for c in point.coordinates) for point in points}
    if len(coords) != len(points):
        raise ValueError(
            "incidence configurations require pairwise distinct coordinates; "
            "repeated labels at one location are rejected"
        )


class LabelledRationalPoint(StrictModel):
    """A labelled rational point in bounded dimension."""

    label: str = Field(min_length=1, max_length=64)
    coordinates: tuple[CanonicalRational, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_dimension(self) -> Self:
        if len(self.coordinates) > MAX_DIMENSION:
            raise ValueError("dimension exceeds bound")
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
                raise ValueError("all points must have the same dimension")
        labels = [p.label for p in self.points]
        if len(labels) != len(set(labels)):
            raise ValueError("point labels must be unique")
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
            raise ValueError("squared distance target must be nonnegative")
        return self


class DistanceGraphResult(StrictModel):
    """Graph whose edges connect pairs at the target squared distance."""

    vertex_count: int = Field(ge=2)
    edges: tuple[tuple[int, int], ...]


IncidenceBoundedInteger = Annotated[
    str,
    StringConstraints(
        pattern=rf"^(?:0|-?[1-9][0-9]{{0,{_INCIDENCE_INPUT_HEIGHT - 1}}})$",
        strict=True,
        max_length=_INCIDENCE_INPUT_HEIGHT + 1,
    ),
]
"""Canonical signed integer whose magnitude carries at most 64 digits.

The bound is published as a standard JSON Schema ``pattern``/``maxLength`` so
over-cap coordinate components are rejected at string validation, before any
nested ``CanonicalRational`` parses, reduces, or reformats them.
"""

IncidenceBoundedDenominator = Annotated[
    str,
    StringConstraints(
        pattern=rf"^[1-9][0-9]{{0,{_INCIDENCE_INPUT_HEIGHT - 1}}}$",
        strict=True,
        max_length=_INCIDENCE_INPUT_HEIGHT,
    ),
]
"""Canonical positive denominator whose magnitude carries at most 64 digits."""


class IncidenceBoundedRational(CanonicalRational):
    """A canonical rational bounded to the incidence-search 64-digit height.

    Operation-local view of the shared ``CanonicalRational``: the shared type
    keeps its global 32,768-digit limit, while this subclass publishes the
    incidence admission cap as enforceable JSON Schema constraints so the
    arithmetic bound fires during field validation, before expansion.
    ``from_attributes`` lets callers supply existing canonical values
    unchanged; over-cap components are rejected before parsing.
    """

    model_config = ConfigDict(from_attributes=True)

    num: IncidenceBoundedInteger = Field(
        description=(
            "Canonical decimal numerator; at most "
            f"{_INCIDENCE_INPUT_HEIGHT} digits for incidence-search admission."
        ),
        examples=["1"],
    )
    den: IncidenceBoundedDenominator = Field(
        description=(
            "Positive canonical decimal denominator, reduced, integers use "
            f"den='1'; at most {_INCIDENCE_INPUT_HEIGHT} digits."
        ),
        examples=["2"],
    )


class IncidencePoint(LabelledRationalPoint):
    """A labelled point whose coordinates carry the incidence digit cap."""

    model_config = ConfigDict(from_attributes=True)

    coordinates: tuple[IncidenceBoundedRational, ...] = Field(min_length=1)


class IncidencePointConfiguration(PointConfiguration):
    """A configuration whose points carry the incidence digit cap.

    Operation-local view of the shared ``PointConfiguration`` with identical
    wire shape; over-cap coordinates are rejected by standard JSON Schema
    constraints before any mathematical work or nested rational expansion.
    """

    model_config = ConfigDict(from_attributes=True)

    points: tuple[IncidencePoint, ...] = Field(min_length=2, max_length=MAX_POINTS)


class CollinearTriplesRequest(StrictModel):
    """Search a planar configuration for collinear triples.

    The configuration must be planar with 3..40 points (the sibling
    concyclic search admits 4..18 points under a joint work budget) and
    each coordinate must stay within the 64-digit operation-specific bound
    so that enumeration with huge Fractions stays bounded; see the
    validator for the precise bound.  The point cap keeps the complete
    worst-case witness set C(40,3) = 9,880 triples inside one bounded
    response.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Search a planar configuration for collinear triples. "
                "Requires a planar configuration with 3..40 points whose "
                "coordinates are pairwise distinct (no repeated coordinate "
                "pairs, even under distinct labels); each coordinate must "
                "stay within the 64-digit operation-specific bound so "
                "enumeration stays bounded.  The 40-point bound keeps the "
                "complete worst-case witness set C(40,3) = 9,880 triples "
                "inside one bounded response."
            )
        }
    )

    configuration: IncidencePointConfiguration = Field(
        description=(
            "Planar point configuration with 3..40 points with pairwise "
            "distinct coordinates; 4..18 points for the sibling concyclic "
            "search. Each coordinate is bounded to 64 digits so that all "
            "exact determinants stay representable.  The point count is "
            "capped so the complete witness set C(n,3) stays within one "
            "bounded response."
        )
    )

    @model_validator(mode="after")
    def require_planar(self) -> Self:
        if not self.configuration.points:
            return self
        if len(self.configuration.points[0].coordinates) != 2:
            raise ValueError("collinear-triple search requires a planar configuration")
        # A collinear triple needs three distinct points; two-point
        # configurations cannot produce witnesses and are rejected at the
        # boundary so the search scope is exact.
        if len(self.configuration.points) < 3:
            raise ValueError("collinear-triple search requires at least three points")
        if len(self.configuration.points) > MAX_COLLINEAR_SEARCH_POINTS:
            raise ValueError(
                "collinear-triple search exceeds the "
                f"{MAX_COLLINEAR_SEARCH_POINTS}-point enumeration bound "
                f"(C({MAX_COLLINEAR_SEARCH_POINTS},3) = {COLLINEAR_WITNESS_BUDGET} "
                "witness triples); reduce the point count"
            )
        for idx, point in enumerate(self.configuration.points):
            for dim, coord in enumerate(point.coordinates):
                _bounded_incidence_coordinate(coord, f"point {idx} coordinate {dim}")
        _require_distinct_incidence_coordinates(self.configuration.points)
        return self


class ConcyclicQuadruplesRequest(StrictModel):
    """Search a planar configuration for concyclic quadruples.

    Requires a planar configuration with 4..18 points with pairwise
    distinct coordinates whose coordinates each stay within the 64-digit
    operation-specific bound and whose joint work measure stays within
    budget: with h the largest decimal digit length over all coordinate
    numerators and denominators, C(n,4)*h must not exceed 65536.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Search a planar configuration for concyclic quadruples. "
                "Requires a planar configuration with 4..18 points "
                "(C(18,4)=3060) whose coordinates are pairwise distinct "
                "(no repeated coordinate pairs, even under distinct "
                "labels); configurations with 19..64 points are rejected. "
                "Each coordinate is bounded to 64 digits, and the joint "
                "work budget C(n,4)*h <= 65536 (h = largest coordinate "
                "digit length) must hold so exact enumeration stays "
                "bounded."
            )
        }
    )

    configuration: IncidencePointConfiguration = Field(
        description=(
            "Planar point configuration with 4..18 points with pairwise "
            "distinct coordinates; the enumeration covers every unordered "
            "quadruple. Each coordinate is bounded to 64 digits, and "
            "C(n,4)*h <= 65536 (h = largest coordinate digit length) so "
            "exact enumeration stays bounded."
        )
    )

    @model_validator(mode="after")
    def require_planar(self) -> Self:
        if not self.configuration.points:
            return self
        if len(self.configuration.points[0].coordinates) != 2:
            raise ValueError(
                "concyclic-quadruple search requires a planar configuration"
            )
        if len(self.configuration.points) < 4:
            raise ValueError("concyclic-quadruple search requires at least four points")
        if len(self.configuration.points) > MAX_QUADRUPLE_SEARCH_POINTS:
            raise ValueError(
                "concyclic-quadruple search exceeds the "
                f"{MAX_QUADRUPLE_SEARCH_POINTS}-point enumeration bound"
            )
        for idx, point in enumerate(self.configuration.points):
            for dim, coord in enumerate(point.coordinates):
                _bounded_incidence_coordinate(coord, f"point {idx} coordinate {dim}")
        _require_distinct_incidence_coordinates(self.configuration.points)
        _require_concyclic_work_bound(self.configuration.points)
        return self


def _incidence_cross(
    o: tuple[Fraction, ...], a: tuple[Fraction, ...], b: tuple[Fraction, ...]
) -> Fraction:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _has_collinear_triple(
    points: list[tuple[Fraction, ...]], quad: tuple[int, ...]
) -> bool:
    i, j, k, m = quad
    return any(
        _incidence_cross(points[a], points[b], points[c]) == 0
        for a, b, c in (
            (i, j, k),
            (i, j, m),
            (i, k, m),
            (j, k, m),
        )
    )


def _incidence_det4(
    points: list[tuple[Fraction, ...]], indices: tuple[int, ...]
) -> Fraction:
    rows = []
    for idx in indices:
        x, y = points[idx]
        rows.append((x * x + y * y, x, y, Fraction(1)))
    total = Fraction(0)
    for col in range(4):
        sub = tuple(tuple(row[c] for c in range(4) if c != col) for row in rows[1:])
        sign = 1 if col % 2 == 0 else -1
        det3 = (
            sub[0][0] * (sub[1][1] * sub[2][2] - sub[1][2] * sub[2][1])
            - sub[0][1] * (sub[1][0] * sub[2][2] - sub[1][2] * sub[2][0])
            + sub[0][2] * (sub[1][0] * sub[2][1] - sub[1][1] * sub[2][0])
        )
        total += sign * rows[0][col] * det3
    return total


def _require_incidence_result_cardinality(
    kind: str,
    configuration: IncidencePointConfiguration,
    dimension: int,
    point_count: int,
) -> None:
    """Bind a retained result to its operation's replay domain."""
    if len(configuration.points) != point_count:
        raise ValueError("point_count must match the retained configuration")
    retained_dimension = len(configuration.points[0].coordinates)
    if retained_dimension != dimension:
        raise ValueError("dimension must match the retained configuration coordinates")
    if retained_dimension != 2:
        raise ValueError("incidence replay requires a planar retained configuration")
    if kind == "CONCYCLIC_QUADRUPLE" and point_count > MAX_QUADRUPLE_SEARCH_POINTS:
        raise ValueError(
            "concyclic result point_count exceeds the "
            f"{MAX_QUADRUPLE_SEARCH_POINTS}-point enumeration bound"
        )
    # Mirror each request's cardinality domain: a result retaining
    # fewer points than its kind's request accepts can never be an
    # outcome of the operation.
    required_point_count = 3 if kind == "COLLINEAR_TRIPLE" else 4
    if point_count < required_point_count:
        raise ValueError(
            f"{kind} results require at least {required_point_count} retained points"
        )


def _require_incidence_witness_agreement(
    holds: bool,
    witnesses: tuple[tuple[int, ...], ...],
) -> None:
    """A result must agree with itself on existence and canonical order."""
    if holds and not witnesses:
        raise ValueError("a holds=True result must list at least one witness")
    if not holds and witnesses:
        raise ValueError("a holds=False result must list no witnesses")
    # Canonical serialization: the search enumerates combinations in
    # lexicographic order, so any permutation of the complete witness
    # set is a second representation of the same exact result.
    if witnesses != tuple(sorted(witnesses)):
        raise ValueError(
            "witnesses must be canonically ordered lexicographically ascending"
        )


def _apply_incidence_result_admission(
    kind: str,
    configuration: IncidencePointConfiguration,
) -> None:
    """Re-apply request-side arithmetic admission to a retained result.

    A deserialized result must not bypass the 64-digit coordinate cap
    through plain PointConfiguration, and its points must stay pairwise
    distinct.
    """
    for idx, point in enumerate(configuration.points):
        for dim, coord in enumerate(point.coordinates):
            _bounded_incidence_coordinate(coord, f"point {idx} coordinate {dim}")
    _require_distinct_incidence_coordinates(configuration.points)
    if kind == "CONCYCLIC_QUADRUPLE":
        _require_concyclic_work_bound(configuration.points)


def _validate_incidence_witnesses(
    kind: str,
    witnesses: tuple[tuple[int, ...], ...],
    points: list[tuple[Fraction, ...]],
    point_count: int,
) -> set[tuple[int, ...]]:
    """Check every listed witness against the exact retained geometry."""
    expected_size = 3 if kind == "COLLINEAR_TRIPLE" else 4
    seen: set[tuple[int, ...]] = set()
    for witness in witnesses:
        if len(witness) != expected_size:
            raise ValueError(f"{kind} witnesses must list {expected_size} indices")
        if witness != tuple(sorted(witness)):
            raise ValueError("witness indices must be sorted ascending")
        if len(set(witness)) != len(witness):
            raise ValueError("witness indices must be distinct")
        if any(i < 0 or i >= point_count for i in witness):
            raise ValueError("witness index out of range")
        if witness in seen:
            raise ValueError("witnesses must be unique")
        seen.add(witness)
        if kind == "COLLINEAR_TRIPLE":
            i, j, k = witness
            if _incidence_cross(points[i], points[j], points[k]) != 0:
                raise ValueError("a collinear witness is not actually collinear")
        else:
            # Concyclic excludes degenerate (collinear) quadruples.
            if _has_collinear_triple(points, witness):
                raise ValueError("a concyclic witness contains a collinear triple")
            if _incidence_det4(points, witness) != 0:
                raise ValueError("a concyclic witness is not actually concyclic")
    return seen


def _expected_incidence_witnesses(
    kind: str,
    points: list[tuple[Fraction, ...]],
    point_count: int,
) -> set[tuple[int, ...]]:
    """Replay the bounded search to certify completeness and absence."""
    from itertools import combinations

    expected: set[tuple[int, ...]] = set()
    if kind == "COLLINEAR_TRIPLE":
        for triple in combinations(range(point_count), 3):
            i, j, k = triple
            if _incidence_cross(points[i], points[j], points[k]) == 0:
                expected.add(triple)
    else:
        for quad in combinations(range(point_count), 4):
            if _has_collinear_triple(points, quad):
                continue
            if _incidence_det4(points, quad) == 0:
                expected.add(quad)
    return expected


class IncidenceSearchResult(StrictModel):
    """Witnesses to a forbidden planar incidence configuration, or none.

    The result retains its source configuration so validation can replay
    every witness exactly against the certified points and certify
    completeness of the reported incidence set.
    """

    configuration: IncidencePointConfiguration
    dimension: int = Field(ge=2, le=2)
    point_count: int = Field(ge=3, le=MAX_COLLINEAR_SEARCH_POINTS)
    holds: bool = Field(
        description="True iff at least one witness incidence exists.",
    )
    witnesses: tuple[tuple[int, ...], ...] = Field(
        default=(),
        max_length=COLLINEAR_WITNESS_BUDGET,
        description=(
            "Complete canonically ordered witness set; capped at "
            f"C({MAX_COLLINEAR_SEARCH_POINTS},3) = {COLLINEAR_WITNESS_BUDGET} "
            "triples so one response stays bounded."
        ),
    )
    kind: Literal["COLLINEAR_TRIPLE", "CONCYCLIC_QUADRUPLE"]

    @model_validator(mode="after")
    def require_consistent_witnesses(self) -> Self:
        _require_incidence_result_cardinality(
            self.kind, self.configuration, self.dimension, self.point_count
        )
        _require_incidence_witness_agreement(self.holds, self.witnesses)
        _apply_incidence_result_admission(self.kind, self.configuration)

        pts = [
            tuple(c.as_fraction() for c in point.coordinates)
            for point in self.configuration.points
        ]
        seen = _validate_incidence_witnesses(
            self.kind, self.witnesses, pts, self.point_count
        )
        expected = _expected_incidence_witnesses(self.kind, pts, self.point_count)
        if seen != expected:
            raise ValueError(
                "witnesses must be the complete set of incidences for the retained configuration"
            )
        if self.holds != bool(expected):
            raise ValueError("holds must match actual incidence existence")
        return self


__all__ = [
    "CollinearTriplesRequest",
    "ConcyclicQuadruplesRequest",
    "DistanceGraphRequest",
    "DistanceGraphResult",
    "DistanceMultiplicityEntry",
    "DistanceProfileRequest",
    "DistanceProfileResult",
    "IncidenceSearchResult",
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
            raise ValueError(
                "pinned line-distance profile requires a planar configuration"
            )
        # A pair of coincident points does not span a line; require distinct
        # coordinates so every pair defines a geometric line.
        coords = {
            tuple(c.as_fraction() for c in pt.coordinates)
            for pt in self.configuration.points
        }
        if len(coords) != len(self.configuration.points):
            raise ValueError(
                "pinned line-distance profile requires distinct point coordinates",
            )
        # Couple the point count to the coordinate heights through the
        # aggregate output budget: C(n,2) lines with height-proportional
        # rational components must stay canonically encodable.
        estimated_bytes = _maximum_pinned_profile_wire_bytes(
            self.configuration, self.anchor
        )
        if estimated_bytes > MAX_PINNED_PROFILE_RESULT_BYTES:
            raise ValueError(
                "the complete pinned line-distance profile would exceed the "
                f"{MAX_PINNED_PROFILE_RESULT_BYTES}-byte aggregate result "
                "budget; reduce the point count or coordinate heights"
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
                raise ValueError("source pairs must be ordered (i < j)")
        if len(set(self.pairs)) != len(self.pairs):
            raise ValueError("source pairs must be unique")
        if self.pairs != tuple(sorted(self.pairs)):
            raise ValueError(
                "source pairs must be sorted so each profile has exactly "
                "one canonical serialization"
            )
        if self.squared_distance.as_fraction() < 0:
            raise ValueError("squared distance must be nonnegative")
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
        raise ValueError(
            "retained configuration must be a planar configuration "
            "(exactly two coordinates per point)"
        )
    if (
        _maximum_pinned_profile_wire_bytes(configuration, anchor)
        > MAX_PINNED_PROFILE_RESULT_BYTES
    ):
        raise ValueError(
            "the complete pinned line-distance profile would exceed the "
            f"{MAX_PINNED_PROFILE_RESULT_BYTES}-byte aggregate result "
            "budget; reduce the point count or coordinate heights"
        )
    if len(configuration.points) != point_count:
        raise ValueError("point_count must match the retained configuration")
    coords = {
        tuple(c.as_fraction() for c in pt.coordinates) for pt in configuration.points
    }
    if len(coords) != len(configuration.points):
        raise ValueError("retained configuration points must have distinct coordinates")
    if point_count > MAX_POINTS:
        raise ValueError("point_count exceeds the configuration bound")
    points = [
        tuple(c.as_fraction() for c in pt.coordinates) for pt in configuration.points
    ]
    return points, tuple(c.as_fraction() for c in anchor)


def _gcd3(a: Fraction, b: Fraction, c: Fraction) -> Fraction:
    from math import gcd

    if a == 0 and b == 0 and c == 0:
        return Fraction(0)
    nums = [a.numerator, b.numerator, c.numerator]
    dens = [a.denominator, b.denominator, c.denominator]
    common_den = 1
    for denominator in dens:
        common_den = common_den * denominator // gcd(common_den, denominator)
    scaled = [n * (common_den // d) for n, d in zip(nums, dens, strict=True)]
    common_num = 0
    for value in scaled:
        common_num = gcd(common_num, abs(value))
    if common_num == 0:
        return Fraction(0)
    return Fraction(common_num, common_den)


def _canonical_line_coefficients(
    p: tuple[Fraction, ...], q: tuple[Fraction, ...]
) -> tuple[Fraction, Fraction, Fraction]:
    dx = q[0] - p[0]
    dy = q[1] - p[1]
    a, b = dy, -dx
    c = -(a * p[0] + b * p[1])
    common = _gcd3(a, b, c)
    if common != 0:
        a, b, c = a / common, b / common, c / common
    for coeff in (a, b, c):
        if coeff != 0:
            if coeff < 0:
                a, b, c = -a, -b, -c
            break
    return a, b, c


def _squared_point_line_distance(
    anchor: tuple[Fraction, ...],
    p: tuple[Fraction, ...],
    q: tuple[Fraction, ...],
) -> Fraction:
    dx = q[0] - p[0]
    dy = q[1] - p[1]
    cross = dx * (anchor[1] - p[1]) - dy * (anchor[0] - p[0])
    return (cross * cross) / (dx * dx + dy * dy)


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
        coeffs = _canonical_line_coefficients(points[i], points[j])
        expected_lines.setdefault(coeffs, []).append((i, j))
        if coeffs not in expected_distances:
            expected_distances[coeffs] = _squared_point_line_distance(
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
            raise ValueError("duplicate lines must be collapsed into one entry")
        seen_lines.add(entry_coeffs)
        if entry_coeffs not in expected_lines:
            raise ValueError("line coefficients do not match any source pair line")
        if tuple(sorted(entry.pairs)) != tuple(sorted(expected_lines[entry_coeffs])):
            raise ValueError("source pairs do not match the line's geometry")
        if entry.squared_distance.as_fraction() != expected_distances[entry_coeffs]:
            raise ValueError("squared distance does not match the source geometry")
        for i, j in entry.pairs:
            if not 0 <= i < j < point_count:
                raise ValueError("source pairs must reference valid point indices")
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
        raise ValueError("lines must be sorted by (squared_distance, coefficients)")


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
                raise ValueError(
                    "the aggregate source-pair ledger exceeds the "
                    f"{MAX_PAIRS}-pair profile bound"
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
            raise ValueError(
                "authored rational components exceed the "
                f"{MAX_PINNED_PROFILE_RESULT_BYTES}-byte aggregate result "
                "budget before parsing"
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
            raise ValueError("lines must cover exactly the set of source pairs once")
        if len(self.lines) != len(expected_lines):
            raise ValueError("lines must correspond to distinct geometric lines")
        _validate_pinned_profile_order(self.lines, expected_lines, expected_distances)

        reconstructed = tuple(
            (
                CanonicalRational.from_fraction(d),
                count,
            )
            for d, count in sorted(multiplicities.items())
        )
        if reconstructed != self.distance_multiplicities:
            raise ValueError(
                "distance multiplicities must partition the lines and be sorted"
            )
        return self
