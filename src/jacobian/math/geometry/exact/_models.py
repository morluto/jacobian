"""Typed wire contracts for exact geometry point-configuration operations."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_POINTS = 64
MAX_DIMENSION = 20
MAX_QUADRUPLE_SEARCH_POINTS = 18
"""Cap on configuration size for C(n,4) quadruple enumeration (3060 subsets)."""

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


# ---------------------------------------------------------------------------
# Configuration-wide incidence search: collinear triples and concyclic quadruples
# ---------------------------------------------------------------------------


class CollinearTriplesRequest(StrictModel):
    """Search a planar configuration for collinear triples.

    The configuration must be planar with 3..64 points (the sibling
    concyclic search admits 4..18 points under a joint work budget) and
    each coordinate must stay within the 64-digit operation-specific bound
    so that enumeration with huge Fractions stays bounded; see the
    validator for the precise bound.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Search a planar configuration for collinear triples. "
                "Requires a planar configuration with 3..64 points; "
                "each coordinate must stay within the 64-digit "
                "operation-specific bound so enumeration stays bounded."
            )
        }
    )

    configuration: PointConfiguration = Field(
        description=(
            "Planar point configuration with 3..64 points; 4..18 points "
            "for the sibling concyclic search. Each coordinate is bounded "
            "to 64 digits so that all exact determinants stay representable."
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
        for idx, point in enumerate(self.configuration.points):
            for dim, coord in enumerate(point.coordinates):
                _bounded_incidence_coordinate(coord, f"point {idx} coordinate {dim}")
        _require_distinct_incidence_coordinates(self.configuration.points)
        return self


class ConcyclicQuadruplesRequest(StrictModel):
    """Search a planar configuration for concyclic quadruples.

    Requires a planar configuration with 4..18 points whose coordinates
    each stay within the 64-digit operation-specific bound and whose joint
    work measure stays within budget: with h the largest decimal digit
    length over all coordinate numerators and denominators, C(n,4)*h must
    not exceed 65536.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Search a planar configuration for concyclic quadruples. "
                "Requires a planar configuration with 4..18 points "
                "(C(18,4)=3060); configurations with 19..64 points are "
                "rejected. Each coordinate is bounded to 64 digits, and "
                "the joint work budget C(n,4)*h <= 65536 (h = largest "
                "coordinate digit length) must hold so exact enumeration "
                "stays bounded."
            )
        }
    )

    configuration: PointConfiguration = Field(
        description=(
            "Planar point configuration with 4..18 points; the enumeration "
            "covers every unordered quadruple. Each coordinate is bounded "
            "to 64 digits, and C(n,4)*h <= 65536 (h = largest coordinate "
            "digit length) so exact enumeration stays bounded."
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


class IncidenceSearchResult(StrictModel):
    """Witnesses to a forbidden planar incidence configuration, or none.

    The result retains its source configuration so validation can replay
    every witness exactly against the certified points and certify
    completeness of the reported incidence set.
    """

    configuration: PointConfiguration
    dimension: int = Field(ge=2, le=2)
    point_count: int = Field(ge=3, le=64)
    holds: bool = Field(
        description="True iff at least one witness incidence exists.",
    )
    witnesses: tuple[tuple[int, ...], ...] = Field(default=())
    kind: Literal["COLLINEAR_TRIPLE", "CONCYCLIC_QUADRUPLE"]

    @model_validator(mode="after")
    def require_consistent_witnesses(self) -> Self:  # noqa: C901
        from fractions import Fraction
        from itertools import combinations

        if len(self.configuration.points) != self.point_count:
            raise ValueError("point_count must match the retained configuration")
        retained_dimension = len(self.configuration.points[0].coordinates)
        if retained_dimension != self.dimension:
            raise ValueError(
                "dimension must match the retained configuration coordinates"
            )
        if retained_dimension != 2:
            raise ValueError(
                "incidence replay requires a planar retained configuration"
            )
        if (
            self.kind == "CONCYCLIC_QUADRUPLE"
            and self.point_count > MAX_QUADRUPLE_SEARCH_POINTS
        ):
            raise ValueError(
                "concyclic result point_count exceeds the "
                f"{MAX_QUADRUPLE_SEARCH_POINTS}-point enumeration bound"
            )
        if self.holds and not self.witnesses:
            raise ValueError("a holds=True result must list at least one witness")
        if not self.holds and self.witnesses:
            raise ValueError("a holds=False result must list no witnesses")

        # Apply the operation's arithmetic admission to the retained
        # configuration before converting and replaying it: a deserialized
        # result must not bypass the 64-digit coordinate cap through plain
        # PointConfiguration, and its points must stay pairwise distinct.
        for idx, point in enumerate(self.configuration.points):
            for dim, coord in enumerate(point.coordinates):
                _bounded_incidence_coordinate(coord, f"point {idx} coordinate {dim}")
        _require_distinct_incidence_coordinates(self.configuration.points)
        if self.kind == "CONCYCLIC_QUADRUPLE":
            _require_concyclic_work_bound(self.configuration.points)

        expected_size = 3 if self.kind == "COLLINEAR_TRIPLE" else 4
        pts = [
            tuple(c.as_fraction() for c in point.coordinates)
            for point in self.configuration.points
        ]

        def cross(
            o: tuple[Fraction, ...], a: tuple[Fraction, ...], b: tuple[Fraction, ...]
        ) -> Fraction:
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        def det4(indices: tuple[int, ...]) -> Fraction:
            rows = []
            for idx in indices:
                x, y = pts[idx]
                rows.append((x * x + y * y, x, y, Fraction(1)))
            total = Fraction(0)
            for col in range(4):
                sub = tuple(
                    tuple(row[c] for c in range(4) if c != col) for row in rows[1:]
                )
                m = sub
                sign = 1 if col % 2 == 0 else -1
                det3 = (
                    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
                )
                total += sign * rows[0][col] * det3
            return total

        seen: set[tuple[int, ...]] = set()
        for witness in self.witnesses:
            if len(witness) != expected_size:
                raise ValueError(
                    f"{self.kind} witnesses must list {expected_size} indices"
                )
            if witness != tuple(sorted(witness)):
                raise ValueError("witness indices must be sorted ascending")
            if len(set(witness)) != len(witness):
                raise ValueError("witness indices must be distinct")
            if any(i < 0 or i >= self.point_count for i in witness):
                raise ValueError("witness index out of range")
            if witness in seen:
                raise ValueError("witnesses must be unique")
            seen.add(witness)
            if self.kind == "COLLINEAR_TRIPLE":
                i, j, k = witness
                if cross(pts[i], pts[j], pts[k]) != 0:
                    raise ValueError("a collinear witness is not actually collinear")
            else:
                i, j, k, m = witness
                # Concyclic excludes degenerate (collinear) quadruples.
                if any(
                    cross(pts[a], pts[b], pts[c]) == 0
                    for a, b, c in (
                        (i, j, k),
                        (i, j, m),
                        (i, k, m),
                        (j, k, m),
                    )
                ):
                    raise ValueError("a concyclic witness contains a collinear triple")
                if det4(witness) != 0:
                    raise ValueError("a concyclic witness is not actually concyclic")
        # Replay the bounded search to certify completeness and absence.
        expected: set[tuple[int, ...]]
        if self.kind == "COLLINEAR_TRIPLE":
            expected = set()
            for triple in combinations(range(self.point_count), 3):
                i, j, k = triple
                if cross(pts[i], pts[j], pts[k]) == 0:
                    expected.add(triple)
        else:
            expected = set()
            for quad in combinations(range(self.point_count), 4):
                i, j, k, m = quad
                if any(
                    cross(pts[a], pts[b], pts[c]) == 0
                    for a, b, c in (
                        (i, j, k),
                        (i, j, m),
                        (i, k, m),
                        (j, k, m),
                    )
                ):
                    continue
                if det4(quad) == 0:
                    expected.add(quad)
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
    "PointConfiguration",
]
