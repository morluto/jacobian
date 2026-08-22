"""Typed wire contracts for exact geometry point-configuration operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_POINTS = 64
MAX_DIMENSION = 20
COORDINATE_DIGITS = 256
"""Per-coordinate digit bound for pinned line-distance profile so squared distances stay representable."""


def _require_bounded_point_configuration(
    configuration: PointConfiguration, anchor: tuple[CanonicalRational, ...] | None = None
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


__all__ = [
    "DistanceGraphRequest",
    "DistanceGraphResult",
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


class PinnedLineDistanceRequest(StrictModel):
    """Compute distances from an anchor to all pair-spanned lines.

    The configuration must be planar (dimension 2) with distinct point
    coordinates; two identically-located points do not span a line. Both
    configuration coordinates and anchor coordinates are bounded to at most
    256 decimal digits per component so all derived squared distances remain
    representable as ``CanonicalRational`` (canonical limit 32,768 digits).
    """

    configuration: PointConfiguration = Field(
        description=(
            "Planar point configuration (dimension 2) with distinct coordinates; "
            "all points must have distinct locations and at most 64 points, "
            "each coordinate at most 256 digits for pinned-line admission."
        )
    )
    anchor: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        description=(
            "Planar rational anchor point (dimension 2); both coordinates at most "
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
        if len(self.anchor) != 2:
            raise ValueError("the anchor must be a planar rational point")
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
        return self


class PinnedLineEntry(StrictModel):
    """One pair-spanned line with its canonical equation and source pairs."""

    line_coefficients: tuple[CanonicalRational, ...] = Field(min_length=3, max_length=3)
    squared_distance: CanonicalRational
    pairs: tuple[tuple[int, int], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_sorted_pairs(self) -> Self:
        for i, j in self.pairs:
            if not i < j:
                raise ValueError("source pairs must be ordered (i < j)")
        if len(set(self.pairs)) != len(self.pairs):
            raise ValueError("source pairs must be unique")
        if self.squared_distance.as_fraction() < 0:
            raise ValueError("squared distance must be nonnegative")
        return self


class PinnedLineDistanceResult(StrictModel):
    """Complete pinned line-distance profile for a point configuration.

    The result retains its source ``configuration`` and ``anchor`` so validation
    can replay the defining geometry: every pair-spanned line is recomputed
    canonically from the retained points and its squared distance from the
    retained anchor is verified.
    """

    configuration: PointConfiguration = Field(
        description=(
            "Source planar point configuration with distinct coordinates; "
            "retained for result binding and replay."
        )
    )
    anchor: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        description=(
            "Retained planar anchor point; both coordinates at most 256 digits."
        ),
    )
    dimension: int = Field(ge=2, le=2)
    point_count: int = Field(ge=2, le=MAX_POINTS)
    lines: tuple[PinnedLineEntry, ...]
    distance_multiplicities: tuple[tuple[CanonicalRational, int], ...]

    @model_validator(mode="after")
    def require_consistent_profile(self) -> Self:  # noqa: C901
        from itertools import combinations
        from math import gcd

        _require_bounded_point_configuration(self.configuration, self.anchor)

        # Bind the profile to its source geometry before any pair accounting.
        if len(self.configuration.points) != self.point_count:
            raise ValueError("point_count must match the retained configuration")
        if len(self.anchor) != 2:
            raise ValueError("the retained anchor must be a planar rational point")
        coords = {
            tuple(c.as_fraction() for c in pt.coordinates)
            for pt in self.configuration.points
        }
        if len(coords) != len(self.configuration.points):
            raise ValueError(
                "retained configuration points must have distinct coordinates"
            )

        # Cap point_count before enumerating expected pairs (schema-visible too).
        if self.point_count > MAX_POINTS:
            raise ValueError("point_count exceeds the configuration bound")

        # Recompute the exact geometry from the retained source.
        points = [
            tuple(c.as_fraction() for c in pt.coordinates)
            for pt in self.configuration.points
        ]
        anchor = tuple(c.as_fraction() for c in self.anchor)

        def _gcd3(a: Fraction, b: Fraction, c: Fraction) -> Fraction:
            if a == 0 and b == 0 and c == 0:
                return Fraction(0)
            nums = [a.numerator, b.numerator, c.numerator]
            dens = [a.denominator, b.denominator, c.denominator]
            common_den = 1
            for d in dens:
                common_den = common_den * d // gcd(common_den, d)
            scaled = [n * (common_den // d) for n, d in zip(nums, dens, strict=True)]
            g = 0
            for v in scaled:
                g = gcd(g, abs(v))
            if g == 0:
                return Fraction(0)
            return Fraction(g, common_den)

        def _canonical_line_coefficients(
            p: tuple[Fraction, ...], q: tuple[Fraction, ...]
        ) -> tuple[Fraction, Fraction, Fraction]:
            dx = q[0] - p[0]
            dy = q[1] - p[1]
            a = dy
            b = -dx
            c = -(a * p[0] + b * p[1])
            g = _gcd3(a, b, c)
            if g != 0:
                a, b, c = a / g, b / g, c / g
            for coeff in (a, b, c):
                if coeff != 0:
                    if coeff < 0:
                        a, b, c = -a, -b, -c
                    break
            return a, b, c

        def _squared_point_line_distance(
            anc: tuple[Fraction, ...],
            p: tuple[Fraction, ...],
            q: tuple[Fraction, ...],
        ) -> Fraction:
            dx = q[0] - p[0]
            dy = q[1] - p[1]
            cross = dx * (anc[1] - p[1]) - dy * (anc[0] - p[0])
            norm_sq = dx * dx + dy * dy
            return (cross * cross) / norm_sq

        expected_lines: dict[tuple[Fraction, Fraction, Fraction], list[tuple[int, int]]] = {}
        expected_distances: dict[tuple[Fraction, Fraction, Fraction], Fraction] = {}
        for i, j in combinations(range(self.point_count), 2):
            coeffs = _canonical_line_coefficients(points[i], points[j])
            expected_lines.setdefault(coeffs, []).append((i, j))
            if coeffs not in expected_distances:
                expected_distances[coeffs] = _squared_point_line_distance(
                    anchor, points[i], points[j]
                )

        expected_pairs = sorted(combinations(range(self.point_count), 2))
        seen_pairs: list[tuple[int, int]] = []
        seen_lines: set[tuple[Fraction, ...]] = set()
        mult: dict[Fraction, int] = {}
        # Map expected coeffs for quick lookup of exact distance/pairs.
        for entry in self.lines:
            coeffs = tuple(c.as_fraction() for c in entry.line_coefficients)
            if coeffs in seen_lines:
                raise ValueError("duplicate lines must be collapsed into one entry")
            seen_lines.add(coeffs)
            # Must be a genuine pair-spanned line from the source.
            if coeffs not in expected_lines:
                raise ValueError("line coefficients do not match any source pair line")
            # Pairs must exactly match the source pairs that generate this line.
            if tuple(sorted(entry.pairs)) != tuple(sorted(expected_lines[coeffs])):
                raise ValueError("source pairs do not match the line's geometry")
            # Squared distance must match the exact anchor-to-line distance.
            expected_d = expected_distances[coeffs]
            if entry.squared_distance.as_fraction() != expected_d:
                raise ValueError("squared distance does not match the source geometry")
            for i, j in entry.pairs:
                if not 0 <= i < j < self.point_count:
                    raise ValueError("source pairs must reference valid point indices")
                seen_pairs.append((i, j))
            d = entry.squared_distance.as_fraction()
            mult[d] = mult.get(d, 0) + 1

        if sorted(seen_pairs) != expected_pairs or len(seen_pairs) != len(
            set(seen_pairs)
        ):
            raise ValueError("lines must cover exactly the set of source pairs once")
        if len(self.lines) != len(expected_lines):
            raise ValueError("lines must correspond to distinct geometric lines")

        # Enforce deterministic ordering: sorted by (squared_distance, coefficients).
        ordered_coeffs = sorted(
            expected_lines.keys(), key=lambda c: (expected_distances[c], c)
        )
        actual_coeffs = [
            tuple(c.as_fraction() for c in e.line_coefficients) for e in self.lines
        ]
        if actual_coeffs != ordered_coeffs:
            raise ValueError("lines must be sorted by (squared_distance, coefficients)")

        reconstructed = tuple(
            (
                CanonicalRational.from_fraction(d),
                count,
            )
            for d, count in sorted(mult.items())
        )
        if reconstructed != self.distance_multiplicities:
            raise ValueError(
                "distance multiplicities must partition the lines and be sorted"
            )
        return self
