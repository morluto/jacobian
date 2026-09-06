"""Typed wire contracts for exact geometry point-configuration operations."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import ConfigDict, Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph
from jacobian.math.matrices.values import RationalMatrix


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

    configuration: PointConfiguration
    entries: tuple[DistanceMultiplicityEntry, ...]


class EuclideanOrbitProfileRequest(StrictModel):
    """Canonicalize one exact rational point configuration."""

    configuration: PointConfiguration


class EuclideanOrbitProfileResult(StrictModel):
    """Canonical unlabeled isometry and similarity forms of one source."""

    configuration: PointConfiguration
    ambient_dimension: int = Field(ge=1, le=MAX_DIMENSION)
    isometry_form: RationalMatrix
    isometry_relabeling: tuple[int, ...]
    similarity_form: RationalMatrix
    similarity_relabeling: tuple[int, ...]
    normalizing_squared_distance: CanonicalRational

    @model_validator(mode="after")
    def require_orbit_profile_shape(self) -> Self:
        size = len(self.configuration.points)
        if self.ambient_dimension != len(self.configuration.points[0].coordinates):
            raise _validation_error(
                "orbit_profile_ambient_dimension",
                "ambient_dimension must match the configuration coordinate axis",
            )
        for form, relabeling in (
            (self.isometry_form, self.isometry_relabeling),
            (self.similarity_form, self.similarity_relabeling),
        ):
            if len(form.entries) != size or any(
                len(row) != size for row in form.entries
            ):
                raise _validation_error(
                    "orbit_profile_form_shape",
                    "every orbit form must be square on the source point axis",
                )
            if tuple(sorted(relabeling)) != tuple(range(size)):
                raise _validation_error(
                    "orbit_profile_relabeling",
                    "every orbit relabeling must be a permutation of the source axis",
                )
        if self.normalizing_squared_distance.as_fraction() <= 0:
            raise _validation_error(
                "orbit_profile_normalizer",
                "normalizing_squared_distance must be positive",
            )
        return self


class DistanceGraphRequest(StrictModel):
    """Build the graph induced by a selected squared distance."""

    configuration: PointConfiguration
    target_squared_distance: CanonicalRational = Field(
        description="Nonnegative squared Euclidean distance to select.",
    )


class DistanceGraphResult(StrictModel):
    """Distance-selected graph retained with its source and target."""

    configuration: PointConfiguration
    target_squared_distance: CanonicalRational
    graph: IndexedSimpleUndirectedGraph


__all__ = [
    "DistanceGraphRequest",
    "DistanceGraphResult",
    "DistanceMultiplicityEntry",
    "DistanceProfileRequest",
    "DistanceProfileResult",
    "EuclideanOrbitProfileRequest",
    "EuclideanOrbitProfileResult",
    "LabelledRationalPoint",
    "PinnedLineDistanceRequest",
    "PinnedLineDistanceResult",
    "PinnedLineEntry",
    "PointConfiguration",
]


# ---------------------------------------------------------------------------
# Pinned line-distance profile
# ---------------------------------------------------------------------------

MAX_PINNED_PROFILE_AUTHORED_RATIONAL_CHARACTERS = 10 * 1024 * 1024
"""Raw rational-character preflight for independently authored profiles."""


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
            "The complete pair ledger is bounded directly by its cardinality."
        ),
        json_schema_extra={
            "coordinate_digit_bound": COORDINATE_DIGITS,
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


class PinnedLineDistanceResult(StrictModel):
    """Complete pinned line-distance profile for a point configuration.

    The result retains its source ``configuration`` and ``anchor`` for
    downstream composition. Parsing checks its canonical wire shape; the
    producing kernel establishes the geometric profile.
    """

    configuration: PinnedLineConfiguration = Field(
        description=(
            "Source planar point configuration, retained for downstream composition."
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
        authored_rational_characters = 0
        for line in lines:
            pair_total, rational_bytes = _count_line_ledger(line)
            total += pair_total
            authored_rational_characters += rational_bytes
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
                    authored_rational_characters += _rational_component_bytes(entry[0])
        # Authored rational components are bounded by the same aggregate
        # result budget as the parsed profile: every valid entry needs at
        # least two characters per canonical rational, so any payload whose
        # raw numerator/denominator characters alone approach the budget is
        # forged padding that must be rejected BEFORE nested parsing.
        if (
            authored_rational_characters
            > MAX_PINNED_PROFILE_AUTHORED_RATIONAL_CHARACTERS
        ):
            raise _validation_error(
                "authored_rational_components_exceed_f_max",
                "authored rational components exceed the "
                "raw rational-character preflight bound before parsing",
            )
        return data

    @model_validator(mode="after")
    def require_consistent_profile(self) -> Self:
        if self.dimension != 2 or any(
            len(point.coordinates) != self.dimension
            for point in self.configuration.points
        ):
            raise _validation_error(
                "retained_configuration_a_planar_configuration_two",
                "retained configuration must be a planar configuration "
                "(exactly two coordinates per point)",
            )
        if self.point_count != len(self.configuration.points):
            raise _validation_error(
                "point_count_retained_configuration",
                "point_count must match the retained configuration",
            )
        seen_pairs = [pair for entry in self.lines for pair in entry.pairs]
        if any(
            not 0 <= first < second < self.point_count for first, second in seen_pairs
        ):
            raise _validation_error(
                "source_pairs_reference_valid_point_indices",
                "source pairs must reference valid point indices",
            )
        expected_pair_count = self.point_count * (self.point_count - 1) // 2
        if len(seen_pairs) != expected_pair_count or len(seen_pairs) != len(
            set(seen_pairs)
        ):
            raise _validation_error(
                "lines_cover_set_source_pairs_once",
                "lines must cover exactly the set of source pairs once",
            )
        coefficients = tuple(
            tuple(value.as_fraction() for value in entry.line_coefficients)
            for entry in self.lines
        )
        if len(coefficients) != len(set(coefficients)):
            raise _validation_error(
                "duplicate_lines_collapsed_entry",
                "duplicate lines must be collapsed into one entry",
            )
        if (
            tuple(
                sorted(
                    self.lines,
                    key=lambda entry: (
                        entry.squared_distance.as_fraction(),
                        tuple(value.as_fraction() for value in entry.line_coefficients),
                    ),
                )
            )
            != self.lines
        ):
            raise _validation_error(
                "lines_sorted_squared_distance_coefficients",
                "lines must be sorted by (squared_distance, coefficients)",
            )
        if any(count <= 0 for _, count in self.distance_multiplicities):
            raise _validation_error(
                "distance_multiplicities_positive",
                "distance multiplicities must be positive",
            )
        distances = tuple(
            value.as_fraction() for value, _ in self.distance_multiplicities
        )
        if distances != tuple(sorted(distances)) or len(distances) != len(
            set(distances)
        ):
            raise _validation_error(
                "distance_multiplicities_sorted",
                "distance multiplicities must be sorted by distinct distance",
            )
        if sum(count for _, count in self.distance_multiplicities) != len(self.lines):
            raise _validation_error(
                "distance_multiplicities_partition_lines_sorted",
                "distance multiplicities must partition the lines and be sorted",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        configuration: PinnedLineConfiguration,
        anchor: tuple[CanonicalRational, ...],
        *,
        lines: tuple[PinnedLineEntry, ...],
        distance_multiplicities: tuple[tuple[CanonicalRational, int], ...],
    ) -> Self:
        """Build a result after the admitted profile kernel established it."""

        return cls.model_construct(
            configuration=configuration,
            anchor=anchor,
            dimension=2,
            point_count=len(configuration.points),
            lines=lines,
            distance_multiplicities=distance_multiplicities,
        )
