"""Typed contracts for exact rational convex-polytope local motion."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import AfterValidator, Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"convex_geometry.{reason}", message)


MAX_CONVEX_DIMENSION = 8
"""Ambient-dimension bound for the local-motion operation."""

MAX_CONVEX_INEQUALITIES = 64
"""Maximum labelled inequalities in one bounded H-polytope."""

MAX_CONVEX_LABEL_LENGTH = 64
"""Maximum length of an axis, inequality, or point identifier."""

MAX_CONVEX_COMPONENT_DIGITS = 1_024
"""Per-component digit bound for inequality, point, and direction entries."""


def _require_scalar_label(value: str) -> str:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise _validation_error(
            "unicode_scalar_label", "labels must contain only Unicode scalar values"
        )
    return value


ConvexAxis = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_CONVEX_LABEL_LENGTH, strict=True),
    AfterValidator(_require_scalar_label),
]
"""One coordinate identifier in an ordered labelled rational space."""


class ConvexSpace(StrictModel):
    """One ordered labelled rational coordinate space."""

    axes: tuple[ConvexAxis, ...] = Field(min_length=1, max_length=MAX_CONVEX_DIMENSION)

    @model_validator(mode="after")
    def require_distinct_axes(self) -> Self:
        if len(set(self.axes)) != len(self.axes):
            raise _validation_error(
                "coordinate_axes_unique", "coordinate axes must be unique"
            )
        return self


class ConvexInequality(StrictModel):
    """One labelled outward-normal inequality ``<a, x> <= b``."""

    inequality_id: Annotated[str, AfterValidator(_require_scalar_label)] = Field(
        min_length=1, max_length=MAX_CONVEX_LABEL_LENGTH
    )
    normal: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_CONVEX_DIMENSION
    )
    bound: CanonicalRational


class ConvexHPolytope(StrictModel):
    """A bounded H-polytope ``{x : A x <= b}`` with labelled inequalities."""

    space: ConvexSpace
    inequalities: tuple[ConvexInequality, ...] = Field(
        min_length=1, max_length=MAX_CONVEX_INEQUALITIES
    )

    @model_validator(mode="after")
    def require_canonical_inequalities(self) -> Self:
        dimension = len(self.space.axes)
        ids = tuple(inequality.inequality_id for inequality in self.inequalities)
        if tuple(sorted(ids)) != ids or len(set(ids)) != len(ids):
            raise _validation_error(
                "inequality_ids",
                "inequality IDs must be unique and strictly ordered",
            )
        if any(len(inequality.normal) != dimension for inequality in self.inequalities):
            raise _validation_error(
                "inequality_dimension",
                "every inequality normal must use the polytope coordinate axis",
            )
        return self


class RationalConvexPoint(StrictModel):
    """One rational point on a labelled convex coordinate space."""

    space: ConvexSpace
    coordinates: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_CONVEX_DIMENSION
    )

    @model_validator(mode="after")
    def require_matching_axis(self) -> Self:
        if len(self.coordinates) != len(self.space.axes):
            raise _validation_error(
                "point_dimension",
                "point coordinates must use the declared coordinate axis",
            )
        return self


class RationalConvexDirection(StrictModel):
    """One rational direction on a labelled convex coordinate space."""

    space: ConvexSpace
    components: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_CONVEX_DIMENSION
    )

    @model_validator(mode="after")
    def require_matching_axis(self) -> Self:
        if len(self.components) != len(self.space.axes):
            raise _validation_error(
                "direction_dimension",
                "direction components must use the declared coordinate axis",
            )
        return self


MotionKind = Literal["STRICTLY_INWARD", "TANGENT", "OUTWARD"]
AggregateMotion = Literal["ILLUMINATES_STRICTLY", "TANGENT_OR_PARTIAL", "MOVES_OUTSIDE"]
PointState = Literal["INTERIOR", "BOUNDARY", "OUTSIDE"]


class ActiveMotionRow(StrictModel):
    """Exact normal-direction pairing of one active inequality."""

    inequality_id: str = Field(min_length=1, max_length=MAX_CONVEX_LABEL_LENGTH)
    dot: CanonicalRational
    kind: MotionKind


class SlackRow(StrictModel):
    """Exact slack ``b_i - <a_i, x>`` of one inequality at the query point."""

    inequality_id: str = Field(min_length=1, max_length=MAX_CONVEX_LABEL_LENGTH)
    slack: CanonicalRational
    active: bool


class DirectionLocalMotionRequest(StrictModel):
    """Decide strict local motion of a boundary point along a direction.

    The polytope, point, and direction must declare one identical labelled
    coordinate space. The point must lie on the polytope boundary and the
    direction must be nonzero; both are admitted by the shared owner helper
    before any ledger is materialized. Each reduced numerator and
    denominator carries at most ``MAX_CONVEX_COMPONENT_DIGITS`` digits.
    """

    polytope: ConvexHPolytope = Field(
        description=(
            "Bounded H-polytope with labelled outward-normal inequalities; "
            "every normal component and bound carries at most "
            f"{MAX_CONVEX_COMPONENT_DIGITS} digits."
        )
    )
    point: RationalConvexPoint = Field(
        description=(
            "Query point on the polytope boundary, on the identical labelled "
            "coordinate space."
        )
    )
    direction: RationalConvexDirection = Field(
        description=(
            "Nonzero rational direction on the identical labelled coordinate "
            "space; positive rescaling preserves the result."
        )
    )


class DirectionLocalMotionResult(StrictModel):
    """Complete exact slack ledger and per-active-inequality motion profile."""

    point_state: PointState
    slacks: tuple[SlackRow, ...] = Field(
        min_length=1, max_length=MAX_CONVEX_INEQUALITIES
    )
    motions: tuple[ActiveMotionRow, ...] = Field(
        min_length=1, max_length=MAX_CONVEX_INEQUALITIES
    )
    aggregate: AggregateMotion

    @model_validator(mode="after")
    def require_ledger_consistency(self) -> Self:
        slack_ids = tuple(row.inequality_id for row in self.slacks)
        if slack_ids != tuple(sorted(slack_ids)) or len(set(slack_ids)) != len(
            slack_ids
        ):
            raise _validation_error(
                "slack_order", "slack rows must be unique and sorted by inequality ID"
            )
        active_ids = tuple(row.inequality_id for row in self.slacks if row.active)
        motion_ids = tuple(row.inequality_id for row in self.motions)
        if motion_ids != active_ids:
            raise _validation_error(
                "motion_coverage",
                "motion rows must cover exactly the active inequalities in order",
            )
        kinds = {row.kind for row in self.motions}
        if self.point_state != "BOUNDARY":
            raise _validation_error(
                "boundary_result",
                "a local-motion result always carries a boundary point",
            )
        if self.aggregate == "ILLUMINATES_STRICTLY":
            expected: set[str] = {"STRICTLY_INWARD"}
        elif self.aggregate == "MOVES_OUTSIDE":
            expected = {"STRICTLY_INWARD", "TANGENT", "OUTWARD"}
            if "OUTWARD" not in kinds:
                raise _validation_error(
                    "aggregate_motion",
                    "MOVES_OUTSIDE requires at least one outward inequality",
                )
        else:
            expected = {"STRICTLY_INWARD", "TANGENT"}
            if "OUTWARD" in kinds:
                raise _validation_error(
                    "aggregate_motion",
                    "TANGENT_OR_PARTIAL excludes outward inequalities",
                )
            if kinds == {"STRICTLY_INWARD"}:
                raise _validation_error(
                    "aggregate_motion",
                    "all-strict motion must aggregate as ILLUMINATES_STRICTLY",
                )
        if self.aggregate != "MOVES_OUTSIDE" and kinds - expected:
            raise _validation_error(
                "aggregate_motion",
                "aggregate motion must match the per-inequality kinds",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        slacks: tuple[SlackRow, ...],
        motions: tuple[ActiveMotionRow, ...],
        aggregate: AggregateMotion,
    ) -> Self:
        """Build a trusted kernel outcome without replaying its ledger."""

        return cls.model_construct(
            point_state="BOUNDARY",
            slacks=slacks,
            motions=motions,
            aggregate=aggregate,
        )


__all__ = [
    "MAX_CONVEX_COMPONENT_DIGITS",
    "MAX_CONVEX_DIMENSION",
    "MAX_CONVEX_INEQUALITIES",
    "MAX_CONVEX_LABEL_LENGTH",
    "ActiveMotionRow",
    "AggregateMotion",
    "ConvexHPolytope",
    "ConvexInequality",
    "ConvexSpace",
    "DirectionLocalMotionRequest",
    "DirectionLocalMotionResult",
    "MotionKind",
    "PointState",
    "RationalConvexDirection",
    "RationalConvexPoint",
    "SlackRow",
]
