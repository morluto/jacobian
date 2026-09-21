"""Native exact convex-polytope local-motion operation."""

from __future__ import annotations

from fractions import Fraction
from typing import NoReturn

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.convex._models import (
    ActiveMotionRow,
    AggregateMotion,
    ConvexHPolytope,
    DirectionLocalMotionResult,
    RationalConvexDirection,
    RationalConvexPoint,
    SlackRow,
)


def _reject(location: str, code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=(location,),
        code=code,
        message=message,
    )


def _refuse(code: str, message: str) -> NoReturn:
    raise OperationResourceAdmissionError(
        location=("polytope",),
        code=code,
        message=message,
    )


def _admit_local_motion(
    polytope: ConvexHPolytope,
    point: RationalConvexPoint,
    direction: RationalConvexDirection,
) -> None:
    """Enforce the shared execution envelope for native and catalog calls."""

    if not isinstance(polytope, ConvexHPolytope):
        _reject(
            "polytope",
            "convex_geometry.local_motion.polytope_not_an_h_polytope",
            "local-motion source must be a labelled convex H-polytope value",
        )
    if not isinstance(point, RationalConvexPoint):
        _reject(
            "point",
            "convex_geometry.local_motion.point_not_a_convex_point",
            "local-motion point must be a labelled convex point value",
        )
    if not isinstance(direction, RationalConvexDirection):
        _reject(
            "direction",
            "convex_geometry.local_motion.direction_not_a_convex_direction",
            "local-motion direction must be a labelled convex direction value",
        )
    if not isinstance(direction.space, type(polytope.space)) or (
        tuple(direction.space.axes) != tuple(polytope.space.axes)
        or tuple(point.space.axes) != tuple(polytope.space.axes)
    ):
        _reject(
            "point",
            "convex_geometry.local_motion.space_mismatch",
            "polytope, point, and direction must use one identical coordinate space",
        )
    try:
        for inequality in polytope.inequalities:
            for component in (*inequality.normal, inequality.bound):
                require_bounded_rational(
                    component,
                    max_digits=1_024,
                    label="H-polytope coefficient",
                )
        for component in (*point.coordinates, *direction.components):
            require_bounded_rational(
                component, max_digits=1_024, label="point/direction component"
            )
    except ValueError as exc:
        _refuse(
            "convex_geometry.local_motion.coefficient_over_envelope",
            str(exc),
        )


def direction_local_motion(
    polytope: ConvexHPolytope,
    point: RationalConvexPoint,
    direction: RationalConvexDirection,
) -> DirectionLocalMotionResult:
    """Decide exact strict local motion of a boundary point along a direction.

    For every active inequality (exact slack zero) the kernel reports the
    exact pairing ``<a_i, u>`` with kind ``STRICTLY_INWARD`` iff negative,
    ``TANGENT`` iff zero, ``OUTWARD`` iff positive. The aggregate is
    ``ILLUMINATES_STRICTLY`` iff every active inequality is strictly inward,
    ``MOVES_OUTSIDE`` iff any active inequality is outward, and
    ``TANGENT_OR_PARTIAL`` otherwise. Points outside the polytope, interior
    points, and zero directions are domain rejections, never motion results.
    """

    _admit_local_motion(polytope, point, direction)
    coords = [c.as_fraction() for c in point.coordinates]
    steps = [c.as_fraction() for c in direction.components]
    if all(step == 0 for step in steps):
        _reject(
            "direction",
            "convex_geometry.local_motion.zero_direction",
            "local-motion direction must be nonzero",
        )
    slack_rows: list[SlackRow] = []
    for inequality in polytope.inequalities:
        normal = [c.as_fraction() for c in inequality.normal]
        bound = inequality.bound.as_fraction()
        slack = bound - sum(a * x for a, x in zip(normal, coords, strict=True))
        slack_rows.append(
            SlackRow(
                inequality_id=inequality.inequality_id,
                slack=CanonicalRational.from_fraction(slack),
                active=slack == 0,
            )
        )
    if any(row.slack.as_fraction() < 0 for row in slack_rows):
        _reject(
            "point",
            "convex_geometry.local_motion.point_outside_polytope",
            "local-motion points outside the polytope are rejected",
        )
    active = [row for row in slack_rows if row.active]
    if not active:
        _reject(
            "point",
            "convex_geometry.local_motion.point_interior",
            "local-motion requires a boundary point; interior points are rejected",
        )
    by_id = {
        inequality.inequality_id: inequality for inequality in polytope.inequalities
    }
    motions: list[ActiveMotionRow] = []
    for row in active:
        normal = [c.as_fraction() for c in by_id[row.inequality_id].normal]
        dot: Fraction = sum(
            (a * u for a, u in zip(normal, steps, strict=True)), Fraction(0)
        )
        kind = "STRICTLY_INWARD" if dot < 0 else ("TANGENT" if dot == 0 else "OUTWARD")
        motions.append(
            ActiveMotionRow(
                inequality_id=row.inequality_id,
                dot=CanonicalRational.from_fraction(dot),
                kind=kind,  # type: ignore[arg-type]
            )
        )
    kinds = {row.kind for row in motions}
    aggregate: AggregateMotion
    if kinds == {"STRICTLY_INWARD"}:
        aggregate = "ILLUMINATES_STRICTLY"
    elif "OUTWARD" in kinds:
        aggregate = "MOVES_OUTSIDE"
    else:
        aggregate = "TANGENT_OR_PARTIAL"
    return DirectionLocalMotionResult._from_kernel(
        slacks=tuple(slack_rows),
        motions=tuple(motions),
        aggregate=aggregate,
    )


__all__ = ["direction_local_motion"]
