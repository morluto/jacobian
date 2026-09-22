"""Native exact convex-polytope local-motion operation."""

from __future__ import annotations

from fractions import Fraction
from typing import NoReturn

from sympy import Matrix, Rational

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.convex._models import (
    MAX_CONVEX_COMPONENT_DIGITS,
    MAX_CONVEX_DIMENSION,
    MAX_CONVEX_INEQUALITIES,
    ActiveFacetProfileResult,
    ActiveMotionRow,
    AggregateMotion,
    ConvexHPolytope,
    ConvexInequality,
    ConvexSpace,
    DirectionCoverageCell,
    DirectionCoverageRequest,
    DirectionCoverageResult,
    DirectionLocalMotionResult,
    RationalConvexDirection,
    RationalConvexPoint,
    SlackRow,
)
from jacobian.math.geometry.polytopes._rational_geometry import (
    RecessionConeComputationError,
    recession_cone_is_trivial,
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


def _admit_bounded_polytope(polytope: ConvexHPolytope) -> None:
    """Admit the H-representation's bounded domain before local kernels."""

    if not isinstance(polytope, ConvexHPolytope):
        _reject(
            "polytope",
            "convex_geometry.polytope.invalid_type",
            "source must be a canonical convex H-polytope value",
        )
    if (
        not isinstance(polytope.space, ConvexSpace)
        or not isinstance(polytope.inequalities, tuple)
        or not isinstance(polytope.space.axes, tuple)
    ):
        _reject(
            "polytope",
            "convex_geometry.polytope.malformed",
            "polytope space and inequalities must be canonical values",
        )
    dimension = len(polytope.space.axes)
    if dimension > MAX_CONVEX_DIMENSION:
        _refuse(
            "convex_geometry.polytope.dimension",
            f"polytope dimension exceeds {MAX_CONVEX_DIMENSION}",
        )
    if len(polytope.inequalities) > MAX_CONVEX_INEQUALITIES:
        _refuse(
            "convex_geometry.polytope.inequality_count",
            f"polytope has more than {MAX_CONVEX_INEQUALITIES} inequalities",
        )
    # Native callers can supply model_construct values.  Reparse the complete
    # nested carrier before reading axes, IDs, or rational components so the
    # backend never sees a forged parent or an over-bound representation.
    try:
        payload = polytope.model_dump(mode="python")
        canonical = ConvexHPolytope.model_validate(payload)
    except Exception:
        _reject(
            "polytope",
            "convex_geometry.polytope.malformed",
            "polytope space, axes, inequalities, and rationals must be canonical",
        )
    if canonical.model_dump(mode="python") != payload:
        _reject(
            "polytope",
            "convex_geometry.polytope.malformed",
            "polytope nested carriers must be canonically validated",
        )
    normals = []
    try:
        for inequality in polytope.inequalities:
            if (
                not isinstance(inequality, ConvexInequality)
                or len(inequality.normal) != dimension
                or any(
                    not isinstance(component, CanonicalRational)
                    for component in (*inequality.normal, inequality.bound)
                )
            ):
                _reject(
                    "polytope",
                    "convex_geometry.polytope.malformed",
                    "inequalities must carry one exact normal per coordinate axis",
                )
            if all(component.as_fraction() == 0 for component in inequality.normal):
                _reject(
                    "polytope",
                    "convex_geometry.polytope.zero_normal",
                    "inequality normals must be nonzero",
                )
            for component in (*inequality.normal, inequality.bound):
                require_bounded_rational(
                    component,
                    max_digits=MAX_CONVEX_COMPONENT_DIGITS,
                    label="H-polytope coefficient",
                )
            normals.append(
                [
                    Rational(component.num, component.den)
                    for component in inequality.normal
                ]
            )
    except OperationDomainValidationError:
        raise
    except ValueError as exc:
        _refuse("convex_geometry.polytope.coefficient_over_envelope", str(exc))
    try:
        bounded = recession_cone_is_trivial(normals, dimension)
    except RecessionConeComputationError as exc:
        raise OperationResourceAdmissionError(
            location=("polytope",),
            code="convex_geometry.polytope.boundedness_undecidable",
            message="exact boundedness admission could not be completed",
        ) from exc
    if not bounded:
        _reject(
            "polytope",
            "convex_geometry.polytope.unbounded",
            "the H-representation must define a bounded polytope",
        )


def _admit_point_carrier(
    point: RationalConvexPoint, *, location: str, code: str
) -> RationalConvexPoint:
    """Reparse a nested point before any tuple arithmetic or axis access."""

    if not isinstance(point, RationalConvexPoint):
        _reject(location, code, "expected a canonical rational convex point")
    try:
        payload = point.model_dump(mode="python")
        canonical = RationalConvexPoint.model_validate(payload)
    except Exception:
        _reject(location, code, "point space, axes, and coordinates are malformed")
    if canonical.model_dump(mode="python") != payload:
        _reject(location, code, "point must be a canonical nested carrier")
    return canonical


def _admit_direction_carrier(
    direction: RationalConvexDirection, *, location: str, code: str
) -> RationalConvexDirection:
    if not isinstance(direction, RationalConvexDirection):
        _reject(location, code, "expected a canonical rational convex direction")
    try:
        payload = direction.model_dump(mode="python")
        canonical = RationalConvexDirection.model_validate(payload)
    except Exception:
        _reject(location, code, "direction space, axes, and components are malformed")
    if canonical.model_dump(mode="python") != payload:
        _reject(location, code, "direction must be a canonical nested carrier")
    return canonical


def _admit_local_motion(
    polytope: ConvexHPolytope,
    point: RationalConvexPoint,
    direction: RationalConvexDirection,
    *,
    check_boundedness: bool = True,
) -> None:
    """Enforce the shared execution envelope for native and catalog calls."""

    if check_boundedness:
        _admit_bounded_polytope(polytope)
    point = _admit_point_carrier(
        point,
        location="point",
        code="convex_geometry.local_motion.point_malformed",
    )
    direction = _admit_direction_carrier(
        direction,
        location="direction",
        code="convex_geometry.local_motion.direction_malformed",
    )
    if not isinstance(polytope.space, ConvexSpace) or (
        direction.space != polytope.space or point.space != polytope.space
    ):
        _reject(
            "point",
            "convex_geometry.local_motion.space_mismatch",
            "polytope, point, and direction must use one identical coordinate space",
        )
    try:
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
    """Admit and decide exact local motion for one boundary query."""

    _admit_local_motion(polytope, point, direction)
    return _direction_local_motion_kernel(polytope, point, direction)


def _direction_local_motion_kernel(
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
        polytope=polytope,
        point=point,
        direction=direction,
        slacks=tuple(slack_rows),
        motions=tuple(motions),
        aggregate=aggregate,
    )


def _admit_axes(polytope: ConvexHPolytope, point: RationalConvexPoint) -> None:
    if not isinstance(polytope, ConvexHPolytope):
        _reject(
            "polytope",
            "convex_geometry.profile.invalid_parent",
            "polytope and point must be canonical convex values",
        )
    point = _admit_point_carrier(
        point,
        location="point",
        code="convex_geometry.profile.point_malformed",
    )
    if point.space != polytope.space:
        _reject(
            "point",
            "convex_geometry.profile.space_mismatch",
            "point and polytope must use one identical labelled space",
        )
    try:
        for value in point.coordinates:
            require_bounded_rational(
                value,
                max_digits=MAX_CONVEX_COMPONENT_DIGITS,
                label="convex profile point component",
            )
    except ValueError as exc:
        _refuse("convex_geometry.profile.coefficient_over_envelope", str(exc))


def active_facet_profile(
    polytope: ConvexHPolytope, point: RationalConvexPoint
) -> ActiveFacetProfileResult:
    _admit_bounded_polytope(polytope)
    _admit_axes(polytope, point)
    coords = [c.as_fraction() for c in point.coordinates]
    rows = []
    for inequality in polytope.inequalities:
        slack = inequality.bound.as_fraction() - sum(
            a.as_fraction() * x for a, x in zip(inequality.normal, coords, strict=True)
        )
        rows.append(
            SlackRow(
                inequality_id=inequality.inequality_id,
                slack=CanonicalRational.from_fraction(slack),
                active=slack == 0,
            )
        )
        if slack < 0:
            _reject(
                "point",
                "convex_geometry.profile.point_outside_polytope",
                "point is outside the polytope",
            )
    active = [row.inequality_id for row in rows if row.active]
    if not active:
        _reject(
            "point",
            "convex_geometry.profile.point_interior",
            "profile requires a boundary point",
        )
    by_id = {row.inequality_id: row for row in polytope.inequalities}
    rank = int(
        Matrix(
            [[c.as_fraction() for c in by_id[label].normal] for label in active]
        ).rank()
    )
    return ActiveFacetProfileResult(
        polytope=polytope,
        point=point,
        slacks=tuple(rows),
        active_inequality_ids=tuple(active),
        active_normal_rank=rank,
    )


def direction_set_coverage(
    request: DirectionCoverageRequest,
) -> DirectionCoverageResult:
    if not isinstance(request, DirectionCoverageRequest):
        _reject(
            "request",
            "convex_geometry.coverage.request_type",
            "coverage requires its canonical request value",
        )
    try:
        payload = request.model_dump(mode="python")
        canonical_request = DirectionCoverageRequest.model_validate(payload)
    except Exception:
        _reject(
            "request",
            "convex_geometry.coverage.malformed",
            "coverage request contains malformed nested carriers",
        )
    if canonical_request.model_dump(mode="python") != payload:
        _reject(
            "request",
            "convex_geometry.coverage.malformed",
            "coverage request must use canonical nested carriers",
        )
    request = canonical_request
    polytope = request.polytope
    _admit_bounded_polytope(polytope)
    if len(request.points) * len(request.directions) > 4096:
        _refuse(
            "convex_geometry.coverage.matrix_size",
            "point-direction matrix exceeds the admitted envelope",
        )
    points = tuple(sorted(request.points, key=lambda row: row.point_id))
    directions = tuple(sorted(request.directions, key=lambda row: row.direction_id))
    if len({row.point_id for row in points}) != len(points) or len(
        {row.direction_id for row in directions}
    ) != len(directions):
        _reject(
            "request",
            "convex_geometry.coverage.duplicate_id",
            "point and direction IDs must be unique",
        )
    if tuple(row.point.space.axes for row in points) != tuple(
        polytope.space.axes for _ in points
    ):
        _reject(
            "points",
            "convex_geometry.coverage.space_mismatch",
            "all points must use the polytope space",
        )
    if tuple(row.direction.space.axes for row in directions) != tuple(
        polytope.space.axes for _ in directions
    ):
        _reject(
            "directions",
            "convex_geometry.coverage.space_mismatch",
            "all directions must use the polytope space",
        )
    cells = []
    strict = set()
    witnesses: dict[str, str] = {}
    for point in points:
        for direction in directions:
            _admit_local_motion(
                polytope,
                point.point,
                direction.direction,
                check_boundedness=False,
            )
            result = _direction_local_motion_kernel(
                polytope, point.point, direction.direction
            )
            cells.append(
                DirectionCoverageCell(
                    point_id=point.point_id,
                    direction_id=direction.direction_id,
                    aggregate=result.aggregate,
                )
            )
            if result.aggregate == "ILLUMINATES_STRICTLY":
                strict.add(point.point_id)
                witnesses.setdefault(point.point_id, direction.direction_id)
    point_ids = tuple(row.point_id for row in points)
    return DirectionCoverageResult(
        polytope=polytope,
        points=points,
        directions=directions,
        point_ids=point_ids,
        direction_ids=tuple(row.direction_id for row in directions),
        cells=tuple(cells),
        strictly_illuminated_point_ids=tuple(i for i in point_ids if i in strict),
        uncovered_point_ids=tuple(i for i in point_ids if i not in strict),
        witness_direction_by_point=tuple(
            (i, witnesses[i]) for i in point_ids if i in witnesses
        ),
    )


__all__ = ["active_facet_profile", "direction_local_motion", "direction_set_coverage"]
