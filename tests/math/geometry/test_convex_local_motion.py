"""Tests for exact convex-polytope direction local motion."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.convex import (
    ConvexHPolytope,
    ConvexInequality,
    ConvexSpace,
    DirectionLocalMotionRequest,
    RationalConvexDirection,
    RationalConvexPoint,
)
from jacobian.math.geometry.convex._tools import TOOLS
from jacobian.math.geometry.convex.operations import direction_local_motion


def _rational(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


def _square() -> ConvexHPolytope:
    space = ConvexSpace(axes=("x", "y"))
    return ConvexHPolytope(
        space=space,
        inequalities=(
            ConvexInequality(
                inequality_id="bottom",
                normal=(_rational(0), _rational(-1)),
                bound=_rational(0),
            ),
            ConvexInequality(
                inequality_id="left",
                normal=(_rational(-1), _rational(0)),
                bound=_rational(0),
            ),
            ConvexInequality(
                inequality_id="right",
                normal=(_rational(1), _rational(0)),
                bound=_rational(1),
            ),
            ConvexInequality(
                inequality_id="top",
                normal=(_rational(0), _rational(1)),
                bound=_rational(1),
            ),
        ),
    )


def _point(x: int, y: int) -> RationalConvexPoint:
    return RationalConvexPoint(
        space=ConvexSpace(axes=("x", "y")),
        coordinates=(_rational(x), _rational(y)),
    )


def _direction(x: int, y: int) -> RationalConvexDirection:
    return RationalConvexDirection(
        space=ConvexSpace(axes=("x", "y")),
        components=(_rational(x), _rational(y)),
    )


class TestKnownAnswer:
    def test_corner_strict_entry_illuminates(self) -> None:
        result = direction_local_motion(_square(), _point(0, 0), _direction(1, 1))
        assert result.point_state == "BOUNDARY"
        assert result.aggregate == "ILLUMINATES_STRICTLY"
        assert [(row.inequality_id, row.kind) for row in result.motions] == [
            ("bottom", "STRICTLY_INWARD"),
            ("left", "STRICTLY_INWARD"),
        ]
        dots = {row.inequality_id: row.dot.as_fraction() for row in result.motions}
        assert dots == {"bottom": Fraction(-1), "left": Fraction(-1)}

    def test_facet_tangent_is_partial(self) -> None:
        result = direction_local_motion(_square(), _point(0, 0), _direction(0, 1))
        assert result.aggregate == "TANGENT_OR_PARTIAL"
        kinds = {row.inequality_id: row.kind for row in result.motions}
        assert kinds == {"bottom": "STRICTLY_INWARD", "left": "TANGENT"}

    def test_outward_step_moves_outside(self) -> None:
        result = direction_local_motion(_square(), _point(1, 1), _direction(1, 0))
        assert result.aggregate == "MOVES_OUTSIDE"
        kinds = {row.inequality_id: row.kind for row in result.motions}
        assert kinds["right"] == "OUTWARD"


class TestBoundary:
    def test_single_active_facet_slack_ledger(self) -> None:
        result = direction_local_motion(_square(), _point(1, 0), _direction(0, 1))
        slacks = {row.inequality_id: row.slack.as_fraction() for row in result.slacks}
        assert slacks == {
            "bottom": Fraction(0),
            "left": Fraction(1),
            "right": Fraction(0),
            "top": Fraction(1),
        }
        assert [row.inequality_id for row in result.motions] == ["bottom", "right"]

    def test_cube_vertex_three_active_facets(self) -> None:
        axes = ("x", "y", "z")
        space = ConvexSpace(axes=axes)
        inequalities = []
        for index, axis in enumerate(axes):
            normal = tuple(_rational(1 if j == index else 0) for j in range(3))
            inequalities.append(
                ConvexInequality(
                    inequality_id=f"hi_{axis}", normal=normal, bound=_rational(1)
                )
            )
            normal = tuple(_rational(-1 if j == index else 0) for j in range(3))
            inequalities.append(
                ConvexInequality(
                    inequality_id=f"lo_{axis}", normal=normal, bound=_rational(0)
                )
            )
        cube = ConvexHPolytope(
            space=space,
            inequalities=tuple(sorted(inequalities, key=lambda i: i.inequality_id)),
        )
        corner = RationalConvexPoint(
            space=space, coordinates=(_rational(0), _rational(0), _rational(0))
        )
        inward = RationalConvexDirection(
            space=space, components=(_rational(1), _rational(1), _rational(1))
        )
        result = direction_local_motion(cube, corner, inward)
        assert result.aggregate == "ILLUMINATES_STRICTLY"
        assert len(result.motions) == 3


class TestAdversarial:
    def test_outside_point_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            direction_local_motion(_square(), _point(2, 0), _direction(-1, 0))

    def test_interior_point_rejected(self) -> None:
        assert _point(0, 0).coordinates[0].as_fraction() == 0
        interior = RationalConvexPoint(
            space=ConvexSpace(axes=("x", "y")),
            coordinates=(
                CanonicalRational(num=1, den=2),
                CanonicalRational(num=1, den=2),
            ),
        )
        with pytest.raises(OperationDomainValidationError):
            direction_local_motion(_square(), interior, _direction(1, 0))

    def test_zero_direction_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            direction_local_motion(_square(), _point(0, 0), _direction(0, 0))

    def test_space_mismatch_rejected(self) -> None:
        other = RationalConvexDirection(
            space=ConvexSpace(axes=("x", "z")),
            components=(_rational(1), _rational(1)),
        )
        with pytest.raises(OperationDomainValidationError):
            direction_local_motion(_square(), _point(0, 0), other)


class TestDefiningInvariant:
    def test_zero_dot_is_tangent_never_strict(self) -> None:
        result = direction_local_motion(_square(), _point(0, 1), _direction(1, 0))
        kinds = {row.inequality_id: row.kind for row in result.motions}
        assert kinds == {"left": "STRICTLY_INWARD", "top": "TANGENT"}
        assert result.aggregate == "TANGENT_OR_PARTIAL"

    def test_positive_rescaling_preserves_result(self) -> None:
        first = direction_local_motion(_square(), _point(0, 0), _direction(1, 2))
        scaled_direction = RationalConvexDirection(
            space=ConvexSpace(axes=("x", "y")),
            components=(
                CanonicalRational(num=3, den=1),
                CanonicalRational(num=6, den=1),
            ),
        )
        second = direction_local_motion(_square(), _point(0, 0), scaled_direction)
        assert [row.kind for row in first.motions] == [
            row.kind for row in second.motions
        ]
        assert first.aggregate == second.aggregate == "ILLUMINATES_STRICTLY"


class TestNativeVsCatalogParity:
    def test_catalog_entry_matches_native(self) -> None:
        request = DirectionLocalMotionRequest(
            polytope=_square(), point=_point(0, 0), direction=_direction(1, 1)
        )
        tool = next(
            tool
            for tool in TOOLS
            if tool.operation_id
            == "convex_geometry.polytope.direction_local_motion.compute"
        )
        assert tool.run(request) == direction_local_motion(
            _square(), _point(0, 0), _direction(1, 1)
        )

    def test_operation_is_published(self) -> None:
        assert "convex_geometry.polytope.direction_local_motion.compute" in {
            tool.operation_id for tool in TOOLS
        }
