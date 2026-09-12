"""Tests for configuration-level geometry operations (#2107, #2106)."""

import json
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry._models import (
    CircumradiusProfileRequest,
    GeneralPositionRequest,
    GeometryCircleResult,
    PointQuadrupleRequest,
    PointTripleRequest,
    RationalPoint2D,
    SpannedCircleProfileRequest,
)
from jacobian.math.geometry._tools import (
    circumradius_profile,
    collinear,
    concyclic,
    general_position_search,
    spanned_circle_profile,
)
from jacobian.math.geometry.operations import verify_collinearity, verify_concyclicity


def _point(x: str, y: str) -> RationalPoint2D:
    return RationalPoint2D(
        x=CanonicalRational.from_fraction(Fraction(x)),
        y=CanonicalRational.from_fraction(Fraction(y)),
    )


class TestGeneralPosition:
    def test_square_concyclic(self) -> None:
        """Four vertices of a square are concyclic."""
        points = [
            _point("0", "0"),
            _point("1", "0"),
            _point("1", "1"),
            _point("0", "1"),
        ]
        result = general_position_search(GeneralPositionRequest(points=tuple(points)))
        assert result.num_points == 4
        assert not result.has_collinear_triple
        assert result.has_concyclic_quadruple
        assert len(result.concyclic_quadruples) == 1
        assert result.concyclic_quadruples[0].indices == (0, 1, 2, 3)
        assert type(result).model_validate_json(result.model_dump_json()) == result

    def test_collinear_triple(self) -> None:
        """Three points on the x-axis are collinear."""
        points = [
            _point("0", "0"),
            _point("1", "0"),
            _point("2", "0"),
            _point("0", "1"),
        ]
        result = general_position_search(GeneralPositionRequest(points=tuple(points)))
        assert result.has_collinear_triple
        assert not result.has_concyclic_quadruple
        assert len(result.collinear_triples) == 1
        assert result.collinear_triples[0].indices == (0, 1, 2)

    def test_general_position(self) -> None:
        """Four points in general position."""
        points = [
            _point("-1", "0"),
            _point("1", "0"),
            _point("0", "2"),
            _point("0", "-2"),
        ]
        result = general_position_search(GeneralPositionRequest(points=tuple(points)))
        assert not result.has_collinear_triple
        assert not result.has_concyclic_quadruple

    def test_triangle_only(self) -> None:
        """A triangle has no collinear triple and no quadruple."""
        points = [_point("0", "0"), _point("1", "0"), _point("0", "1")]
        result = general_position_search(GeneralPositionRequest(points=tuple(points)))
        assert not result.has_collinear_triple
        assert not result.has_concyclic_quadruple

    def test_duplicate_points_rejected(self) -> None:
        """Duplicate points should be rejected."""
        with pytest.raises(ValueError):
            GeneralPositionRequest(
                points=(_point("0", "0"), _point("0", "0"), _point("1", "0"))
            )


def test_point_predicate_claims_round_trip_and_reject_forged_sources() -> None:
    first = _point("0", "0")
    second = _point("1", "0")
    third = _point("2", "0")
    collinear_claim = collinear(
        PointTripleRequest(first=first, second=second, third=third)
    )
    assert verify_collinearity(
        type(collinear_claim).model_validate_json(collinear_claim.model_dump_json())
    )

    payload = collinear_claim.model_dump(mode="json")
    payload["third"] = _point("0", "1").model_dump(mode="json")
    assert not verify_collinearity(
        type(collinear_claim).model_validate_json(json.dumps(payload))
    )

    concyclic_claim = concyclic(
        PointQuadrupleRequest(
            first=_point("1", "0"),
            second=_point("0", "1"),
            third=_point("-1", "0"),
            fourth=_point("0", "-1"),
        )
    )
    assert verify_concyclicity(
        type(concyclic_claim).model_validate_json(concyclic_claim.model_dump_json())
    )


class TestCircumradiusProfile:
    def test_triangle(self) -> None:
        """A single triangle has one circumradius entry."""
        points = [_point("0", "0"), _point("1", "0"), _point("0", "1")]
        result = circumradius_profile(CircumradiusProfileRequest(points=tuple(points)))
        assert result.num_points == 3
        assert len(result.entries) == 1
        assert not result.entries[0].is_degenerate
        assert result.entries[0].radius_squared == CanonicalRational(num=1, den=2)
        assert result.entries[0].indices == (0, 1, 2)
        assert type(result).model_validate_json(result.model_dump_json()) == result

    def test_collinear_triple_degenerate(self) -> None:
        """Collinear triple is marked as degenerate."""
        points = [
            _point("0", "0"),
            _point("1", "0"),
            _point("2", "0"),
        ]
        result = circumradius_profile(CircumradiusProfileRequest(points=tuple(points)))
        assert len(result.entries) == 1
        assert result.entries[0].is_degenerate
        assert result.entries[0].radius_squared is None

    def test_four_points(self) -> None:
        """Four points have C(4,3) = 4 triples."""
        points = [
            _point("0", "0"),
            _point("1", "0"),
            _point("0", "1"),
            _point("1", "1"),
        ]
        result = circumradius_profile(CircumradiusProfileRequest(points=tuple(points)))
        assert result.num_points == 4
        assert len(result.entries) == 4
        for entry in result.entries:
            assert not entry.is_degenerate

    def test_equal_circumradius(self) -> None:
        """All four triangles of a unit square have equal circumradius."""
        points = [
            _point("0", "0"),
            _point("1", "0"),
            _point("1", "1"),
            _point("0", "1"),
        ]
        result = circumradius_profile(CircumradiusProfileRequest(points=tuple(points)))
        radii = set()
        for entry in result.entries:
            if not entry.is_degenerate and entry.radius_squared:
                radii.add((entry.radius_squared.num, entry.radius_squared.den))
        assert len(radii) == 1


class TestAdmissionBounds:
    def test_circumradius_accepts_large_coordinates_when_work_fits(self) -> None:
        points = tuple(
            _point(str(10**63 + 4 * i + 1), str(10**63 + 4 * i + 3)) for i in range(32)
        )
        request = CircumradiusProfileRequest(points=points)
        result = circumradius_profile(request)
        assert len(result.entries) == 4_960

    def test_circumradius_accepts_config_within_output_budget(self) -> None:
        """A moderate configuration still runs end to end."""
        points = tuple(_point(f"{i}", f"{i * i}") for i in range(12))
        result = circumradius_profile(CircumradiusProfileRequest(points=points))
        assert result.num_points == 12
        assert len(result.entries) == 220

    def test_general_position_rejects_work_bound_violation(self) -> None:
        """32 points x 255-digit coordinates exceed the exhaustive work bound."""
        big = 10**254 + 1
        points = tuple(_point(str(big + i), str(big + 2 * i)) for i in range(32))
        request = GeneralPositionRequest(points=points)
        with pytest.raises(OperationDomainValidationError, match="work bound"):
            general_position_search(request)

    def test_general_position_rejects_quartic_point_growth(self) -> None:
        """32 points x 32 digits pass n*digits=1024 but C(32,4)*digits^2 is
        about 36M; the combinatorial count must gate admission instead."""
        points = tuple(_point(str(10**31 + i), str(10**31 + 2 * i)) for i in range(32))
        request = GeneralPositionRequest(points=points)
        with pytest.raises(OperationDomainValidationError, match="work bound"):
            general_position_search(request)

    def test_general_position_accepts_moderate_configurations(self) -> None:
        """Shapes within the C(n,4)*digits^2 budget still run end to end."""
        points = tuple(_point(str(i), str(i * i + 1)) for i in range(16))
        result = general_position_search(GeneralPositionRequest(points=points))
        assert result.num_points == 16

    def test_general_position_accepts_small_high_digit_config(self) -> None:
        """Few points may still carry large coordinates (1 * 256^2 units)."""
        big = 10**254 + 1
        points = (
            _point("0", "0"),
            _point(str(big), "0"),
            _point("0", str(big)),
            _point(str(big), str(big)),
        )
        result = general_position_search(GeneralPositionRequest(points=points))
        assert result.has_concyclic_quadruple
        assert not result.has_collinear_triple


class TestSpannedCircleProfile:
    def test_square_merges_all_triples_and_round_trips(self) -> None:
        points = [
            _point("0", "0"),
            _point("1", "0"),
            _point("1", "1"),
            _point("0", "1"),
        ]
        result = spanned_circle_profile(
            SpannedCircleProfileRequest(points=tuple(points))
        )
        assert len(result.circles) == 1
        circle = result.circles[0]
        assert isinstance(circle.circle, GeometryCircleResult)
        assert circle.circle.center == _point("1/2", "1/2")
        assert circle.circle.radius_squared == CanonicalRational(num=1, den=2)
        assert circle.point_indices == (0, 1, 2, 3)
        assert type(result).model_validate_json(result.model_dump_json()) == result

    def test_collinear_triples_are_omitted_and_distinct_circles_remain_distinct(
        self,
    ) -> None:
        points = [
            _point("0", "0"),
            _point("1", "0"),
            _point("2", "0"),
            _point("0", "1"),
            _point("0", "2"),
        ]
        result = spanned_circle_profile(
            SpannedCircleProfileRequest(points=tuple(points))
        )
        assert len(result.circles) == 5
        assert all(len(circle.point_indices) >= 3 for circle in result.circles)
        assert len({tuple(circle.point_indices) for circle in result.circles}) == 5

    def test_reorders_geometry_rows_but_preserves_exact_circle_values(self) -> None:
        points = [
            _point("0", "0"),
            _point("1", "0"),
            _point("0", "1"),
            _point("1", "1"),
        ]
        reordered = [points[2], points[0], points[3], points[1]]
        first = spanned_circle_profile(
            SpannedCircleProfileRequest(points=tuple(points))
        )
        second = spanned_circle_profile(
            SpannedCircleProfileRequest(points=tuple(reordered))
        )
        assert [(c.circle.center, c.circle.radius_squared) for c in first.circles] == [
            (c.circle.center, c.circle.radius_squared) for c in second.circles
        ]
        assert second.circles[0].point_indices == (0, 1, 2, 3)

    def test_work_ceiling_is_a_resource_admission(self) -> None:
        points = tuple(
            _point(str(index), str(1000 + index)) for index in range(32)
        )
        with pytest.raises(OperationResourceAdmissionError, match="2000000"):
            spanned_circle_profile(SpannedCircleProfileRequest(points=points))


@pytest.mark.parametrize("coordinates", [((0, 0), (1, 0)), ((0, 0), (1, 0), (0, 0))])
def test_native_spanned_circles_reject_invalid_source(
    coordinates: tuple[tuple[int, int], ...],
) -> None:
    from jacobian.math.geometry.operations import spanned_circle_profile as native

    points = tuple(_point(str(x), str(y)) for x, y in coordinates)
    with pytest.raises(OperationDomainValidationError):
        native(points)
