"""Tests for configuration-level geometry operations (#2107, #2106)."""

import json
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS, CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry._models import (
    MAX_SPANNED_CIRCLE_WORK,
    MAX_SPANNED_CIRCLES,
    CircumradiusProfileRequest,
    GeneralPositionRequest,
    GeometryCircleResult,
    PointQuadrupleRequest,
    PointTripleRequest,
    RationalPoint2D,
    SpannedCircleEntry,
    SpannedCircleProfileRequest,
    SpannedCircleProfileResult,
)
from jacobian.math.geometry._tools import (
    circumradius_profile,
    collinear,
    concyclic,
    general_position_search,
    spanned_circle_profile,
)
from jacobian.math.geometry.exact._models import (
    LabelledRationalPoint,
    PointConfiguration,
)
from jacobian.math.geometry.operations import (
    spanned_circle_profile as native_spanned_circle_profile,
)
from jacobian.math.geometry.operations import (
    verify_collinearity,
    verify_concyclicity,
)


def _point(x: str, y: str) -> RationalPoint2D:
    return RationalPoint2D(
        x=CanonicalRational.from_fraction(Fraction(x)),
        y=CanonicalRational.from_fraction(Fraction(y)),
    )


def _configuration(*points: RationalPoint2D) -> PointConfiguration:
    return PointConfiguration(
        points=tuple(
            LabelledRationalPoint(
                label=f"p{index}",
                coordinates=(point.x, point.y),
            )
            for index, point in enumerate(points)
        )
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
            SpannedCircleProfileRequest(configuration=_configuration(*points))
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
            SpannedCircleProfileRequest(configuration=_configuration(*points))
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
            SpannedCircleProfileRequest(configuration=_configuration(*points))
        )
        second = spanned_circle_profile(
            SpannedCircleProfileRequest(configuration=_configuration(*reordered))
        )
        assert [(c.circle.center, c.circle.radius_squared) for c in first.circles] == [
            (c.circle.center, c.circle.radius_squared) for c in second.circles
        ]
        assert second.circles[0].point_indices == (0, 1, 2, 3)

    def test_cocircular_configuration_charges_incidence_once(self) -> None:
        points = tuple(
            RationalPoint2D(
                x=CanonicalRational.from_fraction(Fraction(1 - t * t, 1 + t * t)),
                y=CanonicalRational.from_fraction(Fraction(2 * t, 1 + t * t)),
            )
            for t in range(1, 33)
        )
        result = spanned_circle_profile(
            SpannedCircleProfileRequest(configuration=_configuration(*points))
        )
        assert len(result.circles) == 1
        assert len(result.circles[0].point_indices) == 32

    def test_collinear_configuration_is_admitted_without_incidence_work(self) -> None:
        points = tuple(_point(str(index), str(1000 + index)) for index in range(32))
        result = spanned_circle_profile(
            SpannedCircleProfileRequest(configuration=_configuration(*points))
        )
        assert result.circles == ()

    def test_collinear_reciprocals_are_admitted_independent_of_source_order(
        self,
    ) -> None:
        denominators: list[int] = []
        candidate = 10**10 + 19
        while len(denominators) < 31:
            limit = int(candidate**0.5)
            if all(candidate % prime for prime in range(3, limit + 1, 2)):
                denominators.append(candidate)
            candidate += 2
        origin_first = (
            _point("0", "0"),
            *(_point(f"1/{denominator}", "0") for denominator in denominators),
        )
        reciprocal_first = (
            _point(f"1/{denominators[0]}", "0"),
            _point("0", "0"),
            *(_point(f"1/{denominator}", "0") for denominator in denominators[1:]),
        )
        for points in (origin_first, reciprocal_first):
            result = spanned_circle_profile(
                SpannedCircleProfileRequest(configuration=_configuration(*points))
            )
            assert result.circles == ()

    def test_zero_origin_admits_collinear_reciprocals(self) -> None:
        denominators: list[int] = []
        candidate = 10**10 + 19
        while len(denominators) < 32:
            limit = int(candidate**0.5)
            if all(candidate % prime for prime in range(3, limit + 1, 2)):
                denominators.append(candidate)
            candidate += 2
        points = tuple(_point(f"1/{denominator}", "0") for denominator in denominators)
        result = spanned_circle_profile(
            SpannedCircleProfileRequest(configuration=_configuration(*points))
        )
        assert result.circles == ()

    def test_bounding_box_origin_admits_two_clusters(self) -> None:
        points = tuple(
            _point(str(3 * index), str((7 * index * index + 11 * index) % 97))
            for index in range(16)
        ) + tuple(
            _point(
                str(1998 - 4 * index),
                str(1998 - ((13 * index * index + 5 * index) % 89)),
            )
            for index in range(16)
        )
        result = spanned_circle_profile(
            SpannedCircleProfileRequest(configuration=_configuration(*points))
        )
        assert len(result.circles) > 1

    def test_translated_large_origin_is_admitted_after_shift(self) -> None:
        shift = 10**256
        points = (
            _point(str(shift), str(shift)),
            _point(str(shift + 1), str(shift)),
            _point(str(shift), str(shift + 1)),
        )
        result = spanned_circle_profile(
            SpannedCircleProfileRequest(configuration=_configuration(*points))
        )
        assert len(result.circles) == 1
        assert result.circles[0].point_indices == (0, 1, 2)
        center = result.circles[0].circle.center
        expected = CanonicalRational.from_fraction(Fraction(shift) + Fraction(1, 2))
        assert center.x == expected
        assert center.y == expected

    def test_result_rejects_nonplanar_configuration(self) -> None:
        points = tuple(
            LabelledRationalPoint(
                label=f"p{index}",
                coordinates=(
                    CanonicalRational(num=index, den=1),
                    CanonicalRational(num=0, den=1),
                    CanonicalRational(num=1, den=1),
                ),
            )
            for index in range(3)
        )
        with pytest.raises(ValidationError, match="planar"):
            SpannedCircleProfileResult(
                configuration=PointConfiguration(points=points),
                num_points=3,
                circles=(),
            )

    def test_translated_parabola_is_admitted_like_the_origin_frame(self) -> None:
        points = tuple(
            _point(str(index), str(1000 + index * index)) for index in range(32)
        )
        result = spanned_circle_profile(
            SpannedCircleProfileRequest(configuration=_configuration(*points))
        )
        assert result.configuration == _configuration(*points)
        assert result.circles

    def test_non_collinear_work_ceiling_is_a_resource_admission(self) -> None:
        points = tuple(
            _point(str(index), str(index * index * 10_000)) for index in range(32)
        )
        with pytest.raises(OperationResourceAdmissionError, match="2000000"):
            spanned_circle_profile(
                SpannedCircleProfileRequest(configuration=_configuration(*points))
            )

    def test_result_rejects_more_circles_than_source_triples(self) -> None:
        points = (_point("0", "0"), _point("1", "0"), _point("0", "1"))
        first = GeometryCircleResult(
            center=_point("1/2", "1/2"),
            radius_squared=CanonicalRational(num=1, den=2),
        )
        second = GeometryCircleResult(
            center=_point("2", "2"),
            radius_squared=CanonicalRational(num=8, den=1),
        )
        with pytest.raises(ValidationError, match="source triples"):
            SpannedCircleProfileResult(
                configuration=_configuration(*points),
                num_points=3,
                circles=(
                    SpannedCircleEntry(circle=first, point_indices=(0, 1, 2)),
                    SpannedCircleEntry(circle=second, point_indices=(0, 1, 2)),
                ),
            )

    def test_result_rejects_circles_sharing_a_source_triple(self) -> None:
        points = (
            _point("0", "0"),
            _point("1", "0"),
            _point("0", "1"),
            _point("1", "1"),
        )
        first = GeometryCircleResult(
            center=_point("1/2", "1/2"),
            radius_squared=CanonicalRational(num=1, den=2),
        )
        second = GeometryCircleResult(
            center=_point("2", "2"),
            radius_squared=CanonicalRational(num=8, den=1),
        )
        with pytest.raises(ValidationError, match="at most one spanned circle"):
            SpannedCircleProfileResult(
                configuration=_configuration(*points),
                num_points=4,
                circles=(
                    SpannedCircleEntry(circle=first, point_indices=(0, 1, 2)),
                    SpannedCircleEntry(circle=second, point_indices=(0, 1, 2)),
                ),
            )

    def test_result_rejects_nonplanar_source_points(self) -> None:
        zero = CanonicalRational(num=0, den=1)
        one = CanonicalRational(num=1, den=1)
        configuration = PointConfiguration(
            points=(
                LabelledRationalPoint(label="a", coordinates=(zero, zero, zero)),
                LabelledRationalPoint(label="b", coordinates=(one, zero, zero)),
                LabelledRationalPoint(label="c", coordinates=(zero, one, zero)),
            )
        )
        with pytest.raises(ValidationError, match="planar"):
            SpannedCircleProfileResult(
                configuration=configuration,
                num_points=3,
                circles=(),
            )

    def test_translated_back_output_counts_numerator_and_denominator(self) -> None:
        origin_num = 10**30_000 + 17
        origin_den = 10**30_000 + 19
        points = tuple(
            RationalPoint2D(
                x=CanonicalRational(
                    num=origin_num + index * origin_den, den=origin_den
                ),
                y=CanonicalRational(
                    num=origin_num + (index * index) * origin_den, den=origin_den
                ),
            )
            for index in range(6)
        )
        with pytest.raises(OperationResourceAdmissionError) as error:
            spanned_circle_profile(
                SpannedCircleProfileRequest(configuration=_configuration(*points))
            )
        assert error.value.errors()[0]["type"] == (
            "geometry.spanned_circle_result_digit_bound"
        )
        assert "aggregate" in error.value.errors()[0]["msg"]

    def test_translated_back_centers_are_admitted_before_wiring(self) -> None:
        shift = CanonicalRational(num=6 * 10**32767, den=1)
        points = (
            RationalPoint2D(x=shift, y=shift),
            RationalPoint2D(x=CanonicalRational(num=6 * 10**32767 + 1, den=1), y=shift),
            RationalPoint2D(x=shift, y=CanonicalRational(num=6 * 10**32767 + 1, den=1)),
        )
        with pytest.raises(OperationResourceAdmissionError, match="translated-back"):
            spanned_circle_profile(
                SpannedCircleProfileRequest(configuration=_configuration(*points))
            )

    def test_translated_coordinate_error_uses_request_path(self) -> None:
        points = (
            _point("0", "0"),
            _point(str(10**257), "0"),
            _point("0", "1"),
        )
        with pytest.raises(OperationDomainValidationError) as error:
            native_spanned_circle_profile(_configuration(*points))
        assert error.value.errors()[0]["loc"] == (
            "configuration",
            "points",
            0,
            "coordinates",
            0,
        )

    def test_native_rejects_malformed_configuration_without_index_errors(self) -> None:
        planar = LabelledRationalPoint(
            label="a",
            coordinates=(
                CanonicalRational(num=0, den=1),
                CanonicalRational(num=0, den=1),
            ),
        )
        short = LabelledRationalPoint.model_construct(
            label="b",
            coordinates=(CanonicalRational(num=1, den=1),),
        )
        third = LabelledRationalPoint(
            label="c",
            coordinates=(
                CanonicalRational(num=0, den=1),
                CanonicalRational(num=1, den=1),
            ),
        )
        forged = PointConfiguration.model_construct(points=(planar, short, third))
        with pytest.raises(OperationDomainValidationError):
            native_spanned_circle_profile(forged)
        with pytest.raises(OperationDomainValidationError):
            native_spanned_circle_profile(object())  # type: ignore[arg-type]

    def test_request_publishes_circle_row_envelope(self) -> None:
        schema = SpannedCircleProfileResult.model_json_schema()
        assert schema["properties"]["circles"]["maxItems"] == MAX_SPANNED_CIRCLES
        assert MAX_SPANNED_CIRCLES == 4960
        request_text = SpannedCircleProfileRequest.model_json_schema()["properties"][
            "configuration"
        ]["description"]
        assert str(MAX_CANONICAL_INTEGER_DIGITS) in request_text
        assert str(MAX_SPANNED_CIRCLE_WORK) in request_text
        assert "zero origin" in request_text
        # Imported after `_tools` so `_configuration` is fully initialized.
        from jacobian.math.geometry._configuration import CONFIGURATION_OPERATIONS

        tool = next(
            item
            for item in CONFIGURATION_OPERATIONS
            if item.operation_id == "geometry.points.spanned_circle_profile.compute"
        )
        assert str(MAX_CANONICAL_INTEGER_DIGITS) in tool.description
        assert str(MAX_SPANNED_CIRCLE_WORK) in tool.description
        assert "zero origin" in tool.description

    def test_native_result_keeps_validated_configuration(self) -> None:
        source = _configuration(_point("0", "0"), _point("1", "0"), _point("0", "1"))
        forged = PointConfiguration.model_construct(points=list(source.points))
        result = native_spanned_circle_profile(forged)
        forged.points.clear()  # type: ignore[union-attr]
        assert len(result.configuration.points) == 3
        assert type(result).model_validate_json(result.model_dump_json()) == result

    def test_circle_enumeration_checkpoints(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        observed: list[str] = []

        def _observe(stage: str) -> None:
            observed.append(stage)

        monkeypatch.setattr(
            "jacobian.math.geometry.operations.request_checkpoint", _observe
        )
        monkeypatch.setattr(
            "jacobian.math.geometry.operations._CIRCLE_CHECKPOINT_INTERVAL", 1
        )
        native_spanned_circle_profile(
            _configuration(_point("0", "0"), _point("1", "0"), _point("0", "1"))
        )
        assert any("collinearity" in stage for stage in observed)
        assert any("construction" in stage for stage in observed)
        assert any("incidence" in stage for stage in observed)
        assert any("result construction" in stage for stage in observed)


@pytest.mark.parametrize("coordinates", [((0, 0), (1, 0)), ((0, 0), (1, 0), (0, 0))])
def test_native_spanned_circles_reject_invalid_source(
    coordinates: tuple[tuple[int, int], ...],
) -> None:
    points = tuple(_point(str(x), str(y)) for x, y in coordinates)
    with pytest.raises(OperationDomainValidationError):
        native_spanned_circle_profile(_configuration(*points))
