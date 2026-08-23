"""Tests for the geometry.circumradius.profile.compute operation."""

import math
from fractions import Fraction
from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json, format_canonical_integer
from jacobian.math.geometry._models import (
    CIRCUMRADIUS_INPUT_HEIGHT,
    MAX_CIRCUMRADIUS_PROFILE_RESULT_BYTES,
    CircumradiusProfileRequest,
    CircumradiusProfileResult,
    CircumradiusTripleEntry,
    _maximum_profile_wire_bytes,
)
from jacobian.math.geometry._operations import circumradius_profile
from jacobian.math.geometry.exact._models import (
    DistanceProfileRequest,
    PointConfiguration,
)
from jacobian.math.geometry.exact._operations import compute_distance_profile


def _pt(label: str, x: tuple[str, str], y: tuple[str, str]):
    return {
        "label": label,
        "coordinates": [
            {"num": x[0], "den": x[1]},
            {"num": y[0], "den": y[1]},
        ],
    }


def _request(points):
    return CircumradiusProfileRequest(configuration={"points": tuple(points)})


def _lp(label, xn, xd, yn, yd):
    from jacobian.math.geometry.exact._models import (
        LabelledRationalPoint,
    )

    return LabelledRationalPoint(
        label=label,
        coordinates=(
            CanonicalRational(num=xn, den=xd),
            CanonicalRational(num=yn, den=yd),
        ),
    )


class TestCircumradiusProfile:
    def test_single_triangle_unit_right(self):
        # Right triangle with legs 2: circumradius squared = (hypotenuse/2)^2 = 2
        pts = (
            _lp("a", "0", "1", "0", "1"),
            _lp("b", "2", "1", "0", "1"),
            _lp("c", "0", "1", "2", "1"),
        )
        result = circumradius_profile(_request(pts))
        assert result.point_count == 3
        assert result.triple_count == 1
        entry = result.entries[0]
        assert entry.collinear is False
        assert entry.squared_circumradius == CanonicalRational(num="2", den="1")

    def test_single_triangle_collinear(self):
        # Three collinear points on the x-axis
        pts = (
            _lp("a", "0", "1", "0", "1"),
            _lp("b", "1", "1", "0", "1"),
            _lp("c", "2", "1", "0", "1"),
        )
        result = circumradius_profile(_request(pts))
        assert result.triple_count == 1
        entry = result.entries[0]
        assert entry.collinear is True
        assert entry.squared_circumradius is None

    def test_equilateral_triangle(self):
        # Equilateral triangle with side length sqrt(3): circumradius = 1
        # Vertices: (0,0), (2,0), (1, sqrt(3)) -- but sqrt(3) is irrational.
        # Use a rational equilateral: side 2, circumradius = 2/sqrt(3), so R^2 = 4/3.
        # Rational equilateral: vertices (0,0), (4,0), (2, 2*sqrt(3)) is not rational.
        # Instead test a known rational case: isoceles right triangle (0,0),(2,0),(0,2).
        # Circumradius = hypotenuse/2 = sqrt(8)/2 = sqrt(2), R^2 = 2.
        pts = (
            _lp("a", "0", "1", "0", "1"),
            _lp("b", "2", "1", "0", "1"),
            _lp("c", "0", "1", "2", "1"),
        )
        result = circumradius_profile(_request(pts))
        entry = result.entries[0]
        assert entry.collinear is False
        assert entry.squared_circumradius == CanonicalRational(num="2", den="1")

    def test_four_points_square(self):
        # Unit square: one triple is degenerate? No, any three corners are
        # non-collinear. For a unit square with corners (0,0),(1,0),(1,1),(0,1):
        # Each triple forms a right isoceles triangle with legs 1 and 1 and
        # hypotenuse sqrt(2). Circumradius = hypotenuse/2 = sqrt(2)/2.
        # R^2 = 2/4 = 1/2.
        pts = (
            _lp("a", "0", "1", "0", "1"),
            _lp("b", "1", "1", "0", "1"),
            _lp("c", "1", "1", "1", "1"),
            _lp("d", "0", "1", "1", "1"),
        )
        result = circumradius_profile(_request(pts))
        assert result.point_count == 4
        assert result.triple_count == 4
        for entry in result.entries:
            assert entry.collinear is False
            assert entry.squared_circumradius == CanonicalRational(num="1", den="2")

    def test_mixed_collinear_and_noncollinear(self):
        # (0,0), (1,0), (2,0) collinear; (0,0), (1,0), (0,1) non-collinear
        pts = (
            _lp("a", "0", "1", "0", "1"),
            _lp("b", "1", "1", "0", "1"),
            _lp("c", "2", "1", "0", "1"),
            _lp("d", "0", "1", "1", "1"),
        )
        result = circumradius_profile(_request(pts))
        assert result.triple_count == 4
        collinear_count = sum(1 for e in result.entries if e.collinear)
        non_collinear_count = sum(1 for e in result.entries if not e.collinear)
        # The triple (a,b,c) is collinear; the other three are not.
        assert collinear_count == 1
        assert non_collinear_count == 3
        # Find the collinear entry
        collinear_entry = next(e for e in result.entries if e.collinear)
        assert collinear_entry.labels == ("a", "b", "c")
        assert collinear_entry.indices == (0, 1, 2)
        assert collinear_entry.squared_circumradius is None

    def test_rational_circumradius(self):
        # Triangle (0,0), (3,0), (0,4): right triangle with legs 3,4, hyp 5.
        # Circumradius = 5/2, R^2 = 25/4.
        pts = (
            _lp("a", "0", "1", "0", "1"),
            _lp("b", "3", "1", "0", "1"),
            _lp("c", "0", "1", "4", "1"),
        )
        result = circumradius_profile(_request(pts))
        entry = result.entries[0]
        assert entry.collinear is False
        assert entry.squared_circumradius == CanonicalRational(num="25", den="4")

    def test_indices_and_labels_correct(self):
        # Verify that indices and labels match the input order
        pts = (
            _lp("p0", "0", "1", "0", "1"),
            _lp("p1", "1", "1", "0", "1"),
            _lp("p2", "0", "1", "1", "1"),
            _lp("p3", "1", "1", "1", "1"),
        )
        result = circumradius_profile(_request(pts))
        assert result.triple_count == 4
        # Check all triples have correct indices
        expected_indices = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        expected_labels = [
            ("p0", "p1", "p2"),
            ("p0", "p1", "p3"),
            ("p0", "p2", "p3"),
            ("p1", "p2", "p3"),
        ]
        for i, entry in enumerate(result.entries):
            assert entry.indices == expected_indices[i]
            assert entry.labels == expected_labels[i]

    def test_completeness_count(self):
        # 5 points -> C(5,3) = 10 triples
        pts = (
            _lp("a", "0", "1", "0", "1"),
            _lp("b", "1", "1", "0", "1"),
            _lp("c", "2", "1", "0", "1"),
            _lp("d", "0", "1", "1", "1"),
            _lp("e", "1", "1", "1", "1"),
        )
        result = circumradius_profile(_request(pts))
        assert result.triple_count == 10
        assert len(result.entries) == 10


class TestCircumradiusProfileValidation:
    def test_rejects_duplicate_labels(self):
        pts = (
            _lp("a", "0", "1", "0", "1"),
            _lp("a", "1", "1", "0", "1"),
            _lp("c", "2", "1", "0", "1"),
        )
        with pytest.raises(ValidationError, match="unique"):
            CircumradiusProfileRequest(configuration={"points": tuple(pts)})

    def test_rejects_duplicate_coordinates(self):
        pts = (
            _lp("a", "0", "1", "0", "1"),
            _lp("b", "0", "1", "0", "1"),
            _lp("c", "2", "1", "0", "1"),
        )
        with pytest.raises(ValidationError, match="unique"):
            CircumradiusProfileRequest(configuration={"points": tuple(pts)})

    def test_rejects_too_few_points(self):
        pts = (
            _lp("a", "0", "1", "0", "1"),
            _lp("b", "1", "1", "0", "1"),
        )
        with pytest.raises(ValidationError):
            CircumradiusProfileRequest(configuration={"points": tuple(pts)})

    def test_entry_rejects_collinear_with_radius(self):
        with pytest.raises(ValidationError, match="collinear"):
            CircumradiusTripleEntry(
                labels=("a", "b", "c"),
                indices=(0, 1, 2),
                collinear=True,
                squared_circumradius=CanonicalRational(num="1", den="1"),
            )

    def test_entry_rejects_noncollinear_without_radius(self):
        with pytest.raises(ValidationError, match="collinear"):
            CircumradiusTripleEntry(
                labels=("a", "b", "c"),
                indices=(0, 1, 2),
                collinear=False,
                squared_circumradius=None,
            )

    def test_entry_rejects_nonpositive_radius(self):
        with pytest.raises(ValidationError, match="positive"):
            CircumradiusTripleEntry(
                labels=("a", "b", "c"),
                indices=(0, 1, 2),
                collinear=False,
                squared_circumradius=CanonicalRational(num="0", den="1"),
            )


class TestCatalogContractParity:
    def test_parabola_collision(self):
        from fractions import Fraction

        pts = tuple(
            _lp(f"t{t}", str(t), "1", str(t * t), "1") for t in (1, 2, 4, 19, 29)
        )
        result = circumradius_profile(_request(pts))
        assert result.point_count == 5
        assert result.triple_count == 10
        assert result.degenerate_triple_count == 0
        assert result.nondegenerate_triple_count == 10
        collision = [
            entry.indices
            for entry in result.entries
            if entry.squared_circumradius is not None
            and entry.squared_circumradius.as_fraction() == Fraction(2166905)
        ]
        assert collision == [(0, 1, 4), (1, 2, 3)]
        # The advertised multiplicity ledger groups the replayed radii.
        multiplicities = {
            radius.as_fraction(): count
            for radius, count in result.radius_multiplicities
        }
        assert multiplicities[Fraction(2166905)] == 2
        assert sum(count for _, count in result.radius_multiplicities) == 10
        values = [radius.as_fraction() for radius, _ in result.radius_multiplicities]
        assert values == sorted(values)

    def test_result_rejects_forged_multiplicity_ledger(self):
        from jacobian.math.geometry._models import CircumradiusProfileResult

        pts = (
            _lp("a", "0", "1", "0", "1"),
            _lp("b", "2", "1", "0", "1"),
            _lp("c", "0", "1", "2", "1"),
        )
        honest = circumradius_profile(_request(pts))
        forged_radius = CanonicalRational(num="999", den="1")
        with pytest.raises(ValidationError, match="multiplicit"):
            CircumradiusProfileResult(
                configuration=honest.configuration,
                point_count=3,
                triple_count=1,
                entries=honest.entries,
                radius_multiplicities=((forged_radius, 1),),
                degenerate_triple_count=0,
                nondegenerate_triple_count=1,
            )

    def test_multiplicity_ledger_is_schema_bounded_before_validation(self):
        from jacobian.math.geometry._models import CircumradiusProfileResult

        schema = CircumradiusProfileResult.model_json_schema()["properties"][
            "radius_multiplicities"
        ]["maxItems"]
        assert schema == 41664
        forged = tuple(
            (CanonicalRational(num=str(index), den="1"), 1) for index in range(41665)
        )
        with pytest.raises(ValidationError, match="at most 41664 items"):
            CircumradiusProfileResult(
                configuration=_request(
                    (
                        _lp("a", "0", "1", "0", "1"),
                        _lp("b", "2", "1", "0", "1"),
                        _lp("c", "0", "1", "2", "1"),
                    )
                ),
                point_count=3,
                triple_count=1,
                entries=(),
                radius_multiplicities=forged,
                degenerate_triple_count=0,
                nondegenerate_triple_count=1,
            )

    def test_collinear_triple_is_degenerate(self):
        pts = (
            _lp("a", "0", "1", "0", "1"),
            _lp("b", "1", "1", "0", "1"),
            _lp("c", "2", "1", "0", "1"),
            _lp("d", "0", "1", "1", "1"),
        )
        result = circumradius_profile(_request(pts))
        assert result.triple_count == 4
        degenerate = [entry.indices for entry in result.entries if entry.collinear]
        assert degenerate == [(0, 1, 2)]


def _fraction_wire(value: Fraction) -> dict[str, str]:
    return {
        "num": format_canonical_integer(value.numerator),
        "den": format_canonical_integer(value.denominator),
    }


def _height_point(index: int) -> dict[str, object]:
    """Point with ~60-digit reduced coordinate components (reviewer heights)."""

    x = Fraction(index + 1, 10**59 + 2 * index + 1)
    y = Fraction((index + 1) ** 2, 10**59 + 2 * index + 129)
    return {
        "label": f"p{index}",
        "coordinates": [_fraction_wire(x), _fraction_wire(y)],
    }


class TestAggregateResultBudget:
    def test_reviewer_counterexample_beyond_budget_is_rejected(self):
        points = tuple(_height_point(index) for index in range(64))
        with pytest.raises(ValidationError, match="aggregate result budget"):
            CircumradiusProfileRequest(configuration={"points": tuple(points)})

    def test_same_coordinate_heights_admitted_when_profile_fits_budget(self):
        points = tuple(_height_point(index) for index in range(20))
        result = circumradius_profile(
            CircumradiusProfileRequest(configuration={"points": tuple(points)})
        )
        assert result.point_count == 20
        assert result.triple_count == math.comb(20, 3)
        encoded = len(encode_strict_json(result.model_dump(mode="json")))
        assert encoded <= MAX_CIRCUMRADIUS_PROFILE_RESULT_BYTES

    def test_maximum_point_count_with_small_coordinates_stays_within_budget(self):
        # 59 points with single-digit coordinates is the largest
        # configuration whose complete profile - entries plus the advertised
        # multiplicity ledger and triple counts - still fits the aggregate
        # output budget.
        points = tuple(
            {
                "label": f"p{index}",
                "coordinates": [
                    {"num": str(index), "den": "1"},
                    {"num": str(index * index % 89), "den": "1"},
                ],
            }
            for index in range(59)
        )
        request = CircumradiusProfileRequest(configuration={"points": tuple(points)})
        result = circumradius_profile(request)
        assert result.triple_count == math.comb(59, 3)
        assert len(result.radius_multiplicities) > 0
        wire = result.model_dump(mode="json")
        actual = len(encode_strict_json(wire))
        assert actual <= MAX_CIRCUMRADIUS_PROFILE_RESULT_BYTES
        assert actual <= _maximum_profile_wire_bytes(request.configuration)

    def test_near_degenerate_triple_stays_exact_within_bound(self):
        # Three consecutive parabola points with huge heights: the cross is
        # tiny, so the squared circumradius is enormous but still bounded.
        big = 10**31 + 9
        parameters = (
            Fraction(1),
            Fraction(big, big + 1),
            Fraction(big + 2, big + 3),
        )
        points = tuple(
            {
                "label": f"t{index}",
                "coordinates": [_fraction_wire(t), _fraction_wire(t * t)],
            }
            for index, t in enumerate(parameters)
        )
        result = circumradius_profile(
            CircumradiusProfileRequest(configuration={"points": tuple(points)})
        )
        entry = result.entries[0]
        assert entry.collinear is False
        ax, ay = (Fraction(parameters[0]), parameters[0] ** 2)
        bx, by = (Fraction(parameters[1]), parameters[1] ** 2)
        cx, cy = (Fraction(parameters[2]), parameters[2] ** 2)
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        expected = (
            ((ax - bx) ** 2 + (ay - by) ** 2)
            * ((bx - cx) ** 2 + (by - cy) ** 2)
            * ((ax - cx) ** 2 + (ay - cy) ** 2)
            / (4 * cross * cross)
        )
        assert entry.squared_circumradius is not None
        assert entry.squared_circumradius.as_fraction() == expected
        radius_digits = max(
            len(entry.squared_circumradius.num),
            len(entry.squared_circumradius.den),
        )
        assert radius_digits > 100

    def test_estimate_bounds_actual_encoding_across_mixed_configurations(self):
        configurations = [
            tuple(_height_point(index) for index in range(12)),
            tuple(
                {
                    "label": chr(ord("a") + index),
                    "coordinates": [
                        _fraction_wire(Fraction(index + 1, 97 * (index + 3))),
                        _fraction_wire(Fraction(index * index + 1, 89)),
                    ],
                }
                for index in range(11)
            ),
        ]
        for points in configurations:
            request = CircumradiusProfileRequest(
                configuration={"points": tuple(points)}
            )
            result = circumradius_profile(request)
            actual = len(encode_strict_json(result.model_dump(mode="json")))
            assert actual <= _maximum_profile_wire_bytes(request.configuration)


class TestSchemaPublishedBounds:
    def test_configuration_schema_publishes_coordinate_digit_bound(self):
        schema = CircumradiusProfileRequest.model_json_schema()
        configuration_schema = schema["properties"]["configuration"]
        assert (
            configuration_schema["coordinate_digit_bound"] == CIRCUMRADIUS_INPUT_HEIGHT
        )
        assert CIRCUMRADIUS_INPUT_HEIGHT == 64
        assert (
            configuration_schema["aggregate_result_budget_bytes"]
            == MAX_CIRCUMRADIUS_PROFILE_RESULT_BYTES
        )
        assert "64 canonical decimal digits" in configuration_schema["description"]
        assert (
            str(MAX_CIRCUMRADIUS_PROFILE_RESULT_BYTES)
            in configuration_schema["description"]
        )

    def test_enforced_coordinate_height_matches_published_bound(self):
        oversized = "1" + "0" * CIRCUMRADIUS_INPUT_HEIGHT
        pts = (
            _lp("a", "0", "1", "0", "1"),
            _lp("b", "1", "1", "0", "1"),
            _lp("c", "0", "1", oversized, "1"),
        )
        with pytest.raises(
            ValidationError,
            match=f"exceeds the {CIRCUMRADIUS_INPUT_HEIGHT}-digit bound",
        ):
            CircumradiusProfileRequest(configuration={"points": tuple(pts)})

    def test_label_heavy_near_limit_configuration_is_rejected_by_budget(self):
        # 27 points at ~63-digit coordinate heights with maximum-size
        # four-byte UTF-8 labels: the honest wire estimate must dominate
        # the actual encoding and reject this before any profile work.
        height = 10**62

        def wire(value: Fraction) -> dict[str, str]:
            return {
                "num": format_canonical_integer(value.numerator),
                "den": format_canonical_integer(value.denominator),
            }

        base = "\U0001f600" * 62  # 62 four-byte characters
        points = tuple(
            {
                "label": f"{base}{index:02d}",
                "coordinates": [
                    wire(Fraction(height + 2 * index + 1, height + 2 * index + 3)),
                    wire(Fraction(height + 4 * index + 3, height + 4 * index + 5)),
                ],
            }
            for index in range(27)
        )
        request_points = PointConfiguration(points=points)
        assert (
            _maximum_profile_wire_bytes(request_points)
            > MAX_CIRCUMRADIUS_PROFILE_RESULT_BYTES
        )
        with pytest.raises(ValidationError, match="aggregate result"):
            CircumradiusProfileRequest(configuration={"points": points})

    def test_accepts_the_canonical_configuration_value_unchanged(self):
        configuration = {
            "points": [
                {
                    "label": "a",
                    "coordinates": [
                        {"num": "0", "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                },
                {
                    "label": "b",
                    "coordinates": [
                        {"num": "2", "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                },
                {
                    "label": "c",
                    "coordinates": [
                        {"num": "0", "den": "1"},
                        {"num": "2", "den": "1"},
                    ],
                },
            ]
        }
        distance_result = compute_distance_profile(
            DistanceProfileRequest(configuration=configuration)
        )
        circumradius_request = CircumradiusProfileRequest(configuration=configuration)
        circumradius_result = circumradius_profile(circumradius_request)
        assert distance_result.point_count == 3
        assert circumradius_result.point_count == 3
        assert circumradius_result.triple_count == math.comb(3, 3)
        entry = circumradius_result.entries[0]
        assert entry.squared_circumradius is not None
        assert entry.squared_circumradius.as_fraction() == Fraction(2)


class TestCanonicalConfigurationRetention:
    def test_retained_configuration_feeds_distance_operations_unchanged(self):
        """The retained source must be the canonical PointConfiguration so
        it composes into distance_profile / distance_graph without peeling
        off an inner wrapper."""
        from jacobian.math.geometry.exact._models import DistanceProfileRequest
        from jacobian.math.geometry.exact._operations import (
            compute_distance_profile,
        )

        pts = (
            _lp("a", "0", "1", "0", "1"),
            _lp("b", "2", "1", "0", "1"),
            _lp("c", "0", "1", "2", "1"),
        )
        result = circumradius_profile(_request(pts))
        assert isinstance(result.configuration, PointConfiguration)
        assert result.configuration == _request(pts).configuration

        # The advertised composition: pass the retained value unchanged.
        profile = compute_distance_profile(
            DistanceProfileRequest(configuration=result.configuration)
        )
        assert profile.point_count == 3
        assert sum(entry.pair_count for entry in profile.entries) == 3


class TestNonplanarResultReplayRejection:
    def _honest_payload(self) -> dict[str, object]:
        pts = (
            _lp("A", "0", "1", "0", "1"),
            _lp("B", "2", "1", "0", "1"),
            _lp("C", "0", "1", "2", "1"),
        )
        return circumradius_profile(_request(pts)).model_dump(mode="json")

    def test_three_dimensional_retained_configuration_is_rejected(self):
        # The planar projection of (0,0), (2,0), (0,2) replays to R^2 = 2,
        # but the true triangle (0,0,0), (2,0,0), (0,2,2) has R^2 = 3; the
        # replay must reject the nonplanar retained configuration instead
        # of silently validating the projected radius.
        payload = self._honest_payload()
        for point in payload["configuration"]["points"]:
            point["coordinates"].append({"num": "0", "den": "1"})
        with pytest.raises(ValidationError, match="planar"):
            CircumradiusProfileResult.model_validate(payload)

    def test_one_dimensional_retained_configuration_is_rejected(self):
        payload = self._honest_payload()
        for point in payload["configuration"]["points"]:
            del point["coordinates"][1]
        with pytest.raises(ValidationError, match="planar"):
            CircumradiusProfileResult.model_validate(payload)

    def test_planar_retained_configuration_still_replays(self):
        payload = self._honest_payload()
        result = CircumradiusProfileResult.model_validate(payload)
        assert result.entries[0].squared_circumradius is not None
        assert result.entries[0].squared_circumradius.as_fraction() == Fraction(2)


class TestRetainedConfigurationAdmission:
    """A retained configuration must satisfy request admission before replay."""

    def test_duplicate_coordinates_in_retained_configuration_are_rejected(self):
        # Every entry replays honestly against this source, but the source
        # lies outside CircumradiusProfileRequest's domain because two
        # points share coordinates.
        points = (
            _lp("a", "0", "1", "0", "1"),
            _lp("b", "0", "1", "0", "1"),
            _lp("c", "1", "1", "0", "1"),
        )
        with pytest.raises(ValidationError, match="unique"):
            CircumradiusProfileResult(
                configuration={"points": points},
                point_count=3,
                triple_count=1,
                entries=(
                    CircumradiusTripleEntry(
                        labels=("a", "b", "c"),
                        indices=(0, 1, 2),
                        collinear=True,
                    ),
                ),
                degenerate_triple_count=1,
                nondegenerate_triple_count=0,
            )

    def test_oversized_coordinate_heights_in_retained_configuration_are_rejected(self):
        big = 10**70 + 3
        points = (
            _lp("a", "0", "1", "0", "1"),
            _lp("b", str(big), str(10**69 + 3), "0", "1"),
            _lp("c", "0", "1", "1", "1"),
        )
        ax, ay = Fraction(0), Fraction(0)
        bx, by = Fraction(big, 10**69 + 3), Fraction(0)
        cx, cy = Fraction(0), Fraction(1)
        dab = (ax - bx) ** 2 + (ay - by) ** 2
        dbc = (bx - cx) ** 2 + (by - cy) ** 2
        dac = (ax - cx) ** 2 + (ay - cy) ** 2
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        radius = dab * dbc * dac / (4 * cross * cross)
        with pytest.raises(
            ValidationError,
            match=f"exceeds the {CIRCUMRADIUS_INPUT_HEIGHT}-digit bound",
        ):
            CircumradiusProfileResult(
                configuration={"points": points},
                point_count=3,
                triple_count=1,
                entries=(
                    CircumradiusTripleEntry(
                        labels=("a", "b", "c"),
                        indices=(0, 1, 2),
                        collinear=False,
                        squared_circumradius=CanonicalRational.from_fraction(radius),
                    ),
                ),
                radius_multiplicities=((CanonicalRational.from_fraction(radius), 1),),
                degenerate_triple_count=0,
                nondegenerate_triple_count=1,
            )

    def test_budget_violating_retained_configuration_is_rejected_before_replay(self):
        # All-collinear huge-height points with maximal UTF-8 labels: every
        # entry would replay cleanly, so admission must reject the complete
        # profile's aggregate budget before any replay work.
        height = 10**62
        base = "\U0001f600" * 62
        count = 27
        points = tuple(
            {
                "label": f"{base}{index:02d}",
                "coordinates": [
                    {"num": str(height + index), "den": str(height + index + 1)},
                    {"num": "0", "den": "1"},
                ],
            }
            for index in range(count)
        )
        entries = tuple(
            CircumradiusTripleEntry(
                labels=tuple(points[triple_index]["label"] for triple_index in triple),
                indices=triple,
                collinear=True,
            )
            for triple in combinations(range(count), 3)
        )
        with pytest.raises(ValidationError, match="aggregate result budget"):
            CircumradiusProfileResult(
                configuration={"points": points},
                point_count=count,
                triple_count=len(entries),
                entries=entries,
                degenerate_triple_count=len(entries),
                nondegenerate_triple_count=0,
            )

    def test_honest_retained_configuration_round_trips_through_admission(self):
        pts = (
            _lp("a", "0", "1", "0", "1"),
            _lp("b", "2", "1", "0", "1"),
            _lp("c", "0", "1", "2", "1"),
        )
        payload = circumradius_profile(_request(pts)).model_dump(mode="json")
        result = CircumradiusProfileResult.model_validate(payload)
        assert result.entries[0].squared_circumradius.as_fraction() == Fraction(2)


class TestNfcNormalizedLabelBudget:
    """The wire estimate must charge labels at their NFC-normalized encoding,
    because OperationResult.require_canonical_output applies NFC and it can
    expand a label (review thread: normalize labels before estimating)."""

    def test_nfc_expanding_labels_are_charged_at_admission(self):
        # 39 collinear points labelled with U+0958 x62 + index: raw UTF-8
        # measures ~7.9 MiB, but NFC expands each U+0958 into two
        # characters, so the canonical result exceeds the 10 MiB budget.
        # Admission must reject before any profile work.
        label_unit = "\u0958"  # NFC: one character -> two characters
        points = tuple(
            {
                "label": f"{label_unit * 62}{index:02d}",
                "coordinates": [
                    {"num": str(index), "den": "1"},
                    {"num": "0", "den": "1"},
                ],
            }
            for index in range(39)
        )
        request_points = PointConfiguration(points=points)
        estimate = _maximum_profile_wire_bytes(request_points)
        assert estimate > MAX_CIRCUMRADIUS_PROFILE_RESULT_BYTES
        with pytest.raises(ValidationError, match="aggregate result"):
            CircumradiusProfileRequest(configuration={"points": points})

    def test_ascii_control_of_same_shape_is_still_admitted(self):
        # The rejection above must come from normalization growth, not from
        # an indiscriminately shrunk budget: same shape, ASCII labels fit.
        points = tuple(
            {
                "label": f"{'a' * 62}{index:02d}",
                "coordinates": [
                    {"num": str(index), "den": "1"},
                    {"num": "0", "den": "1"},
                ],
            }
            for index in range(39)
        )
        request = CircumradiusProfileRequest(configuration={"points": points})
        assert len(request.configuration.points) == 39

    def test_estimate_bounds_post_normalization_encoding(self):
        # For a configuration whose labels expand under NFC, the estimate
        # must still dominate the actual canonical (post-NFC) encoding.
        from jacobian.canonical import canonicalize_json

        points = tuple(
            {
                "label": "\u0958" * 40 + chr(ord("a") + index),
                "coordinates": [
                    _fraction_wire(Fraction(index + 1, index + 7)),
                    _fraction_wire(Fraction(2 * index + 1, 13)),
                ],
            }
            for index in range(11)
        )
        request = CircumradiusProfileRequest(configuration={"points": points})
        result = circumradius_profile(request)
        canonical = canonicalize_json(result.model_dump(mode="json"))
        assert len(canonical) <= _maximum_profile_wire_bytes(request.configuration)
