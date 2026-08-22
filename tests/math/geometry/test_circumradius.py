"""Tests for the geometry.circumradius.profile.compute operation."""

import math
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json, format_canonical_integer
from jacobian.math.geometry._models import (
    CIRCUMRADIUS_INPUT_HEIGHT,
    MAX_CIRCUMRADIUS_PROFILE_RESULT_BYTES,
    CircumradiusProfileRequest,
    CircumradiusTripleEntry,
    _maximum_profile_wire_bytes,
)
from jacobian.math.geometry._operations import circumradius_profile


def _pt(label: str, x: tuple[str, str], y: tuple[str, str]):
    return {
        "label": label,
        "point": {
            "x": {"num": x[0], "den": x[1]},
            "y": {"num": y[0], "den": y[1]},
        },
    }


def _request(points):
    return CircumradiusProfileRequest(points=tuple(points))


class _Lp:
    """Shorthand for a labelled point dict."""

    def __init__(self, label, x, y):
        self.label = label
        self.x = x
        self.y = y


def _lp(label, xn, xd, yn, yd):
    from jacobian._exact import CanonicalRational
    from jacobian.math.geometry._models import LabelledPoint2D, RationalPoint2D

    return LabelledPoint2D(
        label=label,
        point=RationalPoint2D(
            x=CanonicalRational(num=xn, den=xd),
            y=CanonicalRational(num=yn, den=yd),
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
            CircumradiusProfileRequest(points=pts)

    def test_rejects_duplicate_coordinates(self):
        pts = (
            _lp("a", "0", "1", "0", "1"),
            _lp("b", "0", "1", "0", "1"),
            _lp("c", "2", "1", "0", "1"),
        )
        with pytest.raises(ValidationError, match="unique"):
            CircumradiusProfileRequest(points=pts)

    def test_rejects_too_few_points(self):
        pts = (
            _lp("a", "0", "1", "0", "1"),
            _lp("b", "1", "1", "0", "1"),
        )
        with pytest.raises(ValidationError):
            CircumradiusProfileRequest(points=pts)

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
        collision = [
            entry.indices
            for entry in result.entries
            if entry.squared_circumradius is not None
            and entry.squared_circumradius.as_fraction() == Fraction(2166905)
        ]
        assert collision == [(0, 1, 4), (1, 2, 3)]

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
        "point": {"x": _fraction_wire(x), "y": _fraction_wire(y)},
    }


class TestAggregateResultBudget:
    def test_reviewer_counterexample_beyond_budget_is_rejected(self):
        points = tuple(_height_point(index) for index in range(64))
        with pytest.raises(ValidationError, match="aggregate result budget"):
            CircumradiusProfileRequest(points=points)

    def test_same_coordinate_heights_admitted_when_profile_fits_budget(self):
        points = tuple(_height_point(index) for index in range(20))
        result = circumradius_profile(CircumradiusProfileRequest(points=points))
        assert result.point_count == 20
        assert result.triple_count == math.comb(20, 3)
        encoded = len(encode_strict_json(result.model_dump(mode="json")))
        assert encoded <= MAX_CIRCUMRADIUS_PROFILE_RESULT_BYTES

    def test_maximum_point_count_with_small_coordinates_stays_within_budget(self):
        points = tuple(
            {
                "label": f"p{index}",
                "point": {
                    "x": {"num": str(index), "den": "1"},
                    "y": {"num": str(index * index % 89), "den": "1"},
                },
            }
            for index in range(64)
        )
        request = CircumradiusProfileRequest(points=points)
        result = circumradius_profile(request)
        assert result.triple_count == math.comb(64, 3)
        wire = result.model_dump(mode="json")
        actual = len(encode_strict_json(wire))
        assert actual <= MAX_CIRCUMRADIUS_PROFILE_RESULT_BYTES
        assert actual <= _maximum_profile_wire_bytes(request)

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
                "point": {
                    "x": _fraction_wire(t),
                    "y": _fraction_wire(t * t),
                },
            }
            for index, t in enumerate(parameters)
        )
        result = circumradius_profile(CircumradiusProfileRequest(points=points))
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
                    "point": {
                        "x": _fraction_wire(Fraction(index + 1, 97 * (index + 3))),
                        "y": _fraction_wire(Fraction(index * index + 1, 89)),
                    },
                }
                for index in range(11)
            ),
        ]
        for points in configurations:
            request = CircumradiusProfileRequest(points=points)
            result = circumradius_profile(request)
            actual = len(encode_strict_json(result.model_dump(mode="json")))
            assert actual <= _maximum_profile_wire_bytes(request)


class TestSchemaPublishedBounds:
    def test_points_schema_publishes_coordinate_digit_bound(self):
        schema = CircumradiusProfileRequest.model_json_schema()
        points_schema = schema["properties"]["points"]
        assert points_schema["coordinate_digit_bound"] == CIRCUMRADIUS_INPUT_HEIGHT
        assert CIRCUMRADIUS_INPUT_HEIGHT == 64
        assert (
            points_schema["aggregate_result_budget_bytes"]
            == MAX_CIRCUMRADIUS_PROFILE_RESULT_BYTES
        )
        assert "64 canonical decimal digits" in points_schema["description"]
        assert (
            str(MAX_CIRCUMRADIUS_PROFILE_RESULT_BYTES) in points_schema["description"]
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
            CircumradiusProfileRequest(points=pts)
