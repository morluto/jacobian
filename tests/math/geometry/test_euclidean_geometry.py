"""Tests for Euclidean geometry operations."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.canonical import encode_strict_json, format_canonical_integer
from jacobian.math.geometry._models import (
    MAX_TRIANGULATION_OUTPUT_CHARS,
    ConvexPolygonTriangulationRequest,
    ConvexPolygonTriangulationResult,
)
from jacobian.math.geometry._models import (
    RationalPoint2D as GeometryRationalPoint2D,
)
from jacobian.math.geometry._triangulation import minimum_weight_triangulation
from jacobian.math.geometry.euclidean._models import (
    AngleEqualityRequest,
    RationalPoint2D,
    SegmentRatioRequest,
    Triangle,
    TriangleSimilarityRequest,
)
from jacobian.math.geometry.euclidean._operations import (
    compute_angle_equality,
    compute_segment_ratio,
    compute_triangle_similarity,
)


def _pt(x, y):
    return RationalPoint2D(x={"num": str(x), "den": "1"}, y={"num": str(y), "den": "1"})


def test_euclidean_points_use_the_canonical_geometry_value():
    point = _pt(1, 2)

    assert RationalPoint2D is GeometryRationalPoint2D
    assert GeometryRationalPoint2D.model_validate(point) is point


class TestSegmentRatio:
    def test_equal_segments(self):
        req = SegmentRatioRequest(
            segment1=(_pt(0, 0), _pt(1, 0)),
            segment2=(_pt(0, 0), _pt(1, 0)),
        )
        result = compute_segment_ratio(req)
        assert result.squared_ratio == "1"

    def test_double_length(self):
        req = SegmentRatioRequest(
            segment1=(_pt(0, 0), _pt(2, 0)),
            segment2=(_pt(0, 0), _pt(1, 0)),
        )
        result = compute_segment_ratio(req)
        assert result.squared_ratio == "4"

    def test_rejects_zero_second_segment(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SegmentRatioRequest(
                segment1=(_pt(0, 0), _pt(1, 0)),
                segment2=(_pt(0, 0), _pt(0, 0)),
            )


class TestRationalWeightTriangulation:
    @staticmethod
    def _ring(*coordinates: tuple[int, int]):
        return tuple(
            {"x": {"num": str(x), "den": "1"}, "y": {"num": str(y), "den": "1"}}
            for x, y in coordinates
        )

    def test_charges_a_selected_diagonal_once(self):
        request = ConvexPolygonTriangulationRequest(
            polygon={
                "points": (
                    {"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}},
                    {"x": {"num": "1", "den": "1"}, "y": {"num": "0", "den": "1"}},
                    {"x": {"num": "1", "den": "1"}, "y": {"num": "1", "den": "1"}},
                    {"x": {"num": "0", "den": "1"}, "y": {"num": "1", "den": "1"}},
                )
            },
            diagonal_weights=(
                {"first": 0, "second": 2, "weight": {"num": "1", "den": "1"}},
                {"first": 1, "second": 3, "weight": {"num": "2", "den": "1"}},
            ),
        )

        result = minimum_weight_triangulation(request)

        assert result.optimum.as_fraction() == 1
        assert tuple((edge.first, edge.second) for edge in result.diagonals) == (
            (0, 2),
        )

    @staticmethod
    def _weights(pairs, assignments: dict[tuple[int, int], tuple[str, str]]):
        return tuple(
            {
                "first": first,
                "second": second,
                "weight": {
                    "num": assignments.get((first, second), ("0", "1"))[0],
                    "den": assignments.get((first, second), ("0", "1"))[1],
                },
            }
            for first, second in pairs
        )

    _PENTAGON = ((0, 0), (1, 0), (2, 1), (1, 2), (0, 1))
    _HEXAGON = ((0, 0), (1, 0), (2, 1), (2, 2), (1, 2), (0, 1))
    _PENTAGON_DIAGONALS = ((0, 2), (0, 3), (1, 3), (1, 4), (2, 4))
    _HEXAGON_DIAGONALS = (
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 5),
        (3, 5),
    )

    def test_boundary_ledger_overflow_is_rejected_at_request_validation(self):
        weights = self._weights(
            self._PENTAGON_DIAGONALS,
            {
                (0, 2): ("1", format_canonical_integer(10**20000 + 1)),
                (0, 3): ("1", format_canonical_integer(10**20000 + 3)),
                (1, 3): ("1", "1"),
            },
        )

        with pytest.raises(ValidationError):
            ConvexPolygonTriangulationRequest(
                polygon={"points": self._ring(*self._PENTAGON)},
                diagonal_weights=weights,
            )

    def test_single_large_weight_remains_admitted(self):
        denominator = 10**30000 + 3
        weights = self._weights(
            self._HEXAGON_DIAGONALS,
            {(0, 2): ("1", format_canonical_integer(denominator))},
        )

        request = ConvexPolygonTriangulationRequest(
            polygon={"points": self._ring(*self._HEXAGON)},
            diagonal_weights=weights,
        )
        result = minimum_weight_triangulation(request)

        assert result.optimum.as_fraction() == 0
        entry = next(
            item for item in result.split_table if (item.start, item.end) == (0, 2)
        )
        assert entry.optimum.as_fraction() == Fraction(1, denominator)
        validated = ConvexPolygonTriangulationResult.model_validate(
            result.model_dump(mode="json")
        )
        assert validated.optimum.as_fraction() == 0
        assert tuple(edge.weight for edge in validated.diagonals) == tuple(
            edge.weight for edge in result.diagonals
        )

    def test_crossing_large_weights_remain_admitted(self):
        # (0,2) and (1,3) cross, so no triangulation - and therefore no
        # split-table ledger sum - can contain both 20,001-digit
        # denominators; feasibility-aware admission must accept them.
        denominator = 10**20000 + 5
        weights = self._weights(
            self._PENTAGON_DIAGONALS,
            {
                (0, 2): ("1", format_canonical_integer(denominator)),
                (1, 3): ("1", format_canonical_integer(denominator)),
            },
        )

        request = ConvexPolygonTriangulationRequest(
            polygon={"points": self._ring(*self._PENTAGON)},
            diagonal_weights=weights,
        )
        result = minimum_weight_triangulation(request)

        assert result.optimum.as_fraction() == 0
        entry = next(
            item for item in result.split_table if (item.start, item.end) == (0, 2)
        )
        assert entry.optimum.as_fraction() == Fraction(1, denominator)
        validated = ConvexPolygonTriangulationResult.model_validate(
            result.model_dump(mode="json")
        )
        assert validated.optimum.as_fraction() == 0

    def test_shared_denominator_weights_stay_admitted_with_ledger_invariant(self):
        # Every non-hull diagonal carries the same 20,001-digit denominator,
        # so each retained ledger optimum is k/P with that same denominator -
        # shared factors cancel and height multiplication would over-reject.
        denominator = format_canonical_integer(10**20000 + 5)
        weights = self._weights(
            self._PENTAGON_DIAGONALS,
            dict.fromkeys(self._PENTAGON_DIAGONALS, ("1", denominator)),
        )

        request = ConvexPolygonTriangulationRequest(
            polygon={"points": self._ring(*self._PENTAGON)},
            diagonal_weights=weights,
        )
        result = minimum_weight_triangulation(request)

        assert result.optimum.as_fraction() == Fraction(2, 10**20000 + 5)
        for entry in result.split_table:
            assert len(entry.optimum.den) == 20001

    def test_noncrossing_large_weight_pair_is_still_rejected(self):
        # Distinct large denominators on the noncrossing pair ((0,2), (0,3))
        # are forced into one retained ledger sum by w(1,3)=1, so their
        # reduced product denominator genuinely exceeds the canonical cap.
        weights = self._weights(
            self._PENTAGON_DIAGONALS,
            {
                (0, 2): ("1", format_canonical_integer(10**20000 + 5)),
                (0, 3): ("1", format_canonical_integer(10**20000 + 7)),
                (1, 3): ("1", "1"),
            },
        )

        with pytest.raises(ValidationError):
            ConvexPolygonTriangulationRequest(
                polygon={"points": self._ring(*self._PENTAGON)},
                diagonal_weights=weights,
            )

    def test_boundary_height_pair_stays_admitted_with_ledger_invariant(self):
        small = 10**16383 + 1
        large = 10**16383 + 7
        huge = 10**100 + 9
        weights = self._weights(
            self._PENTAGON_DIAGONALS,
            {
                (0, 2): ("1", format_canonical_integer(small)),
                (0, 3): ("1", format_canonical_integer(large)),
                (1, 3): (format_canonical_integer(huge), "1"),
            },
        )

        request = ConvexPolygonTriangulationRequest(
            polygon={"points": self._ring(*self._PENTAGON)},
            diagonal_weights=weights,
        )
        result = minimum_weight_triangulation(request)

        entry = next(
            item for item in result.split_table if (item.start, item.end) == (0, 3)
        )
        assert entry.optimum.as_fraction() == Fraction(1, small) + Fraction(1, large)
        assert len(entry.optimum.den) <= 32_768
        assert result.optimum.as_fraction() == 0

    def test_cap_crossing_forced_sum_is_rejected_at_request_validation(self):
        # w(1,3)=1 forces the (0,3) ledger optimum onto the sum of two
        # 16,385-digit denominators whose reduced product has 32,769 digits.
        weights = self._weights(
            self._PENTAGON_DIAGONALS,
            {
                (0, 2): ("1", format_canonical_integer(10**16384 + 1)),
                (0, 3): ("1", format_canonical_integer(10**16384 + 3)),
                (1, 3): ("1", "1"),
            },
        )

        with pytest.raises(ValidationError):
            ConvexPolygonTriangulationRequest(
                polygon={"points": self._ring(*self._PENTAGON)},
                diagonal_weights=weights,
            )

    def test_boundary_height_pair_stays_admitted_at_the_exact_cap(self):
        # The same forced shape one digit below the cap stays admitted: the
        # retained (0,3) optimum is the exact two-term sum, still representable.
        weights = self._weights(
            self._PENTAGON_DIAGONALS,
            {
                (0, 2): ("1", format_canonical_integer(10**16383 + 1)),
                (0, 3): ("1", format_canonical_integer(10**16384 + 3)),
                (1, 3): ("1", "1"),
            },
        )

        request = ConvexPolygonTriangulationRequest(
            polygon={"points": self._ring(*self._PENTAGON)},
            diagonal_weights=weights,
        )
        result = minimum_weight_triangulation(request)

        entry = next(
            item for item in result.split_table if (item.start, item.end) == (0, 3)
        )
        assert entry.optimum.as_fraction() == Fraction(1, 10**16383 + 1) + Fraction(
            1, 10**16384 + 3
        )
        assert len(entry.optimum.den) <= 32_768

    _REVIEW_PENTAGON = ((0, 0), (2, 0), (3, 1), (2, 3), (0, 2))
    _REVIEW_PENTAGON_DIAGONALS = ((0, 2), (0, 3), (1, 3), (1, 4), (2, 4))

    @staticmethod
    def _mixed_extreme_weights(scale: int):
        huge = format_canonical_integer(2 * scale)
        assignments = {
            (0, 2): (format_canonical_integer(scale), "1"),
            (0, 3): ("1", format_canonical_integer(scale + 3)),
            (1, 3): (huge, "1"),
            (1, 4): (huge, "1"),
            (2, 4): (huge, "1"),
        }
        return tuple(
            {
                "first": first,
                "second": second,
                "weight": {
                    "num": assignments[(first, second)][0],
                    "den": assignments[(first, second)][1],
                },
            }
            for first, second in TestRationalWeightTriangulation._REVIEW_PENTAGON_DIAGONALS
        )

    def test_complementary_region_coexistence_is_rejected_at_request_validation(self):
        # Regression: the complementary interval dropped its closing vertex,
        # so anchoring (0,2) hid the coexisting 20,001-digit denominator on
        # (0,3); admission accepted these weights and serializing the ledger
        # sum later raised inside CanonicalRational instead.
        with pytest.raises(ValidationError):
            ConvexPolygonTriangulationRequest(
                polygon={"points": self._ring(*self._REVIEW_PENTAGON)},
                diagonal_weights=self._mixed_extreme_weights(10**20000),
            )

    def test_complementary_region_coexistence_stays_admitted_below_the_cap(self):
        request = ConvexPolygonTriangulationRequest(
            polygon={"points": self._ring(*self._REVIEW_PENTAGON)},
            diagonal_weights=self._mixed_extreme_weights(10**16000),
        )
        result = minimum_weight_triangulation(request)

        expected = Fraction(10**16000) + Fraction(1, 10**16000 + 3)
        assert result.optimum.as_fraction() == expected
        root = next(
            item for item in result.split_table if (item.start, item.end) == (0, 4)
        )
        assert root.optimum.as_fraction() == expected
        validated = ConvexPolygonTriangulationResult.model_validate(
            result.model_dump(mode="json")
        )
        assert validated.optimum == result.optimum

    _UNIFORM_RING = tuple((index, index * index) for index in range(32))
    _UNIFORM_RING_DIAGONALS = tuple(
        (first, second)
        for first in range(32)
        for second in range(first + 1, 32)
        if second != first + 1 and (first, second) != (0, 31)
    )

    def _uniform_ring_weights(self, numerator: str):
        return self._weights(
            self._UNIFORM_RING_DIAGONALS,
            dict.fromkeys(self._UNIFORM_RING_DIAGONALS, (numerator, "1")),
        )

    def test_uniform_heavy_ring_aggregate_overflow_rejected_at_request_validation(
        self,
    ):
        # Regression: every non-hull diagonal carries the same 22,000-digit
        # integer 10^21999, so each derived ledger optimum stays far below
        # the canonical component cap while the aggregate serialization of
        # the full split table and echoed diagonals overflows the output
        # envelope. Admission must reject the request typedly before
        # execution instead of failing canonical output validation after
        # computing the triangulation.
        weights = self._uniform_ring_weights(format_canonical_integer(10**21999))

        with pytest.raises(ValidationError):
            ConvexPolygonTriangulationRequest(
                polygon={"points": self._ring(*self._UNIFORM_RING)},
                diagonal_weights=weights,
            )

    def test_uniform_ring_with_fitting_aggregate_stays_certified(self):
        # The same uniform ring with materially lighter weights keeps the
        # per-entry bound times the entry count inside the published budget,
        # so the request executes, certifies, round-trips, and its encoded
        # result fits the aggregate envelope admission predicted.
        weight = 10**4000
        request = ConvexPolygonTriangulationRequest(
            polygon={"points": self._ring(*self._UNIFORM_RING)},
            diagonal_weights=self._uniform_ring_weights(
                format_canonical_integer(weight)
            ),
        )
        result = minimum_weight_triangulation(request)

        assert result.optimum.as_fraction() == 29 * weight
        validated = ConvexPolygonTriangulationResult.model_validate(
            result.model_dump(mode="json")
        )
        assert validated.optimum == result.optimum
        encoded = encode_strict_json(result.model_dump(mode="json"))
        assert len(encoded) <= MAX_TRIANGULATION_OUTPUT_CHARS

    def test_lone_heavy_weight_ring_aggregate_is_measured_exactly(self):
        # Regression: charging every retained state and echoed diagonal at
        # the largest component height estimated ~29.7 MB here, although
        # only the (0,2) ledger entry retains the 30,001-digit denominator,
        # the root selects zero-weight fan diagonals, and the canonical
        # result encodes to ~62 KB. Admission must sum the exact serialized
        # sizes from the recurrence values and reconstructed selected
        # weights instead, so this ring is admitted and its encoded result
        # stays inside the aggregate envelope.
        denominator = format_canonical_integer(10**30000 + 1)
        request = ConvexPolygonTriangulationRequest(
            polygon={"points": self._ring(*self._UNIFORM_RING)},
            diagonal_weights=self._weights(
                self._UNIFORM_RING_DIAGONALS,
                {(0, 2): ("1", denominator)},
            ),
        )
        result = minimum_weight_triangulation(request)

        assert result.optimum.as_fraction() == 0
        assert len(result.split_table) == 465
        assert len(result.diagonals) == 29
        assert len(result.triangles) == 30
        assert all(edge.weight.as_fraction() == 0 for edge in result.diagonals)
        entry = next(
            item for item in result.split_table if (item.start, item.end) == (0, 2)
        )
        assert entry.optimum.as_fraction() == Fraction(1, 10**30000 + 1)
        validated = ConvexPolygonTriangulationResult.model_validate(
            result.model_dump(mode="json")
        )
        assert validated.optimum == result.optimum
        encoded = encode_strict_json(result.model_dump(mode="json"))
        assert len(encoded) <= MAX_TRIANGULATION_OUTPUT_CHARS

    def test_uniform_heavy_denominator_ring_aggregate_overflow_rejected(self):
        # Every non-hull diagonal carries the same 16,001-digit denominator,
        # so each shared-denominator ledger optimum stays far below the
        # canonical component cap while their combined serialization
        # genuinely overflows the output envelope. Exact size summation must
        # still reject this request at request validation.
        denominator = format_canonical_integer(10**16000 + 1)
        weights = self._weights(
            self._UNIFORM_RING_DIAGONALS,
            dict.fromkeys(self._UNIFORM_RING_DIAGONALS, ("1", denominator)),
        )

        with pytest.raises(ValidationError):
            ConvexPolygonTriangulationRequest(
                polygon={"points": self._ring(*self._UNIFORM_RING)},
                diagonal_weights=weights,
            )


class TestAngleEquality:
    def test_right_angles(self):
        req = AngleEqualityRequest(
            vertex1=_pt(0, 0),
            ray1_a=_pt(1, 0),
            ray1_b=_pt(0, 1),
            vertex2=_pt(0, 0),
            ray2_a=_pt(0, 1),
            ray2_b=_pt(-1, 0),
        )
        result = compute_angle_equality(req)
        assert result.equal is True

    def test_different_angles(self):
        req = AngleEqualityRequest(
            vertex1=_pt(0, 0),
            ray1_a=_pt(1, 0),
            ray1_b=_pt(0, 1),
            vertex2=_pt(0, 0),
            ray2_a=_pt(1, 0),
            ray2_b=_pt(1, 1),
        )
        result = compute_angle_equality(req)
        assert result.equal is False

    def test_supplementary_angles_are_not_equal(self):
        req = AngleEqualityRequest(
            vertex1=_pt(0, 0),
            ray1_a=_pt(1, 0),
            ray1_b=_pt(1, 1),
            vertex2=_pt(0, 0),
            ray2_a=_pt(1, 0),
            ray2_b=_pt(-1, -1),
        )
        result = compute_angle_equality(req)
        assert result.equal is False

    def test_rejects_zero_length_ray(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AngleEqualityRequest(
                vertex1=_pt(0, 0),
                ray1_a=_pt(0, 0),
                ray1_b=_pt(0, 1),
                vertex2=_pt(0, 0),
                ray2_a=_pt(1, 0),
                ray2_b=_pt(0, 1),
            )


class TestTriangleSimilarity:
    def test_similar(self):
        req = TriangleSimilarityRequest(
            triangle1=Triangle(a=_pt(0, 0), b=_pt(1, 0), c=_pt(0, 1)),
            triangle2=Triangle(a=_pt(0, 0), b=_pt(2, 0), c=_pt(0, 2)),
        )
        result = compute_triangle_similarity(req)
        assert result.similar is True

    def test_not_similar(self):
        req = TriangleSimilarityRequest(
            triangle1=Triangle(a=_pt(0, 0), b=_pt(1, 0), c=_pt(0, 1)),
            triangle2=Triangle(a=_pt(0, 0), b=_pt(2, 0), c=_pt(0, 3)),
        )
        result = compute_triangle_similarity(req)
        assert result.similar is False
