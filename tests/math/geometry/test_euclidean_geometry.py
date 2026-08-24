"""Tests for Euclidean geometry operations."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.canonical import format_canonical_integer
from jacobian.math.geometry._models import (
    ConvexPolygonTriangulationRequest,
    ConvexPolygonTriangulationResult,
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

        with pytest.raises(ValidationError, match="nonzero"):
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

        with pytest.raises(ValidationError, match="split-table ledger sums"):
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

    def test_noncrossing_large_weight_pair_is_still_rejected(self):
        # The same two large denominators on the noncrossing pair
        # ((0,2), (0,3)) fit inside one feasible triangulation, so their
        # combined ledger growth genuinely exceeds the canonical cap.
        denominator = format_canonical_integer(10**20000 + 5)
        weights = self._weights(
            self._PENTAGON_DIAGONALS,
            {(0, 2): ("1", denominator), (0, 3): ("1", denominator)},
        )

        with pytest.raises(ValidationError, match="split-table ledger sums"):
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

    def test_one_digit_taller_pair_is_rejected_at_request_validation(self):
        weights = self._weights(
            self._PENTAGON_DIAGONALS,
            {
                (0, 2): ("1", format_canonical_integer(10**16383 + 1)),
                (0, 3): ("1", format_canonical_integer(10**16384 + 3)),
            },
        )

        with pytest.raises(ValidationError, match="split-table ledger sums"):
            ConvexPolygonTriangulationRequest(
                polygon={"points": self._ring(*self._PENTAGON)},
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

        with pytest.raises(ValidationError, match="nonzero"):
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
