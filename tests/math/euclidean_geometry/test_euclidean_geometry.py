"""Tests for Euclidean geometry operations."""

from jacobian.math.euclidean_geometry._models import (
    AngleEqualityRequest,
    RationalPoint2D,
    SegmentRatioRequest,
    Triangle,
    TriangleSimilarityRequest,
)
from jacobian.math.euclidean_geometry._operations import (
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
