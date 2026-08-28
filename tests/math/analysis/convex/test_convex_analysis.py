"""Tests for convex analysis operations."""

from jacobian._exact import CanonicalRational
from jacobian.math.analysis.convex._models import (
    AffinePiece,
    MaxAffineEvalRequest,
    MaxAffineFunction,
    MaxAffineSubdifferentialRequest,
    RationalPoint,
)
from jacobian.math.analysis.convex._operations import (
    compute_max_affine_evaluation,
    compute_subdifferential,
)


def _rational(num: str, den: str = "1") -> CanonicalRational:
    return CanonicalRational(num=num, den=den)


class TestMaxAffineEvaluation:
    def test_simple_max(self) -> None:
        """max(x, -x) at x=2 should give 2."""
        req = MaxAffineEvalRequest(
            function=MaxAffineFunction(
                pieces=(
                    AffinePiece(
                        piece_id="p1",
                        coefficients=(_rational("1"),),
                        intercept=_rational("0"),
                    ),
                    AffinePiece(
                        piece_id="p2",
                        coefficients=(_rational("-1"),),
                        intercept=_rational("0"),
                    ),
                ),
            ),
            point=RationalPoint(coordinates=(_rational("2"),)),
        )
        result = compute_max_affine_evaluation(req)
        assert result.value == "2"
        assert "p1" in result.active_pieces

    def test_tie(self) -> None:
        """max(x, -x) at x=0 should give 0 with both pieces active."""
        req = MaxAffineEvalRequest(
            function=MaxAffineFunction(
                pieces=(
                    AffinePiece(
                        piece_id="p1",
                        coefficients=(_rational("1"),),
                        intercept=_rational("0"),
                    ),
                    AffinePiece(
                        piece_id="p2",
                        coefficients=(_rational("-1"),),
                        intercept=_rational("0"),
                    ),
                ),
            ),
            point=RationalPoint(coordinates=(_rational("0"),)),
        )
        result = compute_max_affine_evaluation(req)
        assert result.value == "0"
        assert len(result.active_pieces) == 2


class TestSubdifferential:
    def test_subdifferential_at_nonzero(self) -> None:
        """At x=2, only p1 is active, so subdifferential is {(1,)}."""
        req = MaxAffineSubdifferentialRequest(
            function=MaxAffineFunction(
                pieces=(
                    AffinePiece(
                        piece_id="p1",
                        coefficients=(_rational("1"),),
                        intercept=_rational("0"),
                    ),
                    AffinePiece(
                        piece_id="p2",
                        coefficients=(_rational("-1"),),
                        intercept=_rational("0"),
                    ),
                ),
            ),
            point=RationalPoint(coordinates=(_rational("2"),)),
        )
        result = compute_subdifferential(req)
        assert len(result.active_gradients) == 1

    def test_subdifferential_at_origin(self) -> None:
        """At x=0, both pieces are active."""
        req = MaxAffineSubdifferentialRequest(
            function=MaxAffineFunction(
                pieces=(
                    AffinePiece(
                        piece_id="p1",
                        coefficients=(_rational("1"),),
                        intercept=_rational("0"),
                    ),
                    AffinePiece(
                        piece_id="p2",
                        coefficients=(_rational("-1"),),
                        intercept=_rational("0"),
                    ),
                ),
            ),
            point=RationalPoint(coordinates=(_rational("0"),)),
        )
        result = compute_subdifferential(req)
        assert len(result.active_gradients) == 2
        assert result.active_gradients[0][0].as_fraction() in {1, -1}

    def test_rejects_mismatched_point_dimension(self) -> None:
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            MaxAffineSubdifferentialRequest.model_validate(
                {
                    "function": {
                        "pieces": [
                            {
                                "piece_id": "p1",
                                "coefficients": [
                                    {"num": "1", "den": "1"},
                                    {"num": "0", "den": "1"},
                                ],
                                "intercept": {"num": "0", "den": "1"},
                            },
                        ],
                    },
                    "point": {"coordinates": [{"num": "1", "den": "1"}]},
                }
            )
        assert (
            exc_info.value.errors()[0]["type"]
            == "convex_analysis.point_dimension_mismatch"
        )
