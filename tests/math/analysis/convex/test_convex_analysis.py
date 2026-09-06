"""Tests for convex analysis operations."""

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.analysis.convex._models import (
    AffinePiece,
    MaxAffineEvalRequest,
    MaxAffineEvalResult,
    MaxAffineFunction,
    MaxAffineSubdifferentialRequest,
    MaxAffineSubdifferentialResult,
    RationalPoint,
)
from jacobian.math.analysis.convex.operations import (
    max_affine_evaluation,
    max_affine_subdifferential,
    verify_max_affine_evaluation,
    verify_max_affine_subdifferential,
)


def compute_max_affine_evaluation(request: MaxAffineEvalRequest) -> MaxAffineEvalResult:
    return max_affine_evaluation(request.function, request.point)


def compute_subdifferential(
    request: MaxAffineSubdifferentialRequest,
) -> MaxAffineSubdifferentialResult:
    return max_affine_subdifferential(request.function, request.point)


def _rational(num: str, den: str = "1") -> CanonicalRational:
    return CanonicalRational(num=num, den=den)


def test_native_surface_accepts_canonical_function_and_point() -> None:
    function = MaxAffineFunction(
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
        )
    )
    point = RationalPoint(coordinates=(_rational("0"),))

    assert max_affine_evaluation(function, point).active_pieces == ("p1", "p2")
    assert len(max_affine_subdifferential(function, point).active_gradients) == 2


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
        assert result.value.as_fraction() == 2
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
        assert result.value.as_fraction() == 0
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


def test_max_affine_claims_retain_sources_and_reject_forgery() -> None:
    function = MaxAffineFunction(
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
        )
    )
    point = RationalPoint(coordinates=(_rational("2"),))

    evaluation = max_affine_evaluation(function, point)
    decoded = type(evaluation).model_validate_json(evaluation.model_dump_json())
    assert verify_max_affine_evaluation(decoded)
    payload = evaluation.model_dump(mode="json")
    payload["active_pieces"] = ["p2"]
    assert not verify_max_affine_evaluation(type(evaluation).model_validate(payload))

    subdifferential = max_affine_subdifferential(function, point)
    decoded_subdifferential = type(subdifferential).model_validate_json(
        subdifferential.model_dump_json()
    )
    assert verify_max_affine_subdifferential(decoded_subdifferential)
    payload = subdifferential.model_dump(mode="json")
    payload["active_gradients"] = [[{"num": "-1", "den": "1"}]]
    assert not verify_max_affine_subdifferential(
        type(subdifferential).model_validate(payload)
    )

    def test_rejects_mismatched_point_dimension(self) -> None:
        request = MaxAffineSubdifferentialRequest.model_validate(
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
        with pytest.raises(OperationDomainValidationError) as exc_info:
            compute_subdifferential(request)
        assert (
            exc_info.value.errors()[0]["type"]
            == "convex_analysis.point_dimension_mismatch"
        )
