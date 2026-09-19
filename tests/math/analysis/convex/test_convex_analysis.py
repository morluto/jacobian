"""Tests for convex analysis operations."""

import json

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
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
)


def compute_max_affine_evaluation(request: MaxAffineEvalRequest) -> MaxAffineEvalResult:
    return max_affine_evaluation(request.function, request.point)


def compute_subdifferential(
    request: MaxAffineSubdifferentialRequest,
) -> MaxAffineSubdifferentialResult:
    return max_affine_subdifferential(request.function, request.point)


def _rational(num: int, den: int = 1) -> CanonicalRational:
    return CanonicalRational(num=num, den=den)


def test_native_surface_accepts_canonical_function_and_point() -> None:
    function = MaxAffineFunction(
        pieces=(
            AffinePiece(
                piece_id="p1",
                coefficients=(_rational(1),),
                intercept=_rational(0),
            ),
            AffinePiece(
                piece_id="p2",
                coefficients=(_rational(-1),),
                intercept=_rational(0),
            ),
        )
    )
    point = RationalPoint(coordinates=(_rational(0),))

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
                        coefficients=(_rational(1),),
                        intercept=_rational(0),
                    ),
                    AffinePiece(
                        piece_id="p2",
                        coefficients=(_rational(-1),),
                        intercept=_rational(0),
                    ),
                ),
            ),
            point=RationalPoint(coordinates=(_rational(2),)),
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
                        coefficients=(_rational(1),),
                        intercept=_rational(0),
                    ),
                    AffinePiece(
                        piece_id="p2",
                        coefficients=(_rational(-1),),
                        intercept=_rational(0),
                    ),
                ),
            ),
            point=RationalPoint(coordinates=(_rational(0),)),
        )
        result = compute_max_affine_evaluation(req)
        assert result.value.as_fraction() == 0
        assert len(result.active_pieces) == 2

    def test_large_source_scalars_are_admitted_when_product_height_is_small(
        self,
    ) -> None:
        height = 20_000
        coefficient = 10**height
        function = MaxAffineFunction(
            pieces=(
                AffinePiece(
                    piece_id="p1",
                    coefficients=(_rational(coefficient),),
                    intercept=_rational(0),
                ),
            )
        )
        point = RationalPoint(coordinates=(_rational(1, coefficient),))

        result = max_affine_evaluation(function, point)

        assert result.value.as_fraction() == 1

    def test_product_at_canonical_height_boundary_is_admitted(self) -> None:
        value = 10**16_383
        function = MaxAffineFunction(
            pieces=(
                AffinePiece(
                    piece_id="p1",
                    coefficients=(_rational(value),),
                    intercept=_rational(0),
                ),
            )
        )
        point = RationalPoint(coordinates=(_rational(value),))

        result = max_affine_evaluation(function, point)

        assert result.value.num == 10**32_766

    @pytest.mark.parametrize(
        "operation", [max_affine_evaluation, max_affine_subdifferential]
    )
    def test_derived_product_height_is_rejected_before_exact_evaluation(
        self, operation
    ) -> None:
        height = 16_384
        value = 10**height
        function = MaxAffineFunction(
            pieces=(
                AffinePiece(
                    piece_id="p1",
                    coefficients=(_rational(value),),
                    intercept=_rational(0),
                ),
            )
        )
        point = RationalPoint(coordinates=(_rational(value),))

        with pytest.raises(OperationResourceAdmissionError) as exc_info:
            operation(function, point)
        assert exc_info.value.errors()[0]["type"] == (
            "convex_analysis.derived_height_exceeded"
        )


class TestSubdifferential:
    def test_subdifferential_at_nonzero(self) -> None:
        """At x=2, only p1 is active, so subdifferential is {(1,)}."""
        req = MaxAffineSubdifferentialRequest(
            function=MaxAffineFunction(
                pieces=(
                    AffinePiece(
                        piece_id="p1",
                        coefficients=(_rational(1),),
                        intercept=_rational(0),
                    ),
                    AffinePiece(
                        piece_id="p2",
                        coefficients=(_rational(-1),),
                        intercept=_rational(0),
                    ),
                ),
            ),
            point=RationalPoint(coordinates=(_rational(2),)),
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
                        coefficients=(_rational(1),),
                        intercept=_rational(0),
                    ),
                    AffinePiece(
                        piece_id="p2",
                        coefficients=(_rational(-1),),
                        intercept=_rational(0),
                    ),
                ),
            ),
            point=RationalPoint(coordinates=(_rational(0),)),
        )
        result = compute_subdifferential(req)
        assert len(result.active_gradients) == 2
        assert result.active_gradients[0][0].as_fraction() in {1, -1}

    def test_rejects_mismatched_point_dimension(self) -> None:
        request = MaxAffineSubdifferentialRequest.model_validate_json(
            json.dumps(
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
        )
        with pytest.raises(OperationDomainValidationError) as exc_info:
            compute_subdifferential(request)
        assert (
            exc_info.value.errors()[0]["type"]
            == "convex_analysis.point_dimension_mismatch"
        )
