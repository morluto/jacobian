"""Contract tests for exact circle inversion of rational planar points."""

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.geometry._models import RationalPoint2D
from jacobian.math.geometry.inversion._models import (
    MAX_INVERSION_INPUT_DIGITS,
    CircleInversionRequest,
    CircleInversionResult,
)
from jacobian.math.geometry.inversion._operations import compute_circle_inversion

_ORIGIN = RationalPoint2D(
    x=CanonicalRational(num="0", den="1"),
    y=CanonicalRational(num="0", den="1"),
)


def _point(x_num, x_den, y_num, y_den) -> RationalPoint2D:
    return RationalPoint2D(
        x=CanonicalRational(num=str(x_num), den=str(x_den)),
        y=CanonicalRational(num=str(y_num), den=str(y_den)),
    )


def _request(
    point_x_num: str,
    point_x_den: str,
    point_y_num: str,
    point_y_den: str,
) -> CircleInversionRequest:
    return CircleInversionRequest(
        center=_ORIGIN,
        power={"num": "1", "den": "1"},
        point=_point(point_x_num, point_x_den, point_y_num, point_y_den),
    )


def test_unit_inversion_known_answer() -> None:
    request = _request("4", "1", "0", "1")
    result = compute_circle_inversion(request)
    assert result.inverted_point.x.num == "1"
    assert result.inverted_point.x.den == "4"
    assert result.inverted_point.y.num == "0"


def test_large_denominator_request_is_rejected_during_validation() -> None:
    # With N = 10^12000 the inverted x-numerator N*(N+1)^2 carries ~36,001
    # digits, beyond the canonical limit; admission must reject the request
    # instead of letting result construction fail.
    with pytest.raises(ValidationError, match=f"{MAX_INVERSION_INPUT_DIGITS}-digit"):
        _request("1", "1" + "0" * 12000, "1", "1" + "0" * 11999 + "1")


def test_boundary_height_request_returns_typed_result() -> None:
    # Inputs at the admitted digit budget stay within the derived output
    # bound: denominators of 2730 digits invert to numerators of ~8191 digits.
    p_den = "1" + "0" * 2729
    q_den = "1" + "0" * 2728 + "7"
    request = _request("1", p_den, "1", q_den)
    result = compute_circle_inversion(request)
    assert len(result.inverted_point.x.num) < 32_768
    revalidated = CircleInversionResult.model_validate(result.model_dump())
    assert revalidated == result


def test_center_cannot_be_inverted() -> None:
    with pytest.raises(ValidationError, match="cannot be inverted"):
        _request("0", "1", "0", "1")


def test_nonpositive_power_is_rejected() -> None:
    with pytest.raises(ValidationError, match="power must be positive"):
        CircleInversionRequest(
            center=_ORIGIN,
            power={"num": "-2", "den": "1"},
            point=_point(4, 1, 0, 1),
        )
