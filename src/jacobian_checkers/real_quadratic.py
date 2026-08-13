"""Independent exact real-quadratic order replay."""

from __future__ import annotations

from fractions import Fraction
from math import isqrt
from typing import Any

from jacobian_checkers.bound_artifacts import bound_request


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_RATIONAL",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _accept(detail: str) -> dict[str, Any]:
    return {
        "accepted": True,
        "conclusion": "TRUE",
        "arithmetic": "EXACT_RATIONAL",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _q(value: object) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise ValueError("rational is malformed")
    numerator = int(value["num"])
    denominator = int(value["den"])
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    result = Fraction(numerator, denominator)
    if str(result.numerator) != value["num"] or str(result.denominator) != value["den"]:
        raise ValueError("rational is not canonical")
    return result


def _wire(value: Fraction) -> dict[str, str]:
    return {"num": str(value.numerator), "den": str(value.denominator)}


def _value(raw: object) -> tuple[Fraction, Fraction, int]:
    if not isinstance(raw, dict) or set(raw) != {
        "rational_part",
        "radical_coefficient",
        "radicand",
    }:
        raise ValueError("real-quadratic value is malformed")
    d = raw["radicand"]
    if type(d) is not int or not 2 <= d <= 1_000_000:
        raise ValueError("radicand is outside checker scope")
    if any(d % (factor * factor) == 0 for factor in range(2, isqrt(d) + 1)):
        raise ValueError("radicand is not square-free")
    return _q(raw["rational_part"]), _q(raw["radical_coefficient"]), d


def _sign(a: Fraction, b: Fraction, d: int) -> int:
    if b == 0:
        return (a > 0) - (a < 0)
    if a == 0:
        return (b > 0) - (b < 0)
    if (a > 0) == (b > 0):
        return (a > 0) - (a < 0)
    left = a * a
    right = b * b * d
    if left == right:
        raise ValueError("square-free magnitudes cannot tie")
    dominant = b if right > left else a
    return (dominant > 0) - (dominant < 0)


def check_real_quadratic_order(request: object) -> dict[str, Any]:
    try:
        source, result = bound_request(
            request,
            operation_id="arithmetic.real_quadratic.order.compute",
            witness_format="arithmetic.real-quadratic.fraction-square-replay",
        )
        if set(source) != {"left", "right"}:
            raise ValueError("request fields are malformed")
        left_a, left_b, d = _value(source["left"])
        right_a, right_b, right_d = _value(source["right"])
        if d != right_d:
            raise ValueError("values do not share a radicand")
        a = left_a - right_a
        b = left_b - right_b
        sign = _sign(a, b, d)
        rational_square = a * a
        radical_square = b * b * d
        expected = {
            "left": source["left"],
            "right": source["right"],
            "difference": {
                "rational_part": _wire(a),
                "radical_coefficient": _wire(b),
                "radicand": d,
            },
            "order": "LT" if sign < 0 else "GT" if sign > 0 else "EQ",
            "sign_basis": (
                "RATIONAL_ONLY"
                if b == 0
                else "RADICAL_ONLY"
                if a == 0
                else "SAME_SIGN"
                if (a > 0) == (b > 0)
                else "OPPOSING_SIGNS_SQUARED_MAGNITUDES"
            ),
            "sign_certificate": {
                "rational_part_squared": _wire(rational_square),
                "radical_part_squared": _wire(radical_square),
                "magnitude_order": (
                    "LT"
                    if rational_square < radical_square
                    else "GT"
                    if rational_square > radical_square
                    else "EQ"
                ),
            },
        }
        if result != expected:
            return _reject("candidate does not match independent exact replay")
        return _accept("independent squared-magnitude replay accepted exact order")
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return _reject("malformed, unsupported, or mismatched checker request")


__all__ = ["check_real_quadratic_order"]
