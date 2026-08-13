"""Independent bounded LCM replay using only the standard library."""

from __future__ import annotations

import re
from typing import Any

from jacobian_checkers.bound_artifacts import bound_request

_INTEGER = re.compile(r"^(?:0|-?[1-9][0-9]*)$")
_MAX_INPUT_DIGITS = 256
_MAX_RESULT_DIGITS = 512


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _accept(detail: str) -> dict[str, Any]:
    return {
        "accepted": True,
        "conclusion": "TRUE",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _canonical_integer(value: object, *, maximum_digits: int) -> int:
    if (
        not isinstance(value, str)
        or len(value.lstrip("-")) > maximum_digits
        or _INTEGER.fullmatch(value) is None
    ):
        raise ValueError("integer is outside checker scope")
    parsed = int(value)
    if str(parsed) != value:
        raise ValueError("integer is not canonical")
    return parsed


def _gcd(left: int, right: int) -> int:
    left = abs(left)
    right = abs(right)
    while right:
        left, right = right, left % right
    return left


def check_integer_lcm(request: object) -> dict[str, Any]:
    try:
        source, result = bound_request(
            request,
            operation_id="integer.compute.lcm",
            witness_format="integer.lcm.euclidean-replay",
        )
        if set(source) != {"left", "right"} or set(result) != {"value"}:
            raise ValueError("LCM source or result fields are malformed")
        left = _canonical_integer(source["left"], maximum_digits=_MAX_INPUT_DIGITS)
        right = _canonical_integer(source["right"], maximum_digits=_MAX_INPUT_DIGITS)
        declared = _canonical_integer(
            result["value"], maximum_digits=_MAX_RESULT_DIGITS
        )
        divisor = _gcd(left, right)
        expected = 0 if divisor == 0 else abs((left // divisor) * right)
        if declared < 0 or declared != expected:
            return _reject("result does not match independent Euclidean replay")
        return _accept("independent Euclidean recurrence replay accepted bounded LCM")
    except (KeyError, TypeError, ValueError, OverflowError):
        return _reject("malformed, unsupported, or mismatched checker request")


__all__ = ["check_integer_lcm"]
