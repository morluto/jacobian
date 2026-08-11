"""Independent Python-FLINT replay for bounded scalar rational arithmetic.

The checked producers use standard-library ``Fraction`` arithmetic. This
module imports neither producer implementation nor Jacobian code; only passive
JSON values cross the checker boundary.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import flint
from flint import fmpq, fmpz

from jacobian_checkers.bound_artifacts import bound_request

_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_MAX_COMPONENT_DIGITS = 32_768
_PYTHON_FLINT_VERSION = "0.9.0"
_FLINT_VERSION = "3.6.0"


def _decision(accepted: bool, detail: str) -> dict[str, Any]:
    return {
        "accepted": accepted,
        "conclusion": "TRUE" if accepted else "UNKNOWN",
        "arithmetic": "EXACT_RATIONAL",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _integer(value: object) -> fmpz:
    if (
        not isinstance(value, str)
        or _INTEGER.fullmatch(value) is None
        or len(value.lstrip("-")) > _MAX_COMPONENT_DIGITS
    ):
        raise ValueError("integer is malformed or outside checker scope")
    return fmpz(value)


def _rational(value: object) -> fmpq:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise ValueError("rational is malformed")
    numerator = _integer(value["num"])
    denominator = _integer(value["den"])
    if denominator <= 0:
        raise ValueError("rational denominator must be positive")
    rational = fmpq(numerator, denominator)
    if rational.numer() != numerator or rational.denom() != denominator:
        raise ValueError("rational is not canonical and reduced")
    return rational


def _replay(
    request: object,
    *,
    operation_id: str,
    witness_format: str,
    operation: Callable[[fmpq, fmpq], fmpq],
) -> dict[str, Any]:
    try:
        if (
            flint.__version__ != _PYTHON_FLINT_VERSION
            or flint.__FLINT_VERSION__ != _FLINT_VERSION
        ):
            return _decision(False, "authorized Python-FLINT runtime is unavailable")
        source, result = bound_request(
            request,
            operation_id=operation_id,
            witness_format=witness_format,
        )
        if set(source) != {"left", "right"} or set(result) != {"value"}:
            raise ValueError("rational operation payload is malformed")
        left = _rational(source["left"])
        right = _rational(source["right"])
        candidate = _rational(result["value"])
        if operation(left, right) != candidate:
            return _decision(
                False,
                "declared result does not match independent Python-FLINT replay",
            )
        return _decision(
            True,
            f"independent Python-FLINT replay accepted {operation_id}",
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return _decision(False, "malformed, unsupported, or mismatched checker request")


def check_rational_sum(request: object) -> dict[str, Any]:
    return _replay(
        request,
        operation_id="rational.compute.sum",
        witness_format="rational.sum.flint-replay",
        operation=lambda left, right: left + right,
    )


def check_rational_difference(request: object) -> dict[str, Any]:
    return _replay(
        request,
        operation_id="rational.compute.difference",
        witness_format="rational.difference.flint-replay",
        operation=lambda left, right: left - right,
    )


def check_rational_product(request: object) -> dict[str, Any]:
    return _replay(
        request,
        operation_id="rational.compute.product",
        witness_format="rational.product.flint-replay",
        operation=lambda left, right: left * right,
    )


def check_rational_quotient(request: object) -> dict[str, Any]:
    return _replay(
        request,
        operation_id="rational.compute.quotient",
        witness_format="rational.quotient.flint-replay",
        operation=lambda left, right: left / right,
    )


__all__ = [
    "check_rational_difference",
    "check_rational_product",
    "check_rational_quotient",
    "check_rational_sum",
]
