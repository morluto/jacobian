"""Independent exact rational LP optimum replay using ``fractions.Fraction``."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

from jacobian_checkers.bound_artifacts import bound_request

_INTEGER = re.compile(r"^(?:0|-?[1-9][0-9]*)$")
_MAX_SOURCE_DIGITS = 128
_MAX_CANDIDATE_DIGITS = 32_768


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


def _integer(value: object, *, maximum_digits: int) -> int:
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


def _q(value: object, *, maximum_digits: int) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise ValueError("rational is malformed")
    numerator = _integer(value["num"], maximum_digits=maximum_digits)
    denominator = _integer(value["den"], maximum_digits=maximum_digits)
    if denominator <= 0:
        raise ValueError("rational denominator must be positive")
    result = Fraction(numerator, denominator)
    if (result.numerator, result.denominator) != (numerator, denominator):
        raise ValueError("rational is not reduced")
    return result


def _vector(value: object, *, length: int, maximum_digits: int) -> list[Fraction]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError("rational vector has the wrong dimension")
    return [_q(item, maximum_digits=maximum_digits) for item in value]


def _dot(left: list[Fraction], right: list[Fraction]) -> Fraction:
    if len(left) != len(right):
        raise ValueError("rational vectors have different dimensions")
    return sum((a * b for a, b in zip(left, right, strict=True)), Fraction())


def check_rational_linear_optimum(request: object) -> dict[str, Any]:
    try:
        source, result = bound_request(
            request,
            operation_id="optimization.linear.rational_optimum.compute",
            witness_format="optimization.linear.rational-optimum.fraction-replay",
        )
        if set(source) != {"program", "wall_seconds"}:
            raise ValueError("LP source fields are malformed")
        if (
            type(source["wall_seconds"]) is not int
            or not 1 <= source["wall_seconds"] <= 60
        ):
            raise ValueError("LP wall-clock budget is malformed")
        program = source["program"]
        if not isinstance(program, dict) or set(program) != {
            "variables",
            "objective",
            "coefficients",
            "rhs",
        }:
            raise ValueError("LP program is malformed")
        variables = program["variables"]
        rows = program["coefficients"]
        if (
            not isinstance(variables, list)
            or not 1 <= len(variables) <= 32
            or len(set(variables)) != len(variables)
            or not isinstance(rows, list)
            or not 1 <= len(rows) <= 64
        ):
            raise ValueError("LP dimensions are outside checker scope")
        width = len(variables)
        height = len(rows)
        objective = _vector(
            program["objective"], length=width, maximum_digits=_MAX_SOURCE_DIGITS
        )
        rhs = _vector(program["rhs"], length=height, maximum_digits=_MAX_SOURCE_DIGITS)
        coefficients = [
            _vector(row, length=width, maximum_digits=_MAX_SOURCE_DIGITS)
            for row in rows
        ]
        expected_fields = {
            "status",
            "primal_candidate",
            "dual_candidate",
            "primal_objective",
            "dual_objective",
            "primal_residuals",
            "dual_slacks",
            "detail",
        }
        if set(result) != expected_fields or result["status"] != "CERTIFICATE_PRODUCED":
            return _reject("candidate does not contain a complete optimum certificate")
        primal = _vector(
            result["primal_candidate"],
            length=width,
            maximum_digits=_MAX_CANDIDATE_DIGITS,
        )
        dual = _vector(
            result["dual_candidate"],
            length=height,
            maximum_digits=_MAX_CANDIDATE_DIGITS,
        )
        primal_residuals = [
            _dot(row, primal) - bound
            for row, bound in zip(coefficients, rhs, strict=True)
        ]
        dual_slacks = [
            objective[column]
            - sum(
                (coefficients[row][column] * dual[row] for row in range(height)),
                Fraction(),
            )
            for column in range(width)
        ]
        primal_objective = _dot(objective, primal)
        dual_objective = _dot(rhs, dual)
        if (
            any(value < 0 for value in primal)
            or any(value != 0 for value in primal_residuals)
            or any(value < 0 for value in dual_slacks)
            or primal_objective != dual_objective
            or _vector(
                result["primal_residuals"],
                length=height,
                maximum_digits=_MAX_CANDIDATE_DIGITS,
            )
            != primal_residuals
            or _vector(
                result["dual_slacks"],
                length=width,
                maximum_digits=_MAX_CANDIDATE_DIGITS,
            )
            != dual_slacks
            or _q(result["primal_objective"], maximum_digits=_MAX_CANDIDATE_DIGITS)
            != primal_objective
            or _q(result["dual_objective"], maximum_digits=_MAX_CANDIDATE_DIGITS)
            != dual_objective
        ):
            return _reject("candidate does not establish exact primal/dual equality")
        return _accept("independent exact rational primal/dual replay accepted optimum")
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return _reject("malformed, unsupported, or mismatched checker request")


__all__ = ["check_rational_linear_optimum"]
