"""Independent exact replay for inline rational-linear candidates.

This module intentionally uses only the Python standard library. It does not
import Jacobian contracts, Python-FLINT, SymPy, or any producer implementation.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

from jacobian_checkers.bound_artifacts import bound_request

_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_VARIABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
MAX_LINEAR_DIMENSION = 32
MAX_RATIONAL_DIGITS = 256


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_RATIONAL",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _rational(value: object) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise ValueError("rational value has an invalid shape")
    numerator = value["num"]
    denominator = value["den"]
    if (
        not isinstance(numerator, str)
        or not isinstance(denominator, str)
        or _INTEGER.fullmatch(numerator) is None
        or _INTEGER.fullmatch(denominator) is None
        or len(numerator.lstrip("-")) > MAX_RATIONAL_DIGITS
        or len(denominator.lstrip("-")) > MAX_RATIONAL_DIGITS
    ):
        raise ValueError("rational value is not bounded canonical integer data")
    result = Fraction(int(numerator), int(denominator))
    if str(result.numerator) != numerator or str(result.denominator) != denominator:
        raise ValueError("rational value is not reduced and canonical")
    return result


def _validate_system(
    payload: object,
) -> tuple[list[list[Fraction]], list[Fraction], list[str]]:
    if not isinstance(payload, dict) or set(payload) != {
        "system_schema_version",
        "domain",
        "relation",
        "variables",
        "coefficients",
        "rhs",
    }:
        raise ValueError("rational linear system has an invalid shape")
    if (
        payload["system_schema_version"] != "1"
        or payload["domain"] != "QQ"
        or payload["relation"] != "AX_EQUALS_B"
    ):
        raise ValueError("rational linear system uses unsupported semantics")
    variables = payload["variables"]
    if (
        not isinstance(variables, list)
        or not 1 <= len(variables) <= MAX_LINEAR_DIMENSION
        or any(
            not isinstance(variable, str) or _VARIABLE.fullmatch(variable) is None
            for variable in variables
        )
        or len(variables) != len(set(variables))
    ):
        raise ValueError("rational linear-system variables are malformed")
    matrix = payload["coefficients"]
    if not isinstance(matrix, dict) or set(matrix) != {
        "matrix_schema_version",
        "domain",
        "entries",
    }:
        raise ValueError("rational coefficient matrix has an invalid shape")
    if matrix["matrix_schema_version"] != "1" or matrix["domain"] != "QQ":
        raise ValueError("rational coefficient matrix uses unsupported semantics")
    entries = matrix["entries"]
    rhs = payload["rhs"]
    if (
        not isinstance(entries, list)
        or not isinstance(rhs, list)
        or not 1 <= len(entries) <= MAX_LINEAR_DIMENSION
        or len(entries) != len(rhs)
        or any(
            not isinstance(row, list) or len(row) != len(variables) for row in entries
        )
    ):
        raise ValueError("rational linear-system dimensions do not match")
    return (
        [[_rational(value) for value in row] for row in entries],
        [_rational(value) for value in rhs],
        variables,
    )


def _check_inline_solution_request(request: dict[str, Any]) -> dict[str, Any]:
    claim, candidate = bound_request(
        request,
        operation_id="linear.rational_solution.compute",
        witness_format="linear.rational_solution",
    )
    coefficients, rhs, variables = _validate_system(claim["system"])
    if set(candidate) not in (
        {"result_schema_version", "values", "method"},
        {"result_schema_version", "status", "values", "method"},
    ):
        return _reject("inline rational solution is malformed")
    if candidate.get("status", "SOLUTION_PRODUCED") != "SOLUTION_PRODUCED":
        return _reject("inline rational solution has no candidate")
    if candidate["method"] != "RREF_FREE_VARIABLES_ZERO":
        return _reject("inline rational solution uses unsupported semantics")
    values = [_rational(value) for value in candidate["values"]]
    if len(values) != len(variables):
        return _reject("inline rational solution is not a total vector")
    if any(
        sum(
            (
                coefficient * value
                for coefficient, value in zip(row, values, strict=True)
            ),
            Fraction(0),
        )
        != expected
        for row, expected in zip(coefficients, rhs, strict=True)
    ):
        return _reject("candidate does not satisfy every bound equation")
    return {
        "accepted": True,
        "conclusion": "TRUE",
        "arithmetic": "EXACT_RATIONAL",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": "replayed every bounded equation over exact rationals",
    }


def _check_inline_inconsistency_request(request: dict[str, Any]) -> dict[str, Any]:
    claim, candidate = bound_request(
        request,
        operation_id="linear.rational_inconsistency.compute",
        witness_format="linear.rational_inconsistency",
    )
    coefficients, rhs, variables = _validate_system(claim["system"])
    if set(candidate) not in (
        {"result_schema_version", "left_witness", "rhs_pairing", "method"},
        {
            "result_schema_version",
            "status",
            "left_witness",
            "rhs_pairing",
            "method",
        },
    ):
        return _reject("inline inconsistency witness is malformed")
    if candidate.get("status", "CERTIFICATE_PRODUCED") != "CERTIFICATE_PRODUCED":
        return _reject("inline inconsistency result has no candidate")
    if candidate["method"] != "DUAL_RREF_PAIRING_ONE":
        return _reject("inline inconsistency witness uses unsupported semantics")
    values = [_rational(value) for value in candidate["left_witness"]]
    pairing = _rational(candidate["rhs_pairing"])
    if len(values) != len(coefficients) or pairing != 1:
        return _reject("inline inconsistency witness is not normalized")
    if any(
        sum(
            (values[row] * coefficients[row][column] for row in range(len(values))),
            Fraction(0),
        )
        != 0
        for column in range(len(variables))
    ):
        return _reject("left witness does not annihilate every column")
    if (
        sum(
            (value * bound for value, bound in zip(values, rhs, strict=True)),
            Fraction(0),
        )
        != pairing
    ):
        return _reject("left witness does not reproduce its right-hand-side pairing")
    return {
        "accepted": True,
        "conclusion": "TRUE",
        "arithmetic": "EXACT_RATIONAL",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": "replayed the normalized left witness over exact rationals",
    }


def check_rational_solution(request: dict[str, Any]) -> dict[str, Any]:
    """Accept only a v2 inline exact vector satisfying every bound equation."""

    try:
        return _check_inline_solution_request(request)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return _reject("malformed rational solution request")


def check_rational_inconsistency(request: dict[str, Any]) -> dict[str, Any]:
    """Accept only a v2 inline normalized left-nullspace witness."""

    try:
        return _check_inline_inconsistency_request(request)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return _reject("malformed rational inconsistency request")
