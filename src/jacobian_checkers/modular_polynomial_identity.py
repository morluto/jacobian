"""Independent formal modular-polynomial replay."""

from __future__ import annotations

import re
from typing import Any, TypedDict

from jacobian_checkers.bound_artifacts import bound_request

_INTEGER = re.compile(r"^(?:0|-?[1-9][0-9]*)$")
_VARIABLE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


class _NormalizedTerm(TypedDict):
    coefficient: int
    exponents: list[int]


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


def _normalize(raw: object, *, variables: int, modulus: int) -> list[_NormalizedTerm]:
    if not isinstance(raw, list) or len(raw) > 64:
        raise ValueError("term list is outside checker scope")
    coefficients: dict[tuple[int, ...], int] = {}
    for term in raw:
        if not isinstance(term, dict) or set(term) != {"coefficient", "exponents"}:
            raise ValueError("term is malformed")
        coefficient = term["coefficient"]
        exponents = term["exponents"]
        if (
            not isinstance(coefficient, str)
            or len(coefficient.lstrip("-")) > 256
            or _INTEGER.fullmatch(coefficient) is None
            or not isinstance(exponents, list)
            or len(exponents) != variables
            or any(
                type(value) is not int or not 0 <= value <= 32 for value in exponents
            )
        ):
            raise ValueError("term is outside checker scope")
        vector = tuple(exponents)
        coefficients[vector] = (
            coefficients.get(vector, 0) + int(coefficient)
        ) % modulus
    return [
        {"coefficient": coefficient, "exponents": list(exponents)}
        for exponents, coefficient in sorted(coefficients.items())
        if coefficient
    ]


def check_modular_polynomial_identity(request: object) -> dict[str, Any]:
    try:
        source, result = bound_request(
            request,
            operation_id="modular.polynomial_identity.compute",
            witness_format="modular.polynomial-identity.stdlib-replay",
        )
        if set(source) != {"modulus", "variables", "left", "right"}:
            raise ValueError("source fields are malformed")
        modulus = source["modulus"]
        variables = source["variables"]
        if (
            type(modulus) is not int
            or not 2 <= modulus <= 1_000_000
            or not isinstance(variables, list)
            or not 1 <= len(variables) <= 6
            or len(set(variables)) != len(variables)
            or any(
                not isinstance(name, str) or _VARIABLE.fullmatch(name) is None
                for name in variables
            )
        ):
            raise ValueError("scope is malformed")
        left = _normalize(source["left"], variables=len(variables), modulus=modulus)
        right = _normalize(source["right"], variables=len(variables), modulus=modulus)
        coefficients = {
            tuple(term["exponents"]): int(term["coefficient"]) for term in left
        }
        for term in right:
            vector = tuple(term["exponents"])
            coefficients[vector] = (
                coefficients.get(vector, 0) - int(term["coefficient"])
            ) % modulus
        residual = [
            {"coefficient": coefficient, "exponents": list(exponents)}
            for exponents, coefficient in sorted(coefficients.items())
            if coefficient
        ]
        expected = {
            "modulus": modulus,
            "variable_order": variables,
            "normalized_left": left,
            "normalized_right": right,
            "residual": residual,
            "identical": not residual,
            "comparison_scope": "FORMAL_COEFFICIENTWISE_IDENTITY",
        }
        if result != expected:
            return _reject("candidate does not match independent coefficient replay")
        return _accept("independent formal coefficientwise replay accepted candidate")
    except (KeyError, TypeError, ValueError, OverflowError):
        return _reject("malformed, unsupported, or mismatched checker request")


__all__ = ["check_modular_polynomial_identity"]
