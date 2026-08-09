"""Independent exact replay for rational polynomial-system assignments."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_VARIABLE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
_MAX_DIMENSION = 4
_MAX_CONSTRAINTS = 64
_MAX_TERMS = 1024
_MAX_EXPONENT = 127


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
        raise ValueError("rational must contain num and den")
    numerator = value["num"]
    denominator = value["den"]
    if (
        not isinstance(numerator, str)
        or not isinstance(denominator, str)
        or _INTEGER.fullmatch(numerator) is None
        or _INTEGER.fullmatch(denominator) is None
    ):
        raise ValueError("rational values must use canonical integers")
    parsed = Fraction(int(numerator), int(denominator))
    if str(parsed.numerator) != numerator or str(parsed.denominator) != denominator:
        raise ValueError("rational value is not reduced and canonical")
    return parsed


def _polynomial(
    value: object, dimension: int
) -> tuple[tuple[Fraction, tuple[int, ...]], ...]:
    if not isinstance(value, dict) or set(value) != {"terms"}:
        raise ValueError("polynomial must contain terms")
    terms = value["terms"]
    if not isinstance(terms, list) or len(terms) > _MAX_TERMS:
        raise ValueError("polynomial term limit exceeded")
    parsed: list[tuple[Fraction, tuple[int, ...]]] = []
    previous: tuple[int, ...] | None = None
    for term in terms:
        if not isinstance(term, dict) or set(term) != {"coefficient", "exponents"}:
            raise ValueError("malformed polynomial term")
        coefficient = _rational(term["coefficient"])
        exponents = term["exponents"]
        if (
            coefficient == 0
            or not isinstance(exponents, list)
            or len(exponents) != dimension
            or any(
                not isinstance(exponent, int)
                or isinstance(exponent, bool)
                or not 0 <= exponent <= _MAX_EXPONENT
                for exponent in exponents
            )
        ):
            raise ValueError("invalid polynomial term")
        exponent_tuple = tuple(exponents)
        if previous is not None and exponent_tuple >= previous:
            raise ValueError("terms are not in canonical descending order")
        previous = exponent_tuple
        parsed.append((coefficient, exponent_tuple))
    return tuple(parsed)


def _evaluate(
    polynomial: tuple[tuple[Fraction, tuple[int, ...]], ...],
    assignment: tuple[Fraction, ...],
) -> Fraction:
    return sum(
        (
            coefficient * _monomial(assignment, exponents)
            for coefficient, exponents in polynomial
        ),
        start=Fraction(0),
    )


def _monomial(
    assignment: tuple[Fraction, ...],
    exponents: tuple[int, ...],
) -> Fraction:
    value = Fraction(1)
    for coordinate, exponent in zip(assignment, exponents, strict=True):
        value *= coordinate**exponent
    return value


def check_solution(request: dict[str, Any]) -> dict[str, Any]:
    """Replay every equation and inequation at one exact assignment."""

    try:
        if request.get("request_version") != "1":
            return _reject("unsupported request version")
        claim = request["claim"]["payload"]
        system_artifact = request["scope"]
        assignment_artifact = request["candidate"]
        certificate = request["certificate"]["payload"]
        if (
            not isinstance(claim, dict)
            or claim.get("claim_schema_version") != "1"
            or claim.get("predicate") != "ASSIGNMENT_SATISFIES_POLYNOMIAL_SYSTEM"
            or claim.get("domain") != "QQ"
            or claim.get("system_uri") != system_artifact.get("artifact_uri")
            or claim.get("assignment_uri") != assignment_artifact.get("artifact_uri")
        ):
            return _reject("unexpected polynomial-system solution claim")
        system = system_artifact["payload"]
        assignment_payload = assignment_artifact["payload"]
        if (
            not isinstance(system, dict)
            or system.get("system_schema_version") != "1"
            or system.get("domain") != "QQ"
            or not isinstance(assignment_payload, dict)
            or assignment_payload.get("assignment_schema_version") != "1"
        ):
            return _reject("unsupported polynomial-system artifacts")
        variables = system.get("variables")
        equations = system.get("equations")
        inequations = system.get("inequations")
        values = assignment_payload.get("values")
        if (
            not isinstance(variables, list)
            or not 1 <= len(variables) <= _MAX_DIMENSION
            or any(
                not isinstance(variable, str) or _VARIABLE.fullmatch(variable) is None
                for variable in variables
            )
            or len(set(variables)) != len(variables)
            or not isinstance(equations, list)
            or not 1 <= len(equations) <= _MAX_CONSTRAINTS
            or not isinstance(inequations, list)
            or len(inequations) > _MAX_CONSTRAINTS
            or not isinstance(values, list)
            or len(values) != len(variables)
        ):
            return _reject("malformed polynomial system or assignment")
        if (
            not isinstance(certificate, dict)
            or certificate.get("certificate_type")
            != "polynomial.system_solution_replay"
            or certificate.get("format_version") != "1"
            or certificate.get("bindings") != request.get("expected_bindings")
        ):
            return _reject("unexpected solution certificate or bindings")
        replay = certificate.get("payload")
        if (
            not isinstance(replay, dict)
            or set(replay)
            != {
                "method",
                "system_uri",
                "assignment_uri",
                "equation_residuals",
                "inequation_values",
            }
            or replay.get("method") != "DIRECT_EXACT_EVALUATION"
            or replay.get("system_uri") != system_artifact["artifact_uri"]
            or replay.get("assignment_uri") != assignment_artifact["artifact_uri"]
        ):
            return _reject("unexpected solution replay payload")
        assignment = tuple(_rational(value) for value in values)
        equation_values = tuple(
            _evaluate(_polynomial(polynomial, len(variables)), assignment)
            for polynomial in equations
        )
        inequation_values = tuple(
            _evaluate(_polynomial(polynomial, len(variables)), assignment)
            for polynomial in inequations
        )
        reported_equation_values = replay["equation_residuals"]
        reported_inequation_values = replay["inequation_values"]
        if (
            not isinstance(reported_equation_values, list)
            or not isinstance(reported_inequation_values, list)
            or tuple(_rational(value) for value in reported_equation_values)
            != equation_values
            or tuple(_rational(value) for value in reported_inequation_values)
            != inequation_values
        ):
            return _reject("reported residuals do not match independent replay")
        satisfies = all(value == 0 for value in equation_values) and all(
            value != 0 for value in inequation_values
        )
        return {
            "accepted": True,
            "conclusion": "TRUE" if satisfies else "FALSE",
            "arithmetic": "EXACT_RATIONAL",
            "method": "CHECKED_CERTIFICATE",
            "coverage": "EXHAUSTIVE",
            "detail": (
                "every equation vanishes and every inequation is nonzero"
                if satisfies
                else "the assignment violates at least one declared constraint"
            ),
            **(
                {
                    "relation_id": "polynomial.relation.satisfies-system",
                    "relationship_source_artifact_uris": [
                        assignment_artifact["artifact_uri"]
                    ],
                    "relationship_target_artifact_uris": [
                        system_artifact["artifact_uri"]
                    ],
                }
                if satisfies
                else {}
            ),
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return _reject("malformed polynomial-system solution request")
