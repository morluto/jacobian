"""Independent exact replay for bounded sparse rational-function identities."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

from jacobian.canonical import format_canonical_integer, parse_canonical_integer

_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_VARIABLE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
_MAX_DIMENSION = 4
_MAX_TERMS = 1024
_MAX_EXPONENT = 127
_MAX_TERM_PAIRS = 4096

Exponent = tuple[int, ...]
Polynomial = dict[Exponent, Fraction]


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_RATIONAL",
        "method": "CHECKED_CERTIFICATE",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _parse_rational(value: object) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise ValueError("rational must contain num and den")
    numerator, denominator = value["num"], value["den"]
    if (
        not isinstance(numerator, str)
        or not isinstance(denominator, str)
        or _INTEGER.fullmatch(numerator) is None
        or _INTEGER.fullmatch(denominator) is None
    ):
        raise ValueError("rational integers are not canonical")
    # Use the limit-independent canonical integer parser so coefficients up to
    # the contract's 32,768-digit bound can be verified; Python's default
    # 4,300-digit int() conversion limit would otherwise raise ValueError.
    parsed = Fraction(
        parse_canonical_integer(numerator), parse_canonical_integer(denominator)
    )
    if (
        format_canonical_integer(parsed.numerator) != numerator
        or format_canonical_integer(parsed.denominator) != denominator
    ):
        raise ValueError("rational is not reduced and canonical")
    return parsed


def _parse_polynomial(value: object, dimension: int) -> Polynomial:
    if not isinstance(value, dict) or set(value) != {"terms"}:
        raise ValueError("polynomial must contain terms")
    terms = value["terms"]
    if not isinstance(terms, list) or len(terms) > _MAX_TERMS:
        raise ValueError("polynomial term list exceeds checker limits")
    result: Polynomial = {}
    last: Exponent | None = None
    for term in terms:
        if not isinstance(term, dict) or set(term) != {"coefficient", "exponents"}:
            raise ValueError("malformed polynomial term")
        coefficient = _parse_rational(term["coefficient"])
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
        if exponent_tuple in result or (last is not None and exponent_tuple >= last):
            raise ValueError("polynomial terms are not in canonical order")
        result[exponent_tuple] = coefficient
        last = exponent_tuple
    return result


def _parse_function(
    value: object, variables: list[str]
) -> tuple[Polynomial, Polynomial]:
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "rational_function_schema_version",
            "domain",
            "variables",
            "numerator",
            "denominator",
        }
        or value.get("rational_function_schema_version") != "1"
        or value.get("domain") != "QQ_FRACTION_FIELD"
        or value.get("variables") != variables
    ):
        raise ValueError("rational-function artifact does not match the field")
    numerator = _parse_polynomial(value["numerator"], len(variables))
    denominator = _parse_polynomial(value["denominator"], len(variables))
    if not denominator:
        raise ValueError("rational-function denominator is zero")
    return numerator, denominator


def _multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    if len(left) * len(right) > _MAX_TERM_PAIRS:
        raise ValueError("cross product exceeds checker limits")
    result: Polynomial = {}
    for left_exp, left_coefficient in left.items():
        for right_exp, right_coefficient in right.items():
            exponent = tuple(a + b for a, b in zip(left_exp, right_exp, strict=True))
            result[exponent] = (
                result.get(exponent, Fraction(0)) + left_coefficient * right_coefficient
            )
            if result[exponent] == 0:
                del result[exponent]
    return result


def check_rational_function_identity(request: dict[str, Any]) -> dict[str, Any]:
    """Decide equality in QQ(x1,...,xn) by independent cross multiplication."""

    try:
        if request.get("request_version") != "1":
            return _reject("unsupported request version")
        claim = request["claim"]["payload"]
        candidate = request["candidate"]
        scope = request["scope"]
        certificate = request["certificate"]["payload"]
        if (
            not isinstance(claim, dict)
            or set(claim)
            != {
                "claim_schema_version",
                "predicate",
                "domain",
                "variables",
                "left_uri",
                "right_uri",
            }
            or claim.get("claim_schema_version") != "1"
            or claim.get("predicate") != "RATIONAL_FUNCTION_IDENTITY"
            or claim.get("domain") != "QQ_FRACTION_FIELD"
            or not isinstance(candidate, dict)
            or not isinstance(scope, dict)
            or request.get("supporting_artifacts", []) != []
            or claim.get("left_uri") != scope.get("artifact_uri")
            or claim.get("right_uri") != candidate.get("artifact_uri")
        ):
            return _reject("unexpected rational-function claim or artifact binding")
        variables = claim.get("variables")
        if (
            not isinstance(variables, list)
            or not 1 <= len(variables) <= _MAX_DIMENSION
            or any(
                not isinstance(v, str) or _VARIABLE.fullmatch(v) is None
                for v in variables
            )
            or len(set(variables)) != len(variables)
        ):
            return _reject("invalid rational-function field variables")
        expected_payload = {
            "method": "CROSS_MULTIPLY_SPARSE_POLYNOMIALS",
            "variables": variables,
            "left_uri": scope["artifact_uri"],
            "right_uri": candidate["artifact_uri"],
        }
        if (
            not isinstance(certificate, dict)
            or certificate.get("certificate_type")
            != "polynomial.rational_function.identity_replay"
            or certificate.get("format_version") != "1"
            or certificate.get("bindings") != request.get("expected_bindings")
            or certificate.get("payload") != expected_payload
        ):
            return _reject("unexpected identity certificate or bindings")
        left_num, left_den = _parse_function(scope["payload"], variables)
        right_num, right_den = _parse_function(candidate["payload"], variables)
        equal = _multiply(left_num, right_den) == _multiply(right_num, left_den)
        return {
            "accepted": True,
            "conclusion": "TRUE" if equal else "FALSE",
            "arithmetic": "EXACT_RATIONAL",
            "method": "CHECKED_CERTIFICATE",
            "coverage": "EXHAUSTIVE",
            "detail": "both sparse cross products were compared exactly over QQ",
            **(
                {
                    "relation_id": "polynomial.relation.rational-function-identity",
                    "relationship_source_artifact_uris": [scope["artifact_uri"]],
                    "relationship_target_artifact_uris": [candidate["artifact_uri"]],
                }
                if equal
                else {}
            ),
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return _reject("malformed rational-function identity replay request")
