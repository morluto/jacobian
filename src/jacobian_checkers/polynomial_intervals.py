"""Independent exact replay for univariate polynomial interval enclosures.

This checker is the independent verification boundary for the
``polynomial.interval.enclosure.verify`` operation. It re-derives the
Bernstein coefficients of one univariate rational polynomial on one closed
rational interval using only ``fractions.Fraction`` arithmetic. It does not
import SymPy and does not depend on the adapter that proposed the enclosure.

The checker confirms that the claimed Bernstein coefficients, lower bound, and
upper bound in the certificate replay match its independent computation. A
``TRUE`` conclusion means the claimed enclosure is the valid Bernstein-
coefficient bound for the declared polynomial on the declared interval. It does
not mean the enclosure equals the exact polynomial range.
"""

from __future__ import annotations

import re
from fractions import Fraction
from math import comb
from typing import Any

_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_VARIABLE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
_MAX_DEGREE = 64
_MAX_TERMS = 1024


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_RATIONAL",
        "method": "CHECKED_CERTIFICATE",
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


def _interval(value: object) -> tuple[Fraction, Fraction]:
    if not isinstance(value, dict) or set(value) != {
        "interval_schema_version",
        "lo",
        "hi",
    }:
        raise ValueError("interval must contain interval_schema_version, lo, hi")
    if value["interval_schema_version"] != "1":
        raise ValueError("unsupported interval schema version")
    lo = _rational(value["lo"])
    hi = _rational(value["hi"])
    if lo >= hi:
        raise ValueError("interval must satisfy lo < hi")
    return lo, hi


def _polynomial_header(value: object) -> str:
    if not isinstance(value, dict) or set(value) != {
        "polynomial_schema_version",
        "domain",
        "variable",
        "polynomial",
    }:
        raise ValueError("univariate polynomial must contain the declared fields")
    if value["polynomial_schema_version"] != "1" or value["domain"] != "QQ":
        raise ValueError("unsupported univariate polynomial schema")
    variable = value["variable"]
    if not isinstance(variable, str) or _VARIABLE.fullmatch(variable) is None:
        raise ValueError("polynomial variable is not a valid identifier")
    return variable


def _polynomial_term(
    term: object,
    previous: int | None,
) -> tuple[int, Fraction]:
    if not isinstance(term, dict) or set(term) != {"coefficient", "exponents"}:
        raise ValueError("malformed polynomial term")
    coefficient = _rational(term["coefficient"])
    exponents = term["exponents"]
    if (
        coefficient == 0
        or not isinstance(exponents, list)
        or len(exponents) != 1
        or not isinstance(exponents[0], int)
        or isinstance(exponents[0], bool)
        or not 0 <= exponents[0] <= _MAX_DEGREE
    ):
        raise ValueError("invalid univariate polynomial term")
    exponent = exponents[0]
    if previous is not None and exponent >= previous:
        raise ValueError("terms are not in canonical descending order")
    return exponent, coefficient


def _polynomial(
    value: object,
) -> tuple[str, dict[int, Fraction]]:
    """Parse a univariate polynomial into ``{exponent: coefficient}`` form."""

    if not isinstance(value, dict):
        raise ValueError("univariate polynomial must contain the declared fields")
    variable = _polynomial_header(value)
    body = value["polynomial"]
    if not isinstance(body, dict) or set(body) != {"terms"}:
        raise ValueError("polynomial body must contain terms")
    terms = body["terms"]
    if not isinstance(terms, list) or len(terms) > _MAX_TERMS:
        raise ValueError("polynomial term limit exceeded")
    parsed: dict[int, Fraction] = {}
    previous: int | None = None
    for term in terms:
        exponent, coefficient = _polynomial_term(term, previous)
        previous = exponent
        if exponent in parsed:
            raise ValueError("polynomial exponent tuples must be unique")
        parsed[exponent] = coefficient
    return variable, parsed


def _degree(parsed: dict[int, Fraction]) -> int:
    return max(parsed) if parsed else 0


def _shift_to_unit_interval(
    parsed: dict[int, Fraction],
    a: Fraction,
    b: Fraction,
) -> dict[int, Fraction]:
    """Substitute ``x = a + (b - a) t`` and return power-basis coefficients in t."""

    width = b - a
    # Evaluate p(a + width * t) by Horner expansion over the monomials x^k.
    # Each x^k becomes (a + width*t)^k = sum_{j=0}^{k} C(k,j) a^(k-j) width^j t^j.
    accumulated: dict[int, Fraction] = {}
    for exponent, coefficient in parsed.items():
        for j in range(exponent + 1):
            term_coeff = (
                coefficient * comb(exponent, j) * (a ** (exponent - j)) * (width**j)
            )
            accumulated[j] = accumulated.get(j, Fraction(0)) + term_coeff
    return accumulated


def _bernstein_coefficients(
    parsed: dict[int, Fraction],
    a: Fraction,
    b: Fraction,
) -> tuple[Fraction, ...]:
    degree = _degree(parsed)
    power = _shift_to_unit_interval(parsed, a, b)
    coefficients: list[Fraction] = []
    for i in range(degree + 1):
        accumulator = Fraction(0)
        for k in range(i + 1):
            ak = power.get(k, Fraction(0))
            if ak == 0:
                continue
            accumulator += ak * Fraction(comb(i, k), comb(degree, k))
        coefficients.append(accumulator)
    return tuple(coefficients)


_ENCLOSURE_REPLAY_KEYS = {
    "method",
    "polynomial_uri",
    "interval",
    "degree",
    "bernstein_coefficients",
    "lo",
    "hi",
}


def _check_enclosure_artifact_types(
    claim_artifact: object,
    candidate_artifact: object,
    scope_artifact: object,
    certificate: object,
) -> str | None:
    if (
        not isinstance(claim_artifact, dict)
        or not isinstance(candidate_artifact, dict)
        or not isinstance(scope_artifact, dict)
        or not isinstance(certificate, dict)
    ):
        return "enclosure replay artifacts are malformed"
    return None


def _check_enclosure_claim(
    claim: object,
    candidate: object,
) -> str | None:
    if (
        not isinstance(claim, dict)
        or not isinstance(candidate, dict)
        or claim.get("claim_schema_version") != "1"
        or claim.get("predicate") != "POLYNOMIAL_INTERVAL_BERNSTEIN_ENCLOSURE"
        or claim.get("domain") != "QQ"
    ):
        return "unexpected polynomial interval enclosure claim"
    return None


def _check_enclosure_certificate(
    certificate: dict[str, Any],
    expected_bindings: object,
) -> str | None:
    if (
        certificate.get("evidence_schema_version") != "1"
        or certificate.get("certificate_type")
        != "polynomial.interval_bernstein_enclosure_replay"
        or certificate.get("format_version") != "1"
        or certificate.get("bindings") != expected_bindings
    ):
        return "unexpected enclosure certificate format or bindings"
    return None


def _check_enclosure_replay(replay: object) -> str | None:
    if (
        not isinstance(replay, dict)
        or set(replay) != _ENCLOSURE_REPLAY_KEYS
        or replay.get("method") != "BERNSTEIN_COEFFICIENT_REPLAY"
    ):
        return "enclosure replay payload is malformed"
    return None


def _check_enclosure_certificate_and_replay(
    certificate: dict[str, Any],
    expected_bindings: object,
) -> str | None:
    error = _check_enclosure_certificate(certificate, expected_bindings)
    if error is not None:
        return error
    return _check_enclosure_replay(certificate.get("payload"))


def _check_enclosure_identities(
    claim: dict[str, Any],
    replay: dict[str, Any],
    polynomial_payload: object,
    polynomial_uri: object,
) -> str | None:
    if (
        not isinstance(polynomial_payload, dict)
        or claim.get("polynomial_uri") != polynomial_uri
        or replay.get("polynomial_uri") != polynomial_uri
    ):
        return "enclosure replay artifact identities do not match"
    return None


def _check_enclosure_degree_candidate_intervals(
    replay: dict[str, Any],
    candidate: dict[str, Any],
    degree: int,
    polynomial_uri: object,
    claim: dict[str, Any],
) -> str | None:
    if replay.get("degree") != degree:
        return "declared degree does not match the source polynomial"
    if candidate.get("polynomial_uri") != polynomial_uri:
        return "candidate enclosure does not bind the source polynomial"
    claim_interval = claim.get("interval")
    replay_interval = replay.get("interval")
    candidate_interval = candidate.get("interval")
    if claim_interval != replay_interval or claim_interval != candidate_interval:
        return "declared intervals do not match across artifacts"
    return None


def _check_enclosure_coefficients(
    replay: dict[str, Any],
    candidate: dict[str, Any],
    degree: int,
) -> tuple[list[Fraction], Fraction, Fraction] | str:
    claimed_coefficients = replay.get("bernstein_coefficients")
    if (
        not isinstance(claimed_coefficients, list)
        or len(claimed_coefficients) != degree + 1
    ):
        return "claimed Bernstein coefficient count is inconsistent"
    claimed_values = [_rational(value) for value in claimed_coefficients]
    claimed_lo = _rational(replay.get("lo"))
    claimed_hi = _rational(replay.get("hi"))
    if claimed_lo != min(claimed_values) or claimed_hi != max(claimed_values):
        return "claimed bounds do not match claimed coefficients"
    if (
        candidate.get("degree") != degree
        or [_rational(value) for value in candidate.get("bernstein_coefficients", [])]
        != claimed_values
        or _rational(candidate.get("lo")) != claimed_lo
        or _rational(candidate.get("hi")) != claimed_hi
    ):
        return "candidate enclosure does not match the replay payload"
    return claimed_values, claimed_lo, claimed_hi


def check_enclosure(request: dict[str, Any]) -> dict[str, Any]:
    """Replay one claimed Bernstein-coefficient enclosure independently."""

    try:
        if request.get("request_version") != "1":
            return _reject("unsupported request version")
        claim_artifact = request["claim"]
        candidate_artifact = request["candidate"]
        scope_artifact = request["scope"]
        certificate = request["certificate"]["payload"]
        error = _check_enclosure_artifact_types(
            claim_artifact, candidate_artifact, scope_artifact, certificate
        )
        if error is not None:
            return _reject(error)
        claim = claim_artifact.get("payload")
        candidate = candidate_artifact.get("payload")
        polynomial_payload = scope_artifact.get("payload")
        error = _check_enclosure_claim(claim, candidate)
        if error is not None:
            return _reject(error)
        error = _check_enclosure_certificate_and_replay(
            certificate, request.get("expected_bindings")
        )
        if error is not None:
            return _reject(error)
        replay = certificate.get("payload")
        polynomial_uri = scope_artifact.get("artifact_uri")
        enclosure_uri = candidate_artifact.get("artifact_uri")
        error = _check_enclosure_identities(
            claim, replay, polynomial_payload, polynomial_uri
        )
        if error is not None:
            return _reject(error)
        _variable, parsed = _polynomial(polynomial_payload)
        degree = _degree(parsed)
        error = _check_enclosure_degree_candidate_intervals(
            replay, candidate, degree, polynomial_uri, claim
        )
        if error is not None:
            return _reject(error)
        a, b = _interval(claim.get("interval"))
        result = _check_enclosure_coefficients(replay, candidate, degree)
        if isinstance(result, str):
            return _reject(result)
        claimed_values, _claimed_lo, _claimed_hi = result
        independent = _bernstein_coefficients(parsed, a, b)
        if tuple(claimed_values) == independent:
            return {
                "accepted": True,
                "conclusion": "TRUE",
                "arithmetic": "EXACT_RATIONAL",
                "method": "CHECKED_CERTIFICATE",
                "coverage": "EXHAUSTIVE",
                "detail": (
                    "Bernstein coefficients, lower bound, and upper bound replayed "
                    "by independent rational arithmetic; the enclosure is a valid "
                    "Bernstein bound, not the exact polynomial range"
                ),
                "relation_id": "polynomial.relation.valid-bernstein-enclosure",
                "relationship_source_artifact_uris": [enclosure_uri],
                "relationship_target_artifact_uris": [polynomial_uri],
            }
        return {
            "accepted": True,
            "conclusion": "FALSE",
            "arithmetic": "EXACT_RATIONAL",
            "method": "CHECKED_CERTIFICATE",
            "coverage": "EXHAUSTIVE",
            "detail": (
                "claimed Bernstein coefficients do not match the independent "
                "rational replay; the claimed enclosure is not the valid "
                "Bernstein-coefficient bound for the declared polynomial and "
                "interval"
            ),
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return _reject("malformed polynomial interval enclosure request")
