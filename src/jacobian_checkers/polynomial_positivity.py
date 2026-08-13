"""Independent exact Sturm-sequence replay for polynomial strict positivity.

This checker is the independent verification boundary for the
``polynomial.interval.positivity.verify`` operation. It recomputes the Sturm
sequence of one univariate rational polynomial from scratch using only
``fractions.Fraction`` arithmetic. It does not import SymPy and does not
depend on the adapter that proposed the decision.

The checker confirms that the claimed sign-change counts, root count,
endpoint-root flag, and positivity decision in the certificate replay match
its independent computation. A ``TRUE`` conclusion means the claimed decision
is correct. A ``FALSE`` conclusion means the claimed decision does not match
the independent Sturm replay.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_VARIABLE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
_MAX_DEGREE = 64
_MAX_TERMS = 1024

Polynomial = dict[int, Fraction]


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


def _polynomial_header(value: object) -> None:
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


def _polynomial(value: object) -> Polynomial:
    """Parse a univariate polynomial into ``{exponent: coefficient}`` form."""

    if not isinstance(value, dict):
        raise ValueError("univariate polynomial must contain the declared fields")
    _polynomial_header(value)
    body = value["polynomial"]
    if not isinstance(body, dict) or set(body) != {"terms"}:
        raise ValueError("polynomial body must contain terms")
    terms = body["terms"]
    if not isinstance(terms, list) or len(terms) > _MAX_TERMS:
        raise ValueError("polynomial term limit exceeded")
    parsed: Polynomial = {}
    previous: int | None = None
    for term in terms:
        exponent, coefficient = _polynomial_term(term, previous)
        previous = exponent
        if exponent in parsed:
            raise ValueError("polynomial exponent tuples must be unique")
        parsed[exponent] = coefficient
    return parsed


def _degree(parsed: Polynomial) -> int:
    return max(parsed) if parsed else 0


def _derivative(parsed: Polynomial) -> Polynomial:
    return {e - 1: c * e for e, c in parsed.items() if e > 0}


def _polynomial_remainder(numerator: Polynomial, divisor: Polynomial) -> Polynomial:
    """Compute remainder(numerator, divisor) via exact rational long division."""

    remainder = dict(numerator)
    divisor = {e: c for e, c in divisor.items() if c != 0}
    if not divisor:
        raise ValueError("polynomial division by zero")
    divisor_degree = max(divisor)
    divisor_lead = divisor[divisor_degree]
    while True:
        remainder = {e: c for e, c in remainder.items() if c != 0}
        if not remainder:
            break
        rem_degree = max(remainder)
        if rem_degree < divisor_degree:
            break
        factor = remainder[rem_degree] / divisor_lead
        shift = rem_degree - divisor_degree
        for exp, coeff in divisor.items():
            target = exp + shift
            remainder[target] = remainder.get(target, Fraction(0)) - factor * coeff
            if remainder[target] == 0:
                del remainder[target]
        if rem_degree in remainder:
            del remainder[rem_degree]
    return remainder


def _sturm_sequence(parsed: Polynomial) -> list[Polynomial]:
    """Compute the Sturm sequence: s_0=p, s_1=p', s_{k+1}=-rem(s_{k-1}, s_k)."""

    if not parsed:
        return [{}]
    degree = _degree(parsed)
    if degree == 0:
        return [dict(parsed)]
    s0 = dict(parsed)
    s1 = _derivative(parsed)
    sequence: list[Polynomial] = [s0, s1]
    while len(sequence) >= 2 and sequence[-1]:
        remainder = _polynomial_remainder(sequence[-2], sequence[-1])
        negated = {e: -c for e, c in remainder.items()}
        if not negated:
            break
        sequence.append(negated)
    if sequence and not sequence[-1]:
        sequence.pop()
    return sequence


def _evaluate(parsed: Polynomial, point: Fraction) -> Fraction:
    total = Fraction(0)
    for exponent, coefficient in parsed.items():
        total += coefficient * (point**exponent)
    return total


def _sign_changes_at(sequence: list[Polynomial], point: Fraction) -> int:
    signs: list[int] = []
    for poly in sequence:
        value = _evaluate(poly, point)
        if value > 0:
            signs.append(1)
        elif value < 0:
            signs.append(-1)
    changes = 0
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            changes += 1
    return changes


def _decide_positivity(
    parsed: Polynomial,
    a: Fraction,
    b: Fraction,
) -> tuple[int, int, int, bool, bool]:
    """Return (V_lo, V_hi, roots_in_open, endpoint_root, positive)."""

    degree = _degree(parsed)
    if degree == 0:
        p_at_a = _evaluate(parsed, a)
        return (0, 0, 0, p_at_a == 0, p_at_a > 0)
    sequence = _sturm_sequence(parsed)
    v_lo = _sign_changes_at(sequence, a)
    v_hi = _sign_changes_at(sequence, b)
    roots_in_open = v_lo - v_hi
    p_at_a = _evaluate(parsed, a)
    endpoint_root = p_at_a == 0
    positive = p_at_a > 0 and roots_in_open == 0 and not endpoint_root
    return (v_lo, v_hi, roots_in_open, endpoint_root, positive)


_POSITIVITY_REPLAY_KEYS = {
    "method",
    "polynomial_uri",
    "interval",
    "degree",
    "sturm_sequence_length",
    "sign_changes_at_lo",
    "sign_changes_at_hi",
    "roots_in_open_interval",
    "endpoint_root",
    "positive",
}


def _check_positivity_artifact_types(
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
        return "positivity replay artifacts are malformed"
    return None


def _check_positivity_claim(
    claim: object,
    candidate: object,
) -> str | None:
    if (
        not isinstance(claim, dict)
        or not isinstance(candidate, dict)
        or claim.get("claim_schema_version") != "1"
        or claim.get("predicate") != "POLYNOMIAL_INTERVAL_STRICT_POSITIVITY"
        or claim.get("domain") != "QQ"
    ):
        return "unexpected polynomial positivity claim"
    return None


def _check_positivity_certificate(
    certificate: dict[str, Any],
    expected_bindings: object,
) -> str | None:
    if (
        certificate.get("evidence_schema_version") != "1"
        or certificate.get("certificate_type")
        != "polynomial.interval_sturm_positivity_replay"
        or certificate.get("format_version") != "1"
        or certificate.get("bindings") != expected_bindings
    ):
        return "unexpected positivity certificate format or bindings"
    return None


def _check_positivity_replay(replay: object) -> str | None:
    if (
        not isinstance(replay, dict)
        or set(replay) != _POSITIVITY_REPLAY_KEYS
        or replay.get("method") != "STURM_SEQUENCE_REPLAY"
    ):
        return "positivity replay payload is malformed"
    return None


def _check_positivity_certificate_and_replay(
    certificate: dict[str, Any],
    expected_bindings: object,
) -> str | None:
    error = _check_positivity_certificate(certificate, expected_bindings)
    if error is not None:
        return error
    return _check_positivity_replay(certificate.get("payload"))


def _check_positivity_identities(
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
        return "positivity replay artifact identities do not match"
    return None


def _check_positivity_degree_intervals(
    replay: dict[str, Any],
    candidate: dict[str, Any],
    degree: int,
    claim: dict[str, Any],
) -> str | None:
    if replay.get("degree") != degree:
        return "declared degree does not match the source polynomial"
    if candidate.get("degree") != degree:
        return "candidate decision degree is inconsistent"
    claim_interval = claim.get("interval")
    replay_interval = replay.get("interval")
    candidate_interval = candidate.get("interval")
    if claim_interval != replay_interval or claim_interval != candidate_interval:
        return "declared intervals do not match across artifacts"
    return None


def _check_positivity_replay_fields(
    replay: dict[str, Any],
) -> tuple[int, int, int, bool, bool] | str:
    claimed_v_lo = replay.get("sign_changes_at_lo")
    claimed_v_hi = replay.get("sign_changes_at_hi")
    claimed_roots = replay.get("roots_in_open_interval")
    claimed_endpoint_root = replay.get("endpoint_root")
    claimed_positive = replay.get("positive")
    if (
        not isinstance(claimed_v_lo, int)
        or not isinstance(claimed_v_hi, int)
        or not isinstance(claimed_roots, int)
        or not isinstance(claimed_endpoint_root, bool)
        or not isinstance(claimed_positive, bool)
        or claimed_v_lo < 0
        or claimed_v_hi < 0
        or claimed_roots < 0
        or claimed_roots != claimed_v_lo - claimed_v_hi
    ):
        return "claimed replay fields are inconsistent"
    return (
        claimed_v_lo,
        claimed_v_hi,
        claimed_roots,
        claimed_endpoint_root,
        claimed_positive,
    )


def _check_positivity_replay_fields_and_candidate(
    replay: dict[str, Any],
    candidate: dict[str, Any],
    claim: dict[str, Any],
) -> tuple[int, int, int, bool, bool] | str:
    result = _check_positivity_replay_fields(replay)
    if isinstance(result, str):
        return result
    (
        claimed_v_lo,
        claimed_v_hi,
        claimed_roots,
        claimed_endpoint_root,
        claimed_positive,
    ) = result
    error = _check_positivity_candidate_and_claim(
        candidate,
        claim,
        claimed_v_lo,
        claimed_v_hi,
        claimed_roots,
        claimed_endpoint_root,
        claimed_positive,
    )
    if error is not None:
        return error
    return result


def _check_positivity_candidate_and_claim(
    candidate: dict[str, Any],
    claim: dict[str, Any],
    claimed_v_lo: int,
    claimed_v_hi: int,
    claimed_roots: int,
    claimed_endpoint_root: bool,
    claimed_positive: bool,
) -> str | None:
    if (
        candidate.get("sign_changes_at_lo") != claimed_v_lo
        or candidate.get("sign_changes_at_hi") != claimed_v_hi
        or candidate.get("roots_in_open_interval") != claimed_roots
        or candidate.get("endpoint_root") != claimed_endpoint_root
        or candidate.get("positive") != claimed_positive
    ):
        return "candidate decision does not match the replay payload"
    if claim.get("positive") != claimed_positive:
        return "claim does not match the replay positivity decision"
    return None


def check_positivity(request: dict[str, Any]) -> dict[str, Any]:
    """Replay one claimed Sturm-sequence positivity decision independently."""

    try:
        if request.get("request_version") != "1":
            return _reject("unsupported request version")
        claim_artifact = request["claim"]
        candidate_artifact = request["candidate"]
        scope_artifact = request["scope"]
        certificate = request["certificate"]["payload"]
        error = _check_positivity_artifact_types(
            claim_artifact, candidate_artifact, scope_artifact, certificate
        )
        if error is not None:
            return _reject(error)
        claim = claim_artifact.get("payload")
        candidate = candidate_artifact.get("payload")
        polynomial_payload = scope_artifact.get("payload")
        error = _check_positivity_claim(claim, candidate)
        if error is not None:
            return _reject(error)
        error = _check_positivity_certificate_and_replay(
            certificate, request.get("expected_bindings")
        )
        if error is not None:
            return _reject(error)
        replay = certificate.get("payload")
        polynomial_uri = scope_artifact.get("artifact_uri")
        decision_uri = candidate_artifact.get("artifact_uri")
        error = _check_positivity_identities(
            claim, replay, polynomial_payload, polynomial_uri
        )
        if error is not None:
            return _reject(error)
        parsed = _polynomial(polynomial_payload)
        degree = _degree(parsed)
        error = _check_positivity_degree_intervals(replay, candidate, degree, claim)
        if error is not None:
            return _reject(error)
        a, b = _interval(claim.get("interval"))
        result = _check_positivity_replay_fields_and_candidate(replay, candidate, claim)
        if isinstance(result, str):
            return _reject(result)
        (
            claimed_v_lo,
            claimed_v_hi,
            claimed_roots,
            claimed_endpoint_root,
            claimed_positive,
        ) = result
        v_lo, v_hi, roots_in_open, endpoint_root, positive = _decide_positivity(
            parsed, a, b
        )
        if (
            v_lo != claimed_v_lo
            or v_hi != claimed_v_hi
            or roots_in_open != claimed_roots
            or endpoint_root != claimed_endpoint_root
            or positive != claimed_positive
        ):
            return {
                "accepted": True,
                "conclusion": "FALSE",
                "arithmetic": "EXACT_RATIONAL",
                "method": "CHECKED_CERTIFICATE",
                "coverage": "EXHAUSTIVE",
                "detail": (
                    "claimed Sturm-sequence replay does not match the independent "
                    "rational recomputation"
                ),
            }
        return {
            "accepted": True,
            "conclusion": "TRUE",
            "arithmetic": "EXACT_RATIONAL",
            "method": "CHECKED_CERTIFICATE",
            "coverage": "EXHAUSTIVE",
            "detail": (
                "Sturm sequence, sign-change counts, root count, and positivity "
                "decision replayed by independent rational arithmetic"
            ),
            "relation_id": "polynomial.relation.valid-positivity-decision",
            "relationship_source_artifact_uris": [decision_uri],
            "relationship_target_artifact_uris": [polynomial_uri],
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return _reject("malformed polynomial positivity request")
