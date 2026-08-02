from __future__ import annotations

import copy
from typing import Any

import pytest
from tests.support.rationals import rational_payload as _rational
from tests.unit.contracts.artifacts import artifact_uri as _uri
from tests.unit.contracts.artifacts import json_digest as _digest

from jacobian_checkers.rational_functions import check_rational_function_identity


def _artifact(
    *,
    uri_character: str,
    object_character: str,
    schema_character: str,
    semantics_uri: str,
    payload: dict[str, Any],
    parents: list[str],
) -> dict[str, Any]:
    return {
        "artifact_uri": _uri(uri_character),
        "object_digest": "sha256:" + object_character * 64,
        "payload_digest": _digest(payload),
        "schema_uri": _uri(schema_character),
        "semantics_uri": semantics_uri,
        "parents": parents,
        "payload": payload,
    }


def _polynomial(terms: list[tuple[int, int, int]]) -> dict[str, Any]:
    """Build a sparse polynomial from (num, den, exponent) tuples."""
    return {
        "terms": [
            {
                "coefficient": _rational(num, den),
                "exponents": [exponent],
            }
            for num, den, exponent in terms
        ]
    }


def _function(
    variables: list[str],
    numerator: dict[str, Any],
    denominator: dict[str, Any],
) -> dict[str, Any]:
    return {
        "rational_function_schema_version": "1",
        "domain": "QQ_FRACTION_FIELD",
        "variables": variables,
        "numerator": numerator,
        "denominator": denominator,
    }


def _request(*, equal: bool = True) -> dict[str, Any]:
    semantics_uri = _uri("e")
    left_uri = _uri("a")
    right_uri = _uri("c")
    variables = ["x"]
    left_payload = _function(
        variables,
        numerator=_polynomial([(1, 1, 2), (-1, 1, 0)]),
        denominator=_polynomial([(1, 1, 1), (-1, 1, 0)]),
    )
    right_numerator = _polynomial([(1, 1, 1), (1 if equal else 2, 1, 0)])
    right_payload = _function(
        variables,
        numerator=right_numerator,
        denominator=_polynomial([(1, 1, 0)]),
    )
    claim = _artifact(
        uri_character="a",
        object_character="1",
        schema_character="b",
        semantics_uri=semantics_uri,
        payload={
            "claim_schema_version": "1",
            "predicate": "RATIONAL_FUNCTION_IDENTITY",
            "domain": "QQ_FRACTION_FIELD",
            "variables": variables,
            "left_uri": left_uri,
            "right_uri": right_uri,
        },
        parents=[],
    )
    scope = _artifact(
        uri_character="a",
        object_character="2",
        schema_character="b",
        semantics_uri=semantics_uri,
        payload=left_payload,
        parents=[],
    )
    candidate = _artifact(
        uri_character="c",
        object_character="3",
        schema_character="d",
        semantics_uri=semantics_uri,
        payload=right_payload,
        parents=[left_uri],
    )
    bindings = {
        "claim_digest": claim["object_digest"],
        "semantics_digest": "sha256:" + "5" * 64,
        "candidate_digest": candidate["object_digest"],
        "scope_digest": scope["object_digest"],
        "encoding_digest": None,
    }
    certificate = _artifact(
        uri_character="f",
        object_character="4",
        schema_character="5",
        semantics_uri=semantics_uri,
        payload={
            "certificate_type": "polynomial.rational_function.identity_replay",
            "format_version": "1",
            "bindings": bindings,
            "payload": {
                "method": "CROSS_MULTIPLY_SPARSE_POLYNOMIALS",
                "variables": variables,
                "left_uri": left_uri,
                "right_uri": right_uri,
            },
        },
        parents=[left_uri, right_uri],
    )
    return {
        "request_version": "1",
        "claim": claim,
        "candidate": candidate,
        "scope": scope,
        "certificate": certificate,
        "expected_bindings": bindings,
        "supporting_artifacts": [],
    }


def _refresh_payload_digest(artifact: dict[str, Any]) -> None:
    artifact["payload_digest"] = _digest(artifact["payload"])


# ---------------------------------------------------------------------------
# Accepted equality and inequality
# ---------------------------------------------------------------------------


def test_checker_accepts_exact_rational_function_identity() -> None:
    decision = check_rational_function_identity(_request(equal=True))

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"
    assert decision["arithmetic"] == "EXACT_RATIONAL"
    assert decision["method"] == "CHECKED_CERTIFICATE"
    assert decision["coverage"] == "EXHAUSTIVE"
    assert decision["relation_id"] == "polynomial.relation.rational-function-identity"


def test_checker_reports_exact_difference_as_false() -> None:
    decision = check_rational_function_identity(_request(equal=False))

    assert decision["accepted"] is True
    assert decision["conclusion"] == "FALSE"
    assert "relation_id" not in decision


# ---------------------------------------------------------------------------
# Fail-closed: malformed or mismatched bindings
# ---------------------------------------------------------------------------


def test_checker_rejects_mismatched_left_uri_binding() -> None:
    request = _request()
    request["claim"]["payload"]["left_uri"] = _uri("9")
    _refresh_payload_digest(request["claim"])

    decision = check_rational_function_identity(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_checker_rejects_mismatched_right_uri_binding() -> None:
    request = _request()
    request["claim"]["payload"]["right_uri"] = _uri("9")
    _refresh_payload_digest(request["claim"])

    decision = check_rational_function_identity(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_checker_rejects_certificate_bindings_mismatch() -> None:
    request = _request()
    request["certificate"]["payload"]["bindings"] = {
        **request["expected_bindings"],
        "claim_digest": "sha256:" + "9" * 64,
    }
    _refresh_payload_digest(request["certificate"])

    decision = check_rational_function_identity(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_checker_rejects_wrong_certificate_type() -> None:
    request = _request()
    request["certificate"]["payload"]["certificate_type"] = "forged"
    _refresh_payload_digest(request["certificate"])

    decision = check_rational_function_identity(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_checker_rejects_wrong_request_version() -> None:
    request = _request()
    request["request_version"] = "2"

    decision = check_rational_function_identity(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_checker_rejects_supporting_artifacts() -> None:
    request = _request()
    request["supporting_artifacts"] = [{"artifact_uri": _uri("z")}]

    decision = check_rational_function_identity(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_checker_rejects_duplicate_variables() -> None:
    request = _request()
    request["claim"]["payload"]["variables"] = ["x", "x"]
    _refresh_payload_digest(request["claim"])

    decision = check_rational_function_identity(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# Fail-closed: malformed payload
# ---------------------------------------------------------------------------


def test_checker_rejects_rational_function_missing_schema_version() -> None:
    request = _request()
    del request["scope"]["payload"]["rational_function_schema_version"]
    _refresh_payload_digest(request["scope"])

    decision = check_rational_function_identity(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_checker_rejects_rational_function_wrong_domain() -> None:
    request = _request()
    request["scope"]["payload"]["domain"] = "QQ"
    _refresh_payload_digest(request["scope"])

    decision = check_rational_function_identity(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_checker_rejects_rational_function_mismatched_variables() -> None:
    request = _request()
    request["scope"]["payload"]["variables"] = ["y"]
    _refresh_payload_digest(request["scope"])

    decision = check_rational_function_identity(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# Fail-closed: malformed rational
# ---------------------------------------------------------------------------


def test_checker_rejects_noncanonical_rational_numerator() -> None:
    request = _request()
    request["scope"]["payload"]["numerator"]["terms"][0]["coefficient"]["num"] = "01"
    _refresh_payload_digest(request["scope"])

    decision = check_rational_function_identity(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_checker_rejects_unreduced_rational() -> None:
    request = _request()
    request["scope"]["payload"]["numerator"]["terms"][0]["coefficient"] = {
        "num": "2",
        "den": "2",
    }
    _refresh_payload_digest(request["scope"])

    decision = check_rational_function_identity(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# Fail-closed: denominator
# ---------------------------------------------------------------------------


def test_checker_rejects_zero_denominator() -> None:
    request = _request()
    request["scope"]["payload"]["denominator"] = {"terms": []}
    _refresh_payload_digest(request["scope"])

    decision = check_rational_function_identity(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_checker_rejects_denominator_with_zero_coefficient_term() -> None:
    request = _request()
    request["scope"]["payload"]["denominator"]["terms"][0]["coefficient"] = {
        "num": "0",
        "den": "1",
    }
    _refresh_payload_digest(request["scope"])

    decision = check_rational_function_identity(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_checker_rejects_non_canonical_term_order() -> None:
    request = _request()
    terms = request["scope"]["payload"]["numerator"]["terms"]
    terms[0], terms[1] = terms[1], terms[0]
    _refresh_payload_digest(request["scope"])

    decision = check_rational_function_identity(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# Fail-closed: overflow
# ---------------------------------------------------------------------------


def _dense_polynomial(term_count: int) -> dict[str, Any]:
    """Build a univariate polynomial with term_count terms in descending order."""
    return {
        "terms": [
            {
                "coefficient": _rational(1, 1),
                "exponents": [term_count - 1 - index],
            }
            for index in range(term_count)
        ]
    }


def test_checker_rejects_cross_product_overflow() -> None:
    request = _request()
    dense = _dense_polynomial(65)
    request["scope"]["payload"]["numerator"] = dense
    request["candidate"]["payload"]["denominator"] = dense
    _refresh_payload_digest(request["scope"])
    _refresh_payload_digest(request["candidate"])

    decision = check_rational_function_identity(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# Thread PRRT_kwDOThEfjc6VuwhT: coefficients beyond Python's 4,300-digit
# int() conversion limit must still be verifiable up to the contract's
# 32,768-digit bound.
# ---------------------------------------------------------------------------


def test_checker_accepts_identity_with_large_canonical_coefficients() -> None:
    """A coefficient with more than 4,300 digits must not trigger
    ValueError from int(); the limit-independent canonical parser must
    handle it and the identity must be verified as TRUE.
    """
    large_digit_count = 5_000
    large_numerator = "1" + "0" * (large_digit_count - 1)
    request = _request(equal=True)
    # Make both sides identical: same numerator and denominator with the
    # large coefficient so the cross-multiplication identity holds.
    identical_function = _function(
        ["x"],
        numerator=_polynomial([(1, 1, 2), (-1, 1, 0)]),
        denominator=_polynomial([(1, 1, 1), (-1, 1, 0)]),
    )
    identical_function["numerator"]["terms"][0]["coefficient"] = {
        "num": large_numerator,
        "den": "1",
    }
    request["scope"]["payload"] = copy.deepcopy(identical_function)
    request["candidate"]["payload"] = copy.deepcopy(identical_function)
    _refresh_payload_digest(request["scope"])
    _refresh_payload_digest(request["candidate"])

    decision = check_rational_function_identity(request)

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"


def test_checker_detects_difference_with_large_canonical_coefficients() -> None:
    """Large coefficients that differ must be detected as FALSE, not
    rejected as UNKNOWN due to int() conversion failure.
    """
    large_digit_count = 5_000
    left_numerator = "1" + "0" * (large_digit_count - 1)
    right_numerator = "2" + "0" * (large_digit_count - 1)
    request = _request(equal=True)
    base_function = _function(
        ["x"],
        numerator=_polynomial([(1, 1, 2), (-1, 1, 0)]),
        denominator=_polynomial([(1, 1, 1), (-1, 1, 0)]),
    )
    left_function = copy.deepcopy(base_function)
    right_function = copy.deepcopy(base_function)
    left_function["numerator"]["terms"][0]["coefficient"] = {
        "num": left_numerator,
        "den": "1",
    }
    right_function["numerator"]["terms"][0]["coefficient"] = {
        "num": right_numerator,
        "den": "1",
    }
    request["scope"]["payload"] = left_function
    request["candidate"]["payload"] = right_function
    _refresh_payload_digest(request["scope"])
    _refresh_payload_digest(request["candidate"])

    decision = check_rational_function_identity(request)

    assert decision["accepted"] is True
    assert decision["conclusion"] == "FALSE"


# ---------------------------------------------------------------------------
# Fail-closed: supporting-artifact and structural mutations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutation",
    [
        lambda request: request.update(request_version="0"),
        lambda request: request["claim"].pop("payload"),
        lambda request: request["certificate"].pop("payload"),
        lambda request: request["scope"].pop("payload"),
        lambda request: request["candidate"].pop("payload"),
        lambda request: request["claim"]["payload"].update(extra=True),
        lambda request: request["certificate"]["payload"]["payload"].update(
            method="DIRECT_SPARSE_REPLAY"
        ),
    ],
    ids=(
        "wrong_version",
        "missing_claim_payload",
        "missing_certificate_payload",
        "missing_scope_payload",
        "missing_candidate_payload",
        "extra_claim_field",
        "wrong_replay_method",
    ),
)
def test_checker_rejects_malformed_or_rebound_payloads(mutation: Any) -> None:
    request = copy.deepcopy(_request())
    mutation(request)
    for key in ("claim", "candidate", "scope", "certificate"):
        if isinstance(request.get(key), dict) and "payload" in request[key]:
            _refresh_payload_digest(request[key])

    decision = check_rational_function_identity(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"
