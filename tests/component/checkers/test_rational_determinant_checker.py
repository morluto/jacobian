from __future__ import annotations

import copy
from typing import Any

import pytest
from tests.support.artifacts import artifact_uri as _uri
from tests.support.artifacts import json_digest as _digest
from tests.support.rationals import rational_payload as _rational

from jacobian_checkers.rational_determinants import check_rational_determinant


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


def _request() -> dict[str, Any]:
    semantics_uri = _uri("e")
    source_payload = {
        "matrix_schema_version": "1",
        "domain": "QQ",
        "entries": [
            [_rational(1), _rational(0), _rational(1)],
            [_rational(2), _rational(-1), _rational(3)],
            [_rational(4), _rational(3), _rational(2)],
        ],
    }
    claim = _artifact(
        uri_character="a",
        object_character="1",
        schema_character="b",
        semantics_uri=semantics_uri,
        payload=source_payload,
        parents=[],
    )
    candidate_payload = {
        "result_schema_version": "1",
        "matrix_uri": claim["artifact_uri"],
        "determinant": _rational(-1),
        "method": "FRACTION_FREE_BAREISS",
        "backend": "sympy",
        "backend_version": "1.14.0",
    }
    candidate = _artifact(
        uri_character="c",
        object_character="2",
        schema_character="d",
        semantics_uri=semantics_uri,
        payload=candidate_payload,
        parents=[claim["artifact_uri"]],
    )
    bindings = {
        "claim_digest": claim["object_digest"],
        "semantics_digest": "sha256:" + "3" * 64,
        "candidate_digest": candidate["object_digest"],
        "scope_digest": None,
        "encoding_digest": None,
    }
    witness_payload = {
        "evidence_schema_version": "1",
        "witness_format": "matrix.rational_determinant",
        "format_version": "1",
        "role": "SUPPORTS_CLAIM",
        "bindings": bindings,
        "payload": {
            "matrix_uri": claim["artifact_uri"],
            "determinant_uri": candidate["artifact_uri"],
        },
    }
    witness = _artifact(
        uri_character="f",
        object_character="4",
        schema_character="5",
        semantics_uri=semantics_uri,
        payload=witness_payload,
        parents=[claim["artifact_uri"], candidate["artifact_uri"]],
    )
    return {
        "request_version": "1",
        "claim": claim,
        "candidate": candidate,
        "scope": None,
        "witness": witness,
        "expected_bindings": bindings,
    }


def _refresh_payload_digest(artifact: dict[str, Any]) -> None:
    artifact["payload_digest"] = _digest(artifact["payload"])


def test_checker_accepts_exact_rational_determinant() -> None:
    decision = check_rational_determinant(_request())

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"
    assert decision["arithmetic"] == "EXACT_RATIONAL"
    assert decision["method"] == "DIRECT_WITNESS"


def test_checker_rejects_mutated_determinant_with_refreshed_digest() -> None:
    request = _request()
    request["candidate"]["payload"]["determinant"] = _rational(1)
    _refresh_payload_digest(request["candidate"])

    decision = check_rational_determinant(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_checker_rejects_rebound_source_identity() -> None:
    request = _request()
    request["candidate"]["payload"]["matrix_uri"] = _uri("9")
    _refresh_payload_digest(request["candidate"])

    decision = check_rational_determinant(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda request: request["claim"]["payload"]["entries"][0][0].update(num="01"),
        lambda request: request["candidate"]["payload"]["determinant"].update(den="0"),
        lambda request: request["candidate"]["payload"].update(extra=True),
        lambda request: request["candidate"]["parents"].append(_uri("7")),
        lambda request: request["witness"]["payload"]["bindings"].update(
            claim_digest="sha256:" + "8" * 64
        ),
    ],
    ids=(
        "noncanonical_rational",
        "zero_denominator",
        "extra_candidate_field",
        "ambiguous_candidate_lineage",
        "rebound_witness",
    ),
)
def test_checker_rejects_malformed_or_rebound_payloads(mutation: Any) -> None:
    request = copy.deepcopy(_request())
    mutation(request)
    for key in ("claim", "candidate", "witness"):
        _refresh_payload_digest(request[key])

    decision = check_rational_determinant(request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"
