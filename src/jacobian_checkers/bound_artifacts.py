"""Passive JSON binding checks shared by independent checker entrypoints."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import rfc8785

_ARTIFACT_URI = re.compile(r"^artifact://sha256/[0-9a-f]{64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_ARTIFACT_KEYS = {
    "artifact_uri",
    "object_digest",
    "payload_digest",
    "schema_uri",
    "semantics_uri",
    "parents",
    "payload",
}
_BINDING_KEYS = {
    "claim_digest",
    "semantics_digest",
    "candidate_digest",
    "scope_digest",
    "encoding_digest",
}


def valid_unscoped_unencoded_bindings(value: object) -> bool:
    """Validate exact five-key bindings without scope or encoding digests."""

    if not isinstance(value, dict) or set(value) != _BINDING_KEYS:
        return False
    if value["scope_digest"] is not None or value["encoding_digest"] is not None:
        return False
    return all(
        isinstance(value[key], str) and _DIGEST.fullmatch(value[key]) is not None
        for key in ("claim_digest", "semantics_digest", "candidate_digest")
    )


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _artifact(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _ARTIFACT_KEYS:
        raise ValueError("artifact metadata is malformed")
    if not all(
        isinstance(value[key], str) and pattern.fullmatch(value[key]) is not None
        for key, pattern in (
            ("artifact_uri", _ARTIFACT_URI),
            ("object_digest", _DIGEST),
            ("payload_digest", _DIGEST),
            ("schema_uri", _ARTIFACT_URI),
            ("semantics_uri", _ARTIFACT_URI),
        )
    ):
        raise ValueError("artifact identifiers are malformed")
    parents = value["parents"]
    if (
        not isinstance(parents, list)
        or len(parents) != len(set(parents))
        or not all(
            isinstance(parent, str) and _ARTIFACT_URI.fullmatch(parent)
            for parent in parents
        )
    ):
        raise ValueError("artifact lineage is malformed")
    if value["payload_digest"] != _digest(value["payload"]):
        raise ValueError("artifact payload digest does not match")
    return value


def _inline_exact_digest(value: dict[str, Any]) -> str:
    """Return the protocol digest for an inline, non-artifact exact value."""

    return (
        "sha256:"
        + hashlib.sha256(
            rfc8785.dumps(
                {
                    "inline_exact_value_version": "1",
                    "schema_uri": value["schema_uri"],
                    "semantics_uri": value["semantics_uri"],
                    "payload": value["payload"],
                }
            )
        ).hexdigest()
    )


def _inline_value(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_uri",
        "semantics_uri",
        "payload",
    }:
        raise ValueError("inline exact value is malformed")
    if not all(
        isinstance(value[key], str) and _ARTIFACT_URI.fullmatch(value[key])
        for key in ("schema_uri", "semantics_uri")
    ):
        raise ValueError("inline exact value identifiers are malformed")
    _inline_exact_digest(value)
    return value


def _bound_inline_request(
    request: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        set(request)
        != {
            "request_version",
            "claim",
            "candidate",
            "semantics",
            "scope",
            "expected_bindings",
        }
        or request["request_version"] != "2"
        or request["scope"] is not None
    ):
        raise ValueError("inline checker request is malformed")
    claim = _inline_value(request["claim"])
    candidate = _inline_value(request["candidate"])
    semantics = _artifact(request["semantics"])
    bindings = request["expected_bindings"]
    if (
        claim["semantics_uri"] != semantics["artifact_uri"]
        or candidate["semantics_uri"] != semantics["artifact_uri"]
        or not valid_unscoped_unencoded_bindings(bindings)
        or bindings["claim_digest"] != _inline_exact_digest(claim)
        or bindings["candidate_digest"] != _inline_exact_digest(candidate)
        or bindings["semantics_digest"] != semantics["object_digest"]
    ):
        raise ValueError("inline exact bindings are malformed or mismatched")
    return claim["payload"], candidate["payload"]


def bound_request(
    request: object, *, operation_id: str, witness_format: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate exact artifact and evidence binding before mathematical replay."""

    if not isinstance(request, dict):
        raise ValueError("checker request is malformed")
    if request.get("request_version") == "2":
        return _bound_inline_request(request)
    if set(request) != {
        "request_version",
        "claim",
        "candidate",
        "semantics",
        "scope",
        "witness",
        "expected_bindings",
    }:
        raise ValueError("checker request is malformed")
    if request["request_version"] != "1" or request["scope"] is not None:
        raise ValueError("checker request version or scope is unsupported")
    claim = _artifact(request["claim"])
    candidate = _artifact(request["candidate"])
    semantics = _artifact(request["semantics"])
    witness = _artifact(request["witness"])
    if (
        len(candidate["parents"]) != 1
        or candidate["parents"][0] != claim["artifact_uri"]
    ):
        raise ValueError("candidate is not bound to the input artifact")
    if (
        claim["semantics_uri"] != candidate["semantics_uri"]
        or claim["semantics_uri"] != witness["semantics_uri"]
    ):
        raise ValueError("artifacts use different semantics")
    bindings = request["expected_bindings"]
    if (
        not valid_unscoped_unencoded_bindings(bindings)
        or bindings["claim_digest"] != claim["object_digest"]
        or bindings["candidate_digest"] != candidate["object_digest"]
        or bindings["semantics_digest"] != semantics["object_digest"]
        or semantics["artifact_uri"] != claim["semantics_uri"]
    ):
        raise ValueError("evidence bindings are malformed or mismatched")
    envelope = witness["payload"]
    if (
        not isinstance(envelope, dict)
        or set(envelope)
        != {
            "evidence_schema_version",
            "witness_format",
            "format_version",
            "role",
            "bindings",
            "payload",
        }
        or envelope["evidence_schema_version"] != "1"
        or envelope["witness_format"] != witness_format
        or envelope["format_version"] != "1"
        or envelope["role"] != "SUPPORTS_CLAIM"
        or envelope["bindings"] != bindings
        or envelope["payload"]
        != {
            "operation_id": operation_id,
            "input_uri": claim["artifact_uri"],
            "result_uri": candidate["artifact_uri"],
        }
        or len(witness["parents"]) != 2
        or set(witness["parents"]) != {claim["artifact_uri"], candidate["artifact_uri"]}
    ):
        raise ValueError("witness is not exactly bound to the operation artifacts")
    return claim["payload"], candidate["payload"]


__all__ = ["bound_request", "valid_unscoped_unencoded_bindings"]
