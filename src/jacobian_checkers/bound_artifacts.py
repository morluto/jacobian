"""Passive JSON binding checks shared by independent checker entrypoints."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

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


def bound_request(
    request: object, *, operation_id: str, witness_format: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate exact artifact and evidence binding before mathematical replay."""

    if not isinstance(request, dict) or set(request) != {
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
        not isinstance(bindings, dict)
        or set(bindings) != _BINDING_KEYS
        or bindings["scope_digest"] is not None
        or bindings["encoding_digest"] is not None
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


__all__ = ["bound_request"]
