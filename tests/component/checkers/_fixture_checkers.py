"""Independent checker entrypoints used by orchestration integration tests."""

from __future__ import annotations

import hashlib
import os
from typing import Any


def fail_with_internal_detail(_request: dict[str, Any]) -> dict[str, Any]:
    raise RuntimeError("provider=fixture internal-checker-id=secret")


def exit_without_response(_request: dict[str, Any]) -> dict[str, Any]:
    os._exit(0)


def return_invalid_decision(_request: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted": True,
        "conclusion": "NOT_A_CONCLUSION",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": "invalid fixture decision",
    }


def imitate_source_change(_request: dict[str, Any]) -> dict[str, Any]:
    raise ValueError("checker source changed during execution")


def emit_large_diagnostic(request: dict[str, Any]) -> dict[str, Any]:
    print("x" * 4096)
    return check_fixture_value(request)


def check_fixture_value(request: dict[str, Any]) -> dict[str, Any]:
    witness = request["witness"]["payload"]
    candidate = request["candidate"]["payload"]
    role = witness.get("role")
    accepted = (
        request.get("request_version") == "1"
        and witness.get("witness_format") == "fixture.value"
        and witness.get("format_version") == "1"
        and role
        in {
            "DEFEATS_CANDIDATE",
            "REFUTES_CLAIM",
            "RESCUES_CANDIDATE",
            "SUPPORTS_CLAIM",
        }
        and witness.get("bindings") == request.get("expected_bindings")
        and witness.get("payload", {}).get("observed") == str(candidate.get("value"))
    )
    return {
        "accepted": accepted,
        "conclusion": (
            "FALSE"
            if accepted and role in {"DEFEATS_CANDIDATE", "REFUTES_CLAIM"}
            else ("TRUE" if accepted else "UNKNOWN")
        ),
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": (
            "fixture value matches the bound candidate"
            if accepted
            else "fixture witness does not match the bound candidate"
        ),
    }


def check_fixture_value_as_true(request: dict[str, Any]) -> dict[str, Any]:
    decision = check_fixture_value(request)
    if decision["accepted"]:
        decision["conclusion"] = "TRUE"
    return decision


def check_parameter_region_certificate(request: dict[str, Any]) -> dict[str, Any]:
    candidate = request["candidate"]["payload"]
    certificate = request["certificate"]["payload"]
    proof = certificate.get("payload", {})
    accepted = (
        request.get("request_version") == "1"
        and certificate.get("certificate_type") == "fixture.parameter_region"
        and certificate.get("format_version") == "1"
        and certificate.get("bindings") == request.get("expected_bindings")
        and proof.get("kind") == candidate.get("kind")
        and proof.get("conditions") == candidate.get("conditions")
        and candidate.get("claim_uri") == request["claim"]["artifact_uri"]
    )
    return {
        "accepted": accepted,
        "conclusion": "TRUE" if accepted else "UNKNOWN",
        "arithmetic": "SYMBOLIC",
        "method": "CHECKED_CERTIFICATE",
        "coverage": "EXHAUSTIVE",
        "detail": (
            "certificate proves the exact parameter-region subject"
            if accepted
            else "certificate does not prove the bound parameter region"
        ),
    }


def check_supporting_artifact_metadata(request: dict[str, Any]) -> dict[str, Any]:
    supporting_artifacts = request.get("supporting_artifacts")
    support = (
        supporting_artifacts[0]
        if isinstance(supporting_artifacts, list) and len(supporting_artifacts) == 1
        else None
    )
    accepted = (
        request.get("request_version") == "1"
        and isinstance(support, dict)
        and support.get("payload_digest")
        == "sha256:" + hashlib.sha256(b'{"kind":"supporting-evidence"}').hexdigest()
        and support.get("parents") == []
    )
    return {
        "accepted": accepted,
        "conclusion": "FALSE" if accepted else "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "CHECKED_CERTIFICATE",
        "coverage": "EXHAUSTIVE",
        "detail": (
            "supporting artifact includes its requested storage metadata"
            if accepted
            else "supporting artifact storage metadata is missing or incomplete"
        ),
    }
