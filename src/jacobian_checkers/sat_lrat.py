"""Independent bounded replay of the Jacobian ASCII LRAT RUP profile."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from typing import Any

_URI = re.compile(r"^artifact://sha256/[0-9a-f]{64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def _sha(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _result(accepted: bool, detail: str) -> dict[str, Any]:
    return {
        "accepted": accepted,
        "conclusion": "TRUE" if accepted else "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "CHECKED_CERTIFICATE",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _unsupported_detail(error: NotImplementedError) -> str:
    detail = str(error)
    return (
        detail
        if detail.startswith("unsupported LRAT feature:")
        else "unsupported LRAT feature: " + detail
    )


def _artifact(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value)
        == {
            "artifact_uri",
            "object_digest",
            "payload_digest",
            "schema_uri",
            "semantics_uri",
            "parents",
            "payload",
        }
        and isinstance(value["artifact_uri"], str)
        and _URI.fullmatch(value["artifact_uri"]) is not None
        and isinstance(value["object_digest"], str)
        and _DIGEST.fullmatch(value["object_digest"]) is not None
        and isinstance(value["payload_digest"], str)
        and value["payload_digest"] == _sha(_json(value["payload"]))
        and isinstance(value["parents"], list)
    )


def _cnf(payload: object) -> tuple[list[tuple[int, ...]], int]:
    if not isinstance(payload, dict) or payload.get("cnf_schema_version") != "1":
        raise ValueError("invalid canonical CNF")
    variables, rows = payload.get("variables"), payload.get("clauses")
    if not isinstance(variables, list) or not isinstance(rows, list):
        raise ValueError("invalid canonical CNF")
    clauses: list[tuple[int, ...]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"literals"}:
            raise ValueError("invalid canonical CNF")
        lits = row["literals"]
        if not isinstance(lits, list) or any(type(x) is not int for x in lits):
            raise ValueError("invalid canonical CNF")
        if any(x == 0 or abs(x) > len(variables) for x in lits):
            raise ValueError("invalid canonical CNF")
        clauses.append(tuple(lits))
    return clauses, len(variables)


def _integers(fields: list[str], number: int, kind: str = "token") -> list[int]:
    try:
        return [int(field) for field in fields]
    except ValueError as exc:
        raise ValueError(f"line {number}: non-integer {kind}") from exc


def _build_lrat_assignment(
    candidate: tuple[int, ...],
    number: int,
) -> dict[int, bool]:
    assignment: dict[int, bool] = {}
    for lit in candidate:
        var, assumed_value = abs(lit), lit < 0
        if var in assignment and assignment[var] != assumed_value:
            raise ValueError(f"line {number}: tautological candidate")
        assignment[var] = assumed_value
    return assignment


def _check_lrat_rup_hints(
    hints: list[int],
    assignment: dict[int, bool],
    active: dict[int, tuple[int, ...]],
    number: int,
) -> bool:
    conflicted = False
    for hint in hints:
        clause = active.get(hint)
        if clause is None:
            raise ValueError(f"line {number}: hint references inactive clause")
        unassigned: list[int] = []
        satisfied: bool = False
        for lit in clause:
            assigned_value = assignment.get(abs(lit))
            if assigned_value is None:
                unassigned.append(lit)
            elif assigned_value == (lit > 0):
                satisfied = True
        if satisfied or len(unassigned) > 1:
            raise ValueError(f"line {number}: hint is not unit or conflicting")
        if not unassigned:
            conflicted = True
            break
        lit = unassigned[0]
        assignment[abs(lit)] = lit > 0
    return conflicted


def _process_lrat_line(
    line: str,
    number: int,
    last_id: int,
    active: dict[int, tuple[int, ...]],
    limits: dict[str, int],
) -> tuple[int, tuple[int, ...], bool]:
    fields = line.split()
    clause_id = _integers(fields[:1], number, "clause id")[0]
    if clause_id <= 0:
        raise ValueError(f"line {number}: invalid clause id")
    if len(fields) > 1 and fields[1] == "d":
        raise NotImplementedError("unsupported LRAT feature: deletions")
    values = _integers(fields, number)
    if clause_id <= last_id or values.count(0) != 2:
        raise ValueError(f"line {number}: invalid addition framing")
    first_zero = values.index(0, 1)
    candidate = tuple(values[1:first_zero])
    hints = values[first_zero + 1 : -1]
    if any(lit == 0 or abs(lit) > limits["variable_count"] for lit in candidate):
        raise ValueError(f"line {number}: invalid literal")
    if len(candidate) > limits["max_clause_literals"]:
        raise OverflowError("LRAT replay exceeded max_clause_literals")
    if not hints or len(hints) > limits["max_hints_per_step"]:
        raise ValueError(f"line {number}: invalid hint count")
    if any(hint <= 0 for hint in hints):
        raise NotImplementedError("negative RAT hints are unsupported in v1")
    assignment = _build_lrat_assignment(candidate, number)
    conflicted = _check_lrat_rup_hints(hints, assignment, active, number)
    if not conflicted:
        raise ValueError(f"line {number}: hints do not establish RUP")
    return clause_id, candidate, not candidate


def _replay(
    clauses: list[tuple[int, ...]], proof: bytes, limits: dict[str, int]
) -> tuple[bool, str]:
    try:
        text = proof.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("proof is not ASCII") from exc
    started = time.monotonic()
    active = dict(enumerate(clauses, 1))
    last_id = len(clauses)
    empty = False
    steps = 0
    for number, raw_line in enumerate(text.splitlines(), 1):
        if (time.monotonic() - started) * 1000 >= limits["timeout_ms"]:
            raise TimeoutError("LRAT replay exceeded timeout_ms")
        line = raw_line.strip()
        if not line or line.startswith("c "):
            continue
        steps += 1
        if steps > limits["max_steps"]:
            raise OverflowError("LRAT replay exceeded max_steps")
        clause_id, candidate, is_empty = _process_lrat_line(
            line, number, last_id, active, limits
        )
        active[clause_id] = candidate
        last_id = clause_id
        empty = empty or is_empty
    return empty, f"replayed {steps} LRAT steps"


_LRAT_REQUEST_KEYS = {
    "request_version",
    "claim",
    "candidate",
    "scope",
    "certificate",
    "expected_bindings",
}
_LRAT_BINDING_KEYS = {
    "claim_digest",
    "semantics_digest",
    "candidate_digest",
    "scope_digest",
    "encoding_digest",
}


def _check_lrat_request_envelope(request: object) -> str | None:
    if not isinstance(request, dict) or set(request) != _LRAT_REQUEST_KEYS:
        return "malformed checker request"
    if request["request_version"] != "1" or request["scope"] is not None:
        return "unsupported checker request"
    return None


def _check_lrat_artifacts_and_bindings(
    claim: dict[str, Any],
    candidate: dict[str, Any],
    certificate: dict[str, Any],
    bindings: object,
) -> str | None:
    if not all(_artifact(item) for item in (claim, candidate, certificate)):
        return "malformed checker artifacts"
    if (
        claim["semantics_uri"] != candidate["semantics_uri"]
        or claim["semantics_uri"] != certificate["semantics_uri"]
        or claim["artifact_uri"] not in candidate["parents"]
        or not {claim["artifact_uri"], candidate["artifact_uri"]}.issubset(
            certificate["parents"]
        )
    ):
        return "proof is missing CNF lineage"
    if (
        not isinstance(bindings, dict)
        or set(bindings) != _LRAT_BINDING_KEYS
        or bindings.get("claim_digest") != claim["object_digest"]
        or bindings.get("candidate_digest") != candidate["object_digest"]
        or bindings.get("scope_digest") is not None
        or bindings.get("encoding_digest") is not None
    ):
        return "evidence bindings do not match"
    return None


def _check_lrat_payload(
    payload: object,
    claim: dict[str, Any],
) -> str | None:
    if (
        not isinstance(payload, dict)
        or payload.get("proof_format") != "LRAT-ASCII"
        or payload.get("proof_format_version") != "jacobian.lrat.rup/v1"
        or payload.get("cnf", {}).get("cnf_artifact_uri") != claim["artifact_uri"]
    ):
        return "unsupported or unbound LRAT artifact"
    expected_cnf_binding = {
        "binding_version": "1",
        "cnf_artifact_uri": claim["artifact_uri"],
        "cnf_object_digest": claim["object_digest"],
        "cnf_payload_digest": claim["payload_digest"],
        "variable_map_digest": claim["payload"]["variable_map_digest"],
        "dimacs_digest": claim["payload"]["dimacs_digest"],
        "projection_format": claim["payload"]["projection_format"],
        "projection_version": claim["payload"]["projection_version"],
        "variable_count": len(claim["payload"]["variables"]),
        "clause_count": len(claim["payload"]["clauses"]),
    }
    if payload["cnf"] != expected_cnf_binding:
        return "LRAT source binding does not match exact CNF"
    return None


def _check_lrat_bytes(
    raw: bytes,
    payload: dict[str, Any],
) -> str | None:
    if (
        base64.b64encode(raw).decode("ascii") != payload["proof_base64"]
        or _sha(raw) != payload.get("proof_digest")
        or len(raw) != payload.get("proof_byte_count")
    ):
        return "LRAT bytes do not match their binding"
    return None


def _check_lrat_certificate(
    envelope: object,
    bindings: object,
    expected_payload: dict[str, Any],
) -> str | None:
    if (
        not isinstance(envelope, dict)
        or envelope.get("certificate_type") != "sat.lrat-proof"
        or envelope.get("format_version") != "1"
        or envelope.get("bindings") != bindings
        or envelope.get("payload") != expected_payload
        or envelope.get("payload_digest") != _sha(_json(expected_payload))
    ):
        return "LRAT certificate is not exactly bound"
    return None


def check_lrat(request: dict[str, Any]) -> dict[str, Any]:
    """Accept only a bound proof whose RUP hints derive the empty clause."""
    try:
        error = _check_lrat_request_envelope(request)
        if error is not None:
            return _result(False, error)
        claim, candidate, certificate = (
            request["claim"],
            request["candidate"],
            request["certificate"],
        )
        bindings = request["expected_bindings"]
        error = _check_lrat_artifacts_and_bindings(
            claim, candidate, certificate, bindings
        )
        if error is not None:
            return _result(False, error)
        payload = candidate["payload"]
        error = _check_lrat_payload(payload, claim)
        if error is not None:
            return _result(False, error)
        raw = base64.b64decode(payload["proof_base64"], validate=True)
        error = _check_lrat_bytes(raw, payload)
        if error is not None:
            return _result(False, error)
        expected_payload = {
            "cnf_uri": claim["artifact_uri"],
            "proof_uri": candidate["artifact_uri"],
            "proof_digest": payload["proof_digest"],
            "limits": payload["limits"],
        }
        error = _check_lrat_certificate(
            certificate["payload"], bindings, expected_payload
        )
        if error is not None:
            return _result(False, error)
        clauses, variable_count = _cnf(claim["payload"])
        limits = dict(payload["limits"])
        limits["variable_count"] = variable_count
        accepted, detail = _replay(clauses, raw, limits)
        return _result(
            accepted, detail if accepted else detail + "; empty clause absent"
        )
    except NotImplementedError as exc:
        return _result(False, _unsupported_detail(exc))
    except TimeoutError:
        return _result(False, "LRAT replay timed out")
    except (KeyError, TypeError, ValueError, OverflowError):
        return _result(False, "malformed, rejected, or over-budget LRAT proof")
