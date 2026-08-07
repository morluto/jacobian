"""Independent exact replay for total SAT assignments.

This module intentionally uses only the Python standard library.  It does not
import Jacobian's SAT contracts or any solver implementation.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

from jacobian.process_policy import (
    ProcessRequest,
    ProcessTermination,
    execute_process,
)
from jacobian.worker_environment import worker_environment
from jacobian_checkers.bound_artifacts import valid_unscoped_unencoded_bindings

_ARTIFACT_URI = re.compile(r"^artifact://sha256/[0-9a-f]{64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_VARIABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$")
_DRAT_INTEGER = re.compile(r"^(?:0|-?[1-9][0-9]*)$")
_ARTIFACT_KEYS = {
    "artifact_uri",
    "object_digest",
    "payload_digest",
    "schema_uri",
    "semantics_uri",
    "parents",
    "payload",
}
DRAT_TRIM_OUTPUT_LIMIT = 1024 * 1024
DRAT_TRIM_TIMEOUT_SECONDS = 120
_DRAT_ASCII_PREFIX = b"c jacobian drat-text/v1 force-ascii 0123456789\n"


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _reject_proof(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "CHECKED_CERTIFICATE",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    # SAT claim payloads contain only ASCII strings, integers, lists, and maps,
    # for which this is the same byte representation as Jacobian's canonical
    # JSON encoder.
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _literal_key(literal: int) -> tuple[int, bool]:
    return abs(literal), literal > 0


def _valid_artifact(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != _ARTIFACT_KEYS:
        return False
    if (
        not isinstance(value["artifact_uri"], str)
        or _ARTIFACT_URI.fullmatch(value["artifact_uri"]) is None
        or not isinstance(value["object_digest"], str)
        or _DIGEST.fullmatch(value["object_digest"]) is None
        or not isinstance(value["payload_digest"], str)
        or _DIGEST.fullmatch(value["payload_digest"]) is None
        or not isinstance(value["schema_uri"], str)
        or _ARTIFACT_URI.fullmatch(value["schema_uri"]) is None
        or not isinstance(value["semantics_uri"], str)
        or _ARTIFACT_URI.fullmatch(value["semantics_uri"]) is None
    ):
        return False
    parents = value["parents"]
    return (
        isinstance(parents, list)
        and len(parents) == len(set(parents))
        and all(
            isinstance(parent, str) and _ARTIFACT_URI.fullmatch(parent) is not None
            for parent in parents
        )
    )


def _validate_cnf(payload: object) -> tuple[list[list[int]], int, str, str]:
    if not isinstance(payload, dict) or set(payload) != {
        "cnf_schema_version",
        "variables",
        "clauses",
        "projection_format",
        "projection_version",
        "variable_map_digest",
        "dimacs_digest",
    }:
        raise ValueError("canonical CNF has an invalid shape")
    if (
        payload["cnf_schema_version"] != "1"
        or payload["projection_format"] != "DIMACS-CNF"
        or payload["projection_version"] != "jacobian.dimacs.cnf/v1"
    ):
        raise ValueError("canonical CNF uses an unsupported format")

    variables = payload["variables"]
    if not isinstance(variables, list) or len(variables) > 1_000_000:
        raise ValueError("canonical CNF variable map is malformed")
    names: list[str] = []
    for expected_id, variable in enumerate(variables, start=1):
        if (
            not isinstance(variable, dict)
            or set(variable) != {"id", "name"}
            or type(variable["id"]) is not int
            or variable["id"] != expected_id
            or not isinstance(variable["name"], str)
            or _VARIABLE_NAME.fullmatch(variable["name"]) is None
            or unicodedata.normalize("NFC", variable["name"]) != variable["name"]
        ):
            raise ValueError("canonical CNF variable map is malformed")
        names.append(variable["name"])
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError("canonical CNF variable map is not canonical")

    clauses = payload["clauses"]
    if not isinstance(clauses, list) or len(clauses) > 1_000_000:
        raise ValueError("canonical CNF clauses are malformed")
    normalized_clauses: list[list[int]] = []
    clause_keys: list[tuple[tuple[int, bool], ...]] = []
    for clause in clauses:
        if (
            not isinstance(clause, dict)
            or set(clause) != {"literals"}
            or not isinstance(clause["literals"], list)
            or len(clause["literals"]) > 1_000_000
        ):
            raise ValueError("canonical CNF clause is malformed")
        literals = clause["literals"]
        if any(
            type(literal) is not int or literal == 0 or abs(literal) > len(variables)
            for literal in literals
        ):
            raise ValueError("canonical CNF literal is malformed")
        if (
            len(literals) != len(set(literals))
            or any(-literal in literals for literal in literals)
            or literals != sorted(literals, key=_literal_key)
        ):
            raise ValueError("canonical CNF clause is not canonical")
        normalized_clauses.append(literals)
        clause_keys.append(tuple(_literal_key(literal) for literal in literals))
    if clause_keys != sorted(clause_keys) or len(clause_keys) != len(set(clause_keys)):
        raise ValueError("canonical CNF clauses are not canonical")

    variable_map_payload = {
        "variable_map_format": "jacobian.sat.variable-map/v1",
        "variables": variables,
    }
    variable_map_digest = _sha256(_canonical_json(variable_map_payload))
    if payload["variable_map_digest"] != variable_map_digest:
        raise ValueError("canonical CNF variable-map digest does not match")
    rows = [f"p cnf {len(variables)} {len(clauses)}\n"]
    for literals in normalized_clauses:
        prefix = " ".join(str(literal) for literal in literals)
        rows.append(f"{prefix} 0\n" if prefix else "0\n")
    dimacs_digest = _sha256("".join(rows).encode("ascii"))
    if payload["dimacs_digest"] != dimacs_digest:
        raise ValueError("canonical CNF DIMACS digest does not match")
    return normalized_clauses, len(variables), variable_map_digest, dimacs_digest


def _validate_assignment(
    payload: object,
    *,
    claim: dict[str, Any],
    variable_count: int,
    clause_count: int,
    variable_map_digest: str,
    dimacs_digest: str,
) -> list[bool]:
    if not isinstance(payload, dict) or set(payload) != {
        "assignment_schema_version",
        "cnf",
        "declared_scope",
        "values",
        "producer",
        "resource_budget",
    }:
        raise ValueError("SAT assignment has an invalid shape")
    if (
        payload["assignment_schema_version"] != "1"
        or payload["declared_scope"] != "FULL_CNF"
        or not isinstance(payload["producer"], dict)
        or not isinstance(payload["resource_budget"], dict)
    ):
        raise ValueError("SAT assignment metadata is malformed")
    binding = payload["cnf"]
    if not isinstance(binding, dict) or set(binding) != {
        "binding_version",
        "cnf_artifact_uri",
        "cnf_object_digest",
        "cnf_payload_digest",
        "variable_map_digest",
        "dimacs_digest",
        "projection_format",
        "projection_version",
        "variable_count",
        "clause_count",
    }:
        raise ValueError("SAT assignment CNF binding is malformed")
    if (
        binding["binding_version"] != "1"
        or not isinstance(binding["cnf_artifact_uri"], str)
        or _ARTIFACT_URI.fullmatch(binding["cnf_artifact_uri"]) is None
        or any(
            not isinstance(binding[key], str) or _DIGEST.fullmatch(binding[key]) is None
            for key in (
                "cnf_object_digest",
                "cnf_payload_digest",
                "variable_map_digest",
                "dimacs_digest",
            )
        )
        or type(binding["variable_count"]) is not int
        or type(binding["clause_count"]) is not int
    ):
        raise ValueError("SAT assignment CNF binding is malformed")
    expected_binding = {
        "binding_version": "1",
        "cnf_artifact_uri": claim["artifact_uri"],
        "cnf_object_digest": claim["object_digest"],
        "cnf_payload_digest": claim["payload_digest"],
        "variable_map_digest": variable_map_digest,
        "dimacs_digest": dimacs_digest,
        "projection_format": "DIMACS-CNF",
        "projection_version": "jacobian.dimacs.cnf/v1",
        "variable_count": variable_count,
        "clause_count": clause_count,
    }
    if binding != expected_binding:
        raise ValueError("SAT assignment is not bound to the exact canonical CNF")
    values = payload["values"]
    if (
        not isinstance(values, list)
        or len(values) != variable_count
        or any(type(value) is not bool for value in values)
    ):
        raise ValueError("SAT assignment is not total and Boolean")
    return values


def _dimacs_bytes(clauses: list[list[int]], variable_count: int) -> bytes:
    rows = [f"p cnf {variable_count} {len(clauses)}\n"]
    for literals in clauses:
        prefix = " ".join(str(literal) for literal in literals)
        rows.append(f"{prefix} 0\n" if prefix else "0\n")
    return "".join(rows).encode("ascii")


def _validate_proof(
    payload: object,
    *,
    claim: dict[str, Any],
    variable_count: int,
    clause_count: int,
    variable_map_digest: str,
    dimacs_digest: str,
) -> bytes:
    if not isinstance(payload, dict) or set(payload) != {
        "proof_schema_version",
        "cnf",
        "declared_scope",
        "proof_format",
        "proof_format_version",
        "proof_encoding",
        "proof_base64",
        "proof_digest",
        "producer",
        "resource_budget",
    }:
        raise ValueError("SAT proof has an invalid shape")
    if (
        payload["proof_schema_version"] != "1"
        or payload["declared_scope"] != "FULL_CNF"
        or payload["proof_format"] != "DRAT"
        or payload["proof_format_version"] != "drat-text/v1"
        or payload["proof_encoding"] != "BASE64"
        or not isinstance(payload["producer"], dict)
        or not isinstance(payload["resource_budget"], dict)
        or not isinstance(payload["proof_base64"], str)
        or len(payload["proof_base64"]) > 8_000_000
        or not isinstance(payload["proof_digest"], str)
        or _DIGEST.fullmatch(payload["proof_digest"]) is None
    ):
        raise ValueError("SAT proof metadata is malformed")
    try:
        raw = base64.b64decode(
            payload["proof_base64"].encode("ascii"),
            validate=True,
        )
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise ValueError("SAT proof base64 is malformed") from exc
    if (
        base64.b64encode(raw).decode("ascii") != payload["proof_base64"]
        or _sha256(raw) != payload["proof_digest"]
    ):
        raise ValueError("SAT proof bytes do not match their exact digest")
    binding = payload["cnf"]
    expected_binding = {
        "binding_version": "1",
        "cnf_artifact_uri": claim["artifact_uri"],
        "cnf_object_digest": claim["object_digest"],
        "cnf_payload_digest": claim["payload_digest"],
        "variable_map_digest": variable_map_digest,
        "dimacs_digest": dimacs_digest,
        "projection_format": "DIMACS-CNF",
        "projection_version": "jacobian.dimacs.cnf/v1",
        "variable_count": variable_count,
        "clause_count": clause_count,
    }
    if binding != expected_binding:
        raise ValueError("SAT proof is not bound to the exact canonical CNF")
    return raw


def _validate_certificate(
    certificate: dict[str, Any],
    *,
    claim: dict[str, Any],
    candidate: dict[str, Any],
    expected_bindings: dict[str, Any],
) -> None:
    envelope = certificate["payload"]
    if not isinstance(envelope, dict) or set(envelope) != {
        "evidence_schema_version",
        "certificate_type",
        "format_version",
        "bindings",
        "payload_digest",
        "payload",
    }:
        raise ValueError("SAT proof certificate has an invalid shape")
    payload = {
        "cnf_uri": claim["artifact_uri"],
        "proof_uri": candidate["artifact_uri"],
    }
    if (
        envelope["evidence_schema_version"] != "1"
        or envelope["certificate_type"] != "sat.unsat-proof"
        or envelope["format_version"] != "1"
        or envelope["bindings"] != expected_bindings
        or envelope["payload"] != payload
        or envelope["payload_digest"] != _sha256(_canonical_json(payload))
    ):
        raise ValueError("SAT proof certificate is not exactly bound")


def _validate_drat_text_profile(proof: bytes) -> None:
    try:
        text = proof.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("DRAT text proof is not ASCII") from exc
    empty_clause_seen = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line == "c" or line.startswith("c "):
            continue
        tokens = line.split()
        deletion = tokens[0] == "d"
        if deletion:
            tokens = tokens[1:]
        if (
            not tokens
            or any(_DRAT_INTEGER.fullmatch(token) is None for token in tokens)
            or tokens[-1] != "0"
            or "0" in tokens[:-1]
        ):
            raise ValueError("DRAT text proof has malformed clause syntax")
        literals = [int(token) for token in tokens[:-1]]
        if (
            any(abs(literal) > (1 << 31) - 1 for literal in literals)
            or len(literals) != len(set(literals))
            or any(-literal in literals for literal in literals)
        ):
            raise ValueError("DRAT text proof has a malformed clause")
        if empty_clause_seen and not deletion:
            raise ValueError(
                "DRAT text proof has a non-deletion step after the empty clause"
            )
        if not literals:
            if deletion:
                raise ValueError("DRAT text proof deletes the empty clause")
            empty_clause_seen = True


def _authorized_runtime() -> tuple[Path, str]:
    executable = os.environ.get("JACOBIAN_CHECKER_EXECUTABLE")
    expected_digest = os.environ.get("JACOBIAN_CHECKER_RUNTIME_DIGEST")
    if (
        executable is None
        or expected_digest is None
        or _DIGEST.fullmatch(expected_digest) is None
    ):
        raise ValueError("DRAT-trim runtime is not operator authorized")
    path = Path(executable).resolve(strict=True)
    if str(path) != executable or not path.is_file() or path.is_symlink():
        raise ValueError("DRAT-trim runtime path is not exact")
    if _sha256(path.read_bytes()) != expected_digest:
        raise ValueError("DRAT-trim runtime digest changed")
    return path, expected_digest


def _bounded_drat_trim(
    executable: Path,
    *,
    cnf: bytes,
    proof: bytes,
    expected_runtime_digest: str,
) -> tuple[bool, str | None]:
    with tempfile.TemporaryDirectory(prefix="jacobian-drat-trim-") as directory:
        root = Path(directory)
        cnf_path = root / "input.cnf"
        proof_path = root / "proof.drat"
        cnf_path.write_bytes(cnf)
        proof_path.write_bytes(_DRAT_ASCII_PREFIX + proof)
        command = [
            str(executable),
            str(cnf_path),
            str(proof_path),
            "-f",
            "-W",
        ]
        result = execute_process(
            ProcessRequest(
                executable=command[0],
                arguments=tuple(command[1:]),
                environment=worker_environment(locale="C"),
                cwd=str(root),
                timeout_seconds=DRAT_TRIM_TIMEOUT_SECONDS,
                stdin_bytes=b"",
                stdout_limit_bytes=DRAT_TRIM_OUTPUT_LIMIT,
                stderr_limit_bytes=DRAT_TRIM_OUTPUT_LIMIT,
            )
        )
        if result.termination is ProcessTermination.TIMED_OUT:
            return False, "DRAT_TIMEOUT"
        if result.termination is ProcessTermination.OUTPUT_LIMIT_EXCEEDED:
            return False, "DRAT_OUTPUT_LIMIT_EXCEEDED"
        if result.termination is ProcessTermination.START_FAILED:
            return False, "DRAT_START_FAILED"
        if _sha256(executable.read_bytes()) != expected_runtime_digest:
            raise ValueError("DRAT-trim runtime changed during replay")
        try:
            lines = [
                line.strip()
                for line in result.stdout.decode("ascii")
                .replace("\r", "\n")
                .splitlines()
                if line.strip()
            ]
            result.stderr.decode("ascii")
        except UnicodeDecodeError:
            return False, "DRAT_INVALID_CHECKER_OUTPUT"
        statuses = [line for line in lines if line.startswith("s ")]
        protocol_lines = all(
            line == "c" or line.startswith(("c ", "s ")) for line in lines
        )
        accepted = (
            result.returncode == 0
            and statuses == ["s VERIFIED"]
            and protocol_lines
            and not result.stderr
        )
        if accepted:
            return True, None
        deletion_warning = any(
            line.startswith("c WARNING: deleted clause") for line in lines
        )
        if deletion_warning:
            return False, "DRAT_DELETION_WARNING_REJECTED"
        return False, "DRAT_PROOF_REJECTED"


def check_unsat_proof(request: dict[str, Any]) -> dict[str, Any]:
    """Accept only exact text DRAT replayed by an authorized external checker."""

    try:
        if not isinstance(request, dict) or set(request) != {
            "request_version",
            "claim",
            "candidate",
            "scope",
            "certificate",
            "expected_bindings",
        }:
            return _reject_proof("malformed checker request")
        if request["request_version"] != "1" or request["scope"] is not None:
            return _reject_proof("unsupported checker request")
        claim = request["claim"]
        candidate = request["candidate"]
        certificate = request["certificate"]
        if not all(_valid_artifact(item) for item in (claim, candidate, certificate)):
            return _reject_proof("checker artifact metadata is malformed")
        expected_bindings = request["expected_bindings"]
        if not valid_unscoped_unencoded_bindings(expected_bindings):
            return _reject_proof("expected evidence bindings are malformed")
        if (
            claim["semantics_uri"] != candidate["semantics_uri"]
            or claim["semantics_uri"] != certificate["semantics_uri"]
            or claim["payload_digest"] != _sha256(_canonical_json(claim["payload"]))
            or candidate["payload_digest"]
            != _sha256(_canonical_json(candidate["payload"]))
            or certificate["payload_digest"]
            != _sha256(_canonical_json(certificate["payload"]))
        ):
            return _reject_proof("checker artifacts are not exactly bound")
        clauses, variable_count, variable_map_digest, dimacs_digest = _validate_cnf(
            claim["payload"]
        )
        raw_proof = _validate_proof(
            candidate["payload"],
            claim=claim,
            variable_count=variable_count,
            clause_count=len(clauses),
            variable_map_digest=variable_map_digest,
            dimacs_digest=dimacs_digest,
        )
        _validate_drat_text_profile(raw_proof)
        if claim["artifact_uri"] not in candidate["parents"] or not {
            claim["artifact_uri"],
            candidate["artifact_uri"],
        }.issubset(set(certificate["parents"])):
            return _reject_proof("SAT proof evidence is missing required lineage")
        if (
            expected_bindings["claim_digest"] != claim["object_digest"]
            or expected_bindings["candidate_digest"] != candidate["object_digest"]
        ):
            return _reject_proof("expected evidence bindings do not match artifacts")
        _validate_certificate(
            certificate,
            claim=claim,
            candidate=candidate,
            expected_bindings=expected_bindings,
        )
        executable, runtime_digest = _authorized_runtime()
        accepted, rejection_code = _bounded_drat_trim(
            executable,
            cnf=_dimacs_bytes(clauses, variable_count),
            proof=raw_proof,
            expected_runtime_digest=runtime_digest,
        )
        if not accepted:
            return _reject_proof(
                f"{rejection_code}: DRAT-trim did not accept the exact bound proof"
            )
        return {
            "accepted": True,
            "conclusion": "TRUE",
            "arithmetic": "EXACT_INTEGER",
            "method": "CHECKED_CERTIFICATE",
            "coverage": "NOT_APPLICABLE",
            "detail": (
                f"DRAT-trim accepted {len(raw_proof)} exact proof bytes against "
                f"{len(clauses)} canonical clauses"
            ),
        }
    except (KeyError, OSError, TypeError, ValueError, OverflowError):
        return _reject_proof("malformed or unauthorized SAT proof checker request")


def check_assignment(request: dict[str, Any]) -> dict[str, Any]:
    """Accept only a total assignment satisfying every exact bound clause."""

    try:
        if not isinstance(request, dict) or set(request) != {
            "request_version",
            "claim",
            "candidate",
            "scope",
            "witness",
            "expected_bindings",
        }:
            return _reject("malformed checker request")
        if request["request_version"] != "1" or request["scope"] is not None:
            return _reject("unsupported checker request")
        claim = request["claim"]
        candidate = request["candidate"]
        witness = request["witness"]
        if not all(_valid_artifact(item) for item in (claim, candidate, witness)):
            return _reject("checker artifact metadata is malformed")
        if not valid_unscoped_unencoded_bindings(request["expected_bindings"]):
            return _reject("expected evidence bindings are malformed")
        if (
            claim["semantics_uri"] != candidate["semantics_uri"]
            or claim["semantics_uri"] != witness["semantics_uri"]
        ):
            return _reject("checker artifacts use different semantics")
        if claim["payload_digest"] != _sha256(_canonical_json(claim["payload"])):
            return _reject("canonical CNF payload digest does not match")

        clauses, variable_count, variable_map_digest, dimacs_digest = _validate_cnf(
            claim["payload"]
        )
        values = _validate_assignment(
            candidate["payload"],
            claim=claim,
            variable_count=variable_count,
            clause_count=len(clauses),
            variable_map_digest=variable_map_digest,
            dimacs_digest=dimacs_digest,
        )
        if claim["artifact_uri"] not in candidate["parents"]:
            return _reject("SAT assignment is missing canonical CNF lineage")

        expected_bindings = request["expected_bindings"]
        if (
            expected_bindings["claim_digest"] != claim["object_digest"]
            or expected_bindings["candidate_digest"] != candidate["object_digest"]
        ):
            return _reject("expected evidence bindings do not match artifacts")
        envelope = witness["payload"]
        if not isinstance(envelope, dict) or set(envelope) != {
            "evidence_schema_version",
            "witness_format",
            "format_version",
            "role",
            "bindings",
            "payload",
        }:
            return _reject("SAT witness envelope is malformed")
        if (
            envelope["evidence_schema_version"] != "1"
            or envelope["witness_format"] != "sat.assignment"
            or envelope["format_version"] != "1"
            or envelope["role"] != "SUPPORTS_CLAIM"
            or envelope["bindings"] != expected_bindings
        ):
            return _reject("SAT witness envelope is not exactly bound")
        witness_payload = envelope["payload"]
        if not isinstance(witness_payload, dict) or witness_payload != {
            "cnf_uri": claim["artifact_uri"],
            "assignment_uri": candidate["artifact_uri"],
        }:
            return _reject("SAT witness points at different artifacts")
        if not {
            claim["artifact_uri"],
            candidate["artifact_uri"],
        }.issubset(set(witness["parents"])):
            return _reject("SAT witness is missing required lineage")

        for clause in clauses:
            if not any(
                values[abs(literal) - 1] if literal > 0 else not values[-literal - 1]
                for literal in clause
            ):
                return _reject("assignment does not satisfy every bound clause")
        return {
            "accepted": True,
            "conclusion": "TRUE",
            "arithmetic": "EXACT_INTEGER",
            "method": "DIRECT_WITNESS",
            "coverage": "NOT_APPLICABLE",
            "detail": (
                f"replayed {len(clauses)} clauses under a total "
                f"{variable_count}-variable assignment"
            ),
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return _reject("malformed checker request")
