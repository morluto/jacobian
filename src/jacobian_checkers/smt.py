"""Independent strict replay for one pinned cvc5 Alethe profile.

This module intentionally uses only the Python standard library. It does not
import Jacobian's SMT contracts, the cvc5 producer, or any solver API.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import tempfile
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
_ARTIFACT_KEYS = {
    "artifact_uri",
    "object_digest",
    "payload_digest",
    "schema_uri",
    "semantics_uri",
    "parents",
    "payload",
}
_ALLOWED_COMMANDS = {
    "set-logic",
    "declare-sort",
    "declare-fun",
    "declare-const",
    "assert",
    "check-sat",
}
_PROFILE = "jacobian.smtlib2.qf-unsat/v1"
_PROOF_FORMAT_VERSION = "cvc5.alethe/1.3.4"
_ALETHE_HOLE_MARKER = b":rule hole"
CARCARA_OUTPUT_LIMIT = 1024 * 1024
CARCARA_TIMEOUT_SECONDS = 90


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "SYMBOLIC",
        "method": "CHECKED_CERTIFICATE",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


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


def _smtlib_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    index = 0
    while index < len(text):
        character = text[index]
        if character.isspace():
            index += 1
            continue
        if character == ";":
            newline = text.find("\n", index)
            index = len(text) if newline < 0 else newline + 1
            continue
        if character in "()":
            tokens.append(character)
            index += 1
            continue
        if character == '"':
            start = index
            index += 1
            while index < len(text):
                if text[index] != '"':
                    index += 1
                    continue
                if index + 1 < len(text) and text[index + 1] == '"':
                    index += 2
                    continue
                index += 1
                tokens.append(text[start:index])
                break
            else:
                raise ValueError("unterminated SMT-LIB string")
            continue
        if character == "|":
            start = index
            index += 1
            while index < len(text) and text[index] != "|":
                if text[index] == "\\":
                    raise ValueError("unsupported quoted-symbol escape")
                index += 1
            if index >= len(text):
                raise ValueError("unterminated SMT-LIB quoted symbol")
            index += 1
            tokens.append(text[start:index])
            continue
        start = index
        while (
            index < len(text) and not text[index].isspace() and text[index] not in "();"
        ):
            if text[index] in '"|':
                raise ValueError("unexpected SMT-LIB quote")
            index += 1
        if start == index:
            raise ValueError("invalid SMT-LIB token")
        tokens.append(text[start:index])
    return tuple(tokens)


def _top_level_commands(text: str) -> tuple[tuple[str, ...], ...]:
    commands: list[tuple[str, ...]] = []
    direct_atoms: list[str] = []
    depth = 0
    for token in _smtlib_tokens(text):
        if token == "(":
            if depth == 0:
                direct_atoms = []
            depth += 1
            if depth > 512:
                raise ValueError("SMT-LIB nesting limit exceeded")
            continue
        if token == ")":
            if depth == 0:
                raise ValueError("unmatched SMT-LIB closing parenthesis")
            if depth == 1:
                if not direct_atoms:
                    raise ValueError("empty SMT-LIB command")
                commands.append(tuple(direct_atoms))
            depth -= 1
            continue
        if depth == 0:
            raise ValueError("SMT-LIB atom outside a command")
        if depth == 1:
            direct_atoms.append(token)
    if depth:
        raise ValueError("unmatched SMT-LIB opening parenthesis")
    return tuple(commands)


def _validate_problem(payload: object) -> tuple[bytes, str]:
    if not isinstance(payload, dict) or set(payload) != {
        "problem_schema_version",
        "profile",
        "input_language",
        "logic",
        "query_scope",
        "smtlib_text",
        "smtlib_digest",
    }:
        raise ValueError("SMT problem has an invalid shape")
    text = payload["smtlib_text"]
    if (
        payload["problem_schema_version"] != "1"
        or payload["profile"] != _PROFILE
        or payload["input_language"] != "SMT-LIB-2.6"
        or payload["logic"] != "QF_UF"
        or payload["query_scope"] != "SINGLE_CHECK_SAT"
        or not isinstance(text, str)
        or not 1 <= len(text) <= 1_000_000
        or not isinstance(payload["smtlib_digest"], str)
        or _DIGEST.fullmatch(payload["smtlib_digest"]) is None
    ):
        raise ValueError("SMT problem metadata is outside the checker profile")
    try:
        raw = text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("SMT-LIB input is not ASCII") from exc
    if (
        not text.endswith("\n")
        or "\r" in text
        or any(byte < 32 and byte not in {9, 10} for byte in raw)
        or _sha256(raw) != payload["smtlib_digest"]
    ):
        raise ValueError("SMT-LIB bytes do not match the exact profile")
    commands = _top_level_commands(text)
    if not commands:
        raise ValueError("SMT-LIB query is empty")
    heads = tuple(command[0] for command in commands)
    if (
        commands[0] != ("set-logic", "QF_UF")
        or heads.count("set-logic") != 1
        or commands[-1] != ("check-sat",)
        or heads.count("check-sat") != 1
        or any(head not in _ALLOWED_COMMANDS for head in heads)
    ):
        raise ValueError("SMT-LIB command sequence is outside the checker profile")
    return raw, payload["smtlib_digest"]


def _valid_producer(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    expected_keys = {
        "runtime_version",
        "provider",
        "availability",
        "version",
        "digest",
        "digest_kind",
        "platform",
        "install_tier",
        "license_id",
        "license_files",
        "features",
        "checker_ids",
        "configuration",
        "distribution_import_name",
        "distribution_required_attributes",
        "diagnostic",
    }
    configuration = value.get("configuration")
    features = value.get("features")
    return (
        set(value) == expected_keys
        and value["runtime_version"] == "1"
        and value["provider"] == "cvc5"
        and value["availability"] == "AVAILABLE"
        and value["version"] == "1.3.4"
        and isinstance(value["digest"], str)
        and _DIGEST.fullmatch(value["digest"]) is not None
        and value["digest_kind"] == "PYTHON_DISTRIBUTION_RECORD"
        and isinstance(value["platform"], str)
        and value["install_tier"] == "T1"
        and value["license_id"] == "BSD-3-Clause"
        and isinstance(value["license_files"], list)
        and isinstance(features, list)
        and "alethe-proof-production" in features
        and value["checker_ids"] == []
        and isinstance(configuration, dict)
        and set(configuration)
        in (
            {"profile", "proof_format"},
            {"distribution", "profile", "proof_format"},
        )
        and configuration.get("distribution", "cvc5") == "cvc5"
        and configuration.get("profile") == _PROFILE
        and configuration.get("proof_format") == _PROOF_FORMAT_VERSION
        and value["diagnostic"] is None
    )


def _validate_proof(
    payload: object,
    *,
    claim: dict[str, Any],
    smtlib_digest: str,
) -> bytes:
    if not isinstance(payload, dict) or set(payload) != {
        "proof_schema_version",
        "problem",
        "declared_scope",
        "proof_format",
        "proof_format_version",
        "proof_encoding",
        "proof_base64",
        "proof_digest",
        "alethe_hole_count",
        "contains_holes",
        "producer",
        "resource_budget",
    }:
        raise ValueError("SMT proof has an invalid shape")
    encoded = payload["proof_base64"]
    if (
        payload["proof_schema_version"] != "1"
        or payload["declared_scope"] != "FULL_QUERY"
        or payload["proof_format"] != "ALETHE"
        or payload["proof_format_version"] != _PROOF_FORMAT_VERSION
        or payload["proof_encoding"] != "BASE64"
        or not isinstance(encoded, str)
        or len(encoded) > 8_000_000
        or not isinstance(payload["proof_digest"], str)
        or _DIGEST.fullmatch(payload["proof_digest"]) is None
        or not _valid_producer(payload["producer"])
        or not isinstance(payload["resource_budget"], dict)
        or set(payload["resource_budget"]) != {"budget_version", "wall_seconds"}
        or payload["resource_budget"]["budget_version"] != "1"
        or type(payload["resource_budget"]["wall_seconds"]) is not int
        or not 1 <= payload["resource_budget"]["wall_seconds"] <= 300
    ):
        raise ValueError("SMT proof metadata is malformed")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise ValueError("SMT proof base64 is malformed") from exc
    hole_count = raw.count(_ALETHE_HOLE_MARKER)
    if (
        base64.b64encode(raw).decode("ascii") != encoded
        or _sha256(raw) != payload["proof_digest"]
        or type(payload["alethe_hole_count"]) is not int
        or payload["alethe_hole_count"] != hole_count
        or type(payload["contains_holes"]) is not bool
        or payload["contains_holes"] != (hole_count > 0)
        or hole_count != 0
    ):
        raise ValueError("SMT proof bytes are malformed or contain holes")
    problem = claim["payload"]
    expected_binding = {
        "binding_version": "1",
        "problem_artifact_uri": claim["artifact_uri"],
        "problem_object_digest": claim["object_digest"],
        "problem_payload_digest": claim["payload_digest"],
        "logic": "QF_UF",
        "profile": _PROFILE,
        "input_language": "SMT-LIB-2.6",
        "smtlib_digest": smtlib_digest,
    }
    if payload["problem"] != expected_binding or problem["logic"] != "QF_UF":
        raise ValueError("SMT proof is not bound to the exact QF_UF problem")
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
        raise ValueError("SMT proof certificate has an invalid shape")
    payload = {
        "problem_uri": claim["artifact_uri"],
        "proof_uri": candidate["artifact_uri"],
    }
    if (
        envelope["evidence_schema_version"] != "1"
        or envelope["certificate_type"] != "smt.unsat-proof"
        or envelope["format_version"] != "1"
        or envelope["bindings"] != expected_bindings
        or envelope["payload"] != payload
        or envelope["payload_digest"] != _sha256(_canonical_json(payload))
    ):
        raise ValueError("SMT proof certificate is not exactly bound")


def _authorized_runtime() -> tuple[Path, str]:
    executable = os.environ.get("JACOBIAN_CHECKER_EXECUTABLE")
    expected_digest = os.environ.get("JACOBIAN_CHECKER_RUNTIME_DIGEST")
    if (
        executable is None
        or expected_digest is None
        or _DIGEST.fullmatch(expected_digest) is None
    ):
        raise ValueError("Carcara runtime is not operator authorized")
    path = Path(executable).resolve(strict=True)
    if str(path) != executable or not path.is_file() or path.is_symlink():
        raise ValueError("Carcara runtime path is not exact")
    if _sha256(path.read_bytes()) != expected_digest:
        raise ValueError("Carcara runtime digest changed")
    return path, expected_digest


def _bounded_carcara(
    executable: Path,
    *,
    problem: bytes,
    proof: bytes,
    expected_runtime_digest: str,
) -> bool:
    with tempfile.TemporaryDirectory(prefix="jacobian-carcara-") as directory:
        root = Path(directory)
        proof_path = root / "proof.alethe"
        problem_path = root / "problem.smt2"
        proof_path.write_bytes(proof)
        problem_path.write_bytes(problem)
        command = [
            str(executable),
            "check",
            "--strict-parsing",
            "--parse-hole-args",
            "--allow-int-real-subtyping",
            "--expand-let-bindings",
            str(proof_path),
            str(problem_path),
        ]
        result = execute_process(
            ProcessRequest(
                executable=command[0],
                arguments=tuple(command[1:]),
                environment=worker_environment(locale="C"),
                cwd=str(root),
                timeout_seconds=CARCARA_TIMEOUT_SECONDS,
                stdin_bytes=b"",
                stdout_limit_bytes=CARCARA_OUTPUT_LIMIT,
                stderr_limit_bytes=CARCARA_OUTPUT_LIMIT,
            )
        )
        if result.termination is not ProcessTermination.EXITED:
            return False
        if _sha256(executable.read_bytes()) != expected_runtime_digest:
            raise ValueError("Carcara runtime changed during replay")
        return (
            result.returncode == 0 and result.stdout == b"valid\n" and not result.stderr
        )


def check_unsat_proof(request: dict[str, Any]) -> dict[str, Any]:
    """Accept only an exact zero-hole QF_UF proof replayed by strict Carcara."""

    try:
        if not isinstance(request, dict) or set(request) != {
            "request_version",
            "claim",
            "candidate",
            "scope",
            "certificate",
            "expected_bindings",
        }:
            return _reject("malformed checker request")
        if request["request_version"] != "1" or request["scope"] is not None:
            return _reject("unsupported checker request")
        claim = request["claim"]
        candidate = request["candidate"]
        certificate = request["certificate"]
        if not all(_valid_artifact(item) for item in (claim, candidate, certificate)):
            return _reject("checker artifact metadata is malformed")
        expected_bindings = request["expected_bindings"]
        if not valid_unscoped_unencoded_bindings(expected_bindings):
            return _reject("expected evidence bindings are malformed")
        if (
            claim["semantics_uri"] != candidate["semantics_uri"]
            or claim["semantics_uri"] != certificate["semantics_uri"]
            or claim["payload_digest"] != _sha256(_canonical_json(claim["payload"]))
            or candidate["payload_digest"]
            != _sha256(_canonical_json(candidate["payload"]))
            or certificate["payload_digest"]
            != _sha256(_canonical_json(certificate["payload"]))
        ):
            return _reject("checker artifacts are not exactly bound")
        raw_problem, smtlib_digest = _validate_problem(claim["payload"])
        raw_proof = _validate_proof(
            candidate["payload"],
            claim=claim,
            smtlib_digest=smtlib_digest,
        )
        if claim["artifact_uri"] not in candidate["parents"] or not {
            claim["artifact_uri"],
            candidate["artifact_uri"],
        }.issubset(set(certificate["parents"])):
            return _reject("SMT proof evidence is missing required lineage")
        if (
            expected_bindings["claim_digest"] != claim["object_digest"]
            or expected_bindings["candidate_digest"] != candidate["object_digest"]
        ):
            return _reject("expected evidence bindings do not match artifacts")
        _validate_certificate(
            certificate,
            claim=claim,
            candidate=candidate,
            expected_bindings=expected_bindings,
        )
        executable, runtime_digest = _authorized_runtime()
        if not _bounded_carcara(
            executable,
            problem=raw_problem,
            proof=raw_proof,
            expected_runtime_digest=runtime_digest,
        ):
            return _reject("strict Carcara did not accept the exact bound proof")
        return {
            "accepted": True,
            "conclusion": "TRUE",
            "arithmetic": "SYMBOLIC",
            "method": "CHECKED_CERTIFICATE",
            "coverage": "NOT_APPLICABLE",
            "detail": (
                f"strict Carcara accepted {len(raw_proof)} exact Alethe bytes "
                "against the full bound QF_UF query"
            ),
        }
    except (KeyError, OSError, TypeError, ValueError, OverflowError):
        return _reject("malformed or unauthorized SMT proof checker request")
