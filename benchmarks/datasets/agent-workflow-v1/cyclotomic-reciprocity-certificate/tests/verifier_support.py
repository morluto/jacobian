"""Canonical fail-closed protocol helpers vendored into Harbor verifiers.

This module uses only the Python standard library. Each task receives an
identical copy in its hidden ``tests`` directory so the verifier remains
self-contained and independent from production Jacobian code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

WORKSPACE = Path("/app")
TESTS = Path("/tests")
SUBMISSION_FIELDS = frozenset(
    {
        "task_id",
        "conclusion",
        "result",
        "claimed_assurance",
        "scope",
        "completeness",
        "evidence",
        "limitations",
    }
)
ASSURANCE_LEVELS = frozenset({"UNVERIFIED", "COMPUTED", "CHECKED", "VERIFIED"})


def sha256_uri(path: Path) -> str:
    """Hash a regular evidence file without following a replacement symlink."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65_536), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def load_submission(
    path: Path = WORKSPACE / "submission.json",
) -> dict[str, Any] | None:
    """Parse a submission as one JSON object, rejecting malformed input."""

    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def strict_submission_contract(
    submission: object,
    *,
    task_id: str,
    conclusion: str,
    completeness: str = "COMPLETE",
    allowed_assurances: frozenset[str] = ASSURANCE_LEVELS,
    verification_record: Literal[
        "required_when_verified", "optional", "forbidden"
    ] = "required_when_verified",
) -> bool:
    """Validate the shared submission envelope without interpreting mathematics."""

    if not isinstance(submission, dict):
        return False
    verified = submission.get("claimed_assurance") == "VERIFIED"
    expected_fields = {frozenset(SUBMISSION_FIELDS)}
    if verification_record == "required_when_verified" and verified:
        expected_fields = {frozenset(SUBMISSION_FIELDS | {"verification_record_uri"})}
    elif verification_record == "optional":
        expected_fields.add(frozenset(SUBMISSION_FIELDS | {"verification_record_uri"}))
    return bool(
        frozenset(submission) in expected_fields
        and submission.get("task_id") == task_id
        and submission.get("conclusion") == conclusion
        and submission.get("completeness") == completeness
        and isinstance(submission.get("result"), dict)
        and isinstance(submission.get("scope"), str)
        and isinstance(submission.get("limitations"), list)
        and all(type(item) is str for item in submission.get("limitations", []))
        and isinstance(submission.get("evidence"), list)
        and isinstance(submission.get("claimed_assurance"), str)
        and submission.get("claimed_assurance") in allowed_assurances
    )


def resolve_evidence(
    descriptor: object,
    *,
    expected_path: str,
    workspace: Path = WORKSPACE,
) -> Path | None:
    """Resolve one digest-bound evidence file without escapes or symlinks."""

    if (
        not isinstance(descriptor, dict)
        or set(descriptor) != {"path", "sha256"}
        or descriptor.get("path") != expected_path
        or not isinstance(descriptor.get("sha256"), str)
    ):
        return None
    relative = Path(expected_path)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    root = workspace.resolve()
    unresolved = workspace / relative
    current = workspace
    try:
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                return None
        target = unresolved.resolve(strict=True)
    except OSError:
        return None
    if not target.is_relative_to(root) or not target.is_file():
        return None
    try:
        if descriptor["sha256"] != sha256_uri(target):
            return None
    except OSError:
        return None
    return target


def read_evidence_json(
    descriptor: object,
    *,
    expected_path: str,
    workspace: Path = WORKSPACE,
) -> dict[str, Any] | None:
    """Resolve and parse a digest-bound evidence object."""

    target = resolve_evidence(
        descriptor,
        expected_path=expected_path,
        workspace=workspace,
    )
    if target is None:
        return None
    try:
        value = json.loads(target.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def evidence_list_is_bound(
    evidence: object,
    *,
    expected_path: str = "evidence/answer.txt",
) -> bool:
    """Require a nonempty list whose every item binds the expected evidence file."""

    return bool(
        isinstance(evidence, list)
        and evidence
        and all(
            resolve_evidence(item, expected_path=expected_path) is not None
            for item in evidence
        )
    )


def authorized_record_is_bound(
    descriptor: object,
    *,
    authorized_path: Path,
    dynamic_fields: frozenset[str] = frozenset({"environment_digest"}),
) -> bool:
    """Bind an exact hidden authorization record while allowing named identities."""

    actual = read_evidence_json(
        descriptor,
        expected_path="evidence/verification-record.json",
    )
    try:
        authorized = json.loads(authorized_path.read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(actual, dict) or not isinstance(authorized, dict):
        return False
    if set(actual) != set(authorized):
        return False
    for key, value in authorized.items():
        if key not in dynamic_fields and actual.get(key) != value:
            return False
    for key in dynamic_fields:
        value = actual.get(key)
        if not valid_sha256_uri(value):
            return False
    return True


def valid_sha256_uri(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def false_verified_claim(
    submission: object,
    *,
    verification_record_bound: bool,
) -> bool:
    """Fail closed when VERIFIED is asserted without exact authorized binding."""

    return bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") == "VERIFIED"
        and not verification_record_bound
    )


__all__ = [
    "ASSURANCE_LEVELS",
    "SUBMISSION_FIELDS",
    "TESTS",
    "WORKSPACE",
    "authorized_record_is_bound",
    "evidence_list_is_bound",
    "false_verified_claim",
    "load_submission",
    "read_evidence_json",
    "resolve_evidence",
    "sha256_uri",
    "strict_submission_contract",
    "valid_sha256_uri",
]
