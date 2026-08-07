"""Fail-closed protocol helpers for one self-contained Harbor verifier."""

from __future__ import annotations

import codecs
import hashlib
import json
import math
import stat
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from referencing.exceptions import Unresolvable

WORKSPACE = Path("/app")
TESTS = Path("/tests")
MAX_SUBMISSION_BYTES = 16 * 1024 * 1024
MAX_INPUT_BYTES = 16 * 1024 * 1024
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


def is_regular_bounded_file(path: Path, *, max_bytes: int | None) -> bool:
    """Reject symlinks, non-regular files, and oversized files before reading."""

    try:
        status = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        return False
    return max_bytes is None or status.st_size <= max_bytes


def sha256_uri(path: Path) -> str:
    """Hash a regular evidence file without following a replacement symlink."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65_536), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


MAX_PUBLIC_CONTRACT_BYTES = 4 * 1024 * 1024


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"out-of-range JSON number: {value}")
    return parsed


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject JSON objects with duplicate names at any nesting level."""

    seen: set[str] = set()
    for key, _ in pairs:
        if key in seen:
            raise ValueError(f"duplicate JSON object key: {key}")
        seen.add(key)
    return dict(pairs)


def _load_public_contract(
    path: Path = TESTS / "public_contract.json",
) -> dict[str, Any] | None:
    if not is_regular_bounded_file(path, max_bytes=MAX_PUBLIC_CONTRACT_BYTES):
        return None
    try:
        contract = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
            parse_float=_finite_json_float,
        )
    except (OSError, ValueError, RecursionError, MemoryError):
        return None
    if not isinstance(contract, dict) or contract.get("schema_version") != "1":
        return None
    schema = contract.get("submission_schema")
    if not isinstance(schema, dict):
        return None
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError:
        return None
    return contract


def load_submission(
    path: Path = WORKSPACE / "submission.json",
    *,
    require_input_binding: bool = True,
) -> dict[str, Any] | None:
    """Parse and completely validate one bounded submission object."""

    if require_input_binding and not workspace_input_is_bound():
        return None
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    contract = _load_public_contract()
    if contract is None:
        return None
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
            parse_float=_finite_json_float,
        )
    except (OSError, ValueError, RecursionError, MemoryError, TypeError):
        return None
    return (
        value
        if isinstance(value, dict) and _public_submission_is_valid(value)
        else None
    )


def _public_submission_is_valid(submission: object) -> bool:
    contract = _load_public_contract()
    if contract is None:
        return False
    schema = contract["submission_schema"]
    try:
        return Draft202012Validator(schema).is_valid(submission)
    except (
        SchemaError,
        Unresolvable,
        ValueError,
        RecursionError,
        MemoryError,
        TypeError,
    ):
        return False


def workspace_input_is_bound(
    visible_path: Path = WORKSPACE / "input.json",
    *,
    tests: Path = TESTS,
) -> bool:
    """Require the agent-visible input to equal the sole frozen verifier input."""

    try:
        candidates = tuple(tests.glob("*input*.json"))
    except OSError:
        return False
    if len(candidates) != 1:
        return False
    frozen_path = candidates[0]
    if not all(
        is_regular_bounded_file(candidate, max_bytes=MAX_INPUT_BYTES)
        for candidate in (frozen_path, visible_path)
    ):
        return False
    try:
        return sha256_uri(frozen_path) == sha256_uri(visible_path)
    except OSError:
        return False


def strict_submission_contract(
    submission: object,
    *,
    task_id: str,
    conclusion: str,
    completeness: str = "COMPLETE",
    evidence_count: int = 1,
    min_limitations: int = 0,
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
    limitations = submission.get("limitations", [])
    return bool(
        _public_submission_is_valid(submission)
        and frozenset(submission) in expected_fields
        and submission.get("task_id") == task_id
        and submission.get("conclusion") == conclusion
        and submission.get("completeness") == completeness
        and isinstance(submission.get("result"), dict)
        and isinstance(submission.get("scope"), str)
        and isinstance(limitations, list)
        and len(limitations) >= min_limitations
        and all(type(item) is str for item in limitations)
        and isinstance(submission.get("evidence"), list)
        and len(submission.get("evidence", [])) == evidence_count
        and isinstance(submission.get("claimed_assurance"), str)
        and submission.get("claimed_assurance") in allowed_assurances
    )


def resolve_evidence(
    descriptor: object,
    *,
    expected_path: str,
    workspace: Path = WORKSPACE,
    max_bytes: int | None = None,
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
    if not target.is_relative_to(root) or not is_regular_bounded_file(
        target, max_bytes=max_bytes
    ):
        return None
    try:
        if descriptor["sha256"] != sha256_uri(target):
            return None
    except OSError:
        return None
    return target


_JSON_WHITESPACE = frozenset(" \t\n\r")
_JSON_WHITESPACE_CHARS = " \t\n\r"


def _drain_stream_tail(stream, decoder) -> None:
    """Reject any non-whitespace content after the parsed JSON value."""

    while True:
        block = stream.read(65_536)
        if not block:
            break
        tail = decoder.decode(block)
        if tail and not all(character in _JSON_WHITESPACE for character in tail):
            raise ValueError("non-whitespace after evidence JSON value")
    tail = decoder.decode(b"", final=True)
    if tail and not all(character in _JSON_WHITESPACE for character in tail):
        raise ValueError("non-whitespace after evidence JSON value")


def _read_streaming_json_value(stream) -> Any:
    """Parse the first JSON value from a binary stream without a byte cap.

    ``json.JSONDecoder.raw_decode`` is fed incrementally in bounded chunks,
    so parsing stops as soon as the top-level value is complete; leading and
    trailing JSON whitespace are handled exactly like ``json.load``, and any
    non-whitespace content after the value is rejected chunk by chunk instead
    of being read into memory. Memory therefore grows only with the size of
    the JSON value itself, never with arbitrary legal whitespace padding: the
    consumed leading-whitespace prefix is discarded as it is skipped.
    """

    decoder = codecs.getincrementaldecoder("utf-8")()
    parser = json.JSONDecoder(object_pairs_hook=_reject_duplicate_keys)
    buffer = ""
    while True:
        block = stream.read(65_536)
        if block:
            buffer += decoder.decode(block)
        # ``raw_decode`` does not skip leading whitespace; discard the
        # consumed leading-whitespace prefix as it arrives so a large legal
        # whitespace prefix before the value is never retained in (or
        # repeatedly copied through) ``buffer``. The guard keeps the JSON
        # value itself untouched once its first non-whitespace byte arrives.
        if buffer[:1] in _JSON_WHITESPACE:
            buffer = buffer.lstrip(_JSON_WHITESPACE_CHARS)
        try:
            value, end = parser.raw_decode(buffer)
        except json.JSONDecodeError:
            if not block:
                raise
            continue
        if not all(character in _JSON_WHITESPACE for character in buffer[end:]):
            raise ValueError("non-whitespace after evidence JSON value")
        _drain_stream_tail(stream, decoder)
        return value


def read_evidence_json(
    descriptor: object,
    *,
    expected_path: str,
    workspace: Path = WORKSPACE,
    max_bytes: int | None = None,
) -> dict[str, Any] | None:
    """Resolve and parse a digest-bound evidence object.

    The file is parsed with a genuinely streaming decoder instead of
    ``read_text()``, so a digest-correct evidence file with arbitrary legal
    JSON whitespace is never materialized in full and no internal byte
    ceiling is imposed: the evidence is already bound by path, digest, and
    schema. The ``max_bytes`` argument only bounds digest resolution.
    """

    target = resolve_evidence(
        descriptor,
        expected_path=expected_path,
        workspace=workspace,
        max_bytes=max_bytes,
    )
    if target is None:
        return None
    try:
        with target.open("rb") as stream:
            value = _read_streaming_json_value(stream)
    except (OSError, ValueError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, dict) else None


def evidence_list_is_bound(
    evidence: object,
    *,
    expected_path: str = "evidence/answer.txt",
    expected_count: int = 1,
    max_bytes: int | None = None,
) -> bool:
    """Require an exact-size list binding the expected evidence file."""

    return bool(
        isinstance(evidence, list)
        and len(evidence) == expected_count
        and all(
            resolve_evidence(item, expected_path=expected_path, max_bytes=max_bytes)
            is not None
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
        authorized = json.loads(
            authorized_path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
        )
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


def _as_unit_score(value: float | bool | int) -> float:
    """Normalize a diagnostic to a unit interval score."""

    if isinstance(value, bool):
        return 1.0 if value else 0.0
    score = float(value)
    if score < 0.0 or score > 1.0 or score != score:  # NaN check
        raise ValueError(f"diagnostic score out of unit interval: {value!r}")
    return score


def aggregate_reward(
    *,
    correctness: float | bool,
    evidence_validity: float | bool,
    scope_accuracy: float | bool = 1.0,
    assurance_calibration: float | bool = 1.0,
    false_certification: bool = False,
    protocol_ok: bool = True,
    require_scope: bool = True,
    soft_assurance: bool = False,
    soft_assurance_base: float = 0.9,
    soft_assurance_weight: float = 0.1,
) -> float:
    """Compute fail-closed aggregate reward from independent diagnostics.

    Mandatory hard gates (protocol, correctness, evidence, and required scope)
    always force ``0.0`` when they fail, as does ``false_certification``.
    When ``soft_assurance`` is false, assurance is also a hard gate and the
    successful aggregate is ``1.0``. When ``soft_assurance`` is true and every
    hard gate passes, the aggregate is
    ``soft_assurance_base + soft_assurance_weight * assurance`` so under-
    claimed assurance can receive documented partial credit without diluting
    evidence or protocol failures.
    """

    if false_certification or not protocol_ok:
        return 0.0
    correctness_score = _as_unit_score(correctness)
    evidence_score = _as_unit_score(evidence_validity)
    scope_score = _as_unit_score(scope_accuracy)
    assurance_score = _as_unit_score(assurance_calibration)
    if correctness_score < 1.0 or evidence_score < 1.0:
        return 0.0
    if require_scope and scope_score < 1.0:
        return 0.0
    if not soft_assurance:
        return 1.0 if assurance_score >= 1.0 else 0.0
    if soft_assurance_base < 0.0 or soft_assurance_weight < 0.0:
        raise ValueError("soft assurance weights must be non-negative")
    if soft_assurance_base + soft_assurance_weight > 1.0 + 1e-12:
        raise ValueError("soft assurance weights must not exceed 1.0 in total")
    return soft_assurance_base + soft_assurance_weight * assurance_score


__all__ = [
    "ASSURANCE_LEVELS",
    "MAX_INPUT_BYTES",
    "MAX_SUBMISSION_BYTES",
    "SUBMISSION_FIELDS",
    "TESTS",
    "WORKSPACE",
    "aggregate_reward",
    "authorized_record_is_bound",
    "evidence_list_is_bound",
    "false_verified_claim",
    "is_regular_bounded_file",
    "load_submission",
    "read_evidence_json",
    "resolve_evidence",
    "sha256_uri",
    "strict_submission_contract",
    "valid_sha256_uri",
    "workspace_input_is_bound",
]
