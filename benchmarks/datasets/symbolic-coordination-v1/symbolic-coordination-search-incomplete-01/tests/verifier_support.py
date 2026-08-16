"""Fail-closed protocol helpers for one self-contained Harbor verifier."""

from __future__ import annotations

import hashlib
import json
import math
import stat
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from referencing.exceptions import Unresolvable

WORKSPACE = Path("/app")
TESTS = Path("/tests")
MAX_SUBMISSION_BYTES = 16 * 1024 * 1024
MAX_INPUT_BYTES = 16 * 1024 * 1024


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


_JSON_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"


def _derive_submission_schema(contract: dict[str, Any]) -> dict[str, Any] | None:
    """Derive the submission JSON Schema from contract declaration fields.

    The public contract no longer stores ``submission_schema`` as a copy.
    Instead, the schema is derived from ``submission_result`` (the result
    schema) and ``witness`` (the witness rule) exactly as the repository-time
    tooling does in ``benchmarks.tooling.public_contract._declared_schema``.
    """
    # Fall back to stored submission_schema for backwards compatibility
    stored = contract.get("submission_schema")
    if isinstance(stored, dict):
        return dict(stored)
    submission_result = contract.get("submission_result")
    if not isinstance(submission_result, dict):
        return None
    properties: dict[str, Any] = {"result": dict(submission_result)}
    required: list[str] = ["result"]
    witness = contract.get("witness") or {}
    if isinstance(witness, dict) and witness.get("max_items", 0):
        allowed_paths = witness.get("allowed_paths", [])
        digest_pattern = witness.get("digest_pattern", r"^sha256:[0-9a-f]{64}$")
        min_items = witness.get("min_items", 0)
        max_items = witness.get("max_items", 0)
        payload_shape = witness.get("payload_shape")
        item_properties: dict[str, Any] = {
            "path": (
                {"const": allowed_paths[0]}
                if len(allowed_paths) == 1
                else {"enum": list(allowed_paths)}
            ),
            "sha256": {"type": "string", "pattern": digest_pattern},
        }
        item_required = ["path", "sha256"]
        if isinstance(payload_shape, dict):
            for key, fragment in payload_shape.items():
                if key in ("path", "sha256"):
                    continue
                item_properties[key] = fragment
                item_required.append(key)
        item_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": item_required,
            "properties": item_properties,
        }
        properties["witness"] = {
            "type": "array",
            "minItems": min_items,
            "maxItems": max_items,
            "items": item_schema,
        }
        if min_items:
            required.append("witness")
    schema: dict[str, Any] = {
        "$schema": _JSON_SCHEMA_DRAFT,
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }
    schema_definitions = contract.get("schema_definitions") or {}
    if schema_definitions:
        schema["$defs"] = dict(schema_definitions)
    return schema


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
    schema = _derive_submission_schema(contract)
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
    """Parse and completely validate one bounded result/witness submission.

    The public protocol is a typed ``result`` plus an optional task-specific
    ``witness`` array. The parsed object is validated against the task's
    agent-visible JSON Schema and returned verbatim. No generic envelope,
    assurance, scope, completeness, or limitations fields are accepted.
    """

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


def load_submission_raw(
    path: Path = WORKSPACE / "submission.json",
    *,
    require_input_binding: bool = True,
) -> dict[str, Any] | None:
    """Parse one bounded submission without applying its public schema."""

    if require_input_binding and not workspace_input_is_bound():
        return None
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
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
    return value if isinstance(value, dict) else None


def submission_matches_public_schema(submission: object) -> bool:
    """Validate a parsed submission against the agent-visible JSON Schema."""

    return _public_submission_is_valid(submission)


def _public_submission_is_valid(submission: object) -> bool:
    contract = _load_public_contract()
    if contract is None:
        return False
    schema = _derive_submission_schema(contract)
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


def read_evidence_json(
    descriptor: object,
    *,
    expected_path: str,
    workspace: Path = WORKSPACE,
    max_bytes: int | None = None,
) -> dict[str, Any] | None:
    """Resolve and parse a digest-bound evidence object."""

    target = resolve_evidence(
        descriptor,
        expected_path=expected_path,
        workspace=workspace,
        max_bytes=max_bytes,
    )
    if target is None:
        return None
    try:
        value = json.loads(
            target.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, ValueError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, dict) else None


def json_value_equal(left: object, right: object) -> bool:
    """Compare JSON values recursively without Python's numeric coercion."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            json_value_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            json_value_equal(item, expected)
            for item, expected in zip(left, right, strict=True)
        )
    return left == right


def witness_list_is_bound(
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


def valid_sha256_uri(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
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
    protocol_ok: bool = True,
) -> float:
    """Binary fail-closed reward from the mathematical predicate and witness.

    Returns ``1.0`` only when the protocol gate, the replayed mathematical
    correctness diagnostic, and the witness-validity diagnostic are all fully
    satisfied; otherwise ``0.0``. There is no soft-assurance, scope, or
    completeness path: diagnostics never earn partial credit.
    """

    if not protocol_ok:
        return 0.0
    try:
        correctness_score = _as_unit_score(correctness)
    except ValueError:
        return 0.0
    if correctness_score < 1.0:
        return 0.0
    return 1.0


def normalize_reward_file(reward_path: Path) -> None:
    """Split a verifier's completed reward payload into scalar and details files."""

    def reject_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise RuntimeError(f"duplicate verifier reward key: {key}")
            value[key] = item
        return value

    def reject_constant(value):
        raise RuntimeError(f"non-finite verifier reward value: {value}")

    payload = json.loads(
        reward_path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict):
        raise RuntimeError("verifier reward payload must be a JSON object")
    if "reward" not in payload:
        raise RuntimeError("verifier reward payload is missing reward")
    reward = payload["reward"]
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or reward != reward
        or abs(reward) == float("inf")
        or not 0.0 <= reward <= 1.0
    ):
        raise RuntimeError("verifier reward must be a finite numeric scalar")
    details = {key: value for key, value in payload.items() if key != "reward"}
    (reward_path.parent / "reward-details.json").write_text(
        json.dumps(details, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    reward_path.write_text(
        json.dumps({"reward": reward}, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


__all__ = [
    "MAX_INPUT_BYTES",
    "MAX_SUBMISSION_BYTES",
    "TESTS",
    "WORKSPACE",
    "aggregate_reward",
    "is_regular_bounded_file",
    "json_value_equal",
    "load_submission",
    "load_submission_raw",
    "normalize_reward_file",
    "read_evidence_json",
    "resolve_evidence",
    "sha256_uri",
    "submission_matches_public_schema",
    "valid_sha256_uri",
    "witness_list_is_bound",
    "workspace_input_is_bound",
]
