"""Clean-room verifier for the perfect-cuboid finite scope audit."""

from __future__ import annotations

import json
from collections import Counter
from math import isqrt
from pathlib import Path
from typing import Any

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    evidence_list_is_bound,
    is_regular_bounded_file,
    load_submission,
    read_evidence_json,
    strict_submission_contract,
    workspace_input_is_bound,
)

TASK_ID = "jacobian/perfect-cuboid-scope-audit"
SCOPE = "perfect-cuboid-scope-audit:case-set-v1"
LIMITATIONS = ["TWELVE_FROZEN_INTEGER_CUBOIDS", "NO_GLOBAL_PERFECT_CUBOID_CONCLUSION"]
CLASSES = {"PERFECT_CUBOID", "EULER_BRICK_ONLY", "SPACE_AND_TWO_FACES", "OTHER"}
MAX_EVIDENCE_BYTES = 2 * 1024 * 1024
scoreable_assurances = frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"})


def _raw_submission() -> dict[str, Any] | None:
    """Parse the raw submission JSON without strict schema validation."""
    if not is_regular_bounded_file(
        Path("/app/submission.json"), max_bytes=MAX_SUBMISSION_BYTES
    ):
        return None
    try:
        value = json.loads(Path("/app/submission.json").read_text())
    except (OSError, ValueError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, dict) else None


def _root(value: int) -> int | None:
    root = isqrt(value)
    return root if root * root == value else None


def _expected(case: dict[str, Any]) -> dict[str, Any]:
    edges = case["edges"]
    face_radicands = [
        edges[0] * edges[0] + edges[1] * edges[1],
        edges[0] * edges[0] + edges[2] * edges[2],
        edges[1] * edges[1] + edges[2] * edges[2],
    ]
    face_roots = [_root(value) for value in face_radicands]
    space_radicand = sum(value * value for value in edges)
    space_root = _root(space_radicand)
    face_count = sum(value is not None for value in face_roots)
    if face_count == 3 and space_root is not None:
        classification = "PERFECT_CUBOID"
    elif face_count == 3:
        classification = "EULER_BRICK_ONLY"
    elif face_count == 2 and space_root is not None:
        classification = "SPACE_AND_TWO_FACES"
    else:
        classification = "OTHER"
    return {
        "id": case["id"],
        "edges": edges,
        "face_radicands": face_radicands,
        "face_roots": face_roots,
        "space_radicand": space_radicand,
        "space_root": space_root,
        "class": classification,
    }


def _frozen() -> dict[str, Any] | None:
    try:
        value = json.loads(Path("/tests/input.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("task_id") != TASK_ID:
        return None
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != 12:
        return None
    return value


def _mathematics(result: Any, frozen: dict[str, Any]) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "cases",
        "case_counts",
        "contains_perfect_cuboid",
        "scope_identity",
    }:
        return False
    rows = result.get("cases")
    if not isinstance(rows, list) or len(rows) != 12:
        return False
    expected = {case["id"]: _expected(case) for case in frozen["cases"]}
    submitted: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("id") in submitted:
            return False
        submitted[row.get("id")] = row
    if set(submitted) != set(expected):
        return False
    if any(not _row_matches(submitted[key], expected[key]) for key in expected):
        return False
    counts = Counter(row["class"] for row in expected.values())
    if set(result["case_counts"]) != CLASSES:
        return False
    if result["case_counts"] != {name: counts[name] for name in CLASSES}:
        return False
    return (
        result["contains_perfect_cuboid"] is (counts["PERFECT_CUBOID"] > 0)
        and result["scope_identity"] == SCOPE
    )


def _row_matches(submitted: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Compare a case while treating aligned face-diagonal pairs as unordered."""

    if not isinstance(submitted, dict) or set(submitted) != set(expected):
        return False
    fixed = set(expected) - {"face_radicands", "face_roots"}
    if any(submitted.get(key) != expected[key] for key in fixed):
        return False
    radicands = submitted.get("face_radicands")
    roots = submitted.get("face_roots")
    if not isinstance(radicands, list) or not isinstance(roots, list):
        return False
    if len(radicands) != 3 or len(roots) != 3:
        return False
    submitted_pairs = sorted(
        zip(radicands, roots, strict=True), key=lambda pair: pair[0]
    )
    expected_pairs = sorted(
        zip(expected["face_radicands"], expected["face_roots"], strict=True),
        key=lambda pair: pair[0],
    )
    return submitted_pairs == expected_pairs


def _reward(value: dict[str, Any]) -> None:
    path = Path("/logs/verifier")
    path.mkdir(parents=True, exist_ok=True)
    (path / "reward.json").write_text(json.dumps(value, sort_keys=True))


def main() -> None:
    input_bound = workspace_input_is_bound()
    frozen = _frozen()
    submission = load_submission(require_input_binding=False)
    raw_submission = _raw_submission()
    contract = strict_submission_contract(
        submission,
        task_id=TASK_ID,
        conclusion="PERFECT_CUBOID_FINITE_SCOPE_AUDIT",
        allowed_assurances=scoreable_assurances,
        verification_record="forbidden",
    )
    mathematics = bool(
        frozen
        and isinstance(submission, dict)
        and _mathematics(submission.get("result"), frozen)
    )
    evidence = bool(
        isinstance(submission, dict)
        and evidence_list_is_bound(
            submission.get("evidence", []), max_bytes=MAX_EVIDENCE_BYTES
        )
    )
    payload = (
        read_evidence_json(
            submission["evidence"][0],
            expected_path="evidence/answer.txt",
            max_bytes=MAX_EVIDENCE_BYTES,
        )
        if evidence
        else None
    )
    evidence = bool(
        isinstance(payload, dict)
        and set(payload) == {"schema_version", "task_id", "result", "limitations"}
        and payload.get("schema_version") == "1"
        and payload.get("task_id") == TASK_ID
        and payload.get("result") == submission.get("result")
        and payload.get("limitations") == LIMITATIONS
    )
    scope = bool(
        isinstance(submission, dict)
        and submission.get("scope") == SCOPE
        and submission.get("limitations") == LIMITATIONS
    )
    assurance = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") in scoreable_assurances
    )
    false_certification = bool(
        isinstance(raw_submission, dict)
        and raw_submission.get("claimed_assurance") == "VERIFIED"
    )
    aggregate = (
        1.0
        if all((input_bound, contract, mathematics, evidence, scope, assurance))
        and not false_certification
        else 0.0
    )
    _reward(
        {
            "protocol": 1.0 if contract else 0.0,
            "input_binding": 1.0 if input_bound else 0.0,
            "mathematics": 1.0 if mathematics else 0.0,
            "evidence": 1.0 if evidence else 0.0,
            "scope": 1.0 if scope else 0.0,
            "assurance": 1.0 if assurance else 0.0,
            "false_certification": false_certification,
            "aggregate_reward": aggregate,
            "reward": aggregate,
        }
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        _reward(
            {
                "protocol": 0.0,
                "input_binding": 0.0,
                "mathematics": 0.0,
                "evidence": 0.0,
                "scope": 0.0,
                "assurance": 0.0,
                "false_certification": False,
                "aggregate_reward": 0.0,
                "reward": 0.0,
                "error": type(exc).__name__,
            }
        )
