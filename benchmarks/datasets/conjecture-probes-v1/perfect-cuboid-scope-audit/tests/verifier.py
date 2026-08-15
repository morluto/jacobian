"""Clean-room verifier for the perfect-cuboid finite scope audit."""

from __future__ import annotations

import json
from collections import Counter
from math import isqrt
from pathlib import Path
from typing import Any

from verifier_support import (
    json_value_equal,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
    resolve_evidence,
    witness_list_is_bound,
    workspace_input_is_bound,
)

TASK_ID = "jacobian/perfect-cuboid-scope-audit"
SCOPE = "perfect-cuboid-scope-audit:case-set-v1"
CLASSES = {"PERFECT_CUBOID", "EULER_BRICK_ONLY", "SPACE_AND_TWO_FACES", "OTHER"}


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
    normalize_reward_file(path / "reward.json")


def _witness_matches_result(witness: object, result: object) -> bool:
    if not witness_list_is_bound(witness, expected_path="evidence/answer.txt"):
        return False
    if resolve_evidence(witness[0], expected_path="evidence/answer.txt") is None:
        return False
    payload = read_evidence_json(witness[0], expected_path="evidence/answer.txt")
    return bool(
        isinstance(payload, dict)
        and set(payload) == {"schema_version", "task_id", "result"}
        and payload.get("schema_version") == "1"
        and payload.get("task_id") == TASK_ID
        and json_value_equal(payload.get("result"), result)
    )


def main() -> None:
    input_bound = workspace_input_is_bound()
    frozen = _frozen()
    submission = load_submission(require_input_binding=False)
    protocol = isinstance(submission, dict)
    mathematics = bool(
        frozen and protocol and _mathematics(submission.get("result"), frozen)
    )
    witness = bool(
        protocol
        and _witness_matches_result(submission.get("witness"), submission.get("result"))
    )
    aggregate = float(input_bound and protocol and mathematics and witness)
    _reward(
        {
            "protocol": float(protocol),
            "input_binding": float(input_bound),
            "mathematics": float(mathematics),
            "witness_validity": float(witness),
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
                "witness_validity": 0.0,
                "aggregate_reward": 0.0,
                "reward": 0.0,
                "error": type(exc).__name__,
            }
        )
