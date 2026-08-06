"""Clean-room verifier for the finite Happy Ending convex-position probe."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

from verifier_support import (
    evidence_list_is_bound,
    load_submission,
    read_evidence_json,
    strict_submission_contract,
    workspace_input_is_bound,
)

TASK_ID = "jacobian/happy-ending-convex-position"
SCOPE = "happy-ending-convex-position:points-v1"
LIMITATIONS = ["THIRTEEN_FROZEN_POINTS", "NO_GENERAL_ERDOS_SZEKERES_CONCLUSION"]


def _cross(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> int:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(set(points))
    lower: list[tuple[int, int]] = []
    for point in ordered:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[int, int]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _frozen() -> dict[str, Any] | None:
    try:
        value = json.loads(Path("/tests/input.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("task_id") != TASK_ID:
        return None
    points = value.get("points")
    return value if isinstance(points, list) and len(points) == 13 else None


def _cyclic(points: list[tuple[int, int]]) -> bool:
    if len(points) < 3:
        return False
    signs = [
        _cross(points[i - 2], points[i - 1], points[i]) for i in range(len(points))
    ]
    return all(value > 0 for value in signs) or all(value < 0 for value in signs)


def _mathematics(result: Any, frozen: dict[str, Any]) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "general_position",
        "convex_subset_counts",
        "maximum_convex_size",
        "maximum_witness_cyclic",
        "scope_identity",
    }:
        return False
    records = frozen["points"]
    ids = [record["id"] for record in records]
    points = [(record["x"], record["y"]) for record in records]
    general = all(_cross(*triple) != 0 for triple in itertools.combinations(points, 3))
    if result.get("general_position") is not general or not general:
        return False
    counts: dict[int, int] = {}
    maximum = 2
    maximum_sets: set[frozenset[str]] = set()
    for size in range(3, 14):
        count = 0
        for subset in itertools.combinations(range(13), size):
            selected = [points[index] for index in subset]
            if len(_hull(selected)) == size:
                count += 1
                if size > maximum:
                    maximum = size
                    maximum_sets.clear()
                if size == maximum:
                    maximum_sets.add(frozenset(ids[index] for index in subset))
        counts[size] = count
    rows = result.get("convex_subset_counts")
    if not isinstance(rows, list) or len(rows) != 11:
        return False
    submitted: dict[int, int] = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"size", "count"}
            or type(row["size"]) is not int
            or type(row["count"]) is not int
            or row["size"] in submitted
        ):
            return False
        submitted[row["size"]] = row["count"]
    witness = result.get("maximum_witness_cyclic")
    if (
        not isinstance(witness, list)
        or len(witness) != maximum
        or len(set(witness)) != maximum
        or any(name not in ids for name in witness)
    ):
        return False
    coordinates = [points[ids.index(name)] for name in witness]
    return (
        submitted == counts
        and result.get("maximum_convex_size") == maximum
        and frozenset(witness) in maximum_sets
        and _cyclic(coordinates)
        and result.get("scope_identity") == SCOPE
    )


def _reward(value: dict[str, Any]) -> None:
    path = Path("/logs/verifier")
    path.mkdir(parents=True, exist_ok=True)
    (path / "reward.json").write_text(json.dumps(value, sort_keys=True))


def main() -> None:
    input_bound = workspace_input_is_bound()
    frozen = _frozen() if input_bound else None
    submission = load_submission()
    contract = strict_submission_contract(
        submission,
        task_id=TASK_ID,
        conclusion="HAPPY_ENDING_FINITE_CONVEX_POSITION",
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"}),
        verification_record="forbidden",
    )
    mathematics = bool(
        contract and frozen and _mathematics(submission["result"], frozen)
    )
    evidence = bool(contract and evidence_list_is_bound(submission["evidence"]))
    payload = (
        read_evidence_json(
            submission["evidence"][0], expected_path="evidence/answer.txt"
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
        contract
        and submission.get("scope") == SCOPE
        and submission.get("limitations") == LIMITATIONS
    )
    assurance = bool(contract and submission.get("claimed_assurance") == "CHECKED")
    false_certification = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") == "VERIFIED"
    )
    aggregate = (
        1.0
        if all((input_bound, contract, mathematics, evidence, scope, assurance))
        and not false_certification
        else 0.0
    )
    _reward(
        {
            "protocol": float(contract),
            "input_binding": float(input_bound),
            "mathematics": float(mathematics),
            "evidence": float(evidence),
            "scope": float(scope),
            "assurance": float(assurance),
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
