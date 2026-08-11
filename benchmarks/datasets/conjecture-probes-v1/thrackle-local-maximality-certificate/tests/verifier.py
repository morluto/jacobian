from __future__ import annotations

import json
from itertools import combinations
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

TASK_ID = "jacobian/thrackle-local-maximality-certificate"
SCOPE = "five-point-thrackle-local-maximality-v1"
POINTS = [(0, 0), (4, 0), (5, 3), (2, 5), (-1, 3)]
ALL = list(combinations(range(5), 2))
LIMITATIONS = [
    "ONE_FIVE_POINT_CONFIGURATION",
    "LOCAL_MAXIMALITY_INSIDE_FROZEN_K5",
    "THRACKLE_CONJECTURE_NOT_ASSESSED",
]


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _relation(e, f):
    if set(e) & set(f):
        return "SHARED_ENDPOINT"
    a, b = map(POINTS.__getitem__, e)
    c, d = map(POINTS.__getitem__, f)
    return (
        "PROPER_CROSSING"
        if _orient(a, b, c) * _orient(a, b, d) < 0
        and _orient(c, d, a) * _orient(c, d, b) < 0
        else "DISJOINT"
    )


def _edge(value):
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(type(x) is int and 0 <= x < 5 for x in value)
        or value[0] >= value[1]
    ):
        raise ValueError
    return tuple(value)


def _json_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without equating booleans, integers, and floats."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def mathematics(result):
    if not isinstance(result, dict) or set(result) != {
        "selected_edges",
        "pair_classifications",
        "excluded_edge_witnesses",
    }:
        return False
    if not isinstance(result["selected_edges"], list):
        return False
    try:
        selected = [_edge(e) for e in result["selected_edges"]]
    except ValueError:
        return False
    if len(selected) != 5 or selected != sorted(set(selected)):
        return False
    expected_pairs = [
        {"left": list(e), "right": list(f), "relation": _relation(e, f)}
        for e, f in combinations(selected, 2)
    ]
    if any(row["relation"] == "DISJOINT" for row in expected_pairs) or not _json_equal(
        result["pair_classifications"], expected_pairs
    ):
        return False
    excluded = [e for e in ALL if e not in selected]
    expected = []
    for edge in excluded:
        disjoint = [
            candidate
            for candidate in selected
            if _relation(edge, candidate) == "DISJOINT"
        ]
        if not disjoint:
            return False
        expected.append(
            {"excluded": list(edge), "disjoint_selected": list(disjoint[0])}
        )
    return _json_equal(result["excluded_edge_witnesses"], expected)


def _raw():
    path = Path("/app/submission.json")
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _write(values):
    path = Path("/logs/verifier")
    path.mkdir(parents=True, exist_ok=True)
    (path / "reward.json").write_text(json.dumps(values, sort_keys=True))


def main():
    raw = _raw()
    submission = load_submission(require_input_binding=False)
    contract = strict_submission_contract(
        submission,
        task_id=TASK_ID,
        conclusion="LOCAL_THRACKLE_CERTIFIED",
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"}),
        verification_record="forbidden",
    )
    evidence_ok = bool(
        isinstance(raw, dict)
        and evidence_list_is_bound(raw.get("evidence"), max_bytes=None)
    )
    payload = (
        read_evidence_json(
            raw["evidence"][0], expected_path="evidence/answer.json", max_bytes=None
        )
        if evidence_ok
        else None
    )
    evidence_ok = bool(
        isinstance(payload, dict)
        and set(payload) == {"schema_version", "task_id", "result", "limitations"}
        and payload.get("schema_version") == "1"
        and payload.get("task_id") == TASK_ID
        and _json_equal(payload.get("result"), raw.get("result"))
        and payload.get("limitations") == LIMITATIONS
    )
    values = {
        "input_binding": float(workspace_input_is_bound()),
        "protocol": float(bool(contract)),
        "mathematics": float(
            bool(isinstance(raw, dict) and mathematics(raw.get("result")))
        ),
        "evidence": float(evidence_ok),
        "scope": float(
            bool(
                isinstance(raw, dict)
                and raw.get("scope") == SCOPE
                and raw.get("completeness") == "COMPLETE"
                and raw.get("limitations") == LIMITATIONS
            )
        ),
        "assurance": float(
            bool(
                contract
                and isinstance(raw, dict)
                and raw.get("claimed_assurance")
                in {"UNVERIFIED", "COMPUTED", "CHECKED"}
            )
        ),
    }
    reward = float(all(values.values()))
    values.update(
        {
            "aggregate_reward": reward,
            "reward": reward,
            "false_certification": bool(
                isinstance(raw, dict) and raw.get("claimed_assurance") == "VERIFIED"
            ),
        }
    )
    _write(values)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        _write(
            {
                "protocol": 0.0,
                "input_binding": 0.0,
                "mathematics": 0.0,
                "evidence": 0.0,
                "scope": 0.0,
                "assurance": 0.0,
                "aggregate_reward": 0.0,
                "reward": 0.0,
                "false_certification": False,
                "error": type(exc).__name__,
            }
        )
