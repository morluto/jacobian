from __future__ import annotations

import json
from itertools import product
from pathlib import Path

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    evidence_list_is_bound,
    is_regular_bounded_file,
    load_submission,
    read_evidence_json,
    strict_submission_contract,
    workspace_input_is_bound,
)

TASK_ID = "jacobian/illumination-strictness-audit"
SCOPE = "cube-illumination-strictness-audit-v1"
LIMITATIONS = [
    "ONE_THREE_DIMENSIONAL_CUBE",
    "VERTEX_SIGN_CONE_MODEL_ONLY",
    "GENERAL_ILLUMINATION_CONJECTURE_NOT_ASSESSED",
]
VERTICES = list(product((-1, 1), repeat=3))


def _weak(v, d):
    return all(a * b <= 0 for a, b in zip(v, d, strict=True))


def _strict(v, d):
    return all(a * b < 0 for a, b in zip(v, d, strict=True))


def _directions(value, n, zeros):
    return (
        isinstance(value, list)
        and len(value) == n
        and len({tuple(d) for d in value}) == n
        and all(
            isinstance(d, list)
            and len(d) == 3
            and all(type(x) is int and x in {-1, 0, 1} for x in d)
            and d.count(0) == zeros
            for d in value
        )
    )


def mathematics(result):
    if not isinstance(result, dict) or set(result) != {
        "flawed_directions",
        "weak_false_positive_pairs",
        "repair_directions",
        "vertex_to_direction",
    }:
        return False
    flawed = result["flawed_directions"]
    repair = result["repair_directions"]
    if not _directions(flawed, 4, 1) or not _directions(repair, 8, 0):
        return False
    weak_cover = all(any(_weak(v, d) for d in flawed) for v in VERTICES)
    expected = [
        {"vertex_index": i, "direction_index": j}
        for i, v in enumerate(VERTICES)
        for j, d in enumerate(flawed)
        if _weak(v, d) and not _strict(v, d)
    ]
    mapping = result["vertex_to_direction"]
    return (
        weak_cover
        and bool(expected)
        and result["weak_false_positive_pairs"] == expected
        and isinstance(mapping, list)
        and len(mapping) == 8
        and all(
            type(j) is int and 0 <= j < 8 and _strict(v, repair[j])
            for v, j in zip(VERTICES, mapping, strict=True)
        )
        and len(set(mapping)) == 8
        and all(sum(_strict(v, d) for d in repair) == 1 for v in VERTICES)
    )


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
        conclusion="WEAK_ILLUMINATION_IS_UNSOUND_AND_REPAIRED",
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"}),
        verification_record="forbidden",
    )
    evidence_ok = bool(
        isinstance(raw, dict)
        and evidence_list_is_bound(raw.get("evidence"), max_bytes=None)
    )
    payload = (
        read_evidence_json(
            raw["evidence"][0], expected_path="evidence/answer.txt", max_bytes=None
        )
        if evidence_ok
        else None
    )
    evidence_ok = bool(
        isinstance(payload, dict)
        and set(payload) == {"schema_version", "task_id", "result", "limitations"}
        and payload.get("schema_version") == "1"
        and payload.get("task_id") == TASK_ID
        and payload.get("result") == raw.get("result")
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
