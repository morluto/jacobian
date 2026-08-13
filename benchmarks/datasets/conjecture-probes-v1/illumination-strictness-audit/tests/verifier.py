from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path
from typing import Any

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    evidence_list_is_bound,
    is_regular_bounded_file,
    load_submission,
    normalize_reward_file,
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
    if not isinstance(value, list) or len(value) != n:
        return False
    if not all(
        isinstance(d, list)
        and len(d) == 3
        and all(type(x) is int and x in {-1, 0, 1} for x in d)
        and d.count(0) == zeros
        for d in value
    ):
        return False
    return len({tuple(d) for d in value}) == n


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


def _evidence_payload_matches_submission(payload: Any, raw: Any) -> bool:
    """Bind every copied evidence field to the submitted JSON value."""
    try:
        return bool(
            isinstance(payload, dict)
            and isinstance(raw, dict)
            and set(payload) == {"schema_version", "task_id", "result", "limitations"}
            and {"task_id", "result", "limitations"} <= raw.keys()
            and payload.get("schema_version") == "1"
            and _json_equal(payload.get("task_id"), raw.get("task_id"))
            and _json_equal(payload.get("result"), raw.get("result"))
            and _json_equal(payload.get("limitations"), raw.get("limitations"))
        )
    except RecursionError:
        return False


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
        and _json_equal(result["weak_false_positive_pairs"], expected)
        and isinstance(mapping, list)
        and len(mapping) == 8
        and all(
            type(j) is int and 0 <= j < 8 and _strict(v, repair[j])
            for v, j in zip(VERTICES, mapping, strict=True)
        )
        and len(set(mapping)) == 8
        and all(sum(_strict(v, d) for d in repair) == 1 for v in VERTICES)
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"out-of-range JSON number: {value}")
    return parsed


def _raw(path: Path = Path("/app/submission.json")):
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=_finite_json_float,
        )
    except (OSError, ValueError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _write(values):
    path = Path("/logs/verifier")
    path.mkdir(parents=True, exist_ok=True)
    reward_path = path / "reward.json"
    reward_path.write_text(json.dumps(values, sort_keys=True))
    normalize_reward_file(reward_path)


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
            raw["evidence"][0],
            expected_path="evidence/answer.json",
            max_bytes=None,
        )
        if evidence_ok
        else None
    )
    evidence_ok = _evidence_payload_matches_submission(payload, raw)
    mathematics_score = float(
        bool(isinstance(raw, dict) and mathematics(raw.get("result")))
    )
    values = {
        "input_binding": float(workspace_input_is_bound()),
        "protocol": float(bool(contract)),
        "correctness": mathematics_score,
        "mathematics": mathematics_score,
        "evidence": float(evidence_ok),
        "evidence_validity": float(evidence_ok),
        "scope": float(
            bool(
                isinstance(raw, dict)
                and raw.get("scope") == SCOPE
                and raw.get("completeness") == "COMPLETE"
                and raw.get("limitations") == LIMITATIONS
            )
        ),
        "scope_accuracy": float(
            bool(
                isinstance(raw, dict)
                and raw.get("scope") == SCOPE
                and raw.get("completeness") == "COMPLETE"
                and raw.get("limitations") == LIMITATIONS
            )
        ),
        "assurance": float(
            bool(
                isinstance(raw, dict)
                and type(raw.get("claimed_assurance")) is str
                and raw.get("claimed_assurance")
                in {"UNVERIFIED", "COMPUTED", "CHECKED"}
            )
        ),
    }
    values["assurance_calibration"] = values["assurance"]
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
                "correctness": 0.0,
                "evidence": 0.0,
                "evidence_validity": 0.0,
                "scope": 0.0,
                "scope_accuracy": 0.0,
                "assurance": 0.0,
                "assurance_calibration": 0.0,
                "aggregate_reward": 0.0,
                "reward": 0.0,
                "false_certification": False,
                "error": type(exc).__name__,
            }
        )
