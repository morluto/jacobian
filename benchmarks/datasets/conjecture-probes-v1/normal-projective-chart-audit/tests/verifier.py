from __future__ import annotations

import json
from fractions import Fraction
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

TASK_ID = "jacobian/normal-projective-chart-audit"
SCOPE = "ellipse-normal-projective-chart-audit-v1"
LIMITATIONS = [
    "ONE_RATIONAL_ELLIPSE",
    "ONE_QUERY_POINT",
    "CONCURRENT_NORMALS_CONJECTURE_NOT_ASSESSED",
]


def _q(x):
    if not isinstance(x, str):
        raise ValueError
    q = Fraction(x)
    if str(q) != x:
        raise ValueError
    return q


def _point(x):
    if not isinstance(x, list) or len(x) != 2:
        raise ValueError
    return _q(x[0]), _q(x[1])


def _param(t):
    return Fraction(2) * (1 - t * t) / (1 + t * t), Fraction(2) * t / (1 + t * t)


def _residuals(point):
    x, y = point
    return x * x / Fraction(4) + y * y - 1, -3 * x * y


def mathematics(result):
    if not isinstance(result, dict) or set(result) != {
        "finite_parameters",
        "finite_points",
        "missing_projective_parameter",
        "missing_point",
        "footpoint_records",
    }:
        return False
    try:
        params = [_q(x) for x in result["finite_parameters"]]
        points = [_point(x) for x in result["finite_points"]]
        missing = _point(result["missing_point"])
        projective = [_q(x) for x in result["missing_projective_parameter"]]
    except (ValueError, ZeroDivisionError):
        return False
    if (
        params != [Fraction(-1), Fraction(0), Fraction(1)]
        or points != [_param(t) for t in params]
        or projective != [Fraction(1), Fraction(0)]
        or missing != (Fraction(-2), Fraction(0))
    ):
        return False
    expected_points = sorted([missing, *points])
    records = result["footpoint_records"]
    if not isinstance(records, list) or len(records) != 4:
        return False
    for row, point in zip(records, expected_points, strict=True):
        if not isinstance(row, dict) or set(row) != {
            "point",
            "ellipse_residual",
            "normal_residual",
        }:
            return False
        try:
            submitted = _point(row["point"])
            er = _q(row["ellipse_residual"])
            nr = _q(row["normal_residual"])
        except (ValueError, ZeroDivisionError):
            return False
        if submitted != point or (er, nr) != _residuals(point) or er or nr:
            return False
    return True


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
        conclusion="AFFINE_NORMAL_COUNT_IS_INCOMPLETE_AND_REPAIRED",
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
