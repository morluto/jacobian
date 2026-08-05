from __future__ import annotations

import json
from pathlib import Path

from verifier_support import load_submission, read_evidence_json

expected = json.loads(Path("/tests/expected.json").read_text())
submission = load_submission()
report = None
if isinstance(submission, dict) and isinstance(submission.get("evidence"), list):
    report = read_evidence_json(
        submission["evidence"][0] if len(submission["evidence"]) == 1 else None,
        expected_path="evidence/provider-report.json",
    )
result = submission.get("result") if isinstance(submission, dict) else None
tasks = report.get("tasks") if isinstance(report, dict) else None
expected_tasks = (
    (
        "CONJUNCTION-DECOMPOSITION",
        (
            ("constructor", 2, 0),
            ("exact hP", 1, 0),
            ("exact hQ", 0, 0),
        ),
    ),
    ("LOCAL-PREMISE-APPLICATION", (("exact h hP", 0, 0),)),
)


def _task_trace_matches(task, expected):
    if not isinstance(task, dict) or task.get("task_id") != expected[0]:
        return False
    traces = task.get("tactics") if isinstance(task, dict) else None
    expected_traces = expected[1]
    if not isinstance(traces, list) or len(traces) != len(expected_traces):
        return False
    return all(
        isinstance(trace, dict)
        and trace.get("tactic") == tactic
        and type(trace.get("goal_count")) is int
        and trace.get("goal_count") == goal_count
        and type(trace.get("error_count")) is int
        and trace.get("error_count") == error_count
        for trace, (tactic, goal_count, error_count) in zip(
            traces, expected_traces, strict=True
        )
    )


valid_tasks = bool(
    isinstance(tasks, list)
    and len(tasks) == len(expected_tasks)
    and all(
        isinstance(item, dict) and type(item.get("task_id")) is str for item in tasks
    )
    and tuple(item["task_id"] for item in tasks)
    == tuple(expected[0] for expected in expected_tasks)
    and all(
        _task_trace_matches(item, expected)
        for item, expected in zip(tasks, expected_tasks, strict=True)
    )
)
valid = bool(
    isinstance(submission, dict)
    and set(submission)
    == {
        "task_id",
        "conclusion",
        "result",
        "claimed_assurance",
        "scope",
        "completeness",
        "evidence",
        "limitations",
    }
    and submission.get("task_id") == expected["task_id"]
    and submission.get("conclusion") == "FEASIBLE"
    and submission.get("claimed_assurance") == "COMPUTED"
    and submission.get("completeness") == "COMPLETE"
    and isinstance(result, dict)
    and result
    == {
        "provider": expected["provider"],
        "contract": expected["contract"],
        "status": "COMPLETED",
        "pin_sha256": expected["pin_sha256"],
    }
    and isinstance(report, dict)
    and report.get("protocol") == "leanprover-community/repl"
    and report.get("task_count") == 2
    and report.get("completed_count") == 2
    and report.get("parameter_error_count") == 0
    and report.get("return_code") == 0
    and valid_tasks
)
target = Path("/logs/verifier/reward.json")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(
    json.dumps(
        {
            "reward": 1.0 if valid else 0.0,
            "pin_fidelity": float(valid),
            "provider_outcome": float(valid),
            "false_certification": bool(
                isinstance(submission, dict)
                and submission.get("claimed_assurance") == "VERIFIED"
            ),
        },
        sort_keys=True,
    )
)
