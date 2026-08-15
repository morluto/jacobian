from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1._verifier import _run_verifier

TASK = "fiber-dimension-semicontinuity-repair"
TASK_PATH = Path(__file__).resolve().parents[3] / (
    "benchmarks/datasets/mathematical-benchmarks-v1/fiber-dimension-semicontinuity-repair"
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _inject_result_json(app: Path, submission: dict) -> None:
    evidence_path = app / "evidence" / "answer.txt"
    text = evidence_path.read_text()
    lines = [line for line in text.splitlines() if not line.startswith("RESULT_JSON:")]
    lines.append(
        "RESULT_JSON:"
        + json.dumps(submission["result"], sort_keys=True, separators=(",", ":"))
    )
    evidence_path.write_text("\n".join(lines) + "\n")
    submission["witness"][0]["sha256"] = _digest(evidence_path)


def _case(tmp_path: Path):
    root = tmp_path / TASK / "computed"
    app = root / "app"
    logs = root / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK_PATH / "environment" / "input.json", app / "input.json")
    submission = json.loads((TASK_PATH / "solution" / "submission.json").read_text())
    for descriptor in submission["witness"]:
        evidence_path = Path(descriptor["path"])
        destination = app / evidence_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TASK_PATH / "solution" / evidence_path.name, destination)
    _inject_result_json(app, submission)
    _write_json(app / "submission.json", submission)
    return TASK_PATH, app, logs


def _rewrite(app: Path, submission: dict) -> None:
    _inject_result_json(app, submission)
    _write_json(app / "submission.json", submission)


def _term(coefficient: str, x_power: int, y_power: int) -> dict[str, object]:
    return {"coefficient": coefficient, "exponents": [x_power, y_power]}


def test_accepts_alternative_ideal_generators_and_fiber_order(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = submission["result"]
    result["ideal_generators"] = [
        {"terms": [_term("1", 2, 0), _term("1", 0, 2)]},
        {"terms": [_term("1", 2, 0), _term("-1", 0, 2)]},
        {"terms": [_term("1", 1, 1)]},
    ]
    result["fiber_checks"].reverse()
    _rewrite(app, submission)
    assert _run_verifier(task, app, logs).reward == 1.0


def test_rejects_strictly_smaller_minor_ideal(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["ideal_generators"] = [
        {"terms": [_term("1", 2, 0)]},
        {"terms": [_term("1", 1, 1)]},
    ]
    _rewrite(app, submission)
    reward = _run_verifier(task, app, logs)
    assert reward.details["correctness"] == 0.0
    assert reward.reward == 0.0


def test_rejects_duplicate_monomials_fail_closed(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["ideal_generators"][0]["terms"] = [
        _term("1", 2, 0),
        _term("1", 2, 0),
    ]
    _rewrite(app, submission)
    reward = _run_verifier(task, app, logs)
    assert reward.details["correctness"] == 0.0


def test_rejects_wrong_fiber_rank(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["fiber_checks"][0]["matrix_rank"] = 1
    _rewrite(app, submission)
    assert _run_verifier(task, app, logs).details["correctness"] == 0.0


def test_visible_input_tamper_fails_closed(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    (app / "input.json").write_text("{}")
    reward = _run_verifier(task, app, logs)
    assert reward.details["correctness"] == 0.0
    assert reward.reward == 0.0


def test_evidence_must_bind_submitted_result(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "answer.txt"
    text = evidence_path.read_text()
    lines = [line for line in text.splitlines() if not line.startswith("RESULT_JSON:")]
    lines.append("RESULT_JSON:{}")
    evidence_path.write_text("\n".join(lines) + "\n")
    submission["witness"][0]["sha256"] = _digest(evidence_path)
    _write_json(app / "submission.json", submission)
    reward = _run_verifier(task, app, logs)
    assert reward.details["correctness"] == 1.0
    assert reward.reward == 0.0
    assert reward.reward == 0.0


def test_oversized_fiber_list_rejected_without_crash(tmp_path: Path) -> None:
    """An oversized fiber_checks list must fail closed without excessive work."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    base = submission["result"]["fiber_checks"][0]
    submission["result"]["fiber_checks"] = [base] * 10000
    _rewrite(app, submission)
    reward = _run_verifier(task, app, logs)
    assert reward.details["correctness"] == 0.0
    assert reward.reward == 0.0
    assert reward.reward == 0.0


def test_rejects_witness_without_result_marker(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text("prose without any structured marker\n")
    submission = json.loads((app / "submission.json").read_text())
    submission["witness"][0]["sha256"] = _digest(evidence_path)
    _write_json(app / "submission.json", submission)
    result = _run_verifier(task, app, logs)
    assert result.reward == 0.0
    assert result.reward == 0.0
