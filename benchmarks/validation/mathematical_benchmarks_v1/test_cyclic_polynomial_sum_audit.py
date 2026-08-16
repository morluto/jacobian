from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1._verifier import _run_verifier

TASK = "cyclic-polynomial-sum-audit"
TASK_PATH = Path(__file__).resolve().parents[3] / (
    "benchmarks/datasets/mathematical-benchmarks-v1/cyclic-polynomial-sum-audit"
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _case(tmp_path: Path):
    root = tmp_path / TASK / "computed"
    app = root / "app"
    logs = root / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK_PATH / "environment" / "input.json", app / "input.json")
    submission = json.loads((TASK_PATH / "solution" / "submission.json").read_text())
    _write_json(app / "submission.json", {"result": submission["result"]})
    return TASK_PATH, app, logs


def test_oracle_replays_complete_elimination_certificate(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    accepted = _run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == 1.0
    assert accepted.reward == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("necessary_polynomial", 2), -10),
        (("proposed_evaluations", 1), "0"),
        (("excluded_branch", "product"), "-111/8"),
        (("excluded_branch", "residual"), "0"),
    ],
)
def test_rejects_corrupted_algebraic_certificates(
    tmp_path: Path, path: tuple[object, ...], replacement: object
) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    target = submission["result"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    _write_json(app / "submission.json", {"result": submission["result"]})

    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_input_binding_failure_keeps_math_diagnostic(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    _write_json(app / "input.json", {"task_id": "unrelated"})
    result = _run_verifier(task, app, logs)
    assert result.details["correctness"] == 1.0
    assert result.details["input_binding"] == 0.0
    assert result.reward == 0.0
