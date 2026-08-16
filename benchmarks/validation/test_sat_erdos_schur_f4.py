from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from benchmarks.validation.public_reproductions_v1._fixtures import _write_json
from benchmarks.validation.public_reproductions_v1._verifier import _run_verifier

ROOT = Path(__file__).parents[2]
TASK = (
    ROOT / "benchmarks" / "datasets" / "public-reproductions-v1" / "sat-erdos-schur-f4"
)


def _case(tmp_path: Path) -> tuple[Path, Path, Path]:
    app = tmp_path / "app"
    logs = tmp_path / "logs"
    app.mkdir()
    logs.mkdir()
    shutil.copy2(TASK / "environment" / "input.json", app / "input.json")
    shutil.copy2(TASK / "solution" / "submission.json", app / "submission.json")
    return TASK, app, logs


def test_rejects_prose_in_place_of_schur_certificates(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"] = {
        "f_value": 45,
        "lower_bound_evidence": "A 4-coloring of 1 through 44 exists.",
        "upper_bound_evidence": "An exhaustive proof for 45 exists.",
    }
    _write_json(submission_path, submission)

    result = _run_verifier(task, app, logs)

    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_accepts_explicit_sum_free_partition_and_independent_upper_check(
    tmp_path: Path,
) -> None:
    task, app, logs = _case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"] = {
        "f_value": 45,
        "lower_bound_partition": [
            [1, 3, 5, 15, 17, 19, 26, 28, 40, 42, 44],
            [6, 8, 13, 18, 20, 22, 23, 25, 27, 32, 34, 37],
            [2, 7, 10, 16, 21, 24, 29, 30, 35, 38, 43],
            [4, 9, 11, 12, 14, 31, 33, 36, 39, 41],
        ],
        "upper_bound_method": "INDEPENDENT_EXHAUSTIVE_CSP",
    }
    _write_json(submission_path, submission)

    accepted = _run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)
