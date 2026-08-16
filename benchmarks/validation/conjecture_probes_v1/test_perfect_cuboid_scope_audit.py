from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from benchmarks.validation._verifier_child import run_verifier_in_child

ROOT = Path(__file__).parents[3]
TASK = ROOT / "benchmarks/datasets/conjecture-probes-v1/perfect-cuboid-scope-audit"


def _case(tmp_path: Path) -> tuple[Path, Path, dict]:
    app = tmp_path / "app"
    logs = tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment/input.json", app / "input.json")
    subprocess.run(
        [sys.executable, str(TASK / "solution/solve.py"), "--root", str(app)],
        check=True,
    )
    return app, logs, json.loads((app / "submission.json").read_text())


def _write(app: Path, submission: dict) -> None:
    (app / "submission.json").write_text(
        json.dumps({"result": submission["result"]}) + "\n"
    )


def _run(app: Path, logs: Path):
    return run_verifier_in_child(task=TASK, app=app, logs=logs)


def test_oracle_and_reordered_cases_receive_full_reward(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["cases"].reverse()
    _write(app, submission)
    reward = _run(app, logs)
    assert reward.details["aggregate_reward"] == 1.0
    assert reward.details["mathematics"] == 1.0


def test_reordered_aligned_face_pairs_receive_full_reward(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    row = submission["result"]["cases"][0]
    order = (2, 0, 1)
    row["face_radicands"] = [row["face_radicands"][index] for index in order]
    row["face_roots"] = [row["face_roots"][index] for index in order]
    _write(app, submission)
    assert _run(app, logs).details["aggregate_reward"] == 1.0


def test_euler_brick_cannot_be_promoted_to_perfect_cuboid(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["cases"][0]["class"] = "PERFECT_CUBOID"
    submission["result"]["contains_perfect_cuboid"] = True
    _write(app, submission)
    reward = _run(app, logs)
    assert reward.details["mathematics"] == 0.0
    assert reward.details["aggregate_reward"] == 0.0


def test_wrong_root_omission_and_duplicate_fail_closed(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["cases"][0]["space_root"] = 271
    _write(app, submission)
    assert _run(app, logs).details["mathematics"] == 0.0
    app, logs, submission = _case(tmp_path / "missing")
    submission["result"]["cases"].pop()
    _write(app, submission)
    assert _run(app, logs).details["aggregate_reward"] == 0.0
    app, logs, submission = _case(tmp_path / "duplicate")
    submission["result"]["cases"][-1] = submission["result"]["cases"][0]
    _write(app, submission)
    assert _run(app, logs).details["aggregate_reward"] == 0.0


def test_tampered_input_is_rejected(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    frozen = json.loads((app / "input.json").read_text())
    frozen["cases"][0]["edges"][0] += 1
    (app / "input.json").write_text(json.dumps(frozen))
    _write(app, submission)
    reward = _run(app, logs)
    assert reward.details["input_binding"] == 0.0
    assert reward.details["mathematics"] == 1.0
    assert reward.details["aggregate_reward"] == 0.0


def test_unused_evidence_file_does_not_affect_reward(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    _write(app, submission)
    (app / "evidence").mkdir(exist_ok=True)
    (app / "evidence/answer.txt").write_text("{}\n")
    reward = _run(app, logs)
    assert reward.details["aggregate_reward"] == 1.0
