from __future__ import annotations

import json
import shutil
from pathlib import Path

from benchmarks.validation._solution import run_solution
from benchmarks.validation._verifier_child import run_verifier_in_child

ROOT = Path(__file__).parents[3]
TASK = ROOT / "benchmarks/datasets/conjecture-probes-v1/happy-ending-convex-position"


def _case(tmp_path: Path) -> tuple[Path, Path, dict]:
    app = tmp_path / "app"
    logs = tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment/input.json", app / "input.json")
    run_solution(TASK, app)
    return app, logs, json.loads((app / "submission.json").read_text())


def _write(app: Path, submission: dict) -> None:
    (app / "submission.json").write_text(
        json.dumps({"result": submission["result"]}) + "\n"
    )


def _run(app: Path, logs: Path):
    return run_verifier_in_child(task=TASK, app=app, logs=logs)


def test_oracle_and_reordered_equivalent_output_pass(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["convex_subset_counts"].reverse()
    witness = submission["result"]["maximum_witness_cyclic"]
    submission["result"]["maximum_witness_cyclic"] = witness[2:] + witness[:2]
    _write(app, submission)
    assert _run(app, logs).details["aggregate_reward"] == 1.0


def test_reversed_cyclic_witness_passes(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["maximum_witness_cyclic"].reverse()
    _write(app, submission)
    assert _run(app, logs).details["aggregate_reward"] == 1.0


def test_wrong_count_and_nonmaximum_witness_fail(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["convex_subset_counts"][3]["count"] += 1
    _write(app, submission)
    assert _run(app, logs).details["mathematics"] == 0.0
    app, logs, submission = _case(tmp_path / "witness")
    submission["result"]["maximum_witness_cyclic"][0] = "P00"
    _write(app, submission)
    assert _run(app, logs).details["aggregate_reward"] == 0.0


def test_input_tampering_fails_closed(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    frozen = json.loads((app / "input.json").read_text())
    frozen["points"][0]["x"] += 1
    (app / "input.json").write_text(json.dumps(frozen))
    _write(app, submission)
    assert _run(app, logs).details["input_binding"] == 0.0


def test_unused_evidence_file_does_not_affect_reward(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    _write(app, submission)
    (app / "evidence").mkdir(exist_ok=True)
    (app / "evidence/answer.txt").write_text("{}\n")
    reward = _run(app, logs)
    assert reward.details["aggregate_reward"] == 1.0
