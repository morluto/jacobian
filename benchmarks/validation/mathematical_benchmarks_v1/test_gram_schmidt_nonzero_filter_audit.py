import json
import shutil
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1._verifier import _run_verifier

TASK = "gram-schmidt-nonzero-filter-audit"


def oracle():
    return json.loads(
        (
            Path("benchmarks/datasets/mathematical-benchmarks-v1")
            / TASK
            / "solution/submission.json"
        ).read_text()
    )


def verify(tmp_path, submission):
    task = Path("benchmarks/datasets/mathematical-benchmarks-v1") / TASK
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment/input.json", app / "input.json")
    (app / "submission.json").write_text(json.dumps({"result": submission["result"]}))
    return _run_verifier(task, app, logs)


def test_oracle_and_orthogonally_transformed_system(tmp_path):
    assert verify(tmp_path / "oracle", oracle()).reward == 1.0
    alt = oracle()
    for v in alt["result"]["vectors"]:
        v[0] *= -1
    for v in alt["result"]["residuals"]:
        v[0]["numerator"] *= -1
    assert verify(tmp_path / "alt", alt).reward == 1.0


def test_reordered_zero_residual_indices_are_accepted(tmp_path):
    reordered = oracle()
    reordered["result"]["zero_residual_indices"].reverse()

    accepted = verify(tmp_path, reordered)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == 1.0


def test_duplicate_zero_residual_index_is_rejected(tmp_path):
    duplicate = oracle()
    duplicate["result"]["zero_residual_indices"] = [4, 4]

    rejected = verify(tmp_path, duplicate)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_malformed_boolean_and_input_tamper_fail_closed(tmp_path):
    bad = oracle()
    bad["result"]["vectors"][0][0] = True
    assert verify(tmp_path / "boolean", bad).reward == 0
    task = Path("benchmarks/datasets/mathematical-benchmarks-v1") / TASK
    app, logs = tmp_path / "tamper/app", tmp_path / "tamper/logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    (app / "input.json").write_text("{}")
    (app / "submission.json").write_text(json.dumps({"result": oracle()["result"]}))
    assert _run_verifier(task, app, logs).reward == 0
