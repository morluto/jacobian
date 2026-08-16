import copy
import json
import shutil
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1._verifier import _run_verifier

TASK = "gf2-matrix-completion-quantifier-audit"


def _oracle():
    return json.loads(
        (
            Path("benchmarks/datasets/mathematical-benchmarks-v1")
            / TASK
            / "solution/submission.json"
        ).read_text()
    )


def _verify(tmp_path, submission):
    task = Path("benchmarks/datasets/mathematical-benchmarks-v1") / TASK
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment/input.json", app / "input.json")
    (app / "submission.json").write_text(json.dumps({"result": submission["result"]}))
    return _run_verifier(task, app, logs)


def test_oracle_passes(tmp_path):
    assert _verify(tmp_path, _oracle()).reward == 1.0


def test_rank_corruption_fails(tmp_path):
    submission = copy.deepcopy(_oracle())
    submission["result"].update(
        low_rank_completion=submission["result"]["full_rank_completion"]
    )
    assert _verify(tmp_path / "rank", submission).reward == 0


def test_boolean_matrix_entries_are_rejected(tmp_path):
    submission = copy.deepcopy(_oracle())
    submission["result"]["pattern"] = [
        [bool(value) for value in row] for row in submission["result"]["pattern"]
    ]
    assert _verify(tmp_path, submission).reward == 0


def test_extra_witness_key_is_rejected(tmp_path):
    task = Path("benchmarks/datasets/mathematical-benchmarks-v1") / TASK
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment/input.json", app / "input.json")
    submission = _oracle()
    submission["witness"] = []
    (app / "submission.json").write_text(json.dumps(submission))
    assert _run_verifier(task, app, logs).reward == 0
