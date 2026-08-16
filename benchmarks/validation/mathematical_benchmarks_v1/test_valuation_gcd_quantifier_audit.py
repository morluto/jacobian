import copy
import json
import shutil
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1._verifier import _run_verifier

TASK = "valuation-gcd-quantifier-audit"


def _oracle():
    return json.loads(
        (
            Path("benchmarks/datasets/mathematical-benchmarks-v1")
            / TASK
            / "solution/submission.json"
        ).read_text()
    )


def _prepare(tmp_path, submission):
    task = Path("benchmarks/datasets/mathematical-benchmarks-v1") / TASK
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment/input.json", app / "input.json")
    (app / "submission.json").write_text(json.dumps({"result": submission["result"]}))
    return task, app, logs


def _verify(tmp_path, submission):
    return _run_verifier(*_prepare(tmp_path, submission))


def test_oracle_and_alternative_repair(tmp_path):
    assert _verify(tmp_path / "oracle", _oracle()).reward == 1.0
    alt = _oracle()
    alt["result"]["repair"] = [
        {"prime": 2, "exponents": [0, 1, 1, 1]},
        {"prime": 7, "exponents": [1, 0, 1, 1]},
        {"prime": 11, "exponents": [1, 1, 0, 1]},
    ]
    assert _verify(tmp_path / "alt", alt).reward == 1.0


def test_rejects_weak_countermodel(tmp_path):
    submission = copy.deepcopy(_oracle())
    submission["result"].update(countermodel=submission["result"]["repair"])
    assert _verify(tmp_path / "counter", submission).reward == 0


def test_rejects_zero_row_that_is_not_a_prime_factor(tmp_path):
    submission = copy.deepcopy(_oracle())
    submission["result"]["countermodel"] = [
        {"prime": 2, "exponents": [0, 0, 0, 0]},
        {"prime": 3, "exponents": [1, 1, 1, 1]},
        {"prime": 5, "exponents": [2, 2, 2, 2]},
    ]
    reward = _verify(tmp_path, submission)
    assert reward.details["correctness"] == 0.0
    assert reward.reward == 0


def test_rejects_out_of_bound_prime_without_crashing(tmp_path):
    for name, prime in [("schema-bound", 101), ("huge", 10**4000)]:
        submission = copy.deepcopy(_oracle())
        submission["result"]["countermodel"][0]["prime"] = prime
        reward = _verify(tmp_path / name, submission)
        assert reward.details["correctness"] == 0.0
        assert reward.reward == 0
        assert (tmp_path / name / "logs" / "reward.json").is_file()


def test_undeclared_witness_key_is_rejected(tmp_path):
    submission = copy.deepcopy(_oracle())
    submission["witness"] = [
        {"path": "evidence/valuation-audit.json", "sha256": "sha256:" + "0" * 64}
    ]
    task = Path("benchmarks/datasets/mathematical-benchmarks-v1") / TASK
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment/input.json", app / "input.json")
    (app / "submission.json").write_text(json.dumps(submission))
    reward = _run_verifier(task, app, logs)
    assert reward.reward == 0
