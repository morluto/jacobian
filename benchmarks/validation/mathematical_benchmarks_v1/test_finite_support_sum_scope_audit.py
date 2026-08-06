import copy
import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1.support import _run_verifier

TASK = "finite-support-sum-scope-audit"


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
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment/input.json", app / "input.json")
    evidence = {
        "schema_version": "1",
        "task_id": f"jacobian/{TASK}",
        "result": submission["result"],
        "limitations": submission["limitations"],
    }
    path = app / "evidence/scope-audit.json"
    path.write_text(json.dumps(evidence, separators=(",", ":")))
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission))
    return _run_verifier(task, app, logs)


def test_oracle_and_alternative_tail_family(tmp_path):
    assert _verify(tmp_path / "oracle", _oracle())["reward"] == 1.0
    alt = _oracle()
    alt["result"]["n"] = 8
    alt["result"]["tail_singletons"] = list(range(20, 28))
    alt["result"]["summand_values"] = [{"numerator": 1, "denominator": 1}] * 8
    alt["result"]["partial_sum_lower_bound"] = 8
    assert _verify(tmp_path / "alt", alt)["reward"] == 1.0


def test_scope_and_assurance_attacks_fail(tmp_path):
    for name, mutate in [
        ("tail", lambda s: s["result"]["tail_singletons"].__setitem__(0, 1)),
        ("assurance", lambda s: s.update(claimed_assurance="VERIFIED")),
    ]:
        submission = copy.deepcopy(_oracle())
        mutate(submission)
        assert _verify(tmp_path / name, submission)["reward"] == 0


def test_accepts_unordered_valid_witnesses(tmp_path):
    submission = _oracle()
    submission["result"]["tail_singletons"] = list(
        reversed(submission["result"]["tail_singletons"])
    )
    submission["result"]["summand_values"] = list(
        reversed(submission["result"]["summand_values"])
    )
    submission["result"]["truncated_checkpoints"] = list(
        reversed(submission["result"]["truncated_checkpoints"])
    )
    assert _verify(tmp_path, submission)["reward"] == 1.0


def test_accepts_equivalent_unreduced_rationals(tmp_path):
    submission = _oracle()
    submission["result"]["summand_values"][0] = {"numerator": 2, "denominator": 2}
    checkpoint = submission["result"]["truncated_checkpoints"][0]["value"]
    checkpoint["numerator"] *= 2
    checkpoint["denominator"] *= 2
    assert _verify(tmp_path, submission)["reward"] == 1.0
