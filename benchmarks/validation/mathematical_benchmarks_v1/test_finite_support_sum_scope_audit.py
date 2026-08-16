import json
import shutil
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1._verifier import _run_verifier

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
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment/input.json", app / "input.json")
    (app / "submission.json").write_text(json.dumps({"result": submission["result"]}))
    return _run_verifier(task, app, logs)


def test_oracle_and_alternative_tail_family(tmp_path):
    assert _verify(tmp_path / "oracle", _oracle()).reward == 1.0
    alt = _oracle()
    alt["result"]["n"] = 8
    alt["result"]["tail_singletons"] = list(range(20, 28))
    alt["result"]["summand_values"] = [{"numerator": 1, "denominator": 1}] * 8
    alt["result"]["partial_sum_lower_bound"] = 8
    assert _verify(tmp_path / "alt", alt).reward == 1.0


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
    assert _verify(tmp_path, submission).reward == 1.0


def test_accepts_equivalent_unreduced_rationals(tmp_path):
    submission = _oracle()
    submission["result"]["summand_values"][0] = {"numerator": 2, "denominator": 2}
    checkpoint = submission["result"]["truncated_checkpoints"][0]["value"]
    checkpoint["numerator"] *= 2
    checkpoint["denominator"] *= 2
    assert _verify(tmp_path, submission).reward == 1.0
