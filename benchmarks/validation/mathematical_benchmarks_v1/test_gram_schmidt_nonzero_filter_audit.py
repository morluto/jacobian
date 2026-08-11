import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1.support import _run_verifier

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
    (app / "evidence").mkdir(parents=True)
    logs.mkdir()
    shutil.copy2(task / "environment/input.json", app / "input.json")
    evidence = {
        "schema_version": "1",
        "task_id": submission["task_id"],
        "result": submission["result"],
        "limitations": submission["limitations"],
    }
    p = app / "evidence/gram-schmidt-audit.json"
    p.write_text(json.dumps(evidence, separators=(",", ":")))
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission))
    return _run_verifier(task, app, logs)


def test_oracle_and_orthogonally_transformed_system(tmp_path):
    assert verify(tmp_path / "oracle", oracle()).reward == 1.0
    alt = oracle()
    for v in alt["result"]["vectors"]:
        v[0] *= -1
    for v in alt["result"]["residuals"]:
        v[0]["numerator"] *= -1
    assert verify(tmp_path / "alt", alt).reward == 1.0


def test_residual_rank_and_assurance_attacks_fail(tmp_path):
    bad = oracle()
    bad["result"]["residuals"][3][0] = {"numerator": 0, "denominator": 1}
    assert verify(tmp_path / "residual", bad).reward == 0
    rank = oracle()
    rank["result"]["vectors"][3] = rank["result"]["vectors"][2]
    assert verify(tmp_path / "rank", rank).reward == 0
    assurance = oracle()
    assurance["claimed_assurance"] = "VERIFIED"
    assert verify(tmp_path / "assurance", assurance).reward == 0


def test_malformed_boolean_and_input_tamper_fail_closed(tmp_path):
    bad = oracle()
    bad["result"]["vectors"][0][0] = True
    assert verify(tmp_path / "boolean", bad).reward == 0
    task = Path("benchmarks/datasets/mathematical-benchmarks-v1") / TASK
    app, logs = tmp_path / "tamper/app", tmp_path / "tamper/logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    (app / "input.json").write_text("{}")
    (app / "submission.json").write_text(json.dumps(oracle()))
    assert _run_verifier(task, app, logs).reward == 0
