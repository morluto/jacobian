import copy
import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.validation.agent_workflow_v1.support import _run_verifier

TASK = "valuation-gcd-quantifier-audit"


def _oracle():
    return json.loads(
        (
            Path("benchmarks/datasets/agent-workflow-v1")
            / TASK
            / "solution/submission.json"
        ).read_text()
    )


def _prepare(tmp_path, submission):
    task = Path("benchmarks/datasets/agent-workflow-v1") / TASK
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
    evidence_path = app / "evidence/valuation-audit.json"
    evidence_path.write_text(json.dumps(evidence, separators=(",", ":")))
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission))
    return task, app, logs


def _verify(tmp_path, submission):
    return _run_verifier(*_prepare(tmp_path, submission))


def test_oracle_and_alternative_repair(tmp_path):
    assert _verify(tmp_path / "oracle", _oracle())["reward"] == 1.0
    alt = _oracle()
    alt["result"]["repair"] = [
        {"prime": 2, "exponents": [0, 1, 1, 1]},
        {"prime": 7, "exponents": [1, 0, 1, 1]},
        {"prime": 11, "exponents": [1, 1, 0, 1]},
    ]
    assert _verify(tmp_path / "alt", alt)["reward"] == 1.0


def test_rejects_weak_countermodel_and_false_assurance(tmp_path):
    for name, mutate in [
        ("counter", lambda s: s["result"].update(countermodel=s["result"]["repair"])),
        ("assurance", lambda s: s.update(claimed_assurance="VERIFIED")),
    ]:
        submission = copy.deepcopy(_oracle())
        mutate(submission)
        assert _verify(tmp_path / name, submission)["reward"] == 0


def test_rejects_zero_row_that_is_not_a_prime_factor(tmp_path):
    submission = copy.deepcopy(_oracle())
    submission["result"]["countermodel"] = [
        {"prime": 2, "exponents": [0, 0, 0, 0]},
        {"prime": 3, "exponents": [1, 1, 1, 1]},
        {"prime": 5, "exponents": [2, 2, 2, 2]},
    ]
    reward = _verify(tmp_path, submission)
    assert reward["correctness"] == 0.0
    assert reward["reward"] == 0


def test_rejects_out_of_bound_prime_without_crashing(tmp_path):
    for name, prime in [("schema-bound", 101), ("huge", 10**4000)]:
        submission = copy.deepcopy(_oracle())
        submission["result"]["countermodel"][0]["prime"] = prime
        reward = _verify(tmp_path / name, submission)
        assert reward["correctness"] == 0.0
        assert reward["reward"] == 0
        assert (tmp_path / name / "logs" / "reward.json").is_file()


def test_evidence_result_requires_exact_json_types(tmp_path):
    submission = copy.deepcopy(_oracle())
    task, app, logs = _prepare(tmp_path, submission)
    evidence_path = app / "evidence/valuation-audit.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["result"]["countermodel"][0]["exponents"][0] = True
    evidence_path.write_text(json.dumps(evidence, separators=(",", ":")))
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission))
    reward = _run_verifier(task, app, logs)
    assert reward["correctness"] == 1.0
    assert reward["evidence_validity"] == 0.0
    assert reward["reward"] == 0


def test_math_diagnostic_survives_envelope_failures(tmp_path):
    for name, mutate, expected_correctness in [
        ("extra-field", lambda s: s.update(extra=True), 0.0),
        (
            "conclusion",
            lambda s: s.update(conclusion="INSUFFICIENT_EVIDENCE"),
            1.0,
        ),
        ("completeness", lambda s: s.update(completeness="UNKNOWN"), 1.0),
    ]:
        submission = copy.deepcopy(_oracle())
        mutate(submission)
        reward = _verify(tmp_path / name, submission)
        assert reward["correctness"] == expected_correctness, name
        assert reward["reward"] == 0
