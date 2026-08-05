import copy
import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.validation.agent_workflow_v1.support import _run_verifier

TASK = "prime-power-divisibility-gap-audit"


def _oracle() -> dict[str, object]:
    return json.loads(
        (
            Path("benchmarks/datasets/agent-workflow-v1")
            / TASK
            / "solution/submission.json"
        ).read_text()
    )


def _prepare(tmp_path: Path, submission: dict[str, object]):
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
    evidence_path = app / "evidence/divisibility-audit.json"
    evidence_path.write_text(json.dumps(evidence, separators=(",", ":")))
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission))
    return task, app, logs


def _verify(tmp_path: Path, submission: dict[str, object]):
    return _run_verifier(*_prepare(tmp_path, submission))


def _integral_json_numbers(value: object) -> object:
    if type(value) is int:
        return float(value)
    if isinstance(value, list):
        return [_integral_json_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: _integral_json_numbers(item) for key, item in value.items()}
    return value


def test_oracle_and_alternative_countermodel(tmp_path: Path) -> None:
    assert _verify(tmp_path / "oracle", _oracle())["reward"] == 1.0
    alternative = copy.deepcopy(_oracle())
    alternative["result"].update(
        {
            "prime": 3,
            "exponent": 3,
            "coprime_factor": 1,
            "modulus": 27,
            "cycle_count": 9,
            "cycle_groups": [
                {"multiplicity": 8, "cycle_sum": 3},
                {"multiplicity": 1, "cycle_sum": 6},
            ],
            "total_sum": 30,
            "p_valuation_modulus": 3,
            "p_valuation_total": 1,
        }
    )
    assert _verify(tmp_path / "alternative", alternative)["reward"] == 1.0
    reversed_groups = copy.deepcopy(_oracle())
    reversed_groups["result"]["cycle_groups"].reverse()
    assert _verify(tmp_path / "reversed", reversed_groups)["reward"] == 1.0


def test_rejects_corrupt_prime_power_and_cycle_ledger(tmp_path: Path) -> None:
    for name, mutation in [
        ("valuation", lambda result: result.update(p_valuation_total=2)),
        (
            "multiplicity",
            lambda result: result["cycle_groups"][0].update(multiplicity=10),
        ),
        ("local", lambda result: result["cycle_groups"][1].update(cycle_sum=3)),
    ]:
        submission = copy.deepcopy(_oracle())
        mutation(submission["result"])
        assert _verify(tmp_path / name, submission)["reward"] == 0.0


def test_rejects_tiny_or_false_certification_shortcuts(tmp_path: Path) -> None:
    tiny = copy.deepcopy(_oracle())
    tiny["result"].update(
        {
            "prime": 2,
            "exponent": 2,
            "coprime_factor": 1,
            "modulus": 4,
            "cycle_count": 2,
            "cycle_groups": [
                {"multiplicity": 1, "cycle_sum": 2},
                {"multiplicity": 1, "cycle_sum": 4},
            ],
            "total_sum": 6,
            "p_valuation_modulus": 2,
            "p_valuation_total": 1,
        }
    )
    assert _verify(tmp_path / "tiny", tiny)["reward"] == 0.0
    verified = copy.deepcopy(_oracle())
    verified["claimed_assurance"] = "VERIFIED"
    assert _verify(tmp_path / "verified", verified)["false_certification"] is True


def test_schema_bypass_values_fail_closed(tmp_path: Path) -> None:
    for name, mutation in [
        ("zero-total", lambda result: result.update(total_sum=0)),
        ("boolean-valuation", lambda result: result.update(p_valuation_total=True)),
        ("large-prime", lambda result: result.update(prime=31)),
        ("coprime-factor", lambda result: result.update(coprime_factor=3)),
        (
            "zero-cycle-sum",
            lambda result: result.update(
                cycle_groups=[
                    {"multiplicity": 3, "cycle_sum": 0},
                    {"multiplicity": 1, "cycle_sum": 2},
                ],
                total_sum=2,
            ),
        ),
    ]:
        submission = copy.deepcopy(_oracle())
        mutation(submission["result"])
        reward = _verify(tmp_path / name, submission)
        assert reward["correctness"] == 0.0
        assert reward["reward"] == 0.0


def test_accepts_schema_valid_integral_json_numbers(tmp_path: Path) -> None:
    submission = copy.deepcopy(_oracle())
    submission["result"] = _integral_json_numbers(submission["result"])

    reward = _verify(tmp_path, submission)

    assert reward["correctness"] == 1.0
    assert reward["evidence_validity"] == 1.0
    assert reward["reward"] == 1.0


def test_rejects_decimal_token_rounded_by_binary_float(tmp_path: Path) -> None:
    submission = copy.deepcopy(_oracle())
    task, app, logs = _prepare(tmp_path, submission)
    evidence_path = app / "evidence/divisibility-audit.json"
    evidence_text = evidence_path.read_text().replace(
        '"prime":2', '"prime":2.0000000000000001', 1
    )
    evidence_path.write_text(evidence_text)
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    )
    submission_text = json.dumps(submission).replace(
        '"prime": 2', '"prime": 2.0000000000000001', 1
    )
    (app / "submission.json").write_text(submission_text)

    reward = _run_verifier(task, app, logs)
    assert reward["correctness"] == 0.0
    assert reward["reward"] == 0.0


def test_protocol_and_input_failures_preserve_diagnostics(tmp_path: Path) -> None:
    bad_conclusion = copy.deepcopy(_oracle())
    bad_conclusion["conclusion"] = "WRONG"
    reward = _verify(tmp_path / "protocol", bad_conclusion)
    assert reward["correctness"] == 1.0
    assert reward["evidence_validity"] == 1.0
    assert reward["scope_accuracy"] == 1.0
    assert reward["assurance_calibration"] == 1.0
    assert reward["protocol_compliance"] == 0.0
    assert reward["reward"] == 0.0

    bad_assurance = copy.deepcopy(_oracle())
    bad_assurance["claimed_assurance"] = "UNVERIFIED"
    reward = _verify(tmp_path / "assurance", bad_assurance)
    assert reward["correctness"] == 1.0
    assert reward["evidence_validity"] == 1.0
    assert reward["scope_accuracy"] == 1.0
    assert reward["assurance_calibration"] == 0.0
    assert reward["protocol_compliance"] == 1.0
    assert reward["reward"] == 0.0

    submission = copy.deepcopy(_oracle())
    task, app, logs = _prepare(tmp_path / "input", submission)
    (app / "input.json").write_text("{}")
    reward = _run_verifier(task, app, logs)
    assert reward["input_binding"] == 0.0
    assert reward["correctness"] == 1.0
    assert reward["reward"] == 0.0


def test_malformed_assurance_preserves_scope_diagnostic(tmp_path: Path) -> None:
    submission = copy.deepcopy(_oracle())
    submission["claimed_assurance"] = []

    reward = _verify(tmp_path, submission)

    assert reward["protocol_compliance"] == 0.0
    assert reward["correctness"] == 1.0
    assert reward["evidence_validity"] == 1.0
    assert reward["scope_accuracy"] == 1.0
    assert reward["assurance_calibration"] == 0.0
    assert reward["reward"] == 0.0
    assert reward["false_certification"] is False


def test_evidence_comparison_uses_exact_json_types(tmp_path: Path) -> None:
    submission = copy.deepcopy(_oracle())
    task, app, logs = _prepare(tmp_path, submission)
    evidence_path = app / "evidence/divisibility-audit.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["result"]["p_valuation_total"] = True
    evidence_path.write_text(json.dumps(evidence, separators=(",", ":")))
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission))

    reward = _run_verifier(task, app, logs)
    assert reward["correctness"] == 1.0
    assert reward["evidence_validity"] == 0.0
    assert reward["reward"] == 0.0


def test_evidence_is_bound_to_the_expected_task(tmp_path: Path) -> None:
    submission = copy.deepcopy(_oracle())
    submission["task_id"] = "jacobian/other-task"
    reward = _verify(tmp_path, submission)
    assert reward["correctness"] == 1.0
    assert reward["evidence_validity"] == 0.0
    assert reward["protocol_compliance"] == 0.0
    assert reward["reward"] == 0.0


def test_evidence_schema_version_is_public_and_required(tmp_path: Path) -> None:
    contract = json.loads(
        (
            Path("benchmarks/datasets/agent-workflow-v1")
            / TASK
            / "tests/public_contract.json"
        ).read_text()
    )
    envelope = contract["schema_definitions"]["evidence_envelope"]
    assert envelope["properties"]["schema_version"] == {"const": "1"}
    assert "schema_version" in envelope["required"]

    submission = copy.deepcopy(_oracle())
    task, app, logs = _prepare(tmp_path, submission)
    evidence_path = app / "evidence/divisibility-audit.json"
    evidence = json.loads(evidence_path.read_text())
    evidence.pop("schema_version")
    evidence_path.write_text(json.dumps(evidence, separators=(",", ":")))
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission))

    reward = _run_verifier(task, app, logs)
    assert reward["correctness"] == 1.0
    assert reward["evidence_validity"] == 0.0
    assert reward["protocol_compliance"] == 1.0
    assert reward["reward"] == 0.0


def test_large_valid_evidence_remains_admissible(tmp_path: Path) -> None:
    submission = copy.deepcopy(_oracle())
    task, app, logs = _prepare(tmp_path, submission)
    evidence_path = app / "evidence/divisibility-audit.json"
    evidence = evidence_path.read_text()
    evidence_path.write_text(" " * 8192 + evidence + "\n")
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission))

    reward = _run_verifier(task, app, logs)
    assert reward["evidence_validity"] == 1.0
    assert reward["reward"] == 1.0


def test_rejects_extra_limitation_even_when_evidence_repeats_it(tmp_path: Path) -> None:
    submission = copy.deepcopy(_oracle())
    submission["limitations"].append("EXTRA")
    reward = _verify(tmp_path, submission)
    assert reward["correctness"] == 1.0
    assert reward["evidence_validity"] == 1.0
    assert reward["limitations_accuracy"] == 0.0
    assert reward["protocol_compliance"] == 0.0
    assert reward["reward"] == 0.0
