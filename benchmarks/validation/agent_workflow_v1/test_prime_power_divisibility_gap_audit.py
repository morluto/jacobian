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


def test_oracle_and_alternative_countermodel(tmp_path: Path) -> None:
    assert _verify(tmp_path / "oracle", _oracle())["reward"] == 1.0
    alternative = copy.deepcopy(_oracle())
    alternative["result"].update(
        {
            "prime": 3,
            "exponent": 2,
            "coprime_factor": 2,
            "modulus": 18,
            "cycle_count": 6,
            "cycle_groups": [
                {"multiplicity": 5, "cycle_sum": 3},
                {"multiplicity": 1, "cycle_sum": 6},
            ],
            "total_sum": 21,
            "p_valuation_modulus": 2,
            "p_valuation_total": 1,
        }
    )
    assert _verify(tmp_path / "alternative", alternative)["reward"] == 1.0


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
