import hashlib
import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "inversion-aggregate-mask-audit"


def rewrite(app: Path, s: dict):
    e = {
        "schema_version": "1",
        "task_id": s["task_id"],
        "result": s["result"],
        "limitations": s["limitations"],
    }
    raw = json.dumps(e, separators=(",", ":")).encode()
    (app / "evidence/inversion-audit.json").write_bytes(raw)
    s["evidence"][0]["sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    support._write_json(app / "submission.json", s)


def test_accepts_alternative_witness(tmp_path: Path):
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    s = json.loads((app / "submission.json").read_text())
    s["result"].update(
        {
            "witness_permutation": [3, 2, 1, 0],
            "implemented_count": 0,
            "intended_count": 6,
        }
    )
    rewrite(app, s)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_rejects_pointwise_count_tamper(tmp_path: Path):
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    s = json.loads((app / "submission.json").read_text())
    s["result"]["implemented_count"] = 5
    rewrite(app, s)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_rejects_aggregate_only_nonwitness(tmp_path: Path):
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    s = json.loads((app / "submission.json").read_text())
    s["result"].update(
        {
            "witness_permutation": [1, 0, 3, 2],
            "implemented_count": 4,
            "intended_count": 2,
        }
    )
    rewrite(app, s)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_rejects_boolean_permutation_witness(tmp_path: Path):
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    s = json.loads((app / "submission.json").read_text())
    s["result"]["witness_permutation"] = [False, True, 2, 3]
    rewrite(app, s)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_rejects_float_result_counts(tmp_path: Path):
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    s = json.loads((app / "submission.json").read_text())
    s["result"]["implemented_count"] = 6.0
    s["result"]["implemented_aggregate"] = 72.0
    rewrite(app, s)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_verified_claim_preserves_correctness_diagnostic(tmp_path: Path):
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    s = json.loads((app / "submission.json").read_text())
    s["claimed_assurance"] = "VERIFIED"
    rewrite(app, s)
    result = support._run_verifier(task, app, logs)
    assert result["reward"] == 0.0
    assert result["correctness"] == 1.0
    assert result["false_certification"] is True


def test_rejects_oversized_evidence(tmp_path: Path):
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    (app / "evidence/inversion-audit.json").write_bytes(b"x" * (16 * 1024 * 1024 + 1))
    result = support._run_verifier(task, app, logs)
    assert result["reward"] == 0.0
