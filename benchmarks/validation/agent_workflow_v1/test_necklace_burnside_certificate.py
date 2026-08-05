import importlib.util
import json
import sys
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "benchmarks/datasets/agent-workflow-v1/necklace-burnside-certificate"
TASK_NAME = "necklace-burnside-certificate"

LIMITATIONS = ["FINITE_LENGTH_16_INSTANCE", "NO_GENERAL_ENUMERATION_THEOREM"]


def load_verifier():
    sys.path.insert(0, str(TASK / "tests"))
    spec = importlib.util.spec_from_file_location(
        "necklace_verifier", TASK / "tests/verifier.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK_NAME, "computed")


def _evidence_object(result: dict) -> dict:
    return {
        "schema_version": "1",
        "task_id": "jacobian/necklace-burnside-certificate",
        "result": result,
        "limitations": LIMITATIONS,
    }


def _rewrite(app: Path, submission: dict, result: dict | None = None) -> None:
    """Rebind the JSON evidence object and submission digest in concert."""
    bound_result = result if result is not None else submission["result"]
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(
        json.dumps(
            _evidence_object(bound_result), sort_keys=True, separators=(",", ":")
        )
    )
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)


def test_independent_orbit_derivation():
    result = load_verifier().derive()
    assert result["valid_labelled_words"] == 2206
    assert result["burnside_numerator"] == 2816
    assert result["orbit_count"] == 88
    assert len(result["canonical_representatives"]) == 88


def test_wraparound_and_reflection_are_material():
    verifier = load_verifier()
    assert not verifier.valid(tuple(map(int, "0010101010101010")))
    result = verifier.derive()
    assert result["reflection_fixed_counts"] == [42, 26] * 8


def test_corrupt_fixed_count_or_orbit_representative_is_rejected():
    verifier = load_verifier()
    result = verifier.derive()
    result["rotation_fixed_counts"][0] -= 1
    assert not verifier.matches(result)
    result = verifier.derive()
    result["canonical_representatives"].pop()
    assert not verifier.matches(result)


def test_contract_has_no_verified_upgrade():
    contract = json.loads((TASK / "tests/public_contract.json").read_text())
    assert contract["allowed_assurance"] == ["COMPUTED"]


def test_canonical_solution_receives_full_reward(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 1.0
    assert result["evidence_validity"] == 1.0
    assert result["scope_accuracy"] == 1.0
    assert result["assurance_calibration"] == 1.0
    assert result["reward"] == 1.0
    assert result["false_certification"] is False


def test_rejects_boolean_fixed_counts(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = dict(submission["result"])
    result["rotation_fixed_counts"] = [bool(x) for x in result["rotation_fixed_counts"]]
    submission["result"] = result
    _rewrite(app, submission, result)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_float_fixed_counts(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = dict(submission["result"])
    result["rotation_fixed_counts"] = [
        float(x) for x in result["rotation_fixed_counts"]
    ]
    submission["result"] = result
    _rewrite(app, submission, result)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_assurance_failure_preserves_evidence_and_scope(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "VERIFIED"
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 1.0
    assert result["evidence_validity"] == 1.0
    assert result["scope_accuracy"] == 1.0
    assert result["assurance_calibration"] == 0.0
    assert result["reward"] == 0.0


def test_evidence_result_must_match_submission_result(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    corrupted = dict(submission["result"])
    corrupted["orbit_count"] = 999
    submission["result"] = corrupted
    _rewrite(app, submission, corrupted)

    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 0.0
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_evidence_limitations_must_match_submission(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = ["CHANGED_LIMITATION"]
    support._write_json(app / "submission.json", submission)

    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_protocol_compliance_is_reported(tmp_path: Path) -> None:
    """An extra envelope field breaks the contract; protocol_compliance
    must be 0 while math/evidence/scope/assurance remain independently
    evaluated."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["extra_field"] = "unexpected"
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["protocol_compliance"] == 0.0
    assert result["correctness"] == 0.0
    assert result["evidence_validity"] == 0.0
    assert result["scope_accuracy"] == 0.0
    assert result["reward"] == 0.0


def test_float_in_result_reports_protocol_failure(tmp_path: Path) -> None:
    """A float where an integer is required is a schema violation that
    must be reported as protocol_compliance = 0."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    corrupted = dict(submission["result"])
    corrupted["orbit_count"] = 88.0
    submission["result"] = corrupted
    _rewrite(app, submission, corrupted)
    result = support._run_verifier(task, app, logs)
    assert result["protocol_compliance"] == 0.0
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


@pytest.mark.parametrize(
    "corruption",
    (
        "empty",
        "duplicate",
        "non_binary",
    ),
)
def test_representative_schema_violations_report_protocol_failure(
    tmp_path: Path, corruption: str
) -> None:
    """The result-shape check must enforce the advertised representative
    schema: at least one entry, all unique, and every entry matching
    ^[01]{16}$.  Violations are protocol failures, not only math errors."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    corrupted = dict(submission["result"])
    reps = list(corrupted["canonical_representatives"])
    if corruption == "empty":
        corrupted["canonical_representatives"] = []
    elif corruption == "duplicate":
        corrupted["canonical_representatives"] = [reps[0], reps[0]]
    else:
        corrupted["canonical_representatives"] = ["x" * 16]
    submission["result"] = corrupted
    _rewrite(app, submission, corrupted)
    result = support._run_verifier(task, app, logs)
    assert result["protocol_compliance"] == 0.0
    assert result["reward"] == 0.0


def test_input_tamper_preserves_assurance_diagnostics(tmp_path: Path) -> None:
    """When the workspace input is altered, mathematical correctness remains
    independently evaluated (the result is still canonical) while input
    binding is reported as its own failed diagnostic; assurance/scope
    diagnostics remain independently evaluated rather than collapsing to
    zero, and aggregate reward is gated to zero."""
    task, app, logs = _case(tmp_path)
    (app / "input.json").write_text("{}")
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 1.0
    assert result["input_binding"] == 0.0
    assert result["scope_accuracy"] == 1.0
    assert result["assurance_calibration"] == 1.0
    assert result["reward"] == 0.0


def test_oversized_evidence_with_valid_digest_is_accepted(tmp_path: Path) -> None:
    """No task-local evidence-size cap: a large but well-formed and
    digest-bound evidence object with the correct result still receives
    full evidence_validity and reward. The JSON is padded with whitespace
    so the parsed object still has exactly the four required keys."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "answer.txt"
    obj = _evidence_object(submission["result"])
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    padded = payload.replace(",", ", " + " " * (2 * 1024 * 1024 // 10))
    evidence_path.write_text(padded)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 1.0
    assert result["correctness"] == 1.0
    assert result["reward"] == 1.0
