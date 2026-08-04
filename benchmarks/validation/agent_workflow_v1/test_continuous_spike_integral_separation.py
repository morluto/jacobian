import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

from benchmarks.validation.agent_workflow_v1 import support

ROOT = Path(__file__).resolve().parents[3]
TASK = (
    ROOT / "benchmarks/datasets/agent-workflow-v1/continuous-spike-integral-separation"
)
TASK_NAME = "continuous-spike-integral-separation"


def load_verifier():
    sys.path.insert(0, str(TASK / "tests"))
    spec = importlib.util.spec_from_file_location(
        "continuous_spike_verifier", TASK / "tests/verifier.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    sys.modules.pop("verifier_support", None)
    return module


def candidate(alpha=Fraction(1, 4)):
    verifier = load_verifier()
    return {
        "alpha": str(alpha),
        "baseline_power": 2,
        "spike_height": "1",
        "spikes": [verifier.expected_spike(n, alpha) for n in range(1, 13)],
        "integral_classification": {
            "spike_area_series": "alpha*sum(1/n)",
            "status": "DIVERGENT",
        },
        "sample_series_classification": {
            "sample_series": "sum(1/n^2)",
            "status": "CONVERGENT",
        },
    }


def test_accepts_alternative_width_scales():
    verifier = load_verifier()
    assert verifier.valid_result(candidate(Fraction(1, 4)))
    assert verifier.valid_result(candidate(Fraction(1, 7)))


def test_rejects_width_touching_integer_and_corrupt_area():
    verifier = load_verifier()
    assert not verifier.valid_result(candidate(Fraction(1, 2)))
    bad = candidate()
    bad["spikes"][6]["area"] = "1"
    assert not verifier.valid_result(bad)


def test_rejects_finite_or_reversed_classification():
    verifier = load_verifier()
    bad = candidate()
    bad["spikes"] = bad["spikes"][:-1]
    assert not verifier.valid_result(bad)
    bad = candidate()
    bad["sample_series_classification"]["status"] = "DIVERGENT"
    assert not verifier.valid_result(bad)


def test_contract_allows_unverified_and_computed():
    schema = json.loads((TASK / "environment/submission_schema.json").read_text())
    assert schema["properties"]["claimed_assurance"] == {
        "enum": ["UNVERIFIED", "COMPUTED"]
    }


def test_rejects_boolean_spike_index():
    verifier = load_verifier()
    bad = candidate()
    bad["spikes"][0]["n"] = True
    assert not verifier.valid_result(bad)


def test_accepts_permuted_spike_order():
    verifier = load_verifier()
    good = candidate()
    good["spikes"] = list(reversed(good["spikes"]))
    assert verifier.valid_result(good)


def test_rejects_rational_exceeding_published_bound():
    verifier = load_verifier()
    bad = candidate()
    bad["alpha"] = "1/" + "0" * 100
    assert not verifier.valid_result(bad)


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK_NAME, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    support._write_json(app / "submission.json", submission)


def test_canonical_computed_submission_passes(tmp_path: Path):
    task, app, logs = _case(tmp_path)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 1.0
    assert result["evidence_validity"] == 1.0
    assert result["scope_accuracy"] == 1.0
    assert result["assurance_calibration"] == 1.0
    assert result["reward"] == 1.0
    assert result["false_certification"] is False


def test_overclaimed_verified_keeps_correctness_decoupled(tmp_path: Path):
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "VERIFIED"
    submission["verification_record_uri"] = {
        "path": "evidence/verification-record.json",
        "sha256": support._digest(app / "evidence" / "answer.txt"),
    }
    _rewrite(app, submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 1.0
    assert result["evidence_validity"] == 1.0
    assert result["scope_accuracy"] == 1.0
    assert result["assurance_calibration"] == 0.0
    assert result["false_certification"] is True
    assert result["reward"] == 0.0


def test_unverified_assurance_passes_with_partial_credit(tmp_path: Path):
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "UNVERIFIED"
    _rewrite(app, submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 1.0
    assert result["evidence_validity"] == 1.0
    assert result["scope_accuracy"] == 1.0
    assert result["assurance_calibration"] == 1.0
    assert result["reward"] == 1.0
    assert result["false_certification"] is False


def test_rejects_empty_evidence(tmp_path: Path):
    task, app, logs = _case(tmp_path)
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text("")
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    _rewrite(app, submission)
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_rejects_unrelated_evidence(tmp_path: Path):
    task, app, logs = _case(tmp_path)
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text("lorem ipsum dolor sit amet\n")
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    _rewrite(app, submission)
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0
