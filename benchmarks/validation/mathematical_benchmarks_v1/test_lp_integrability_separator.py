import importlib.util
import json
import sys
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import support

ROOT = Path(__file__).resolve().parents[3]
TASK = (
    ROOT / "benchmarks/datasets/mathematical-benchmarks-v1/lp-integrability-separator"
)


def load_verifier():
    sys.path.insert(0, str(TASK / "tests"))
    spec = importlib.util.spec_from_file_location(
        "lp_integrability_verifier", TASK / "tests/verifier.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def result(beta="1", log_exponent="-2", integral="1"):
    return {
        "beta": beta,
        "origin_power_coefficient": "-1/2",
        "infinity_power_coefficient": "-1/2",
        "p2_log_exponent": log_exponent,
        "p2_integral_each": integral,
        "critical_p": "2",
        "lower_regime": {
            "p_interval": "0<p<2",
            "obstruction": "INFINITY_POWER_TAIL",
        },
        "upper_regime": {
            "p_interval": "p>2",
            "obstruction": "ORIGIN_POWER_SINGULARITY",
        },
    }


def test_accepts_alternative_rational_parameter():
    assert load_verifier().valid_result(result("3/4", "-3/2", "2"))


def test_rejects_boundary_and_corrupt_integral():
    verifier = load_verifier()
    assert not verifier.valid_result(result("1/2", "-1", "1"))
    assert not verifier.valid_result(result("3/4", "-3/2", "3"))


def test_rejects_wrong_tail_obstruction():
    candidate = result()
    candidate["lower_regime"]["obstruction"] = "ORIGIN_POWER_SINGULARITY"
    assert not load_verifier().valid_result(candidate)


def test_contract_has_computed_ceiling():
    contract = json.loads((TASK / "tests/public_contract.json").read_text())
    assert contract["allowed_assurance"] == ["COMPUTED"]


# --- T3: accept mathematically equivalent rationals ---


def test_accepts_noncanonical_rational_representations():
    verifier = load_verifier()
    assert verifier.valid_result(result("1/1", "-2/1", "1/1"))
    assert verifier.valid_result(result("2/2", "-4/2", "2/2"))


# --- T1: assurance failures must not collapse correctness ---


def test_valid_result_independent_of_assurance():
    """valid_result must pass for a mathematically correct result even if the
    submission claims VERIFIED (which the contract rejects)."""
    verifier = load_verifier()
    assert verifier.valid_result(result())


# --- T2: evidence content validation ---


def _evidence_case(tmp_path: Path, text: str):
    """Prepare a computed case with a custom evidence body and return the
    verifier result, so evidence resolution goes through the /app -> tmp_path
    mapping used by support._run_verifier."""
    task, app, logs = support._prepare_case(
        tmp_path, "lp-integrability-separator", "computed"
    )
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(text)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    return support._run_verifier(task, app, logs)


def test_valid_evidence_rejects_empty_text(tmp_path: Path):
    result = _evidence_case(tmp_path, "")
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_valid_evidence_rejects_unrelated_text(tmp_path: Path):
    result = _evidence_case(tmp_path, "hello world\n")
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_valid_evidence_rejects_missing_result_json_marker(tmp_path: Path):
    text = (
        "At p=2 the substitution gives the integral. "
        "The origin diverges for p>2 and infinity diverges for 0<p<2. "
        "The log factor controls integrability.\n"
    )
    result = _evidence_case(tmp_path, text)
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_valid_evidence_rejects_mismatched_result_json(tmp_path: Path):
    wrong = result("3/4", "-3/2", "2")
    text = (
        "RESULT_JSON:" + json.dumps(wrong) + "\n"
        "At p=2 the substitution gives the integral. "
        "The origin diverges for p>2 and infinity diverges for 0<p<2. "
        "The log factor controls integrability.\n"
    )
    res = _evidence_case(tmp_path, text)
    assert res["evidence_validity"] == 0.0
    assert res["reward"] == 0.0


def test_valid_evidence_accepts_bound_result(tmp_path: Path):
    task, app, logs = support._prepare_case(
        tmp_path, "lp-integrability-separator", "computed"
    )
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 1.0
    assert result["reward"] == 1.0


def test_assurance_failure_preserves_evidence_and_scope(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "lp-integrability-separator", "computed"
    )
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "VERIFIED"
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 1.0
    assert result["evidence_validity"] == 1.0
    assert result["scope_accuracy"] == 1.0
    assert result["assurance_calibration"] == 0.0
    assert result["reward"] == 0.0


def test_rational_field_max_length_is_documented():
    schema = json.loads((TASK / "environment/submission_schema.json").read_text())
    for field in ("beta", "p2_log_exponent", "p2_integral_each"):
        assert schema["properties"]["result"]["properties"][field]["maxLength"] == 80
    text = (TASK / "instruction.md").read_text().casefold()
    assert "80 characters" in text
