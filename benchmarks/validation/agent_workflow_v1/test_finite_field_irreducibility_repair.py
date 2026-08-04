from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.agent_workflow_v1 import support

TASK = "finite-field-irreducibility-repair"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def test_reference_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def _reject(tmp_path: Path, mutate) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    mutate(submission)
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_wrong_bad_factor_is_rejected(tmp_path: Path) -> None:
    _reject(tmp_path, lambda s: s["result"].__setitem__("bad_factor", [0, 1]))


def test_corrupt_p2_remainder_is_rejected(tmp_path: Path) -> None:
    _reject(tmp_path, lambda s: s["result"]["p2_remainder"].__setitem__(0, 0))


def test_reducible_repair_prime_is_rejected(tmp_path: Path) -> None:
    _reject(tmp_path, lambda s: s["result"].__setitem__("repair_prime", 3))


def test_false_verified_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["claimed_assurance"] = "VERIFIED"
    support._write_json(path, submission)
    result = support._run_verifier(task, app, logs)
    assert result["reward"] == 0.0
    assert result["false_certification"] is True


def test_false_verified_keeps_other_diagnostics_independent(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["claimed_assurance"] = "VERIFIED"
    support._write_json(path, submission)
    result = support._run_verifier(task, app, logs)
    assert result["reward"] == 0.0
    assert result["correctness"] == 1.0
    assert result["evidence_validity"] == 1.0
    assert result["scope_accuracy"] == 1.0
    assert result["assurance_calibration"] == 0.0


def test_boolean_fixed_coefficients_are_rejected(tmp_path: Path) -> None:
    _reject(
        tmp_path,
        lambda s: s["result"].update(
            bad_reduction=[True, False, False, False, True],
            bad_factor=[True, True],
            rabin_gcd_degree=False,
        ),
    )


def test_unrelated_explanation_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    evidence = app / "evidence" / "answer.txt"
    marker = next(
        line
        for line in evidence.read_text().splitlines()
        if line.startswith("RESULT_JSON:")
    )
    evidence.write_text(
        ("reducible rabin gcd rational density " * 30) + "\n" + marker + "\n"
    )
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_density_limitation_is_exact(tmp_path: Path) -> None:
    _reject(
        tmp_path,
        lambda s: s.__setitem__("limitations", ["The density is not purple."]),
    )


def test_visible_input_tampering_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    data = json.loads((app / "input.json").read_text())
    data["source"]["row"] = 3
    support._write_json(app / "input.json", data)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0
