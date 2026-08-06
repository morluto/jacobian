from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = (
    Path(__file__).parents[2]
    / "datasets"
    / "mathematical-benchmarks-v1"
    / "multiplicative-grid-extremum"
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _case(tmp_path: Path) -> tuple[Path, Path, dict]:
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir()
    logs.mkdir()
    shutil.copy2(TASK / "environment" / "input.json", app / "input.json")
    subprocess.run(
        [sys.executable, str(TASK / "solution" / "oracle.py"), str(app)],
        check=True,
    )
    return app, logs, json.loads((app / "submission.json").read_text())


def _bind(app: Path, submission: dict) -> None:
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text(
        "RESULT_JSON: "
        + json.dumps(submission["result"], sort_keys=True, separators=(",", ":"))
        + "\nThe witness and projection data are replayed independently.\n",
        encoding="utf-8",
    )
    submission["evidence"] = [
        {"path": "evidence/answer.txt", "sha256": _digest(evidence)}
    ]
    _write_json(app / "submission.json", submission)


def _run(app: Path, logs: Path) -> dict:
    return support._run_verifier(TASK, app, logs)


def test_oracle_receives_full_reward(tmp_path: Path) -> None:
    app, logs, _ = _case(tmp_path)
    result = _run(app, logs)
    assert result["reward"] == 1.0
    assert all(
        result[key] == 1.0
        for key in (
            "protocol_compliance",
            "correctness",
            "evidence_validity",
            "scope_accuracy",
            "assurance_calibration",
        )
    )


def test_alternative_coprime_rescaling_is_accepted(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["numbers"] = [
        5 * number for number in submission["result"]["numbers"]
    ]
    for factor in submission["result"]["factorizations"]:
        factor["core"] = 5
    _bind(app, submission)
    assert _run(app, logs)["reward"] == 1.0


def test_wrong_or_incomplete_edge_sets_are_rejected(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["good_pairs"].pop()
    _bind(app, submission)
    assert _run(app, logs)["correctness"] == 0.0


def test_wrong_factorization_is_rejected(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["factorizations"][0]["core"] = 7
    _bind(app, submission)
    assert _run(app, logs)["correctness"] == 0.0


def test_wrong_projection_summary_is_rejected(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["projection_summary"]["nonempty_rows"] = 9
    _bind(app, submission)
    assert _run(app, logs)["correctness"] == 0.0


def test_bool_integer_attack_is_rejected(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["numbers"][0] = True
    _bind(app, submission)
    assert _run(app, logs)["correctness"] == 0.0


def test_unrelated_evidence_preserves_math_dimension(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text("RESULT_JSON: {}\nunrelated\n", encoding="utf-8")
    submission["evidence"][0]["sha256"] = _digest(evidence)
    _write_json(app / "submission.json", submission)
    result = _run(app, logs)
    assert result["correctness"] == 1.0
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_wrong_digest_is_rejected(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["evidence"][0]["sha256"] = "sha256:" + "0" * 64
    _write_json(app / "submission.json", submission)
    assert _run(app, logs)["evidence_validity"] == 0.0


def test_scope_attack_preserves_correctness(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["scope"] = "some integers"
    _write_json(app / "submission.json", submission)
    result = _run(app, logs)
    assert result["correctness"] == 0.0
    assert result["scope_accuracy"] == 0.0
    assert result["reward"] == 0.0


def test_unverified_preserves_math_but_not_full_reward(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["claimed_assurance"] = "UNVERIFIED"
    _write_json(app / "submission.json", submission)
    result = _run(app, logs)
    assert result["correctness"] == 1.0
    assert result["evidence_validity"] == 1.0
    assert result["assurance_calibration"] == 0.0
    assert result["reward"] == 0.0


def test_false_verified_claim_is_rejected(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["claimed_assurance"] = "VERIFIED"
    _write_json(app / "submission.json", submission)
    result = _run(app, logs)
    assert result["correctness"] == 1.0
    assert result["false_certification"] is True
    assert result["reward"] == 0.0


def test_tampered_input_and_malformed_submission_fail_closed(tmp_path: Path) -> None:
    app, logs, _ = _case(tmp_path)
    source = json.loads((app / "input.json").read_text())
    source["claimed_maximum"] = 181
    _write_json(app / "input.json", source)
    assert _run(app, logs)["reward"] == 0.0

    (app / "submission.json").write_text("{", encoding="utf-8")
    result = _run(app, logs)
    assert result["reward"] == 0.0
    assert (logs / "reward.json").is_file()


def test_oversized_evidence_is_rejected_deterministically(tmp_path: Path) -> None:
    """A digest-bound oversized evidence file must yield reward 0 with reward.json."""

    app, logs, submission = _case(tmp_path)
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text(
        "RESULT_JSON: "
        + json.dumps(submission["result"], sort_keys=True, separators=(",", ":"))
        + "\nThe witness and projection data are replayed independently.\n",
        encoding="utf-8",
    )
    # Grow the file past the verifier evidence bound (16 MiB) without
    # allocating disk: a sparse hole reports the oversized st_size that
    # is_regular_bounded_file must reject before any read or hash.
    import os

    os.truncate(evidence, 20 * 1024 * 1024)
    submission["evidence"] = [
        {"path": "evidence/answer.txt", "sha256": _digest(evidence)}
    ]
    _write_json(app / "submission.json", submission)
    result = _run(app, logs)
    assert result["reward"] == 0.0
    assert result["evidence_validity"] == 0.0
    assert (logs / "reward.json").is_file()


def test_bool_summary_value_is_rejected(tmp_path: Path) -> None:
    """projection_summary.component_count: true must not equal 1 for reward."""

    app, logs, submission = _case(tmp_path)
    submission["result"]["projection_summary"]["component_count"] = True
    _bind(app, submission)
    result = _run(app, logs)
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_float_claimed_maximum_is_rejected(tmp_path: Path) -> None:
    """claimed_maximum: 180.0 must not equal 180 for reward."""

    app, logs, submission = _case(tmp_path)
    submission["result"]["claimed_maximum"] = 180.0
    _bind(app, submission)
    result = _run(app, logs)
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_evidence_marker_is_sufficient(tmp_path: Path) -> None:
    """The contract does not score arbitrary prose beyond the bound marker."""

    app, logs, submission = _case(tmp_path)
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text(
        "RESULT_JSON: "
        + json.dumps(submission["result"], sort_keys=True, separators=(",", ":"))
        + "\nx\n",
        encoding="utf-8",
    )
    submission["evidence"] = [
        {"path": "evidence/answer.txt", "sha256": _digest(evidence)}
    ]
    _write_json(app / "submission.json", submission)
    result = _run(app, logs)
    assert result["evidence_validity"] == 1.0
    assert result["correctness"] == 1.0
    assert result["reward"] == 1.0


def test_envelope_defect_preserves_correctness_dimension(tmp_path: Path) -> None:
    """An extra top-level field must zero protocol but not correctness."""

    app, logs, submission = _case(tmp_path)
    _bind(app, submission)
    submission["unexpected_field"] = "extra"
    _write_json(app / "submission.json", submission)
    result = _run(app, logs)
    assert result["protocol_compliance"] == 0.0
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_result_shape_defect_lowers_protocol_compliance(tmp_path: Path) -> None:
    """A bool in numbers must zero protocol_compliance while math is also wrong."""

    app, logs, submission = _case(tmp_path)
    submission["result"]["numbers"][0] = True
    _bind(app, submission)
    result = _run(app, logs)
    assert result["protocol_compliance"] == 0.0
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_deeply_nested_evidence_json_does_not_crash(tmp_path: Path) -> None:
    """A RecursionError in evidence JSON parsing must not crash the verifier."""

    app, logs, submission = _case(tmp_path)
    evidence = app / "evidence" / "answer.txt"
    nested = "[" * 200 + "]" * 200
    evidence.write_text(
        "RESULT_JSON: " + nested + "\nThe witness is replayed independently.\n",
        encoding="utf-8",
    )
    submission["evidence"] = [
        {"path": "evidence/answer.txt", "sha256": _digest(evidence)}
    ]
    _write_json(app / "submission.json", submission)
    result = _run(app, logs)
    assert result["evidence_validity"] == 0.0
    assert result["correctness"] == 1.0
    assert result["reward"] == 0.0
    assert (logs / "reward.json").is_file()


def test_type_mismatched_evidence_json_is_rejected(tmp_path: Path) -> None:
    """Evidence JSON must not coerce booleans or integral floats to integers."""

    app, logs, submission = _case(tmp_path)
    evidence = app / "evidence" / "answer.txt"
    evidence_result = copy.deepcopy(submission["result"])
    evidence_result["projection_summary"]["component_count"] = True
    evidence.write_text(
        "RESULT_JSON: "
        + json.dumps(evidence_result, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    submission["evidence"] = [
        {"path": "evidence/answer.txt", "sha256": _digest(evidence)}
    ]
    _write_json(app / "submission.json", submission)
    result = _run(app, logs)
    assert result["correctness"] == 1.0
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_duplicate_pairs_lower_protocol_compliance(tmp_path: Path) -> None:
    """A duplicate pair is schema-invalid even when the mathematical result fails."""

    app, logs, submission = _case(tmp_path)
    submission["result"]["good_pairs"][1] = submission["result"]["good_pairs"][0]
    _bind(app, submission)
    result = _run(app, logs)
    assert result["protocol_compliance"] == 0.0
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_envelope_defect_preserves_assurance_calibration(tmp_path: Path) -> None:
    """A protocol defect does not make a correctly calibrated claim incorrect."""

    app, logs, submission = _case(tmp_path)
    submission["unexpected_field"] = "extra"
    _write_json(app / "submission.json", submission)
    result = _run(app, logs)
    assert result["protocol_compliance"] == 0.0
    assert result["assurance_calibration"] == 0.0
    assert result["reward"] == 0.0
