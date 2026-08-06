from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "emerald-path-family-audit"


def _load(app: Path) -> dict[str, object]:
    return json.loads((app / "submission.json").read_text())


def _bind_evidence(app: Path, submission: dict[str, object]) -> None:
    result = submission["result"]
    digest = hashlib.sha256(
        json.dumps(result["trace"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    text = (
        "\n".join(
            [
                "emerald-path-family-certificate-v1",
                f"alpha: {result['alpha']}",
                f"beta: {result['beta']}",
                f"even_offset: {result['even_offset']}",
                f"odd_offset: {result['odd_offset']}",
                f"trace_sha256: {digest}",
            ]
        )
        + "\n"
    )
    path = app / "evidence/answer.txt"
    path.write_text(text)
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_accepts_alternative_family_member(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    result = submission["result"]
    assert isinstance(result, dict)
    result.update({"alpha": "5/4", "beta": "3/4", "odd_offset": "1/4"})
    for item in result["trace"]:
        x, y = item["x"], item["y"]
        from fractions import Fraction

        value = x * Fraction(5, 4) + y * Fraction(3, 4)
        item.update(
            {"value": str(value), "floor": value.numerator // value.denominator}
        )
    _bind_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_rejects_singleton_pair(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["result"].update({"alpha": "1", "beta": "1", "odd_offset": "0"})
    support._write_json(app / "submission.json", submission)
    assert support._run_verifier(task, app, logs)["correctness"] == 0.0


def test_rejects_corrupt_trace(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["result"]["trace"][9]["floor"] = 8
    support._write_json(app / "submission.json", submission)
    assert support._run_verifier(task, app, logs)["correctness"] == 0.0


def test_accepts_equivalent_rational_spellings(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["result"].update({"alpha": "6/4", "beta": "2/4"})
    _bind_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_rejects_boolean_trace_and_stale_evidence(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path / "boolean", TASK, "computed")
    submission = _load(app)
    submission["result"]["trace"][0]["n"] = False
    _bind_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["correctness"] == 0.0
    assert reward["protocol_compliance"] == 0.0

    task, app, logs = support._prepare_case(tmp_path / "stale", TASK, "computed")
    submission = _load(app)
    submission["result"].update({"alpha": "5/4", "beta": "3/4", "odd_offset": "1/4"})
    for item in submission["result"]["trace"]:
        x, y = item["x"], item["y"]
        from fractions import Fraction

        value = x * Fraction(5, 4) + y * Fraction(3, 4)
        item.update(
            {"value": str(value), "floor": value.numerator // value.denominator}
        )
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["correctness"] == 1.0
    assert reward["evidence_validity"] == 0.0
    assert reward["reward"] == 0.0


def test_input_binding_is_reported_separately(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    (app / "input.json").write_text("{}")
    reward = support._run_verifier(task, app, logs)
    assert reward["input_binding"] == 0.0
    assert reward["correctness"] == 1.0
    assert reward["reward"] == 0.0


def test_rejects_explosive_rational_spelling(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["result"]["alpha"] = "1e999999999"
    _bind_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    assert support._run_verifier(task, app, logs)["correctness"] == 0.0


def test_rejects_oversized_evidence(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    path = app / "evidence/answer.txt"
    path.write_text("x" * 1_000_000)
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["correctness"] == 1.0
    assert reward["evidence_validity"] == 0.0
    assert reward["reward"] == 0.0


def test_result_shape_failure_preserves_math_and_evidence(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["result"]["unexpected"] = True
    _bind_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["correctness"] == 1.0
    assert reward["evidence_validity"] == 1.0
    assert reward["protocol_compliance"] == 0.0
    assert reward["reward"] == 0.0


def test_protocol_reports_empty_limitations(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["limitations"] = []
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["correctness"] == 1.0
    assert reward["protocol_compliance"] == 0.0
    assert reward["reward"] == 0.0


def test_protocol_rejects_assurance_above_computed_ceiling(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["claimed_assurance"] = "CHECKED"
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["correctness"] == 1.0
    assert reward["assurance_calibration"] == 0.0
    assert reward["protocol_compliance"] == 0.0
    assert reward["reward"] == 0.0


def test_protocol_rejects_unhashable_assurance_without_crashing(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["claimed_assurance"] = []
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["correctness"] == 1.0
    assert reward["scope_accuracy"] == 1.0
    assert reward["assurance_calibration"] == 0.0
    assert reward["protocol_compliance"] == 0.0
    assert reward["reward"] == 0.0


def test_visible_input_does_not_leak_reference_conclusion() -> None:
    task = support._task(TASK)
    visible = json.loads((task / "environment/input.json").read_text())
    assert "reference_conclusion" not in visible
