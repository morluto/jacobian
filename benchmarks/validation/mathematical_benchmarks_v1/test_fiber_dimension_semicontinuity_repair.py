from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "fiber-dimension-semicontinuity-repair"


def _load(app: Path) -> dict[str, object]:
    return json.loads((app / "submission.json").read_text())


def _bind_evidence(app: Path, submission: dict[str, object]) -> None:
    result = submission["result"]
    rows = []
    for check in result["fiber_checks"]:
        point = check["point"]
        rows.append(
            (Fraction(point["x"]), Fraction(point["y"]), check["cokernel_dimension"])
        )
    rows.sort()
    text = (
        "\n".join(
            [
                "fiber-dimension-fitting-repair-v1",
                f"tensor-repair: {result['tensor_repair']}",
                f"global-repair: {result['global_repair']}",
                f"generator-count: {len(result['ideal_generators'])}",
                "fiber-dimensions: "
                + ";".join(f"{x},{y}:{dimension}" for x, y, dimension in rows),
            ]
        )
        + "\n"
    )
    path = app / "evidence/answer.txt"
    path.write_text(text)
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )


def _term(coefficient: str, x_power: int, y_power: int) -> dict[str, object]:
    return {"coefficient": coefficient, "exponents": [x_power, y_power]}


def test_accepts_alternative_ideal_generators_and_fiber_order(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    result = submission["result"]
    result["ideal_generators"] = [
        {"terms": [_term("1", 2, 0), _term("1", 0, 2)]},
        {"terms": [_term("1", 2, 0), _term("-1", 0, 2)]},
        {"terms": [_term("1", 1, 1)]},
    ]
    result["fiber_checks"].reverse()
    _bind_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_rejects_strictly_smaller_minor_ideal(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["result"]["ideal_generators"] = [
        {"terms": [_term("1", 2, 0)]},
        {"terms": [_term("1", 1, 1)]},
    ]
    _bind_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["correctness"] == 0.0
    assert reward["reward"] == 0.0


def test_rejects_duplicate_monomials_fail_closed(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["result"]["ideal_generators"][0]["terms"] = [
        _term("1", 2, 0),
        _term("1", 2, 0),
    ]
    _bind_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["protocol_compliance"] == 0.0
    assert reward["correctness"] == 0.0


def test_rejects_wrong_fiber_rank(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["result"]["fiber_checks"][0]["matrix_rank"] = 1
    _bind_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    assert support._run_verifier(task, app, logs)["correctness"] == 0.0


def test_visible_input_tamper_fails_closed(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    (app / "input.json").write_text("{}")
    reward = support._run_verifier(task, app, logs)
    assert reward["input_binding"] == 0.0
    assert reward["correctness"] == 1.0
    assert reward["reward"] == 0.0


def test_scope_diagnostic_is_independent_of_assurance_type(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["claimed_assurance"] = []
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["scope_accuracy"] == 1.0
    assert reward["assurance_calibration"] == 0.0
    assert reward["reward"] == 0.0


def test_unsupported_verified_claim_is_rejected(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["claimed_assurance"] = "VERIFIED"
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["false_certification"] is True
    assert reward["assurance_calibration"] == 0.0
    assert reward["reward"] == 0.0


def test_evidence_must_bind_submitted_generator_count(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    path = app / "evidence/answer.txt"
    path.write_text(
        path.read_text().replace("generator-count: 3", "generator-count: 2")
    )
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["correctness"] == 1.0
    assert reward["evidence_validity"] == 0.0
    assert reward["reward"] == 0.0


def test_oversized_fiber_list_rejected_without_crash(tmp_path: Path) -> None:
    """An oversized fiber_checks list must fail closed without excessive work."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    # Replace fiber_checks with a far larger list to exercise the cardinality guard.
    base = submission["result"]["fiber_checks"][0]
    submission["result"]["fiber_checks"] = [base] * 10000
    _bind_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["correctness"] == 0.0
    assert reward["evidence_validity"] == 0.0
    assert reward["reward"] == 0.0


def test_task_metadata_declares_diagnostics() -> None:
    """Diagnostics are declared in task-local metadata, not global registries."""
    assert TASK not in support.INPUT_BINDING_DECOUPLED_TASKS
    assert TASK not in support.SCOPE_INDEPENDENT_ASSURANCE_TASKS
    assert support.is_input_binding_decoupled(TASK) is True
    assert support.is_scope_independent_assurance(TASK) is True
    metadata = support.load_task_contract_metadata(TASK)
    assert metadata.get("input_binding_decoupled") is True
    assert metadata.get("scope_independent_assurance") is True
