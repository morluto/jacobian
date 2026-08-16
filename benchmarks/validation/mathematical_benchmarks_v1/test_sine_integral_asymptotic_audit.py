from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures, _verifier

TASK = "sine-integral-asymptotic-audit"


def _submission(app: Path) -> dict[str, object]:
    return json.loads((app / "submission.json").read_text())


def _rewrite(app: Path, submission: dict[str, object]) -> None:
    _fixtures._write_json(app / "submission.json", {"result": submission["result"]})


def test_accepts_reordered_equivalent_terms(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = _submission(app)
    result = submission["result"]
    assert isinstance(result, dict)
    result["tail_terms"] = list(reversed(result["tail_terms"]))
    result["si_terms"] = list(reversed(result["si_terms"]))
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0


def test_rejects_published_wrong_sine_sign(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = _submission(app)
    result = submission["result"]
    assert isinstance(result, dict)
    for term in result["si_terms"]:
        if term["function"] == "SIN" and term["power"] == 2:
            term["coefficient"] = 1
    result["corrected_sine_coefficient"] = 1
    _fixtures._write_json(app / "submission.json", submission)
    assert _verifier._run_verifier(task, app, logs).details["correctness"] == 0.0


def test_rejects_corrupt_remainder_bound(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = _submission(app)
    result = submission["result"]
    assert isinstance(result, dict)
    result["absolute_remainder_bound"]["numerator"] = 23
    _fixtures._write_json(app / "submission.json", submission)
    assert _verifier._run_verifier(task, app, logs).details["correctness"] == 0.0


def test_undeclared_witness_key_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = _submission(app)
    submission["witness"] = [
        {"path": "evidence/answer.txt", "sha256": "sha256:" + "0" * 64}
    ]
    _fixtures._write_json(app / "submission.json", submission)
    result = _verifier._run_verifier(task, app, logs)
    assert result.reward == 0.0


def test_rejects_noninteger_remainder_power(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = _submission(app)
    result = submission["result"]
    assert isinstance(result, dict)
    result["tail_remainder"]["power"] = 6.0
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).details["correctness"] == 0.0


def test_rejects_noninteger_si_remainder_coefficient(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = _submission(app)
    result = submission["result"]
    assert isinstance(result, dict)
    result["si_remainder"]["coefficient"] = 120.0
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).details["correctness"] == 0.0


def test_rejects_boolean_reported_coefficient(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = _submission(app)
    result = submission["result"]
    assert isinstance(result, dict)
    result["published_sine_coefficient"] = True
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).details["correctness"] == 0.0


def test_reordered_terms_without_ceremonial_file_are_accepted(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = _submission(app)
    result = submission["result"]
    assert isinstance(result, dict)
    result["tail_terms"] = list(reversed(result["tail_terms"]))
    result["si_terms"] = list(reversed(result["si_terms"]))
    _rewrite(app, submission)
    verdict = _verifier._run_verifier(task, app, logs)
    assert verdict.details["correctness"] == 1.0
    assert verdict.reward == 1.0


def test_input_tamper_is_reported_separately(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    frozen = json.loads((app / "input.json").read_text())
    frozen["source"]["row"] = 999
    _fixtures._write_json(app / "input.json", frozen)
    verdict = _verifier._run_verifier(task, app, logs)
    assert verdict.details["correctness"] == 1.0
    assert verdict.reward == 0.0


def test_unhashable_term_function_fails_closed(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = _submission(app)
    submission["result"]["tail_terms"][0]["function"] = []
    _rewrite(app, submission)
    verdict = _verifier._run_verifier(task, app, logs)
    assert verdict.details["correctness"] == 0.0
    assert verdict.reward == 0.0


def test_undeclared_witness_path_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = _submission(app)
    submission["witness"] = [{"path": "../answer.txt", "sha256": "sha256:" + "0" * 64}]
    _fixtures._write_json(app / "submission.json", submission)
    verdict = _verifier._run_verifier(task, app, logs)
    assert verdict.reward == 0.0
