from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "sine-integral-asymptotic-audit"


def _submission(app: Path) -> dict[str, object]:
    return json.loads((app / "submission.json").read_text())


def _bind_evidence(app: Path, submission: dict[str, object]) -> None:
    result = submission["result"]
    assert isinstance(result, dict)
    digest = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    answer = "\n".join(
        [
            "sine-integral-certificate-v1",
            f"result_sha256: {digest}",
            f"published_sine_coefficient: {result['published_sine_coefficient']}",
            f"corrected_sine_coefficient: {result['corrected_sine_coefficient']}",
            "",
        ]
    )
    (app / "evidence" / "answer.txt").write_text(answer)
    evidence = submission["evidence"]
    assert isinstance(evidence, list) and isinstance(evidence[0], dict)
    evidence[0]["sha256"] = f"sha256:{hashlib.sha256(answer.encode()).hexdigest()}"


def test_accepts_reordered_equivalent_terms(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _submission(app)
    result = submission["result"]
    assert isinstance(result, dict)
    result["tail_terms"] = list(reversed(result["tail_terms"]))
    result["si_terms"] = list(reversed(result["si_terms"]))
    _bind_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_rejects_published_wrong_sine_sign(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _submission(app)
    result = submission["result"]
    assert isinstance(result, dict)
    for term in result["si_terms"]:
        if term["function"] == "SIN" and term["power"] == 2:
            term["coefficient"] = 1
    result["corrected_sine_coefficient"] = 1
    support._write_json(app / "submission.json", submission)
    assert support._run_verifier(task, app, logs)["correctness"] == 0.0


def test_rejects_corrupt_remainder_bound(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _submission(app)
    result = submission["result"]
    assert isinstance(result, dict)
    result["absolute_remainder_bound"]["numerator"] = 23
    support._write_json(app / "submission.json", submission)
    assert support._run_verifier(task, app, logs)["correctness"] == 0.0


def test_rejects_duplicate_evidence(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _submission(app)
    submission["evidence"].append(dict(submission["evidence"][0]))
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["reward"] == 0.0
    assert result["correctness"] == 1.0


def test_rejects_noninteger_remainder_power(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _submission(app)
    result = submission["result"]
    assert isinstance(result, dict)
    result["tail_remainder"]["power"] = 6.0
    _bind_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    assert support._run_verifier(task, app, logs)["correctness"] == 0.0


def test_rejects_noninteger_si_remainder_coefficient(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _submission(app)
    result = submission["result"]
    assert isinstance(result, dict)
    result["si_remainder"]["coefficient"] = 120.0
    _bind_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    assert support._run_verifier(task, app, logs)["correctness"] == 0.0


def test_rejects_boolean_reported_coefficient(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _submission(app)
    result = submission["result"]
    assert isinstance(result, dict)
    result["published_sine_coefficient"] = True
    _bind_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    assert support._run_verifier(task, app, logs)["correctness"] == 0.0


def test_rejects_stale_evidence_binding(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _submission(app)
    result = submission["result"]
    assert isinstance(result, dict)
    result["tail_terms"] = list(reversed(result["tail_terms"]))
    result["si_terms"] = list(reversed(result["si_terms"]))
    support._write_json(app / "submission.json", submission)
    verdict = support._run_verifier(task, app, logs)
    assert verdict["correctness"] == 1.0
    assert verdict["evidence_validity"] == 0.0
    assert verdict["reward"] == 0.0


def test_input_tamper_is_reported_separately(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    frozen = json.loads((app / "input.json").read_text())
    frozen["source"]["row"] = 999
    support._write_json(app / "input.json", frozen)
    verdict = support._run_verifier(task, app, logs)
    assert verdict["correctness"] == 1.0
    assert verdict["input_binding"] == 0.0
    assert verdict["reward"] == 0.0


def test_malformed_limitations_fail_closed(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _submission(app)
    submission["limitations"] = None
    support._write_json(app / "submission.json", submission)
    verdict = support._run_verifier(task, app, logs)
    assert verdict["limitations_accuracy"] == 0.0
    assert verdict["protocol_compliance"] == 0.0
    assert verdict["reward"] == 0.0


def test_unhashable_term_function_fails_closed(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _submission(app)
    submission["result"]["tail_terms"][0]["function"] = []
    _bind_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    verdict = support._run_verifier(task, app, logs)
    assert verdict["correctness"] == 0.0
    assert verdict["reward"] == 0.0


def test_bad_evidence_descriptor_is_protocol_failure(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _submission(app)
    submission["evidence"][0]["path"] = "../answer.txt"
    support._write_json(app / "submission.json", submission)
    verdict = support._run_verifier(task, app, logs)
    assert verdict["protocol_compliance"] == 0.0
    assert verdict["evidence_validity"] == 0.0
    assert verdict["reward"] == 0.0
