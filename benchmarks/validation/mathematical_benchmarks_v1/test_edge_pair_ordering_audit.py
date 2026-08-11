from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "edge-pair-ordering-audit"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)


ROOT = Path(__file__).resolve().parents[3]
VERIFIER = (
    ROOT
    / "benchmarks/datasets/mathematical-benchmarks-v1/edge-pair-ordering-audit/tests/verifier.py"
)


def module():
    sys.path.insert(0, str(VERIFIER.parent))
    try:
        spec = importlib.util.spec_from_file_location("edge_pair_verifier", VERIFIER)
        assert spec and spec.loader
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded
    finally:
        sys.path.remove(str(VERIFIER.parent))
        sys.modules.pop("verifier_support", None)


def test_exhaustive_values():
    verifier = module()
    assert [verifier.exhaustive(n) for n in range(3, 7)] == [12, 384, 15360, 983040]


def test_unordered_public_answer_is_rejected():
    verifier = module()
    assert verifier.exhaustive(3) != 6
    assert verifier.exhaustive(6) != 491520


def test_reference_submission_passes_with_protocol_diagnostic(tmp_path: Path):
    task, app, logs = _case(tmp_path)
    result = support._run_verifier(task, app, logs)
    assert result.details["protocol_compliance"] == 1.0
    assert result.details["correctness"] == 1.0
    assert result.reward == 1.0


def test_empty_evidence_is_rejected_without_losing_mathematical_diagnostic(
    tmp_path: Path,
):
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    (app / "evidence" / "answer.txt").write_text("")
    submission["evidence"][0]["sha256"] = support._digest(
        app / "evidence" / "answer.txt"
    )
    support._write_json(app / "submission.json", submission)

    result = support._run_verifier(task, app, logs)
    assert result.details["correctness"] == 1.0
    assert result.details["evidence_validity"] == 0.0
    assert result.reward == 0.0


def test_extra_result_field_is_protocol_failure_only(tmp_path: Path):
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["unexpected"] = "ignored for mathematical replay"
    _rewrite(app, submission)

    result = support._run_verifier(task, app, logs)
    assert result.details["protocol_compliance"] == 0.0
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_duplicate_probe_is_rejected_as_malformed_and_incorrect(tmp_path: Path):
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["probe_values"][3]["n"] = 3
    _rewrite(app, submission)

    result = support._run_verifier(task, app, logs)
    assert result.details["protocol_compliance"] == 0.0
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_non_string_pair_semantics_does_not_crash(tmp_path: Path):
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["pair_semantics"] = ["ORDERED"]
    _rewrite(app, submission)
    result = support._run_verifier(task, app, logs)
    assert result.details["protocol_compliance"] == 0.0
    assert result.reward == 0.0


def test_keyword_only_evidence_is_rejected(tmp_path: Path):
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    marker = "RESULT_JSON: " + json.dumps(
        submission["result"], sort_keys=True, separators=(",", ":")
    )
    (app / "evidence" / "answer.txt").write_text(
        "ordered unordered incident pair factor free edge coefficient finite\n"
        + marker
        + "\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(
        app / "evidence" / "answer.txt"
    )
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result.details["evidence_validity"] == 0.0
    assert result.reward == 0.0


def test_recursive_evidence_marker_does_not_crash(tmp_path: Path):
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    nested: list = []
    for _ in range(600):
        nested = [nested]
    marker = "RESULT_JSON: " + json.dumps(nested, separators=(",", ":"))
    (app / "evidence" / "answer.txt").write_text(
        "ordered unordered factor free edge finite two\n" + marker + "\n"
    )
    submission["evidence"][0]["sha256"] = support._digest(
        app / "evidence" / "answer.txt"
    )
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result.details["evidence_validity"] == 0.0
    assert result.reward == 0.0


def test_scope_accuracy_independent_of_conclusion_mismatch(tmp_path: Path):
    """An envelope defect (wrong conclusion) must not erase scope_accuracy."""

    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["conclusion"] = "UNSUPPORTED"
    _rewrite(app, submission)
    result = support._run_verifier(task, app, logs)
    assert result.details["protocol_compliance"] == 0.0
    assert result.details["scope_accuracy"] == 0.0
    assert result.reward == 0.0


def test_wrong_limitation_reported_as_protocol_failure(tmp_path: Path):
    """A wrong limitation must produce protocol_compliance 0.0, not a silent zero."""

    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = ["wrong limitation text"]
    _rewrite(app, submission)
    result = support._run_verifier(task, app, logs)
    assert result.details["protocol_compliance"] == 0.0
    assert result.reward == 0.0
