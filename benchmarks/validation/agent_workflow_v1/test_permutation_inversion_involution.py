from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.validation.agent_workflow_v1 import support

TASK = "permutation-inversion-involution"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    evidence = {
        "schema_version": "1",
        "task_id": submission["task_id"],
        "result": submission["result"],
        "limitations": submission["limitations"],
    }
    raw = json.dumps(evidence, separators=(",", ":")).encode()
    (app / "evidence" / "permutation-involution-certificate.json").write_bytes(raw)
    submission["evidence"][0]["sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    support._write_json(app / "submission.json", submission)


def test_accepts_alternative_trace_set(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["traces"] = list(reversed(submission["result"]["traces"]))
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_rejects_locally_plausible_wrong_transform(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["transformation"] = "REVERSE_POSITIONS"
    submission["result"]["value_multiplier"] = 1
    submission["result"]["value_offset"] = 0
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_accepts_reverse_position_involution(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = submission["result"]
    result["transformation"] = "REVERSE_POSITIONS"
    result["value_multiplier"] = 1
    result["value_offset"] = 0
    for trace in result["traces"]:
        permutation = trace["permutation"]
        transformed = list(reversed(permutation))
        trace["transformed"] = transformed
        trace["transformed_inversions"] = sum(
            transformed[i] > transformed[j] for i in range(7) for j in range(i + 1, 7)
        )
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_rejects_corrupted_trace(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["traces"][3]["inversions"] += 1
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_oversized_evidence_emits_zero_reward(tmp_path: Path) -> None:
    """An evidence file exceeding MAX_EVIDENCE_BYTES is rejected before hashing
    or parsing, the verifier emits reward.json, and reward is zero."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "permutation-involution-certificate.json"
    evidence_path.write_bytes(b"x" * (16 * 1024 * 1024 + 1))
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_unhashable_permutation_trace_emits_zero_reward(tmp_path: Path) -> None:
    """A trace whose permutation contains a nested list would crash tuple/set
    operations without the exact-integer validation; the verifier must reject it
    and emit reward.json with reward zero."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["traces"][0]["permutation"] = [1, 2, 3, 4, 5, 6, [7]]
    _rewrite(app, submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_boolean_permutation_entry_is_rejected(tmp_path: Path) -> None:
    """A boolean entry is equal to an integer under Python equality but must be
    rejected as a non-exact-integer permutation element."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["traces"][0]["permutation"] = [True, 2, 3, 4, 5, 6, 7]
    _rewrite(app, submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_boolean_result_field_is_rejected(tmp_path: Path) -> None:
    """A boolean ``False`` equals integer ``0`` under Python equality but must
    be rejected in the integer result fields."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["fixed_point_count"] = False
    _rewrite(app, submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_type_sensitive_evidence_comparison(tmp_path: Path) -> None:
    """An evidence certificate that replaces an integer ``0`` with boolean
    ``False`` must be rejected even though Python treats them as equal."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = {
        "schema_version": "1",
        "task_id": submission["task_id"],
        "result": dict(submission["result"]),
        "limitations": submission["limitations"],
    }
    evidence["result"]["fixed_point_count"] = False
    raw = json.dumps(evidence, separators=(",", ":")).encode()
    (app / "evidence" / "permutation-involution-certificate.json").write_bytes(raw)
    submission["evidence"][0]["sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_boolean_trace_inversions_is_rejected(tmp_path: Path) -> None:
    """A boolean ``False`` equals integer ``0`` under Python equality but must
    be rejected in the trace inversion-count fields."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["traces"][0]["inversions"] = False
    _rewrite(app, submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_boolean_trace_transformed_entry_is_rejected(tmp_path: Path) -> None:
    """A boolean entry in the ``transformed`` list equals an integer under
    Python equality but must be rejected as a non-exact-integer value."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["traces"][0]["transformed"][0] = True
    _rewrite(app, submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_decouples_scope_from_assurance(tmp_path: Path) -> None:
    """An otherwise exact submission claiming ``VERIFIED`` must retain
    scope_accuracy while failing assurance and reward."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "invalid")
    submission = json.loads((app / "submission.json").read_text())
    _rewrite(app, submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 1.0
    assert result["evidence_validity"] == 1.0
    assert result["scope_accuracy"] == 1.0
    assert result["assurance_calibration"] == 0.0
    assert result["reward"] == 0.0
