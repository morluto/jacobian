from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.agent_workflow_v1 import support

TASK = "ratio-test-boundary-separation"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def test_reference_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    result = support._run_verifier(task, app, logs)
    assert result["reward"] == 1.0


def test_alternative_checkpoints_pass(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["convergent_witness"]["checkpoints"] = [
        {"N": n, "partial_sum": f"{n}/{n + 1}", "tail": f"1/{n + 1}"}
        for n in (2, 5, 17, 40)
    ]
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    result = support._run_verifier(task, app, logs)
    assert result["reward"] == 1.0


def test_corrupt_block_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["divergent_witness"]["blocks"][3]["count"] += 1
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_corrupt_partial_sum_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["convergent_witness"]["checkpoints"][1]["partial_sum"] = "3/4"
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_one_witness_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    del submission["result"]["convergent_witness"]
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_false_verified_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["claimed_assurance"] = "VERIFIED"
    support._write_json(path, submission)
    result = support._run_verifier(task, app, logs)
    assert result["reward"] == 0.0
    assert result["false_certification"] is True


def test_small_checkpoints_below_threshold_pass(tmp_path: Path) -> None:
    """The undocumented max(seen) >= 12 threshold is removed: all-small checkpoints pass."""

    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["convergent_witness"]["checkpoints"] = [
        {"N": n, "partial_sum": f"{n}/{n + 1}", "tail": f"1/{n + 1}"}
        for n in (1, 2, 3, 4)
    ]
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_large_checkpoint_closed_form_passes(tmp_path: Path) -> None:
    """A large N is accepted in bounded time via the telescoping closed form."""

    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    large = 10**8
    submission["result"]["convergent_witness"]["checkpoints"] = [
        {"N": n, "partial_sum": f"{n}/{n + 1}", "tail": f"1/{n + 1}"}
        for n in (1, 2, 3, large)
    ]
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_evidence_prose_outside_result_marker_is_required(tmp_path: Path) -> None:
    """Evidence with only the marker line and no explanatory prose is rejected."""

    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    evidence_path = app / "evidence" / "answer.txt"
    marker = "RESULT_JSON:" + json.dumps(
        submission["result"], sort_keys=True, separators=(",", ":")
    )
    evidence_path.write_text("dyadic\n" + marker + "\n", encoding="utf-8")
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_affirmative_proof_assistant_claim_is_rejected(tmp_path: Path) -> None:
    """A 'Noteworthy proof assistant ...' limitation must not satisfy the negation."""

    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["limitations"] = [
        "Noteworthy proof assistant verification was performed."
    ]
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_visible_schema_does_not_leak_witness_values() -> None:
    """The agent-visible schema is structural and omits hidden witness const values."""

    schema = json.loads(
        (support.TASKS / TASK / "environment" / "submission_schema.json").read_text()
    )
    result_props = schema["properties"]["result"]["properties"]
    divergent = result_props["divergent_witness"]["properties"]
    convergent = result_props["convergent_witness"]["properties"]
    for field in ("term", "ratio", "ratio_error"):
        assert divergent[field] == {"type": "string"}
    assert divergent["blocks"]["items"]["properties"]["block_lower_bound"] == {
        "type": "string"
    }
    for field in ("term", "telescoping_identity", "ratio", "ratio_error"):
        assert convergent[field] == {"type": "string"}
    # The ratio-limit premise is part of the problem statement, not a hidden witness.
    assert result_props["ratio_limit"] == {"const": "1"}
