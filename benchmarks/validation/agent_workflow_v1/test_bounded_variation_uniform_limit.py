from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.agent_workflow_v1 import support

TASK = "bounded-variation-uniform-limit"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def test_reference_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_alternative_scale_and_indices_pass(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["scale_q"] = 5
    submission["result"]["uniform_certificate"]["sup_norm_denominator_coefficient"] = 5
    submission["result"]["checkpoints"] = []
    for n in (3, 7, 13, 21):
        k = 5 * n
        submission["result"]["checkpoints"].append(
            {
                "n": n,
                "frequency": k,
                "amplitude": f"1/{k}",
                "interior_segments": 2 * k - 1,
                "endpoint_contribution": f"2/{k}",
                "interior_contribution": f"{4 * k - 2}/{k}",
                "total_variation": "4",
            }
        )
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def _mutate(tmp_path: Path, mutation) -> float:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    mutation(submission)
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    return support._run_verifier(task, app, logs)["reward"]


def test_wrong_segment_count_is_rejected(tmp_path: Path) -> None:
    assert (
        _mutate(
            tmp_path,
            lambda s: s["result"]["checkpoints"][0].__setitem__("interior_segments", 6),
        )
        == 0.0
    )


def test_wrong_variation_is_rejected(tmp_path: Path) -> None:
    assert (
        _mutate(
            tmp_path,
            lambda s: s["result"]["variation_formula"].__setitem__(
                "total_variation", "0"
            ),
        )
        == 0.0
    )


def test_duplicate_indices_are_rejected(tmp_path: Path) -> None:
    def duplicate(submission):
        submission["result"]["checkpoints"][1] = dict(
            submission["result"]["checkpoints"][0]
        )

    assert _mutate(tmp_path, duplicate) == 0.0


def test_false_verified_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["claimed_assurance"] = "VERIFIED"
    support._write_json(path, submission)
    result = support._run_verifier(task, app, logs)
    assert result["reward"] == 0.0
    assert result["false_certification"] is True


def test_visible_input_tampering_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    data = json.loads((app / "input.json").read_text())
    data["source"]["row"] = 601
    support._write_json(app / "input.json", data)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_unstructured_argument_claim_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["argument"]["implication"] = "does not force"
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_oversized_evidence_is_rejected_before_digest_binding(tmp_path: Path) -> None:
    """The task-specific evidence bound is checked before hashing."""

    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text("x" * (1_048_576 + 1))
    submission["evidence"][0]["sha256"] = "sha256:" + "0" * 64
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_bool_in_integer_certificate_is_rejected(tmp_path: Path) -> None:
    """``True == 1`` in Python must not let a boolean pass as an integer."""

    def mutate(submission):
        submission["result"]["uniform_certificate"]["sup_norm_numerator"] = True

    assert _mutate(tmp_path, mutate) == 0.0


def test_bool_in_endpoint_jump_multiplier_is_rejected(tmp_path: Path) -> None:
    def mutate(submission):
        submission["result"]["variation_formula"]["endpoint_jump_multiplier"] = True

    assert _mutate(tmp_path, mutate) == 0.0


def test_equivalent_sequence_serialization_passes(tmp_path: Path) -> None:
    """``sin(n*q*x)/(n*q)`` is mathematically identical to ``sin(q*n*x)/(q*n)``."""

    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["sequence"] = "sin(n*q*x)/(n*q)"
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_equivalent_limitation_wording_passes(tmp_path: Path) -> None:
    """The public structured limitation value is accepted."""

    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["limitations"] = ["NO_PROOF_ASSISTANT_VERIFICATION"]
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_protocol_failure_preserves_math_correctness(tmp_path: Path) -> None:
    """An extra top-level field must zero reward but not collapse correctness."""

    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["extra_field"] = "protocol violation"
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    result = support._run_verifier(task, app, logs)
    assert result["reward"] == 0.0
    assert result["protocol"] is False
    assert result["correctness"] == 0.0
    assert result["evidence_validity"] == 0.0
    assert result["scope_accuracy"] == 0.0
