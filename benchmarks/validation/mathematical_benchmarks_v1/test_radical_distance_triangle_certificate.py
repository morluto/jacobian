from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "radical-distance-triangle-certificate"


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
    (app / "evidence" / "radical-distance-certificate.json").write_bytes(raw)
    submission["evidence"][0]["sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    support._write_json(app / "submission.json", submission)


def test_accepts_the_exact_geometric_certificate(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_rejects_sampling_as_universal_proof(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["method"] = "NUMERICAL_SAMPLING"
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_rejects_corrupted_center_expansion(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["expanded_radicands"][0]["constant"] = 2
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_accepts_the_reversed_center_certificate(tmp_path: Path) -> None:
    """The two centers and their expansions may be listed in either order
    without changing the distance model, lower bound, or equality witness."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = submission["result"]
    result["scaled_centers"] = [
        list(result["scaled_centers"][1]),
        list(result["scaled_centers"][0]),
    ]
    result["expanded_radicands"] = [
        dict(result["expanded_radicands"][1]),
        dict(result["expanded_radicands"][0]),
    ]
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_accepts_schema_valid_integral_float_centers(tmp_path: Path) -> None:
    """JSON Schema's ``integer`` type accepts integral floats like ``-1.0``,
    so the verifier must accept them while still rejecting booleans."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = submission["result"]
    result["scaled_centers"] = [
        [float(v) for v in result["scaled_centers"][0]],
        [float(v) for v in result["scaled_centers"][1]],
    ]
    result["center_distance_squared"] = float(result["center_distance_squared"])
    result["lower_bound"] = float(result["lower_bound"])
    result["equality_witness"] = [float(v) for v in result["equality_witness"]]
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_rejects_boolean_radicand_coefficients(tmp_path: Path) -> None:
    """Boolean ``true`` must not spoof integer ``1`` in radicand coefficients."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["expanded_radicands"][0]["a2"] = True
    _rewrite(app, submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_rejects_boolean_center_entries(tmp_path: Path) -> None:
    """Boolean center entries must be rejected even though ``True == 1``."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["scaled_centers"][0][0] = True
    _rewrite(app, submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_decouples_correctness_from_unsupported_assurance(tmp_path: Path) -> None:
    """An otherwise exact certificate claiming ``VERIFIED`` must retain
    mathematical correctness, evidence validity, and scope accuracy while
    failing assurance and aggregate reward."""
    task, app, logs = support._prepare_case(tmp_path, TASK, "invalid")
    submission = json.loads((app / "submission.json").read_text())
    _rewrite(app, submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 0.0
    assert result["evidence_validity"] == 0.0
    assert result["scope_accuracy"] == 0.0
    assert result["assurance_calibration"] == 0.0
    assert result["reward"] == 0.0


def test_oversized_evidence_is_rejected_without_crashing(tmp_path: Path) -> None:
    """An evidence file exceeding the byte bound is rejected before reading
    or hashing, so reward.json is still written with zero reward."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "radical-distance-certificate.json"
    evidence_path.write_text("x" * (17 * 1024 * 1024))
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0


def test_rejects_swapped_expansions_with_unswapped_centers(tmp_path: Path) -> None:
    """Swapping only the expanded radicands while leaving centers unchanged
    breaks the center-to-expansion pairing and must be rejected."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = submission["result"]
    result["expanded_radicands"] = [
        dict(result["expanded_radicands"][1]),
        dict(result["expanded_radicands"][0]),
    ]
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 0.0


def test_accepts_schema_valid_integral_float_coefficients(tmp_path: Path) -> None:
    """JSON Schema's ``integer`` type accepts integral floats like ``1.0``,
    so the verifier must accept them in radicand coefficient records while
    still rejecting booleans."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    for record in submission["result"]["expanded_radicands"]:
        for key in record:
            record[key] = float(record[key])
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_rejects_evidence_limitations_unbound_from_submission(tmp_path: Path) -> None:
    """Evidence limitations must match the submitted limitations, not just the
    canonical constant. A submission that changes limitations while evidence
    retains the canonical list must fail evidence_validity."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    canonical_limitations = list(submission["limitations"])
    submission["limitations"] = ["FAKE_LIMITATION"]
    # Write evidence with the canonical limitations, not the submitted ones.
    evidence = {
        "schema_version": "1",
        "task_id": submission["task_id"],
        "result": submission["result"],
        "limitations": canonical_limitations,
    }
    raw = json.dumps(evidence, separators=(",", ":")).encode()
    (app / "evidence" / "radical-distance-certificate.json").write_bytes(raw)
    submission["evidence"][0]["sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 0.0
    assert result["reward"] == 0.0
