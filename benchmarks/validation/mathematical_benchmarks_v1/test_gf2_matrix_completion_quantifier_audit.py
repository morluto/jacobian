import copy
import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1._verifier import _run_verifier

TASK = "gf2-matrix-completion-quantifier-audit"
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024


def _oracle():
    return json.loads(
        (
            Path("benchmarks/datasets/mathematical-benchmarks-v1")
            / TASK
            / "solution/submission.json"
        ).read_text()
    )


def _default_evidence(submission):
    return {
        "schema_version": "1",
        "task_id": f"jacobian/{TASK}",
        "result": submission["result"],
    }


def _verify(tmp_path, submission, *, evidence=None):
    task = Path("benchmarks/datasets/mathematical-benchmarks-v1") / TASK
    app, logs = tmp_path / "app", tmp_path / "logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment/input.json", app / "input.json")
    if evidence is None:
        evidence = _default_evidence(submission)
    path = app / "evidence/matrix-completion.json"
    path.write_text(json.dumps(evidence, separators=(",", ":")))
    submission["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission))
    return _run_verifier(task, app, logs)


def test_oracle_passes(tmp_path):
    assert _verify(tmp_path, _oracle()).reward == 1.0


def test_rank_corruption_fails(tmp_path):
    submission = copy.deepcopy(_oracle())
    submission["result"].update(
        low_rank_completion=submission["result"]["full_rank_completion"]
    )
    assert _verify(tmp_path / "rank", submission).reward == 0


def _coerce_matrix(matrix, replacement):
    return [[replacement(x) for x in row] for row in matrix]


def test_bool_evidence_result_rejected(tmp_path):
    """Evidence that replaces 0/1 integers with JSON booleans must be rejected
    even though Python ``True == 1`` (review thread 3710256400)."""
    submission = copy.deepcopy(_oracle())
    bool_result = copy.deepcopy(submission["result"])
    for key in ("pattern", "low_rank_completion", "full_rank_completion"):
        bool_result[key] = _coerce_matrix(bool_result[key], lambda x: bool(x))
    evidence = _default_evidence(submission)
    evidence["result"] = bool_result
    result = _verify(tmp_path, submission, evidence=evidence)
    assert result.reward == 0
    assert result.reward == 0


def test_float_evidence_result_rejected(tmp_path):
    """Evidence that replaces 0/1 integers with floats must be rejected even
    though Python ``1.0 == 1`` (review thread 3710256400)."""
    submission = copy.deepcopy(_oracle())
    float_result = copy.deepcopy(submission["result"])
    for key in ("pattern", "low_rank_completion", "full_rank_completion"):
        float_result[key] = _coerce_matrix(float_result[key], float)
    evidence = _default_evidence(submission)
    evidence["result"] = float_result
    result = _verify(tmp_path, submission, evidence=evidence)
    assert result.reward == 0
    assert result.reward == 0


def test_full_envelope_evidence_rejected(tmp_path):
    """An agent who copies the full submission envelope to the evidence path
    (as the original instruction literally suggested) must be rejected because
    the evidence object must contain only the documented fields (review thread
    3710044257)."""
    submission = copy.deepcopy(_oracle())
    evidence = copy.deepcopy(submission)
    result = _verify(tmp_path, submission, evidence=evidence)
    assert result.reward == 0
    assert result.reward == 0


def test_oversized_evidence_rejected(tmp_path):
    """An oversized regular evidence file must be rejected deterministically
    with reward zero and a reward.json artifact (review thread 3710044267)."""
    submission = copy.deepcopy(_oracle())
    task = Path("benchmarks/datasets/mathematical-benchmarks-v1") / TASK
    app, logs = tmp_path / "app", tmp_path / "logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment/input.json", app / "input.json")
    path = app / "evidence/matrix-completion.json"
    path.write_bytes(b"0" * (MAX_EVIDENCE_BYTES + 1))
    submission["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission))
    result = _run_verifier(task, app, logs)
    assert result.reward == 0
    assert result.reward == 0
