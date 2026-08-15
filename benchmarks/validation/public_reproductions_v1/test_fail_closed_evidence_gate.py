"""Golden fail-closed tasks: invalid witness digests must zero aggregate reward."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.public_reproductions_v1._fixtures import (
    _prepare_case,
    _write_json,
)
from benchmarks.validation.public_reproductions_v1._verifier import _run_verifier

_GOLDEN_FAIL_CLOSED = (
    "balanced-row-permutation",
    "coin-process-potential",
    "closed-set-distance-strengthening-audit",
    "reduced-point",
    "gaussian-complex-cancellation",
    "reliability-triangle-fair",
    "smith-rectangular",
)


@pytest.mark.parametrize("task_name", _GOLDEN_FAIL_CLOSED)
def test_wrong_witness_digest_zeros_reward_with_visible_diagnostics(
    tmp_path: Path,
    task_name: str,
) -> None:
    task, app, logs = _prepare_case(tmp_path, task_name, "computed")
    accepted = _run_verifier(task, app, logs)
    assert accepted.reward == pytest.approx(1.0)
    assert accepted.details.get(
        "witness_validity", accepted.details.get("evidence", 1.0)
    ) == pytest.approx(1.0)

    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    assert isinstance(submission.get("witness"), list) and submission["witness"]
    submission["witness"][0]["sha256"] = "sha256:" + ("0" * 64)
    _write_json(submission_path, submission)

    rejected = _run_verifier(task, app, logs)
    evidence = rejected.details.get(
        "witness_validity", rejected.details.get("evidence")
    )
    assert evidence == pytest.approx(0.0)
    assert rejected.reward == pytest.approx(0.0)
    if "correctness" in rejected.details:
        assert rejected.details["correctness"] in {0.0, 1.0}
