"""Golden fail-closed tasks: invalid evidence digests must zero aggregate reward.

These tasks already use min-gate / hard evidence policy. They prove the
two-layer scoring contract before the Phase 1 leaky-template migration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.public_reproductions_v1 import support

# Tasks that hard-gate evidence into aggregate reward (pre-existing min-gate or
# Phase 1 aggregate_reward migration).
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
def test_wrong_evidence_digest_zeros_reward_with_visible_diagnostics(
    tmp_path: Path,
    task_name: str,
) -> None:
    task, app, logs = support._prepare_case(tmp_path, task_name, "computed")
    accepted = support._run_verifier(task, app, logs)
    assert accepted.reward == pytest.approx(1.0)
    assert accepted.details.get(
        "evidence_validity", accepted.details.get("evidence", 1.0)
    ) == pytest.approx(1.0)

    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    assert isinstance(submission.get("evidence"), list) and submission["evidence"]
    submission["evidence"][0]["sha256"] = "sha256:" + ("0" * 64)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    evidence = rejected.details.get(
        "evidence_validity", rejected.details.get("evidence")
    )
    assert evidence == pytest.approx(0.0)
    assert rejected.reward == pytest.approx(0.0)
    # Correct mathematics may remain visible; aggregate must still fail closed.
    if "correctness" in rejected.details:
        assert rejected.details["correctness"] in {0.0, 1.0}
