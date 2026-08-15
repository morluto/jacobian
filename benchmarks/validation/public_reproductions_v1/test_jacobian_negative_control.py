"""Regression coverage for the corrupted Jacobian collision witness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.public_reproductions_v1._fixtures import (
    _prepare_case,
    _write_json,
)
from benchmarks.validation.public_reproductions_v1._verifier import _run_verifier


def test_negative_control_recomputes_the_claimed_image(tmp_path: Path) -> None:
    task, app, logs = _prepare_case(tmp_path, "jacobian-negative-control", "claimed")
    assert _run_verifier(task, app, logs).reward == pytest.approx(1.0)

    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"] = {
        "both_points_map_to_claimed_image": True,
        "noninvertibility_verified": True,
    }
    _write_json(submission_path, submission)

    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == pytest.approx(0.0)
    assert rejected.reward == pytest.approx(0.0)
