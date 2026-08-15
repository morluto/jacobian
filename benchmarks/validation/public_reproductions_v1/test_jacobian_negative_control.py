"""Regression coverage for the corrupted Jacobian collision witness."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from benchmarks.validation.public_reproductions_v1._fixtures import (
    _prepare_case,
    _write_json,
)
from benchmarks.validation.public_reproductions_v1._verifier import _run_verifier


def test_negative_control_recomputes_the_claimed_image(tmp_path: Path) -> None:
    task, app, logs = _prepare_case(tmp_path, "jacobian-negative-control", "claimed")
    accepted = _run_verifier(task, app, logs)
    assert accepted.reward == pytest.approx(1.0)
    submission = json.loads((app / "submission.json").read_text())
    assert submission["result"] == {
        "both_points_map_to_claimed_image": False,
        "noninvertibility_verified": True,
    }

    submission["result"] = {
        "both_points_map_to_claimed_image": True,
        "noninvertibility_verified": True,
    }
    _write_json(app / "submission.json", submission)

    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == pytest.approx(0.0)
    assert rejected.reward == pytest.approx(0.0)


def test_false_false_on_the_frozen_collision_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _prepare_case(
        tmp_path, "jacobian-negative-control", "false-false"
    )
    submission = json.loads((app / "submission.json").read_text())
    submission["result"] = {
        "both_points_map_to_claimed_image": False,
        "noninvertibility_verified": False,
    }
    _write_json(app / "submission.json", submission)
    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == pytest.approx(0.0)
    assert rejected.reward == pytest.approx(0.0)


def test_distinct_non_colliding_points_do_not_verify_noninvertibility(
    tmp_path: Path,
) -> None:
    task, app, logs = _prepare_case(
        tmp_path, "jacobian-negative-control", "no-collision"
    )
    copied_task = tmp_path / "jacobian-negative-control-no-collision"
    shutil.copytree(task, copied_task)
    frozen = json.loads((copied_task / "tests" / "input.json").read_text())
    frozen["second_point"] = [
        {"num": "0", "den": "1"},
        {"num": "0", "den": "1"},
        {"num": "0", "den": "1"},
    ]
    _write_json(copied_task / "tests" / "input.json", frozen)
    _write_json(app / "input.json", frozen)

    submission = json.loads((app / "submission.json").read_text())
    submission["result"] = {
        "both_points_map_to_claimed_image": False,
        "noninvertibility_verified": True,
    }
    _write_json(app / "submission.json", submission)
    rejected = _run_verifier(copied_task, app, logs)
    assert rejected.details["correctness"] == pytest.approx(0.0)
    assert rejected.reward == pytest.approx(0.0)

    submission["result"]["noninvertibility_verified"] = False
    _write_json(app / "submission.json", submission)
    accepted = _run_verifier(copied_task, app, logs)
    assert accepted.details["correctness"] == pytest.approx(1.0)
    assert accepted.reward == pytest.approx(1.0)


def test_expected_fixture_does_not_rewrite_noninvertibility(tmp_path: Path) -> None:
    task, app, logs = _prepare_case(
        tmp_path, "jacobian-negative-control", "expected-mutation"
    )
    copied_task = tmp_path / "jacobian-negative-control-expected"
    shutil.copytree(task, copied_task)
    expected_path = copied_task / "tests" / "expected.json"
    expected = json.loads(expected_path.read_text())
    expected["expected_noninvertibility_verified"] = False
    _write_json(expected_path, expected)
    assert _run_verifier(copied_task, app, logs).reward == pytest.approx(1.0)
