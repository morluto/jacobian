"""Regression coverage for the Jacobian collision witness."""

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


def test_collision_witness_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _prepare_case(tmp_path, "jacobian-negative-control", "claimed")
    accepted = _run_verifier(task, app, logs)
    assert accepted.reward == pytest.approx(1.0)


def test_wrong_image_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _prepare_case(tmp_path, "jacobian-negative-control", "claimed")
    submission = json.loads((app / "submission.json").read_text())
    # Corrupt the first image so it no longer matches the computed map
    submission["result"]["collision"]["first_image"][0]["num"] = 999
    _write_json(app / "submission.json", submission)
    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == pytest.approx(0.0)
    assert rejected.reward == pytest.approx(0.0)


def test_identical_points_do_not_verify_noninvertibility(
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
    # Both points are now identical — a collision requires distinct points
    submission["result"]["collision"]["second_point"] = [
        {"num": 0, "den": 1},
        {"num": 0, "den": 1},
        {"num": 0, "den": 1},
    ]
    submission["result"]["collision"]["second_image"] = [
        {"num": 0, "den": 1},
        {"num": 0, "den": 1},
        {"num": 0, "den": 1},
    ]
    _write_json(app / "submission.json", submission)
    rejected = _run_verifier(copied_task, app, logs)
    assert rejected.details["correctness"] == pytest.approx(0.0)
    assert rejected.reward == pytest.approx(0.0)


def test_expected_fixture_does_not_rewrite_the_collision(tmp_path: Path) -> None:
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


def test_unrelated_collision_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _prepare_case(tmp_path, "jacobian-negative-control", "claimed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["collision"]["first_point"] = [
        {"num": -2, "den": 1},
        {"num": 1, "den": 1},
        {"num": 2, "den": 1},
    ]
    submission["result"]["collision"]["second_point"] = [
        {"num": 0, "den": 1},
        {"num": 1, "den": 1},
        {"num": -4, "den": 1},
    ]
    _write_json(app / "submission.json", submission)
    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == pytest.approx(0.0)
    assert rejected.reward == pytest.approx(0.0)


def test_short_collision_vector_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _prepare_case(tmp_path, "jacobian-negative-control", "claimed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["collision"]["first_point"] = [
        {"num": -1, "den": 4},
        {"num": 0, "den": 1},
    ]
    _write_json(app / "submission.json", submission)
    rejected = _run_verifier(task, app, logs)
    assert rejected.reward == pytest.approx(0.0)
