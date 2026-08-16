from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.public_reproductions_v1._fixtures import (
    _prepare_case,
    _write_json,
)
from benchmarks.validation.public_reproductions_v1._verifier import _run_verifier

TASK = "symmetry-cycle-rotation"


def _case(tmp_path: Path):
    return _prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    _write_json(app / "submission.json", {"result": submission["result"]})


def test_reference_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    assert _run_verifier(task, app, logs).reward == 1.0


def test_reordered_edges_and_endpoints_pass(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["edge_orbits"] = [
        [["b", "c"], ["d", "c"], ["d", "a"], ["b", "a"]]
    ]
    _rewrite(app, submission)
    assert _run_verifier(task, app, logs).reward == 1.0


def test_edge_moved_to_separate_orbit_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    edges = submission["result"]["edge_orbits"][0]
    submission["result"]["edge_orbits"] = [edges[:3], edges[3:]]
    _rewrite(app, submission)
    assert _run_verifier(task, app, logs).reward == 0.0


def test_malformed_edge_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["edge_orbits"][0][0] = ["a"]
    _rewrite(app, submission)
    assert _run_verifier(task, app, logs).reward == 0.0


def test_undeclared_witness_key_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["witness"] = [
        {"path": "evidence/answer.txt", "sha256": "sha256:" + "0" * 64}
    ]
    _write_json(app / "submission.json", submission)
    assert _run_verifier(task, app, logs).reward == 0.0


def test_result_only_submission_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    _rewrite(app, submission)
    assert _run_verifier(task, app, logs).reward == 1.0
