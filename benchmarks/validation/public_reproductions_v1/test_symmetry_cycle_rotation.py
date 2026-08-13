from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.public_reproductions_v1 import support

TASK = "symmetry-cycle-rotation"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)


def test_reference_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    assert support._run_verifier(task, app, logs).reward == 1.0


def test_reordered_edges_and_endpoints_pass(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["edge_orbits"] = [
        [["b", "c"], ["d", "c"], ["d", "a"], ["b", "a"]]
    ]
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs).reward == 1.0


def test_edge_moved_to_separate_orbit_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    edges = submission["result"]["edge_orbits"][0]
    submission["result"]["edge_orbits"] = [edges[:3], edges[3:]]
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs).reward == 0.0


def test_malformed_edge_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["edge_orbits"][0][0] = ["a"]
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs).reward == 0.0
