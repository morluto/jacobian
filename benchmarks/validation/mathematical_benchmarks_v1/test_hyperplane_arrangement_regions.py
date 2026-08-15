from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "hyperplane-arrangement-regions"


def _case(tmp_path: Path):
    return _fixtures._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    _fixtures._bind_result_evidence(app, submission)
    _fixtures._write_json(app / "submission.json", submission)


def test_reference_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0


def test_alternative_order_and_scaling_pass(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    planes = submission["result"]["ordered_planes"]
    planes.reverse()
    for plane in planes:
        plane["coefficients"] = [3 * value for value in plane["coefficients"]]
    for item, increment in zip(planes, [1, 2, 4, 7, 0, 6, 9, 6, 12, 16], strict=True):
        item["increment"] = increment
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0


def test_generic_position_increment_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["ordered_planes"][-1]["increment"] = 22
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_missing_duplicate_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["duplicate_groups"] = []
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_corrupted_plane_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["ordered_planes"][8]["coefficients"][3] = 2
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_unhashable_duplicate_member_is_cleanly_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["duplicate_groups"] = [["cube_z1", []]]
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_float_region_total_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["regions"] = 64.0
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_bound_evidence_accepts_noncanonical_explanation_wording(
    tmp_path: Path,
) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    _fixtures._bind_result_evidence(app, submission)
    evidence = app / "evidence" / "answer.txt"
    marker = next(
        line
        for line in evidence.read_text().splitlines()
        if line.startswith("RESULT_JSON:")
    )
    evidence.write_text(
        marker
        + "\nThe construction is checked by exact affine restrictions and the final count."
    )
    submission["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest()
    )
    _fixtures._write_json(app / "submission.json", submission)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0
