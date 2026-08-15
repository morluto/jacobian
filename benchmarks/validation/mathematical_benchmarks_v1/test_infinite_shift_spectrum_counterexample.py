from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "infinite-shift-spectrum-counterexample"


def _case(tmp_path: Path):
    return _fixtures._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    _fixtures._write_json(app / "submission.json", submission)


def test_uses_result_only_protocol(tmp_path: Path) -> None:
    _fixtures.assert_result_witness_protocol(tmp_path, TASK)


def test_accepts_reversed_operator_orientation(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = submission["result"]
    result["orientation"] = "S_LEFT_T_RIGHT"
    result["zero_eigenvalue_product"] = "TS"
    result["identity_product"] = "ST"
    for action in result["actions"]:
        index = action["basis_index"]
        action.update(
            {
                "s_output": None if index == 0 else index - 1,
                "t_output": index + 1,
                "st_output": index,
                "ts_output": None if index == 0 else index,
            }
        )
    _rewrite(app, submission)

    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0


def test_rejects_corrupted_composition(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["actions"][5]["st_output"] = 4
    _rewrite(app, submission)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0
