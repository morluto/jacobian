from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures, _verifier

TASK = "chebotarev-fixed-point-proof-audit"


def test_result_witness_protocol(tmp_path: Path) -> None:
    _fixtures.assert_result_witness_protocol(tmp_path, TASK)


def test_rejects_mod2_factor_string(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["mod2_factor"] = "(x+1)^4"
    _fixtures._write_json(path, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_rejects_mod2_factor_multiplicity_above_schema_bound(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["mod2_factor"]["multiplicity"] = 9
    _fixtures._write_json(path, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0
