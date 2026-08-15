from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "divisibility-construction-witness"


def test_rejects_float_values_for_integer_witness(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"] = {
        key: float(value) for key, value in submission["result"].items()
    }
    _fixtures._bind_result_evidence(app, submission)
    _fixtures._write_json(app / "submission.json", submission)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0
