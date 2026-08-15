from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "generated-lemma-vacuity-audit"


def test_enforces_visible_divisor_witness_bounds(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    audit = submission["result"]["common_divisor_audit"]
    audit["a"] = 1_000_001
    audit["b"] = 1_000_002
    audit["dividends"] = [
        4 * audit["a"] * audit["b"] - 1,
        2 * audit["a"] - 1,
        2 * audit["a"] + 1,
    ]
    audit["original_premise_holds"] = False
    _fixtures._write_json(app / "submission.json", submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
