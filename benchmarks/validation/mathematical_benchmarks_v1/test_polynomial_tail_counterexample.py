from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "polynomial-tail-counterexample"


def test_rejects_non_array_witness_fields(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"] = {
        "p_coefficients": {"2": None, "8/3": None, "2/3": None, "0": None},
        "q_coefficients": {"1": None, "199": None, "9900": None},
        "p_roots": {"-1": None, "-1/3": None, "0": None},
        "q_roots": {"-100": None, "-99": None},
        "x1": "0",
        "x2": "1/100",
    }
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_corrupted_x2(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["x2"] = "1"
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0
