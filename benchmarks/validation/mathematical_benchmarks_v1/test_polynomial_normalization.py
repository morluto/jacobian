from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures, _verifier

TASK = "polynomial-normalization"


def test_rejects_string_coerced_and_noncanonical_coefficients(
    tmp_path: Path,
) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "string-coefficient")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["terms"][0]["coefficient"] = "3"
    _fixtures._write_json(app / "submission.json", submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0

    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "noncanonical")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["terms"][0]["coefficient"] = {
        "numerator": 6,
        "denominator": 2,
    }
    _fixtures._write_json(app / "submission.json", submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0
