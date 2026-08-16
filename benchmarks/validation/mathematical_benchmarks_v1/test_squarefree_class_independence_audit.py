from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures, _verifier

TASK = "squarefree-class-independence-audit"


def test_accepts_permuted_quadratic_residues(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "permuted")
    submission = json.loads((app / "submission.json").read_text())
    residues = submission["result"]["modular_obstruction"]["quadratic_residues"]
    submission["result"]["modular_obstruction"]["quadratic_residues"] = list(
        reversed(residues)
    )
    _fixtures._write_json(app / "submission.json", submission)
    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == 1.0
