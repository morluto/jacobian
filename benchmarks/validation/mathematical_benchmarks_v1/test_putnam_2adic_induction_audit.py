from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "putnam-2adic-induction-audit"


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("valuation_induction", "sub_one_term_lower_bounds", 1), [1, 2]),
        (("target_transfer", "b_difference"), [2, 3]),
        (("finite_testing_role",), "FINITE_CASES_PROVE_ALL_K"),
    ],
)
def test_rejects_corrupted_induction_certificates(
    tmp_path: Path,
    path: tuple[object, ...],
    replacement: object,
) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    target = submission["result"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0
