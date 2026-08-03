from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support

TASK = "lcm-highly-abundant-scope-audit"


def test_accepts_alternative_earlier_index(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    earlier = submission["result"]["witnesses"][1]
    earlier.update(
        {
            "n": 73,
            "lcm_factorization": earlier["lcm_factorization"]
            + [{"prime": 73, "exponent": 1}],
            "lcm_value": 410555180440430163438262940577600,
            "competitor": 409087987237258561004281340832000,
            "sigma_lcm": 3068535475037360330537152020480000,
            "sigma_competitor": 3071037991057009848454773473280000,
        }
    )
    support._write_json(app / "submission.json", submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("witnesses", 0, "sigma_competitor"), 1),
        (("witnesses", 1, "n"), 97),
        (("witnesses", 1, "exponent_deltas", 0, "prime"), 4),
        (("minimality_claim",), "CONFIRMED"),
    ],
)
def test_rejects_corrupted_or_overclaimed_certificates(
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
