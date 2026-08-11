from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "series-domain-junk-zero"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)


@pytest.mark.parametrize(
    "bound_prose",
    [
        "2^(4k-1)",
        "2**(4*k-1)",
    ],
)
def test_accepts_alternative_denominator_bound_notation(
    tmp_path: Path,
    bound_prose: str,
) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = submission["result"]
    result["reciprocal_denominator"] = 5
    result["real_part"] = "1/5"
    result["general_block_power_exponent"] = {
        "level_coefficient": 4,
        "constant": -1,
    }
    for block in result["blocks"]:
        block["block_sum_power_lower_bound"] = 2 ** (4 * block["level"] - 1)
    (app / "evidence" / "answer.txt").write_text(
        f"For q=5 the general dyadic block lower bound is {bound_prose}, "
        "which proves divergence. "
        "The block sums do not tend to zero. The returned zero is a fallback "
        "artifact, not an analytic-continuation zero.\n"
    )
    _rewrite(app, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_rejects_corrupted_general_bound(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["general_block_power_exponent"]["constant"] = 0
    _rewrite(app, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_affirmative_analytic_claim_in_evidence(tmp_path: Path) -> None:
    """Evidence that affirmatively claims analytic continuation is rejected."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    (app / "evidence" / "answer.txt").write_text(
        "For q=3 the general dyadic block lower bound is 2^(2k-1), "
        "which proves divergence. "
        "The block sums do not tend to zero. The returned zero is a fallback "
        "artifact. This verifies the analytic continuation and proves a "
        "genuine zeta zero.\n"
    )
    _rewrite(app, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 1.0
    assert rejected.details["evidence_validity"] == 0.0
