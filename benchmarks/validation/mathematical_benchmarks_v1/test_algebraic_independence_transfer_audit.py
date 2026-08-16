from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "algebraic-independence-transfer-audit"


def _prepare(tmp_path: Path):
    return _fixtures._prepare_case(tmp_path, TASK, "computed")


def _mutate(tmp_path: Path, mutation):
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    mutation(submission)
    _fixtures._write_json(app / "submission.json", submission)
    return _verifier._run_verifier(task, app, logs)


def test_oracle_transfer_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    reward = _verifier._run_verifier(task, app, logs)
    assert reward.details["correctness"] == 1.0
    assert reward.reward == 1.0


def test_visible_input_tamper_preserves_math_diagnostic(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    (app / "input.json").write_text("{}")
    reward = _verifier._run_verifier(task, app, logs)
    assert reward.details["correctness"] == 0.0
    assert reward.reward == 0.0


def test_alternative_term_order_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["norm_polynomial"].reverse()
    _fixtures._write_json(app / "submission.json", submission)
    reward = _verifier._run_verifier(task, app, logs)
    assert reward.reward == 1.0


def test_corrupted_inverse_coefficient_is_rejected(tmp_path: Path) -> None:
    reward = _mutate(
        tmp_path,
        lambda s: s["result"]["d2_delta_numerator"][0].__setitem__(
            "coefficient", {"numerator": 12, "denominator": 1}
        ),
    )
    assert reward.details["correctness"] == 0.0
    assert reward.reward == 0.0


def test_duplicate_monomial_is_rejected(tmp_path: Path) -> None:
    reward = _mutate(
        tmp_path,
        lambda s: s["result"]["q_numerator"].append(
            dict(s["result"]["q_numerator"][0])
        ),
    )
    assert reward.details["correctness"] == 0.0
    assert reward.reward == 0.0


def test_corrupted_conjugate_norm_is_rejected(tmp_path: Path) -> None:
    reward = _mutate(
        tmp_path,
        lambda s: s["result"]["norm_polynomial"][4].__setitem__(
            "coefficient", {"numerator": 2, "denominator": 1}
        ),
    )
    assert reward.details["correctness"] == 0.0
    assert reward.reward == 0.0


def test_unreduced_rational_is_accepted(tmp_path: Path) -> None:
    reward = _mutate(
        tmp_path,
        lambda s: s["result"]["p_numerator"][0].__setitem__(
            "coefficient", {"numerator": 2, "denominator": 2}
        ),
    )
    assert reward.details["correctness"] == 1.0
    assert reward.reward == 1.0


def test_malformed_result_does_not_crash(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"] = "not-a-dict"
    _fixtures._write_json(app / "submission.json", submission)
    reward = _verifier._run_verifier(task, app, logs)
    assert reward.details["correctness"] == 0.0
    assert reward.reward == 0.0
