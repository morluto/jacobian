from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1._verifier import _run_verifier

_TASK = "convergence-mode-separation"
TASK = Path(__file__).resolve().parents[3] / (
    "benchmarks/datasets/mathematical-benchmarks-v1/convergence-mode-separation"
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _case(tmp_path: Path):
    root = tmp_path / _TASK / "computed"
    app = root / "app"
    logs = root / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment" / "input.json", app / "input.json")
    submission = json.loads((TASK / "solution" / "submission.json").read_text())
    _write_json(app / "submission.json", submission)
    return TASK, app, logs


def test_typed_oracle_and_unreduced_event_mass_pass(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path / "oracle")
    assert _run_verifier(task, app, logs).reward == pytest.approx(1.0)

    task, app, logs = _case(tmp_path / "unreduced")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["levels"][0]["event_mass"] = {
        "numerator": 2,
        "denominator": 4,
    }
    _write_json(app / "submission.json", submission)
    assert _run_verifier(task, app, logs).reward == pytest.approx(1.0)


def test_string_probe_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path / "string")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["probes"][0]["point"] = "1/3"
    _write_json(app / "submission.json", submission)
    assert _run_verifier(task, app, logs).reward == pytest.approx(0.0)


def test_unreduced_probe_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path / "unreduced")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["probes"][0]["point"] = {
        "numerator": 2,
        "denominator": 6,
    }
    _write_json(app / "submission.json", submission)
    assert _run_verifier(task, app, logs).reward == pytest.approx(1.0)


def test_rejects_unbounded_research_status_fact(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["research_scope"] = {"underlying_problem": "ADJUDICATED"}
    _write_json(app / "submission.json", submission)

    rejected = _run_verifier(task, app, logs)
    assert rejected.reward == pytest.approx(0.0)


def test_result_requires_checked_structural_convergence_arguments(
    tmp_path: Path,
) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["pointwise_argument"] = {
        "hit_count_per_level": 1,
        "miss_count_per_level": "UNSPECIFIED",
    }
    _write_json(app / "submission.json", submission)

    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == pytest.approx(0.0)


def test_equivalent_event_mass_formulas_pass_and_near_misses_fail(
    tmp_path: Path,
) -> None:
    task, app, logs = _case(tmp_path / "half")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["probability_argument"]["event_mass_formula"] = {
        "coefficient": {"numerator": 1, "denominator": 1},
        "base": {"numerator": 1, "denominator": 2},
        "exponent_coefficient": 1,
        "variable": "k",
    }
    _write_json(app / "submission.json", submission)
    assert _run_verifier(task, app, logs).reward == pytest.approx(1.0)

    task, app, logs = _case(tmp_path / "unreduced")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["probability_argument"]["event_mass_formula"] = {
        "coefficient": {"numerator": 2, "denominator": 2},
        "base": {"numerator": 2, "denominator": 1},
        "exponent_coefficient": -1,
        "variable": "k",
    }
    _write_json(app / "submission.json", submission)
    assert _run_verifier(task, app, logs).reward == pytest.approx(1.0)

    task, app, logs = _case(tmp_path / "wrong-base")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["probability_argument"]["event_mass_formula"]["base"] = {
        "numerator": 3,
        "denominator": 1,
    }
    _write_json(app / "submission.json", submission)
    assert _run_verifier(task, app, logs).reward == pytest.approx(0.0)
