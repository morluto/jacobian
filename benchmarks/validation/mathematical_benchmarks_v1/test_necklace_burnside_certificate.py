import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1._verifier import _run_verifier

ROOT = Path(__file__).resolve().parents[3]
TASK = (
    ROOT
    / "benchmarks/datasets/mathematical-benchmarks-v1/necklace-burnside-certificate"
)
TASK_NAME = "necklace-burnside-certificate"


def load_verifier():
    saved_path = sys.path[:]
    saved_modules = dict(sys.modules)
    try:
        sys.path.insert(0, str(TASK / "tests"))
        spec = importlib.util.spec_from_file_location(
            "necklace_verifier", TASK / "tests/verifier.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = saved_path
        sys.modules.clear()
        sys.modules.update(saved_modules)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _case(tmp_path: Path):
    root = tmp_path / TASK_NAME / "computed"
    app = root / "app"
    logs = root / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment" / "input.json", app / "input.json")
    submission = json.loads((TASK / "solution" / "submission.json").read_text())
    _write_json(app / "submission.json", {"result": submission["result"]})
    return TASK, app, logs


def test_independent_orbit_derivation():
    result = load_verifier().derive()
    assert result["valid_labelled_words"] == 2206
    assert result["burnside_numerator"] == 2816
    assert result["orbit_count"] == 88
    assert len(result["canonical_representatives"]) == 88


def test_wraparound_and_reflection_are_material():
    verifier = load_verifier()
    assert not verifier.valid(tuple(map(int, "0010101010101010")))
    result = verifier.derive()
    assert result["reflection_fixed_counts"] == [42, 26] * 8


def test_corrupt_fixed_count_or_orbit_representative_is_rejected():
    verifier = load_verifier()
    result = verifier.derive()
    result["rotation_fixed_counts"][0] -= 1
    assert not verifier.matches(result)
    result = verifier.derive()
    result["canonical_representatives"].pop()
    assert not verifier.matches(result)


def test_canonical_solution_receives_full_reward(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    result = _run_verifier(task, app, logs)
    assert result.details["correctness"] == 1.0
    assert result.reward == 1.0


def test_rejects_boolean_fixed_counts(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = dict(submission["result"])
    result["rotation_fixed_counts"] = [bool(x) for x in result["rotation_fixed_counts"]]
    _write_json(app / "submission.json", {"result": result})

    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_float_fixed_counts(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = dict(submission["result"])
    result["rotation_fixed_counts"] = [
        float(x) for x in result["rotation_fixed_counts"]
    ]
    _write_json(app / "submission.json", {"result": result})

    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_extra_envelope_field_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["extra_field"] = "unexpected"
    _write_json(app / "submission.json", submission)
    result = _run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_float_in_result_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    corrupted = dict(submission["result"])
    corrupted["orbit_count"] = 88.0
    _write_json(app / "submission.json", {"result": corrupted})
    result = _run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


@pytest.mark.parametrize(
    "corruption",
    (
        "empty",
        "duplicate",
        "non_binary",
    ),
)
def test_representative_schema_violations_are_rejected(
    tmp_path: Path, corruption: str
) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    corrupted = dict(submission["result"])
    reps = list(corrupted["canonical_representatives"])
    if corruption == "empty":
        corrupted["canonical_representatives"] = []
    elif corruption == "duplicate":
        corrupted["canonical_representatives"] = [reps[0], reps[0]]
    else:
        corrupted["canonical_representatives"] = ["x" * 16]
    _write_json(app / "submission.json", {"result": corrupted})
    result = _run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_input_tamper_gates_reward(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    (app / "input.json").write_text("{}")
    result = _run_verifier(task, app, logs)
    assert result.details["correctness"] == 1.0
    assert result.reward == 0.0
