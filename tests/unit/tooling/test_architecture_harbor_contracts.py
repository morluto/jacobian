"""Architecture checker contracts for Harbor public task schemas."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from tools.check_architecture import check_architecture

_ROOT = Path(__file__).resolve().parents[3]
_REAL_TASK = (
    _ROOT
    / "benchmarks/datasets/mathematical-benchmarks-v1/finite-field-irreducibility-repair"
)
_REAL_CONJECTURE_TASK = (
    _ROOT
    / "benchmarks/datasets/conjecture-probes-v1/vizing-bounded-cartesian-products"
)


def _write(root: Path, relative: str, source: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _copy_real_task(root: Path, task_name: str) -> None:
    """Copy a real mathematical-benchmarks-v1 task into the test tree."""
    dest = root / "benchmarks/datasets/mathematical-benchmarks-v1" / task_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(_REAL_TASK, dest)


def _copy_real_conjecture_task(root: Path, task_name: str) -> None:
    dest = root / "benchmarks/datasets/conjecture-probes-v1" / task_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(_REAL_CONJECTURE_TASK, dest)


def test_public_contract_drift_is_flagged(tmp_path: Path) -> None:
    _copy_real_task(tmp_path, "test-task")
    schema_path = (
        tmp_path
        / "benchmarks/datasets/mathematical-benchmarks-v1/test-task/environment/submission_schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["required"] = ["task_id"]
    schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")

    report = check_architecture(tmp_path)
    drift = [v for v in report.violations if v.code == "public-contract-drift"]
    assert len(drift) >= 1
    assert "test-task" in drift[0].path


def test_public_contract_no_drift_passes(tmp_path: Path) -> None:
    _copy_real_task(tmp_path, "test-task")
    report = check_architecture(tmp_path)
    drift = [v for v in report.violations if v.code == "public-contract-drift"]
    assert drift == []


def test_conjecture_public_contract_no_drift_passes(tmp_path: Path) -> None:
    _copy_real_conjecture_task(tmp_path, "test-task")
    report = check_architecture(tmp_path)
    drift = [v for v in report.violations if v.code == "public-contract-drift"]
    assert drift == []


def test_missing_public_contract_is_a_violation(tmp_path: Path) -> None:
    task_dir = (
        tmp_path
        / "benchmarks/datasets/mathematical-benchmarks-v1"
        / "missing-contract-task"
    )
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "task.toml").write_text("", encoding="utf-8")

    report = check_architecture(tmp_path)
    drift = [v for v in report.violations if v.code == "public-contract-drift"]
    assert len(drift) == 1
    assert "missing" in drift[0].message.lower()
    assert "missing-contract-task" in drift[0].path


def test_non_task_directory_does_not_require_contract(tmp_path: Path) -> None:
    """Directories without task.toml (e.g. jobs/, members/) are not tasks."""
    meta_dir = tmp_path / "benchmarks/datasets/mathematical-benchmarks-v1" / "metadata"
    (meta_dir / "tests").mkdir(parents=True)
    report = check_architecture(tmp_path)
    drift = [v for v in report.violations if v.code == "public-contract-drift"]
    assert drift == []
