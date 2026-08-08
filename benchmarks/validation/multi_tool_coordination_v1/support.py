from __future__ import annotations

import hashlib
import json
import os
import pathlib
import runpy
import shutil
import sys
from pathlib import Path

from benchmarks.validation._verifier_child import VerifierOutput, _read_verifier_output

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "benchmarks/datasets/multi-tool-coordination-v1"


def task(task_id: str) -> Path:
    path = DATASET / task_id
    assert path.is_dir()
    return path


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def prepare(tmp_path: Path, task_id: str) -> tuple[Path, Path, Path]:
    task_path = task(task_id)
    root = tmp_path / task_id
    app = root / "app"
    logs = root / "logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task_path / "environment/input.json", app / "input.json")
    shutil.copy2(
        task_path / "solution/certificate.json", app / "evidence/certificate.json"
    )
    shutil.copy2(task_path / "solution/submission.json", app / "submission.json")
    return task_path, app, logs


def rebind_evidence(app: Path, submission: dict) -> None:
    evidence = {
        "schema_version": "1",
        "task_id": submission["task_id"],
        "result": submission["result"],
        "scope": submission["scope"],
        "completeness": submission["completeness"],
        "limitations": submission["limitations"],
    }
    evidence_path = app / "evidence/certificate.json"
    write_json(evidence_path, evidence)
    submission["evidence"] = [
        {"path": "evidence/certificate.json", "sha256": digest(evidence_path)}
    ]
    write_json(app / "submission.json", submission)


def run_verifier(task_path: Path, app: Path, logs: Path) -> VerifierOutput:
    concrete_path = type(pathlib.Path())
    original_path = pathlib.Path
    original_dont_write_bytecode = sys.dont_write_bytecode
    mounts = {"/app": app, "/tests": task_path / "tests", "/logs/verifier": logs}

    def mapped_path(value: os.PathLike[str] | str = ".") -> Path:
        raw = os.fspath(value)
        for prefix, target in mounts.items():
            if raw == prefix:
                return concrete_path(target)
            if raw.startswith(prefix + "/"):
                return concrete_path(target) / raw.removeprefix(prefix + "/")
        return concrete_path(raw)

    try:
        pathlib.Path = mapped_path  # type: ignore[assignment]
        sys.dont_write_bytecode = True
        sys.modules.pop("verifier_support", None)
        sys.path.insert(0, str(task_path / "tests"))
        runpy.run_path(str(task_path / "tests/verifier.py"), run_name="__main__")
    finally:
        sys.path.remove(str(task_path / "tests"))
        sys.modules.pop("verifier_support", None)
        sys.dont_write_bytecode = original_dont_write_bytecode
        pathlib.Path = original_path
    return _read_verifier_output(logs)
