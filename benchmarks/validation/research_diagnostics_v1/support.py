from __future__ import annotations

import hashlib
import json
import os
import pathlib
import runpy
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parents[3]
DATASET = ROOT / "benchmarks" / "datasets" / "research-diagnostics-v1"
TASK_EVIDENCE = {
    "jcb-postdoc-004": "counterexample.json",
    "jcb-postdoc-014": "syzygy-certificate.json",
}


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def prepare_case(
    tmp_path: Path,
    task_name: str,
) -> tuple[Path, Path, Path]:
    task = DATASET / task_name
    root = tmp_path / task_name
    app = root / "app"
    logs = root / "logs"
    evidence = app / "evidence"
    evidence.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment" / "input.json", app / "input.json")
    filename = TASK_EVIDENCE[task_name]
    shutil.copy2(task / "solution" / filename, evidence / filename)
    submission = json.loads((task / "solution" / "submission.json").read_text())
    submission["evidence"][0]["sha256"] = digest(evidence / filename)
    write_json(app / "submission.json", submission)
    return task, app, logs


def run_verifier(task: Path, app: Path, logs: Path) -> dict:
    concrete_path = type(pathlib.Path())
    original_path = pathlib.Path
    original_dont_write_bytecode = sys.dont_write_bytecode
    mounts = {
        "/app": app,
        "/tests": task / "tests",
        "/logs/verifier": logs,
    }

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
        sys.path.insert(0, str(task / "tests"))
        runpy.run_path(str(task / "tests" / "verifier.py"), run_name="__main__")
    finally:
        sys.path.remove(str(task / "tests"))
        sys.modules.pop("verifier_support", None)
        sys.dont_write_bytecode = original_dont_write_bytecode
        pathlib.Path = original_path
    return json.loads((logs / "reward.json").read_text())
