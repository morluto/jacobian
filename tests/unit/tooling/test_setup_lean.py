from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[3]


def _load(name: str) -> ModuleType:
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_lean_setup_installs_only_the_pinned_toolchain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup_lean = _load("setup_lean")
    repo = tmp_path / "repo"
    (repo / "lean").mkdir(parents=True)
    (repo / "lean" / "lean-toolchain").write_text(
        "leanprover/lean4:v4.31.0\n", encoding="utf-8"
    )
    commands: list[tuple[tuple[str, ...], Path]] = []
    monkeypatch.setattr(setup_lean.shutil, "which", lambda name: f"/bin/{name}")

    def record(arguments: object, cwd: Path) -> int:
        commands.append((tuple(arguments), cwd))
        return 0

    setup_lean.setup_lean(repo, run=record)

    assert commands == [
        (("elan", "toolchain", "install", "leanprover/lean4:v4.31.0"), repo)
    ]
    assert all("update" not in command for command, _cwd in commands)


def test_lean_setup_requires_elan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup_lean = _load("setup_lean")
    repo = tmp_path / "repo"
    (repo / "lean").mkdir(parents=True)
    (repo / "lean" / "lean-toolchain").write_text("leanprover/lean4:v4.31.0\n")
    monkeypatch.setattr(setup_lean.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="requires elan"):
        setup_lean.setup_lean(repo, run=lambda *_args: 0)


def test_doctor_does_not_inspect_python_package_versions() -> None:
    source = (ROOT / "tools" / "doctor_external_tools.py").read_text(encoding="utf-8")

    assert "importlib.metadata" not in source
    assert "pyproject.toml" not in source
    assert "_matches_spec" not in source
    assert "uv sync" not in source
