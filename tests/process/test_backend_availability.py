"""Bounded version diagnostics for optional native executables."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from jacobian.backends import BackendName, check_backend


@pytest.mark.parametrize(
    ("backend", "output", "expected"),
    [
        ("singular", "44100", "AVAILABLE"),
        ("singular", "43100", "UNSUPPORTED"),
        ("singular", "garbage", "CHECK_FAILED"),
        ("qepcad", "QEPCAD - Version B 1.74, build", "AVAILABLE"),
        ("qepcad", "QEPCAD - Version B 1.73, build", "UNSUPPORTED"),
        ("qepcad", "garbage", "CHECK_FAILED"),
    ],
)
@pytest.mark.skipif(os.name != "posix", reason="POSIX executable fixture")
def test_version_diagnostic_runs_bounded_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: BackendName,
    output: str,
    expected: str,
) -> None:
    executable = tmp_path / "backend"
    executable.write_text(f"#!{sys.executable}\nprint({output!r})\n")
    executable.chmod(0o755)
    support = tmp_path / "support"
    support.mkdir()
    (support / "default.qepcadrc").write_text("")
    monkeypatch.setenv("QEPCAD_ROOT", str(support))
    monkeypatch.setattr(shutil, "which", lambda command: str(executable))
    assert check_backend(backend).status == expected


@pytest.mark.parametrize("backend", ["singular", "qepcad"])
def test_installed_supported_backend_can_be_checked(backend: BackendName) -> None:
    if shutil.which("Singular" if backend == "singular" else "qepcad") is None:
        pytest.skip("optional native executable is not installed")
    result = check_backend(backend)
    assert result.status == "AVAILABLE", result


def test_availability_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    if shutil.which("Singular") is None:
        pytest.skip("optional native executable is not installed")
    assert check_backend("singular").status == "AVAILABLE"
    monkeypatch.setattr(shutil, "which", lambda command: None)
    assert check_backend("singular").status == "MISSING"
