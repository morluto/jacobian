from __future__ import annotations

import zipfile
from pathlib import Path

from tools.check_wheel_contents import wheel_content_failures


def _source(root: Path, relative: str) -> None:
    path = root / "src" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _wheel(path: Path, members: tuple[str, ...]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for member in members:
            archive.writestr(member, "")


def test_wheel_matches_installed_source_modules(tmp_path: Path) -> None:
    _source(tmp_path, "jacobian/__init__.py")
    _source(tmp_path, "jacobian/math/__init__.py")
    wheel = tmp_path / "jacobian.whl"
    _wheel(wheel, ("jacobian/__init__.py", "jacobian/math/__init__.py"))

    assert wheel_content_failures(tmp_path, wheel) == ()


def test_wheel_rejects_missing_and_repository_only_modules(tmp_path: Path) -> None:
    _source(tmp_path, "jacobian/__init__.py")
    _source(tmp_path, "jacobian/math/__init__.py")
    wheel = tmp_path / "jacobian.whl"
    _wheel(
        wheel,
        (
            "jacobian/__init__.py",
            "jacobian/contracts/value.py",
            "deploy/smoke.py",
            "benchmarks/tooling/telemetry.py",
        ),
    )

    failures = wheel_content_failures(tmp_path, wheel)
    assert "wheel is missing installed module: jacobian/math/__init__.py" in failures
    assert any("retired runtime path" in failure for failure in failures)
    assert sum("repository-only tooling" in failure for failure in failures) == 2
