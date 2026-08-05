from __future__ import annotations

import errno
import importlib
from pathlib import Path

import pytest

import jacobian.implementation as implementation
from jacobian.implementation import (
    ImplementationError,
    checker_source_digest,
    package_source_digest,
)


def test_digest_binds_helper_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "digest_fixture"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "plugin.py").write_text(
        "from .helper import VALUE\n\ndef run(_request):\n    return VALUE\n",
        encoding="utf-8",
    )
    helper = package / "helper.py"
    helper.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    before = package_source_digest("digest_fixture.plugin:run")
    helper.write_text("VALUE = 2\n", encoding="utf-8")
    after = package_source_digest("digest_fixture.plugin:run")

    assert before != after


def test_digest_binds_package_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "data_digest_fixture"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "plugin.py").write_text(
        "def run(_request):\n    return {}\n",
        encoding="utf-8",
    )
    data = package / "parameters.json"
    data.write_text('{"limit": 1}\n', encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    before = package_source_digest("data_digest_fixture.plugin:run")
    data.write_text('{"limit": 2}\n', encoding="utf-8")
    after = package_source_digest("data_digest_fixture.plugin:run")

    assert before != after


def test_checker_digest_binds_execution_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "checker_fixture"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "checker.py").write_text(
        "def run(_request):\n    return {}\n", encoding="utf-8"
    )
    runtime = tmp_path / "checker_runtime_fixture"
    runtime.mkdir()
    (runtime / "__init__.py").write_text("", encoding="utf-8")
    dependency = runtime / "worker.py"
    dependency.write_text("def main():\n    return 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        implementation,
        "_CHECKER_RUNTIME_ENTRYPOINT",
        "checker_runtime_fixture.worker:main",
    )
    importlib.invalidate_caches()

    before = checker_source_digest("checker_fixture.checker:run")
    dependency.write_text("VALUE = 2\n", encoding="utf-8")
    importlib.invalidate_caches()
    after = checker_source_digest("checker_fixture.checker:run")

    assert before != after


def test_digest_resolution_does_not_execute_package_initializers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "initializer-ran"
    package = tmp_path / "initializer_fixture"
    package.mkdir()
    (package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n",
        encoding="utf-8",
    )
    (package / "plugin.py").write_text(
        "def run(_request):\n    return {}\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    package_source_digest("initializer_fixture.plugin:run")

    assert not marker.exists()


def test_digest_rejects_package_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "symlink_fixture"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("def run(_request):\n    return {}\n", encoding="utf-8")
    link = package / "plugin.py"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        if exc.errno not in {errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP}:
            raise
        pytest.skip(f"this platform cannot create test symlinks: {exc}")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    with pytest.raises(
        ImplementationError,
        match=r"cannot resolve module|symlink",
    ):
        package_source_digest("symlink_fixture.plugin:run")


def test_digest_rejects_entrypoint_path_traversal() -> None:
    with pytest.raises(
        ImplementationError,
        match="entrypoint must use the form",
    ):
        package_source_digest("../outside:run")
