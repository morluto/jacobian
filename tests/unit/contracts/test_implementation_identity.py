from __future__ import annotations

import errno
import importlib
from pathlib import Path

import pytest

import jacobian.checker_identity as checker_identity
from jacobian.checker_identity import (
    CheckerManifestError,
    build_checker_manifest,
    checker_implementation_digest,
    require_manifest_unchanged,
)
from jacobian.implementation import (
    ImplementationError,
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


def test_checker_manifest_binds_declared_execution_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "checker_fixture"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "checker.py").write_text(
        "def run(_request):\n    return {}\n", encoding="utf-8"
    )
    helper = package / "helper.py"
    helper.write_text("VALUE = 1\n", encoding="utf-8")
    (package / "checker.py").write_text(
        "from .helper import VALUE\n\ndef run(_request):\n    return VALUE\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    before = checker_implementation_digest(
        build_checker_manifest(
            "checker_fixture.checker:run",
            provider_runtime=None,
            passive_contract_uris=(),
        )
    )
    helper.write_text("VALUE = 2\n", encoding="utf-8")
    importlib.invalidate_caches()
    after = checker_implementation_digest(
        build_checker_manifest(
            "checker_fixture.checker:run",
            provider_runtime=None,
            passive_contract_uris=(),
        )
    )

    assert before != after


def test_checker_manifest_separates_source_closures_and_worker_distributions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "checker_closure_fixture"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "checker.py").write_text(
        "from .helper import VALUE\n\ndef run(_request):\n    return VALUE\n",
        encoding="utf-8",
    )
    (package / "helper.py").write_text("VALUE = {}\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    manifest = build_checker_manifest(
        "checker_closure_fixture.checker:run",
        provider_runtime=None,
        passive_contract_uris=(),
    )

    assert {item.module for item in manifest.checker_source_modules} >= {
        "checker_closure_fixture.checker",
        "checker_closure_fixture.helper",
    }
    assert "jacobian.checker_worker" in {
        item.module for item in manifest.worker_source_modules
    }
    distributions = {
        item.distribution.lower().replace("_", "-")
        for item in manifest.python_distributions
    }
    assert {"pydantic", "pydantic-core", "rfc8785"} <= distributions


def test_checker_manifest_rejects_tampered_python_distribution_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "distribution_identity_fixture"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "checker.py").write_text(
        "def run(_request):\n    return {}\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    manifest = build_checker_manifest(
        "distribution_identity_fixture.checker:run",
        provider_runtime=None,
        passive_contract_uris=(),
    )
    original = manifest.python_distributions[0]
    tampered_distribution = original.model_copy(update={"version": "tampered-version"})
    tampered = manifest.model_copy(
        update={
            "python_distributions": (
                tampered_distribution,
                *manifest.python_distributions[1:],
            )
        }
    )

    with pytest.raises(CheckerManifestError, match="changed after authorization"):
        require_manifest_unchanged(tampered)


def test_python_distribution_identity_binds_installed_file_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "bound_dependency"
    package.mkdir()
    module = package / "__init__.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    metadata_root = tmp_path / "bound_dependency-1.0.dist-info"
    metadata_root.mkdir()
    (metadata_root / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: bound-dependency\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (metadata_root / "RECORD").write_text(
        "bound_dependency/__init__.py,,\n"
        "bound_dependency-1.0.dist-info/METADATA,,\n"
        "bound_dependency-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    distribution = checker_identity.metadata.PathDistribution(metadata_root)
    monkeypatch.setattr(
        checker_identity.metadata,
        "distribution",
        lambda _name: distribution,
    )

    before = checker_identity._measure_python_distribution("bound-dependency")
    module.write_text("VALUE = 2\n", encoding="utf-8")
    after = checker_identity._measure_python_distribution("bound-dependency")

    assert before != after


def test_checker_manifest_ignores_unrelated_checker_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "isolated_checker_fixture"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "checker.py").write_text(
        "def run(_request):\n    return {}\n", encoding="utf-8"
    )
    unrelated = package / "unrelated.py"
    unrelated.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    before = checker_implementation_digest(
        build_checker_manifest(
            "isolated_checker_fixture.checker:run",
            provider_runtime=None,
            passive_contract_uris=(),
        )
    )
    unrelated.write_text("VALUE = 2\n", encoding="utf-8")
    importlib.invalidate_caches()
    after = checker_implementation_digest(
        build_checker_manifest(
            "isolated_checker_fixture.checker:run",
            provider_runtime=None,
            passive_contract_uris=(),
        )
    )

    assert before == after


def test_checker_manifest_rejects_a_changed_declared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "changed_checker_fixture"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "checker.py").write_text(
        "from .helper import VALUE\n\ndef run(_request):\n    return VALUE\n",
        encoding="utf-8",
    )
    helper = package / "helper.py"
    helper.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    manifest = build_checker_manifest(
        "changed_checker_fixture.checker:run",
        provider_runtime=None,
        passive_contract_uris=(),
    )

    helper.write_text("VALUE = 2\n", encoding="utf-8")
    importlib.invalidate_caches()

    with pytest.raises(CheckerManifestError, match="changed after authorization"):
        require_manifest_unchanged(manifest)


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
