from __future__ import annotations

from pathlib import Path

import pytest

from jacobian import __version__
from jacobian.adapters.mcp.deployment_identity import (
    DEPLOYMENT_REVISION_FILE_ENV,
    load_deployment_identity,
)

_REVISION = "a" * 40


def _release(tmp_path: Path) -> tuple[Path, Path]:
    release = tmp_path / "releases" / "candidate"
    implementation = release / "src" / "jacobian" / "server.py"
    implementation.parent.mkdir(parents=True)
    implementation.write_text("# installed implementation\n", encoding="utf-8")
    marker = release / ".git-revision"
    marker.write_text(f"{_REVISION}\n", encoding="ascii")
    return marker, implementation


def test_managed_deployment_identity_binds_release_marker_to_package_tree(
    tmp_path: Path,
) -> None:
    marker, implementation = _release(tmp_path)

    identity = load_deployment_identity(
        environment={DEPLOYMENT_REVISION_FILE_ENV: str(marker)},
        implementation_path=implementation,
    )

    assert identity is not None
    assert identity.revision == _REVISION
    assert identity.package_version == __version__
    assert identity.evidence == "release-marker"


def test_unmanaged_server_has_no_deployment_identity(tmp_path: Path) -> None:
    assert (
        load_deployment_identity(environment={}, implementation_path=tmp_path) is None
    )


def test_deployment_identity_rejects_marker_outside_running_package(
    tmp_path: Path,
) -> None:
    marker, _implementation = _release(tmp_path)
    outside = tmp_path / "other" / "server.py"
    outside.parent.mkdir()
    outside.write_text("# stale implementation\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not contain"):
        load_deployment_identity(
            environment={DEPLOYMENT_REVISION_FILE_ENV: str(marker)},
            implementation_path=outside,
        )


def test_deployment_identity_rejects_noncanonical_marker(tmp_path: Path) -> None:
    marker, implementation = _release(tmp_path)
    marker.write_text(f"{_REVISION[:12]}\n", encoding="ascii")

    with pytest.raises(RuntimeError, match="not canonical"):
        load_deployment_identity(
            environment={DEPLOYMENT_REVISION_FILE_ENV: str(marker)},
            implementation_path=implementation,
        )
