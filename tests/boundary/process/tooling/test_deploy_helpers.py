"""Behavioral coverage for deployment preflight, smoke, and retention helpers."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx2
import pytest
from deploy.preflight_state import (
    _drop_privileges,
    _require_probe_access,
    inspect_selected_state,
)
from deploy.preflight_state import main as preflight_main
from deploy.release_retention import prune_releases

from jacobian._deployment_smoke import (
    TRANSIENT_SMOKE_EXIT,
    TransientSmokeError,
    exit_for_smoke_failure,
    is_transient_transport_failure,
    raise_for_http_error,
)
from jacobian.persistence.migrations import CURRENT_STATE_FORMAT_REVISION
from jacobian.storage.repository import ArtifactRepository


def _completed_release(root: Path, name: str, *, modified_ns: int) -> Path:
    release = root / name
    release.mkdir()
    (release / ".git-revision").write_text("a" * 40 + "\n", encoding="utf-8")
    (release / ".release-profile").write_text("lean\n", encoding="utf-8")
    os.utime(release, ns=(modified_ns, modified_ns))
    return release


def test_state_preflight_accepts_missing_and_current_tenant_state(
    tmp_path: Path,
) -> None:
    missing = inspect_selected_state(tmp_path, ("missing-tenant",))
    assert missing[0]["status"] == "MISSING"
    assert missing[0]["blocking"] is False

    tenant_id = "current-tenant"
    tenant_key = hashlib.sha256(tenant_id.encode()).hexdigest()
    with ArtifactRepository(tmp_path / "tenants" / tenant_key):
        pass

    current = inspect_selected_state(tmp_path, (tenant_id,))
    assert current[0]["status"] == "COMPATIBLE"
    assert current[0]["persisted_revision"] == CURRENT_STATE_FORMAT_REVISION
    assert current[0]["blocking"] is False


@pytest.mark.parametrize("database_present", (False, True))
def test_state_preflight_rejects_an_incomplete_existing_tenant(
    tmp_path: Path,
    database_present: bool,
) -> None:
    tenant_id = "incomplete-restored-tenant"
    tenant_key = hashlib.sha256(tenant_id.encode()).hexdigest()
    state = tmp_path / "tenants" / tenant_key
    state.mkdir(parents=True)
    if database_present:
        (state / "metadata.sqlite3").touch()

    report = inspect_selected_state(tmp_path, (tenant_id,))[0]

    assert report["status"] == "CORRUPT"
    assert report["blocking"] is True
    assert report["diagnostic"] == (
        "the tenant directory exists without initialized metadata; "
        "restore or remove the incomplete tenant state"
    )


@pytest.mark.parametrize("missing_blob", ("manifest", "payload"))
def test_state_preflight_rejects_missing_referenced_blobs(
    tmp_path: Path, missing_blob: str
) -> None:
    tenant_id = "incomplete-blob-restore"
    tenant_key = hashlib.sha256(tenant_id.encode()).hexdigest()
    state = tmp_path / "tenants" / tenant_key
    with ArtifactRepository(state) as repository:
        artifact_uri = repository.register_descriptor(
            kind="schema",
            name="restored",
            version="1",
            definition={"type": "object"},
        )
    with sqlite3.connect(state / "metadata.sqlite3") as connection:
        row = connection.execute(
            "SELECT manifest_digest, payload_digest FROM artifacts "
            "WHERE artifact_uri = ?",
            (artifact_uri,),
        ).fetchone()
    assert row is not None
    digest = str(row[0 if missing_blob == "manifest" else 1]).removeprefix("sha256:")
    (state / "blobs" / "sha256" / digest[:2] / digest[2:]).unlink()

    report = inspect_selected_state(tmp_path, (tenant_id,))[0]

    assert report["status"] == "CORRUPT"
    assert report["blocking"] is True
    assert report["diagnostic"] == (
        "artifact metadata references a missing blob; restore the complete tenant state"
    )


def test_state_preflight_rejects_state_below_the_supported_floor(
    tmp_path: Path,
) -> None:
    tenant_id = "legacy-tenant"
    tenant_key = hashlib.sha256(tenant_id.encode()).hexdigest()
    state = tmp_path / "tenants" / tenant_key
    with ArtifactRepository(state):
        pass
    with sqlite3.connect(state / "metadata.sqlite3") as connection:
        connection.execute("DELETE FROM jacobian_schema_migrations WHERE revision > 8")

    report = inspect_selected_state(tmp_path, (tenant_id,))[0]
    assert report["status"] == "UNSUPPORTED"
    assert report["persisted_revision"] == 8
    assert report["blocking"] is True


def test_state_preflight_cli_does_not_print_bearer_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    token = "sentinel-secret-token-that-must-stay-private"
    token_file = tmp_path / "tokens.json"
    token_file.write_text(
        json.dumps(
            {
                "tokens": [
                    {
                        "tenant_id": "tenant-a",
                        "token": token,
                        "scopes": ["jacobian:use"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preflight_state.py",
            "--state-root",
            str(tmp_path / "state"),
            "--auth-tokens-file",
            str(token_file),
        ],
    )
    assert preflight_main() == 0
    output = capsys.readouterr()

    assert '"status": "MISSING"' in output.out
    assert token not in output.out
    assert token not in output.err


def test_state_preflight_drops_to_the_service_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    account = SimpleNamespace(pw_uid=1234, pw_gid=5678)
    monkeypatch.setattr("deploy.preflight_state.pwd.getpwnam", lambda _name: account)
    monkeypatch.setattr("deploy.preflight_state.os.geteuid", lambda: 0)
    monkeypatch.setattr(
        "deploy.preflight_state.os.initgroups",
        lambda name, gid: calls.append(("initgroups", (name, gid))),
    )
    monkeypatch.setattr(
        "deploy.preflight_state.os.setgid",
        lambda gid: calls.append(("setgid", gid)),
    )
    monkeypatch.setattr(
        "deploy.preflight_state.os.setuid",
        lambda uid: calls.append(("setuid", uid)),
    )

    _drop_privileges("jacobian")

    assert calls == [
        ("initgroups", ("jacobian", 5678)),
        ("setgid", 5678),
        ("setuid", 1234),
    ]


def test_state_preflight_rejects_a_tenant_database_the_service_cannot_stat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant_id = "unreadable-tenant"
    tenant_key = hashlib.sha256(tenant_id.encode()).hexdigest()
    state = tmp_path / "tenants" / tenant_key
    state.mkdir(parents=True)
    database = state / "metadata.sqlite3"
    database.touch()
    original_stat = Path.stat

    def guarded_stat(path: Path, *args: object, **kwargs: object) -> os.stat_result:
        if path == database:
            raise PermissionError("service account cannot traverse copied state")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)

    with pytest.raises(PermissionError, match="tenant database"):
        _require_probe_access(tmp_path, (tenant_id,))


def test_state_preflight_rejects_a_readable_but_unwritable_tenant_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant_id = "read-only-database-tenant"
    tenant_key = hashlib.sha256(tenant_id.encode()).hexdigest()
    state = tmp_path / "tenants" / tenant_key
    with ArtifactRepository(state):
        pass
    database = state / "metadata.sqlite3"
    original_access = os.access

    def guarded_access(
        path: os.PathLike[str],
        mode: int,
        *,
        effective_ids: bool = False,
    ) -> bool:
        if Path(path) == database and mode & os.W_OK:
            return False
        return original_access(path, mode, effective_ids=effective_ids)

    monkeypatch.setattr("deploy.preflight_state.os.access", guarded_access)

    with pytest.raises(
        PermissionError, match=r"tenant database.*not readable and writable"
    ):
        _require_probe_access(tmp_path, (tenant_id,))


@pytest.mark.parametrize(
    "relative_path",
    (Path("staging"), Path("blobs/sha256"), Path("blobs/sha256/ab")),
)
def test_state_preflight_rejects_an_unwritable_runtime_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: Path,
) -> None:
    tenant_id = "read-only-runtime-tenant"
    tenant_key = hashlib.sha256(tenant_id.encode()).hexdigest()
    state = tmp_path / "tenants" / tenant_key
    with ArtifactRepository(state):
        pass
    denied = state / relative_path
    denied.mkdir(parents=True, exist_ok=True)
    original_access = os.access

    def guarded_access(
        path: os.PathLike[str],
        mode: int,
        *,
        effective_ids: bool = False,
    ) -> bool:
        if Path(path) == denied and mode & os.W_OK:
            return False
        return original_access(path, mode, effective_ids=effective_ids)

    monkeypatch.setattr("deploy.preflight_state.os.access", guarded_access)

    with pytest.raises(
        PermissionError, match="not readable, writable, and traversable"
    ):
        _require_probe_access(tmp_path, (tenant_id,))


def test_state_preflight_rejects_an_unreadable_existing_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant_id = "unreadable-blob-tenant"
    tenant_key = hashlib.sha256(tenant_id.encode()).hexdigest()
    state = tmp_path / "tenants" / tenant_key
    with ArtifactRepository(state):
        pass
    blob = state / "blobs" / "sha256" / "ab" / ("0" * 62)
    blob.parent.mkdir()
    blob.write_bytes(b"restored artifact")
    original_access = os.access

    def guarded_access(
        path: os.PathLike[str],
        mode: int,
        *,
        effective_ids: bool = False,
    ) -> bool:
        if Path(path) == blob and mode & os.R_OK:
            return False
        return original_access(path, mode, effective_ids=effective_ids)

    monkeypatch.setattr("deploy.preflight_state.os.access", guarded_access)

    with pytest.raises(PermissionError, match=r"tenant blob file.*not readable"):
        _require_probe_access(tmp_path, (tenant_id,))


def test_state_preflight_requires_a_missing_tenant_to_be_creatable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenants_root = tmp_path / "tenants"
    tenants_root.mkdir()
    original_access = os.access

    def guarded_access(
        path: os.PathLike[str],
        mode: int,
        *,
        effective_ids: bool = False,
    ) -> bool:
        if Path(path) == tenants_root and mode & os.W_OK:
            return False
        return original_access(path, mode, effective_ids=effective_ids)

    monkeypatch.setattr("deploy.preflight_state.os.access", guarded_access)

    with pytest.raises(PermissionError, match="tenant state root"):
        _require_probe_access(tmp_path, ("missing-tenant",))


def test_release_retention_keeps_active_and_explicit_previous_release(
    tmp_path: Path,
) -> None:
    releases = tmp_path / "releases"
    releases.mkdir()
    oldest = _completed_release(releases, "111111111111-lean", modified_ns=1)
    newest = _completed_release(releases, "222222222222-lean", modified_ns=3)
    active = _completed_release(releases, "333333333333-lean", modified_ns=2)
    unknown = releases / "operator-notes"
    unknown.mkdir()
    incomplete = releases / "444444444444-lean"
    incomplete.mkdir()
    current = tmp_path / "current"
    current.symlink_to(active)

    pruned = prune_releases(
        releases,
        current,
        retain=2,
        preserve_releases=(oldest,),
    )

    assert pruned == (newest,)
    assert oldest.is_dir()
    assert active.is_dir()
    assert not newest.exists()
    assert unknown.is_dir()
    assert incomplete.is_dir()


def test_release_retention_refuses_to_prune_without_the_active_release(
    tmp_path: Path,
) -> None:
    releases = tmp_path / "releases"
    releases.mkdir()
    active = releases / "not-a-release"
    active.mkdir()
    current = tmp_path / "current"
    current.symlink_to(active)

    with pytest.raises(ValueError, match="active release is not a completed"):
        prune_releases(releases, current, retain=1)


def test_smoke_retry_classification_is_transport_only() -> None:
    transient = ExceptionGroup(
        "transport",
        [httpx2.ConnectError("refused"), httpx2.ReadTimeout("cold start")],
    )
    deterministic = ExceptionGroup(
        "contract",
        [httpx2.ConnectError("refused"), RuntimeError("version mismatch")],
    )

    assert is_transient_transport_failure(transient) is True
    assert is_transient_transport_failure(TransientSmokeError("cold worker")) is True
    assert is_transient_transport_failure(deterministic) is False
    assert is_transient_transport_failure(RuntimeError("catalog mismatch")) is False


@pytest.mark.parametrize(
    ("status_code", "expected"),
    ((401, False), (403, False), (500, False), (502, True), (503, True), (504, True)),
)
def test_smoke_retry_classification_preserves_http_status(
    status_code: int, expected: bool
) -> None:
    request = httpx2.Request("POST", "https://math.example.org/mcp")
    response = httpx2.Response(status_code, request=request)
    with pytest.raises(httpx2.HTTPStatusError) as exc_info:
        response.raise_for_status()

    assert is_transient_transport_failure(exc_info.value) is expected


@pytest.mark.anyio
async def test_smoke_response_hook_surfaces_http_status() -> None:
    request = httpx2.Request("POST", "https://math.example.org/mcp")
    response = httpx2.Response(503, request=request)

    with pytest.raises(httpx2.HTTPStatusError):
        await raise_for_http_error(response)


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    (
        (httpx2.ConnectError("refused"), TRANSIENT_SMOKE_EXIT),
        (RuntimeError("revision mismatch"), 1),
    ),
)
def test_smoke_failure_exit_codes_are_stable(
    failure: Exception,
    expected_code: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        exit_for_smoke_failure("smoke", failure)

    assert exc_info.value.code == expected_code
    assert str(failure) in capsys.readouterr().err
