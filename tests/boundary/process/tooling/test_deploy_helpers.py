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
    with ArtifactRepository(tmp_path / "tenants" / tenant_key) as repository:
        schema_uri = repository.register_descriptor(
            kind="schema",
            name="current-schema",
            version="1",
            definition={"type": "object"},
        )
        semantics_uri = repository.register_descriptor(
            kind="semantics",
            name="current-semantics",
            version="1",
            definition={"meaning": "deployment preflight fixture"},
        )
        parent_uri = repository.register_descriptor(
            kind="schema",
            name="current-parent",
            version="1",
            definition={"type": "boolean"},
        )
        repository.put(
            schema_uri=schema_uri,
            semantics_uri=semantics_uri,
            payload={"restored": True},
            parents=(parent_uri,),
        )

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


@pytest.mark.parametrize("blob_kind", ("manifest", "payload"))
@pytest.mark.parametrize("damage", ("missing", "corrupt"))
def test_state_preflight_rejects_damaged_referenced_blobs(
    tmp_path: Path, blob_kind: str, damage: str
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
    digest = str(row[0 if blob_kind == "manifest" else 1]).removeprefix("sha256:")
    blob_path = state / "blobs" / "sha256" / digest[:2] / digest[2:]
    if damage == "missing":
        blob_path.unlink()
    else:
        blob_path.write_bytes(b"incomplete restore")

    report = inspect_selected_state(tmp_path, (tenant_id,))[0]

    assert report["status"] == "CORRUPT"
    assert report["blocking"] is True
    assert report["diagnostic"] == (
        f"artifact metadata references a {damage} blob; restore the complete "
        "tenant state"
    )


def test_state_preflight_rejects_inconsistent_manifest_bindings(
    tmp_path: Path,
) -> None:
    tenant_id = "inconsistent-artifact-restore"
    tenant_key = hashlib.sha256(tenant_id.encode()).hexdigest()
    state = tmp_path / "tenants" / tenant_key
    with ArtifactRepository(state) as repository:
        first_uri = repository.register_descriptor(
            kind="schema",
            name="first",
            version="1",
            definition={"type": "object"},
        )
        second_uri = repository.register_descriptor(
            kind="schema",
            name="second",
            version="1",
            definition={"type": "string"},
        )
    with sqlite3.connect(state / "metadata.sqlite3") as connection:
        second_payload_digest = connection.execute(
            "SELECT payload_digest FROM artifacts WHERE artifact_uri = ?",
            (second_uri,),
        ).fetchone()
        assert second_payload_digest is not None
        connection.execute(
            "UPDATE artifacts SET payload_digest = ? WHERE artifact_uri = ?",
            (second_payload_digest[0], first_uri),
        )

    report = inspect_selected_state(tmp_path, (tenant_id,))[0]

    assert report["status"] == "CORRUPT"
    assert report["blocking"] is True
    assert report["diagnostic"] == ("artifact manifest differs from committed metadata")


@pytest.mark.parametrize(
    ("size_delta", "reconciliation_required", "diagnostic"),
    (
        (1, 0, "artifact blob quota differs from restored blob storage"),
        (
            0,
            1,
            "artifact blob quota metadata requires recovery before deployment",
        ),
    ),
)
def test_state_preflight_rejects_unreconciled_blob_quota(
    tmp_path: Path,
    size_delta: int,
    reconciliation_required: int,
    diagnostic: str,
) -> None:
    tenant_id = "stale-blob-quota"
    tenant_key = hashlib.sha256(tenant_id.encode()).hexdigest()
    state = tmp_path / "tenants" / tenant_key
    with ArtifactRepository(state) as repository:
        repository.register_descriptor(
            kind="schema",
            name="quota-fixture",
            version="1",
            definition={"type": "object"},
        )
    with sqlite3.connect(state / "metadata.sqlite3") as connection:
        connection.execute(
            "UPDATE blob_quota SET size_bytes = size_bytes + ?, "
            "reconciliation_required = ? WHERE id = 0",
            (size_delta, reconciliation_required),
        )

    report = inspect_selected_state(tmp_path, (tenant_id,))[0]

    assert report["status"] == "CORRUPT"
    assert report["blocking"] is True
    assert report["diagnostic"] == diagnostic


@pytest.mark.parametrize("missing_table", ("blob_quota", "operation_catalog_snapshots"))
def test_state_preflight_rejects_an_incomplete_current_schema(
    tmp_path: Path, missing_table: str
) -> None:
    tenant_id = "incomplete-current-schema"
    tenant_key = hashlib.sha256(tenant_id.encode()).hexdigest()
    state = tmp_path / "tenants" / tenant_key
    with ArtifactRepository(state):
        pass
    with sqlite3.connect(state / "metadata.sqlite3") as connection:
        connection.execute(f'DROP TABLE "{missing_table}"')

    report = inspect_selected_state(tmp_path, (tenant_id,))[0]

    assert report["status"] == "CORRUPT"
    assert report["blocking"] is True
    assert report["diagnostic"] == "tenant database is missing required schema"


def test_state_preflight_rejects_duplicate_quota_without_schema_constraints(
    tmp_path: Path,
) -> None:
    tenant_id = "constraint-damaged-quota"
    tenant_key = hashlib.sha256(tenant_id.encode()).hexdigest()
    state = tmp_path / "tenants" / tenant_key
    with ArtifactRepository(state):
        pass
    with sqlite3.connect(state / "metadata.sqlite3") as connection:
        connection.execute("ALTER TABLE blob_quota RENAME TO damaged_blob_quota")
        connection.execute(
            """
            CREATE TABLE blob_quota (
                id INTEGER,
                size_bytes INTEGER NOT NULL,
                reconciliation_required INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        connection.execute(
            """
            INSERT INTO blob_quota(id, size_bytes, reconciliation_required)
            SELECT id, size_bytes, reconciliation_required FROM damaged_blob_quota
            """
        )
        connection.execute(
            """
            INSERT INTO blob_quota(id, size_bytes, reconciliation_required)
            SELECT id, size_bytes, reconciliation_required FROM damaged_blob_quota
            """
        )
        connection.execute("DROP TABLE damaged_blob_quota")

    report = inspect_selected_state(tmp_path, (tenant_id,))[0]

    assert report["status"] == "CORRUPT"
    assert report["blocking"] is True
    assert report["diagnostic"] == (
        "tenant database is missing required schema constraints"
    )


def test_state_preflight_does_not_accept_check_text_in_a_default_literal(
    tmp_path: Path,
) -> None:
    tenant_id = "constraint-damaged-quota-checks"
    tenant_key = hashlib.sha256(tenant_id.encode()).hexdigest()
    state = tmp_path / "tenants" / tenant_key
    with ArtifactRepository(state):
        pass
    with sqlite3.connect(state / "metadata.sqlite3") as connection:
        connection.execute("ALTER TABLE blob_quota RENAME TO damaged_blob_quota")
        connection.execute(
            """
            CREATE TABLE blob_quota (
                id INTEGER PRIMARY KEY,
                size_bytes INTEGER NOT NULL,
                reconciliation_required INTEGER NOT NULL DEFAULT 1,
                decoy TEXT DEFAULT 'CHECK (id = 0) CHECK (size_bytes >= 0)
                    CHECK (reconciliation_required IN (0, 1))'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO blob_quota(id, size_bytes, reconciliation_required)
            SELECT id, size_bytes, reconciliation_required FROM damaged_blob_quota
            """
        )
        connection.execute("DROP TABLE damaged_blob_quota")

    report = inspect_selected_state(tmp_path, (tenant_id,))[0]

    assert report["status"] == "CORRUPT"
    assert report["blocking"] is True
    assert report["diagnostic"] == (
        "tenant database is missing required schema constraints"
    )


def test_state_preflight_rejects_a_missing_migration_name_unique_key(
    tmp_path: Path,
) -> None:
    tenant_id = "constraint-damaged-migration-ledger"
    tenant_key = hashlib.sha256(tenant_id.encode()).hexdigest()
    state = tmp_path / "tenants" / tenant_key
    with ArtifactRepository(state):
        pass
    with sqlite3.connect(state / "metadata.sqlite3") as connection:
        connection.execute(
            "ALTER TABLE jacobian_schema_migrations RENAME TO damaged_schema_migrations"
        )
        connection.execute(
            """
            CREATE TABLE jacobian_schema_migrations (
                revision INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                checksum TEXT NOT NULL,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            INSERT INTO jacobian_schema_migrations(
                revision, name, checksum, applied_at
            )
            SELECT revision, name, checksum, applied_at
            FROM damaged_schema_migrations
            """
        )
        connection.execute("DROP TABLE damaged_schema_migrations")

    report = inspect_selected_state(tmp_path, (tenant_id,))[0]

    assert report["status"] == "CORRUPT"
    assert report["blocking"] is True
    assert report["diagnostic"] == (
        "tenant database is missing required schema constraints"
    )


def test_state_preflight_rejects_a_missing_catalog_foreign_key(
    tmp_path: Path,
) -> None:
    tenant_id = "constraint-damaged-catalog-foreign-key"
    tenant_key = hashlib.sha256(tenant_id.encode()).hexdigest()
    state = tmp_path / "tenants" / tenant_key
    with ArtifactRepository(state):
        pass
    with sqlite3.connect(state / "metadata.sqlite3") as connection:
        connection.execute(
            "ALTER TABLE operation_checker_bindings "
            "RENAME TO damaged_operation_checker_bindings"
        )
        connection.execute(
            """
            CREATE TABLE operation_checker_bindings (
                snapshot_revision INTEGER NOT NULL,
                operation_id TEXT NOT NULL,
                binding_index INTEGER NOT NULL CHECK (binding_index >= 0),
                checker_id TEXT NOT NULL,
                manifest_digest TEXT NOT NULL,
                PRIMARY KEY (snapshot_revision, operation_id, binding_index)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO operation_checker_bindings(
                snapshot_revision,
                operation_id,
                binding_index,
                checker_id,
                manifest_digest
            )
            SELECT
                snapshot_revision,
                operation_id,
                binding_index,
                checker_id,
                manifest_digest
            FROM damaged_operation_checker_bindings
            """
        )
        connection.execute("DROP TABLE damaged_operation_checker_bindings")

    report = inspect_selected_state(tmp_path, (tenant_id,))[0]

    assert report["status"] == "CORRUPT"
    assert report["blocking"] is True
    assert report["diagnostic"] == (
        "tenant database is missing required schema constraints"
    )


def test_state_preflight_rejects_a_missing_artifact_timestamp_default(
    tmp_path: Path,
) -> None:
    tenant_id = "column-damaged-artifact-default"
    tenant_key = hashlib.sha256(tenant_id.encode()).hexdigest()
    state = tmp_path / "tenants" / tenant_key
    with ArtifactRepository(state):
        pass
    with sqlite3.connect(state / "metadata.sqlite3") as connection:
        connection.execute("DROP TABLE artifact_parents")
        connection.execute("ALTER TABLE artifacts RENAME TO damaged_artifacts")
        connection.execute(
            """
            CREATE TABLE artifacts (
                artifact_uri TEXT PRIMARY KEY,
                manifest_digest TEXT NOT NULL UNIQUE,
                object_digest TEXT NOT NULL,
                payload_digest TEXT NOT NULL,
                schema_uri TEXT NOT NULL,
                semantics_uri TEXT NOT NULL,
                canonicalizer_digest TEXT NOT NULL,
                summary TEXT NOT NULL,
                committed_at TEXT NOT NULL
            )
            """
        )
        connection.execute("DROP TABLE damaged_artifacts")
        connection.execute(
            """
            CREATE TABLE artifact_parents (
                artifact_uri TEXT NOT NULL,
                position INTEGER NOT NULL,
                parent_uri TEXT NOT NULL,
                PRIMARY KEY (artifact_uri, position),
                FOREIGN KEY (artifact_uri)
                    REFERENCES artifacts(artifact_uri)
                    ON DELETE RESTRICT
            )
            """
        )

    report = inspect_selected_state(tmp_path, (tenant_id,))[0]

    assert report["status"] == "CORRUPT"
    assert report["blocking"] is True
    assert report["diagnostic"] == (
        "tenant database has incompatible required column definitions"
    )


def test_state_preflight_binds_state_format_to_the_migration_ledger(
    tmp_path: Path,
) -> None:
    tenant_id = "mismatched-state-format"
    tenant_key = hashlib.sha256(tenant_id.encode()).hexdigest()
    state = tmp_path / "tenants" / tenant_key
    with ArtifactRepository(state):
        pass
    with sqlite3.connect(state / "metadata.sqlite3") as connection:
        connection.execute(
            "UPDATE jacobian_state_format SET format_revision = 11 WHERE id = 0"
        )

    report = inspect_selected_state(tmp_path, (tenant_id,))[0]

    assert report["status"] == "CORRUPT"
    assert report["blocking"] is True
    assert report["diagnostic"] == (
        "tenant state-format metadata does not match the migration ledger"
    )


def test_state_preflight_accepts_the_supported_migration_source_schema(
    tmp_path: Path,
) -> None:
    tenant_id = "revision-eleven-tenant"
    tenant_key = hashlib.sha256(tenant_id.encode()).hexdigest()
    state = tmp_path / "tenants" / tenant_key
    with ArtifactRepository(state):
        pass
    with sqlite3.connect(state / "metadata.sqlite3") as connection:
        for table in (
            "operation_checker_bindings",
            "active_operation_catalog",
            "operation_catalog_entries",
            "operation_catalog_snapshots",
        ):
            connection.execute(f'DROP TABLE "{table}"')
        connection.execute("DELETE FROM jacobian_schema_migrations WHERE revision = 12")
        connection.execute(
            "UPDATE jacobian_state_format SET format_revision = 11 WHERE id = 0"
        )

    report = inspect_selected_state(tmp_path, (tenant_id,))[0]

    assert report["status"] == "MIGRATION_PENDING"
    assert report["persisted_revision"] == 11
    assert report["blocking"] is False


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


def test_state_preflight_rejects_symlinked_tenant_state(tmp_path: Path) -> None:
    tenant_id = "symlinked-tenant"
    tenant_key = hashlib.sha256(tenant_id.encode()).hexdigest()
    external_state = tmp_path / "external-state"
    with ArtifactRepository(external_state):
        pass
    tenants_root = tmp_path / "state" / "tenants"
    tenants_root.mkdir(parents=True)
    (tenants_root / tenant_key).symlink_to(external_state, target_is_directory=True)

    with pytest.raises(PermissionError, match="must not contain symbolic links"):
        _require_probe_access(tmp_path / "state", (tenant_id,))


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
