"""Read-only compatibility gate for tenant state selected by a deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import sqlite3
import stat
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote

from jacobian.adapters.mcp.remote import load_static_token_file
from jacobian.canonical import CanonicalizationError, loads_strict_json
from jacobian.contracts.artifacts import ArtifactManifest
from jacobian.persistence.decoding import (
    PersistenceCorruptionError,
    decode_persisted_model,
)
from jacobian.persistence.migrations import (
    CURRENT_STATE_FORMAT_REVISION,
    STATE_MIGRATIONS,
    SUPPORTED_STATE_FLOOR,
)
from jacobian.persistence.state_health import inspect_state_health
from jacobian.storage.errors import ArtifactNotFoundError
from jacobian.storage.identity import (
    BOOTSTRAP_SCHEMA_URI,
    BOOTSTRAP_SEMANTICS_URI,
    OBJECT_FORMAT_VERSION,
    digest_from_uri,
    framed_digest,
)
from jacobian.storage.models import StorageLimits

_MAX_BLOB_BYTES = StorageLimits().max_artifact_bytes
_MAX_TOTAL_BLOB_BYTES = StorageLimits().max_total_blob_bytes
_COMMON_STATE_SCHEMA = {
    "jacobian_schema_migrations": frozenset(
        {"revision", "name", "checksum", "applied_at"}
    ),
    "jacobian_state_format": frozenset({"id", "format_revision", "recorded_at"}),
    "artifacts": frozenset(
        {
            "artifact_uri",
            "manifest_digest",
            "object_digest",
            "payload_digest",
            "schema_uri",
            "semantics_uri",
            "canonicalizer_digest",
            "summary",
            "committed_at",
        }
    ),
    "artifact_parents": frozenset({"artifact_uri", "position", "parent_uri"}),
    "blob_quota": frozenset({"id", "size_bytes", "reconciliation_required"}),
    "checkers": frozenset(
        {"checker_id", "registration_json", "authorized", "implementation_digest"}
    ),
    "checker_audit": frozenset(
        {"sequence", "checker_id", "action", "reason", "recorded_at"}
    ),
}
_CURRENT_STATE_SCHEMA = {
    "operation_catalog_snapshots": frozenset(
        {
            "revision",
            "package_version",
            "format_version",
            "checker_binding_digest",
            "diagnostics_json",
            "created_at",
        }
    ),
    "operation_catalog_entries": frozenset(
        {
            "snapshot_revision",
            "operation_id",
            "search_card_json",
            "descriptor_json",
            "input_schema_json",
            "output_schema_json",
            "declaration_module",
            "declaration_digest",
        }
    ),
    "active_operation_catalog": frozenset({"id", "snapshot_revision"}),
    "operation_checker_bindings": frozenset(
        {
            "snapshot_revision",
            "operation_id",
            "binding_index",
            "checker_id",
            "manifest_digest",
        }
    ),
}


def _blob_path(state_dir: Path, digest: str) -> Path | None:
    value = digest.removeprefix("sha256:")
    if (
        not digest.startswith("sha256:")
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        return None
    return state_dir / "blobs" / "sha256" / value[:2] / value[2:]


def _read_referenced_blob(
    state_dir: Path, digest: str
) -> tuple[bytes | None, str | None]:
    path = _blob_path(state_dir, digest)
    if path is None:
        return None, "artifact metadata contains an invalid blob reference"
    if path.is_symlink() or not path.is_file():
        return (
            None,
            "artifact metadata references a missing blob; restore the complete "
            "tenant state",
        )

    measured = hashlib.sha256()
    contents = bytearray()
    with path.open("rb") as blob:
        while chunk := blob.read(1024 * 1024):
            measured.update(chunk)
            contents.extend(chunk)
            if len(contents) > _MAX_BLOB_BYTES:
                return None, "artifact metadata references an oversized blob"
    if f"sha256:{measured.hexdigest()}" != digest:
        return (
            None,
            "artifact metadata references a corrupt blob; restore the complete "
            "tenant state",
        )
    return bytes(contents), None


def _decode_manifest(
    state_dir: Path, artifact_uri: str, manifest_digest: str
) -> tuple[ArtifactManifest | None, str | None]:
    manifest_bytes, diagnostic = _read_referenced_blob(state_dir, manifest_digest)
    if diagnostic is not None:
        return None, diagnostic
    if manifest_bytes is None:
        return None, "artifact metadata contains an unreadable manifest"
    try:
        return (
            decode_persisted_model(
                ArtifactManifest,
                manifest_bytes,
                record_kind="artifact_manifest",
                record_id=artifact_uri,
                field="manifest_json",
            ),
            None,
        )
    except PersistenceCorruptionError:
        return None, "artifact metadata contains an invalid manifest"


def _manifest_metadata_diagnostic(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    manifest: ArtifactManifest,
) -> str | None:
    artifact_uri = str(row["artifact_uri"])
    parent_rows = connection.execute(
        """
        SELECT
            parent.parent_uri,
            committed.artifact_uri AS committed_parent_uri
        FROM artifact_parents AS parent
        LEFT JOIN artifacts AS committed
            ON committed.artifact_uri = parent.parent_uri
        WHERE parent.artifact_uri = ?
        ORDER BY parent.position
        """,
        (artifact_uri,),
    ).fetchall()
    if any(parent["committed_parent_uri"] is None for parent in parent_rows):
        return "artifact metadata references an uncommitted parent"
    database_parents = tuple(str(parent["parent_uri"]) for parent in parent_rows)
    if manifest.parents != database_parents:
        return "artifact manifest parents differ from committed metadata"
    if (
        manifest.object_digest != str(row["object_digest"])
        or manifest.payload_digest != str(row["payload_digest"])
        or manifest.schema_uri != str(row["schema_uri"])
        or manifest.semantics_uri != str(row["semantics_uri"])
        or manifest.canonicalizer_digest != str(row["canonicalizer_digest"])
        or manifest.summary != str(row["summary"])
    ):
        return "artifact manifest differs from committed metadata"

    descriptor_uris = (str(manifest.schema_uri), str(manifest.semantics_uri))
    if descriptor_uris != (BOOTSTRAP_SCHEMA_URI, BOOTSTRAP_SEMANTICS_URI):
        committed_references = {
            str(reference["artifact_uri"])
            for reference in connection.execute(
                "SELECT artifact_uri FROM artifacts WHERE artifact_uri IN (?, ?)",
                descriptor_uris,
            ).fetchall()
        }
        if set(descriptor_uris) != committed_references:
            return "artifact metadata references an uncommitted descriptor"
    return None


def _payload_binding_diagnostic(
    state_dir: Path, manifest: ArtifactManifest
) -> str | None:
    payload_bytes, diagnostic = _read_referenced_blob(
        state_dir, str(manifest.payload_digest)
    )
    if diagnostic is not None:
        return diagnostic
    if payload_bytes is None:
        return "artifact metadata contains an unreadable payload"
    recomputed_object_digest = framed_digest(
        OBJECT_FORMAT_VERSION,
        (
            str(manifest.schema_uri).encode(),
            str(manifest.semantics_uri).encode(),
            str(manifest.canonicalizer_digest).encode(),
            payload_bytes,
        ),
    )
    if recomputed_object_digest != manifest.object_digest:
        return "artifact payload differs from its mathematical object digest"
    try:
        loads_strict_json(payload_bytes)
    except CanonicalizationError:
        return "artifact metadata contains an invalid payload"
    return None


def _artifact_binding_diagnostic(
    connection: sqlite3.Connection,
    state_dir: Path,
    row: sqlite3.Row,
) -> str | None:
    artifact_uri = str(row["artifact_uri"])
    try:
        manifest_digest = digest_from_uri(artifact_uri)
    except ArtifactNotFoundError:
        return "artifact metadata contains an invalid blob reference"
    if manifest_digest != str(row["manifest_digest"]):
        return "artifact metadata contains an inconsistent blob reference"

    manifest, diagnostic = _decode_manifest(state_dir, artifact_uri, manifest_digest)
    if diagnostic is not None:
        return diagnostic
    if manifest is None:
        return "artifact metadata contains an unreadable manifest"
    diagnostic = _manifest_metadata_diagnostic(connection, row, manifest)
    if diagnostic is not None:
        return diagnostic
    return _payload_binding_diagnostic(state_dir, manifest)


def _measure_cas_blob(
    state_dir: Path, prefix: Path, blob: Path
) -> tuple[int | None, str | None]:
    if blob.is_symlink() or not blob.is_file():
        return None, "artifact blob storage contains a non-regular entry"
    digest = f"sha256:{prefix.name}{blob.name}"
    if _blob_path(state_dir, digest) != blob:
        return None, "artifact blob storage contains an invalid content address"

    measured = hashlib.sha256()
    size_bytes = 0
    with blob.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            measured.update(chunk)
            size_bytes += len(chunk)
            if size_bytes > _MAX_BLOB_BYTES:
                return None, "artifact blob storage contains an oversized blob"
    if f"sha256:{measured.hexdigest()}" != digest:
        return None, "artifact blob storage contains a corrupt blob"
    return size_bytes, None


def _blob_tree_measurement(state_dir: Path) -> tuple[int | None, str | None]:
    blob_root = state_dir / "blobs" / "sha256"
    if not blob_root.is_dir() or blob_root.is_symlink():
        return None, "artifact blob storage is missing or invalid"

    total_bytes = 0
    for prefix in _iter_directory(blob_root, label="artifact blob storage"):
        if prefix.is_symlink() or not prefix.is_dir():
            return None, "artifact blob storage contains an invalid prefix"
        for blob in _iter_directory(prefix, label="artifact blob prefix"):
            size_bytes, diagnostic = _measure_cas_blob(state_dir, prefix, blob)
            if diagnostic is not None:
                return None, diagnostic
            if size_bytes is None:
                return None, "artifact blob storage contains an unreadable blob"
            total_bytes += size_bytes
            if total_bytes > _MAX_TOTAL_BLOB_BYTES:
                return None, "artifact blob storage exceeds the runtime quota"
    return total_bytes, None


def _quota_diagnostic(connection: sqlite3.Connection, state_dir: Path) -> str | None:
    row = connection.execute(
        "SELECT size_bytes, reconciliation_required FROM blob_quota WHERE id = 0"
    ).fetchone()
    if row is None:
        return "artifact blob quota metadata is missing"
    size_bytes = row["size_bytes"]
    reconciliation_required = row["reconciliation_required"]
    if (
        not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or not isinstance(reconciliation_required, int)
        or isinstance(reconciliation_required, bool)
    ):
        return "artifact blob quota metadata is invalid"
    if reconciliation_required != 0:
        return "artifact blob quota metadata requires recovery before deployment"

    measured_bytes, diagnostic = _blob_tree_measurement(state_dir)
    if diagnostic is not None:
        return diagnostic
    if measured_bytes != size_bytes:
        return "artifact blob quota differs from restored blob storage"
    return None


def _schema_diagnostic(
    connection: sqlite3.Connection, persisted_revision: int
) -> str | None:
    quick_check = connection.execute("PRAGMA quick_check(1)").fetchone()
    if quick_check is None or str(quick_check[0]) != "ok":
        return "tenant database failed the SQLite integrity check"
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        return "tenant database contains broken foreign-key references"

    required_schema = dict(_COMMON_STATE_SCHEMA)
    if persisted_revision >= CURRENT_STATE_FORMAT_REVISION:
        required_schema.update(_CURRENT_STATE_SCHEMA)
    for table, required_columns in required_schema.items():
        actual_columns = {
            str(column["name"])
            for column in connection.execute(f'PRAGMA table_info("{table}")')
        }
        if not required_columns.issubset(actual_columns):
            return "tenant database is missing required schema"

    format_rows = connection.execute(
        "SELECT id, format_revision FROM jacobian_state_format LIMIT 2"
    ).fetchall()
    if (
        len(format_rows) != 1
        or format_rows[0]["id"] != 0
        or format_rows[0]["format_revision"] != persisted_revision
    ):
        return "tenant state-format metadata does not match the migration ledger"
    return None


def _referenced_blob_diagnostic(state_dir: Path, persisted_revision: int) -> str | None:
    database_path = state_dir / "metadata.sqlite3"
    try:
        uri = f"file:{quote(str(database_path.resolve()), safe='/')}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            connection.row_factory = sqlite3.Row
            diagnostic = _schema_diagnostic(connection, persisted_revision)
            if diagnostic is not None:
                return diagnostic
            rows = connection.execute("SELECT * FROM artifacts")
            for row in rows:
                diagnostic = _artifact_binding_diagnostic(connection, state_dir, row)
                if diagnostic is not None:
                    return diagnostic
            return _quota_diagnostic(connection, state_dir)
    except (OSError, sqlite3.DatabaseError) as exc:
        return f"could not inspect artifact blob references: {exc}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the configured tenant stores without creating files or applying "
            "migrations."
        )
    )
    parser.add_argument("--state-root", required=True, type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--tenant-id", action="append")
    selection.add_argument("--auth-tokens-file", type=Path)
    parser.add_argument("--run-as-user")
    return parser


def _tenant_ids(args: argparse.Namespace) -> tuple[str, ...]:
    if args.auth_tokens_file is not None:
        return tuple(
            dict.fromkeys(
                grant.tenant_id
                for grant in load_static_token_file(args.auth_tokens_file)
                if "jacobian:use" in grant.scopes
            )
        )
    return tuple(dict.fromkeys(args.tenant_id))


def inspect_selected_state(
    state_root: Path, tenant_ids: tuple[str, ...]
) -> tuple[dict[str, object], ...]:
    """Return bounded health reports without exposing authentication tokens."""

    reports: list[dict[str, object]] = []
    for tenant_id in tenant_ids:
        tenant_key = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()
        state_dir = state_root / "tenants" / tenant_key
        health = inspect_state_health(
            state_dir,
            STATE_MIGRATIONS,
            supported_floor=SUPPORTED_STATE_FLOOR,
            current_revision=CURRENT_STATE_FORMAT_REVISION,
        )
        report = {"tenant_key": tenant_key, **health.as_dict()}
        if state_dir.exists() and health.status in {"MISSING", "UNINITIALIZED"}:
            report.update(
                status="CORRUPT",
                diagnostic=(
                    "the tenant directory exists without initialized metadata; "
                    "restore or remove the incomplete tenant state"
                ),
                blocking=True,
            )
        elif not health.blocking and health.status in {
            "COMPATIBLE",
            "MIGRATION_PENDING",
        }:
            persisted_revision = health.persisted_revision
            blob_diagnostic: str | None
            if persisted_revision is None:
                blob_diagnostic = "the migration ledger has no persisted revision"
            else:
                blob_diagnostic = _referenced_blob_diagnostic(
                    state_dir, persisted_revision
                )
            if blob_diagnostic is not None:
                report.update(
                    status="CORRUPT",
                    diagnostic=blob_diagnostic,
                    blocking=True,
                )
        reports.append(report)
    return tuple(reports)


def _drop_privileges(user_name: str) -> None:
    account = pwd.getpwnam(user_name)
    effective_uid = os.geteuid()
    if effective_uid == account.pw_uid:
        return
    if effective_uid != 0:
        raise PermissionError(
            f"cannot inspect state as {user_name!r} from effective uid {effective_uid}"
        )
    os.initgroups(user_name, account.pw_gid)
    os.setgid(account.pw_gid)
    os.setuid(account.pw_uid)


def _existing_path_stat(path: Path, *, label: str) -> os.stat_result | None:
    try:
        if path.is_symlink():
            raise PermissionError(f"{label} must not be a symbolic link")
        return path.stat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise PermissionError(f"{label} is not accessible: {exc}") from exc


def _require_no_symlink_components(path: Path, *, label: str) -> None:
    absolute = path.absolute()
    candidate = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        candidate /= component
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise PermissionError(f"{label} is not accessible: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise PermissionError(f"{label} must not contain symbolic links")


def _require_directory_access(path: Path, *, label: str) -> bool:
    metadata = _existing_path_stat(path, label=label)
    if metadata is None:
        return False
    if not stat.S_ISDIR(metadata.st_mode):
        raise PermissionError(f"{label} is not a directory")
    if not os.access(path, os.R_OK | os.W_OK | os.X_OK, effective_ids=True):
        raise PermissionError(
            f"{label} is not readable, writable, and traversable by the service identity"
        )
    return True


def _require_file_access(path: Path, *, label: str) -> None:
    metadata = _existing_path_stat(path, label=label)
    if metadata is None:
        return
    if not stat.S_ISREG(metadata.st_mode):
        raise PermissionError(f"{label} is not a regular file")
    if not os.access(path, os.R_OK | os.W_OK, effective_ids=True):
        raise PermissionError(
            f"{label} is not readable and writable by the service identity"
        )


def _require_readable_file(path: Path, *, label: str) -> None:
    metadata = _existing_path_stat(path, label=label)
    if metadata is None or not stat.S_ISREG(metadata.st_mode):
        return
    if not os.access(path, os.R_OK, effective_ids=True):
        raise PermissionError(f"{label} is not readable by the service identity")


def _iter_directory(path: Path, *, label: str) -> Iterator[Path]:
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                yield Path(entry.path)
    except OSError as exc:
        raise PermissionError(f"{label} is not accessible: {exc}") from exc


def _require_blob_prefix_access(blob_root: Path, *, tenant_key: str) -> None:
    if not blob_root.is_dir():
        return
    for prefix in _iter_directory(blob_root, label=f"tenant blob root {tenant_key}"):
        if prefix.is_symlink():
            raise PermissionError(
                f"tenant blob prefix {tenant_key} must not be a symbolic link"
            )
        if prefix.is_dir():
            _require_directory_access(
                prefix,
                label=f"tenant blob prefix {tenant_key} ({prefix.name})",
            )
            for blob in _iter_directory(
                prefix, label=f"tenant blob prefix {tenant_key}"
            ):
                _require_readable_file(
                    blob,
                    label=(f"tenant blob file {tenant_key} ({prefix.name} prefix)"),
                )


def _require_existing_tenant_access(state_dir: Path, *, tenant_key: str) -> None:
    database = state_dir / "metadata.sqlite3"
    _require_file_access(database, label=f"tenant database {tenant_key}")
    for suffix in ("-journal", "-shm", "-wal", ".lifecycle.lock"):
        _require_file_access(
            database.with_name(database.name + suffix),
            label=f"tenant database runtime file {tenant_key} ({suffix})",
        )
    for name in (".blob-quota.lock", ".transaction-recovery"):
        _require_file_access(
            state_dir / name,
            label=f"tenant runtime file {tenant_key} ({name})",
        )

    for relative in (Path("blobs"), Path("blobs/sha256"), Path("staging")):
        _require_directory_access(
            state_dir / relative,
            label=f"tenant runtime directory {tenant_key} ({relative})",
        )
    _require_blob_prefix_access(state_dir / "blobs" / "sha256", tenant_key=tenant_key)


def _require_probe_access(state_root: Path, tenant_ids: tuple[str, ...]) -> None:
    for tenant_id in tenant_ids:
        tenant_key = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()
        state_dir = state_root / "tenants" / tenant_key
        state_label = f"tenant state {tenant_key}"
        _require_no_symlink_components(state_dir, label=state_label)
        if not _require_directory_access(state_dir, label=state_label):
            tenants_root = state_dir.parent
            if _existing_path_stat(tenants_root, label="tenant state root") is not None:
                _require_directory_access(tenants_root, label="tenant state root")
            elif _existing_path_stat(state_root, label="state root") is not None:
                _require_directory_access(state_root, label="state root")
            continue
        _require_existing_tenant_access(state_dir, tenant_key=tenant_key)


def main() -> int:
    args = _parser().parse_args()
    tenant_ids = _tenant_ids(args)
    if not tenant_ids:
        raise SystemExit("no jacobian:use tenant is configured")
    if args.run_as_user is not None:
        try:
            _drop_privileges(args.run_as_user)
        except (KeyError, OSError) as exc:
            raise SystemExit(
                f"could not assume state service identity: {exc}"
            ) from None
    try:
        _require_probe_access(args.state_root, tenant_ids)
    except OSError as exc:
        raise SystemExit(f"configured tenant state is unreadable: {exc}") from None
    reports = inspect_selected_state(args.state_root, tenant_ids)
    print(json.dumps({"state_preflight": reports}, indent=2, sort_keys=True))
    return 1 if any(bool(report["blocking"]) for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
