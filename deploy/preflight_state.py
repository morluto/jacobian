"""Read-only compatibility gate for tenant state selected by a deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import stat
from collections.abc import Iterator
from pathlib import Path

from jacobian.adapters.mcp.remote import load_static_token_file
from jacobian.persistence.migrations import (
    CURRENT_STATE_FORMAT_REVISION,
    STATE_MIGRATIONS,
    SUPPORTED_STATE_FLOOR,
)
from jacobian.persistence.state_health import inspect_state_health


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
        health = inspect_state_health(
            state_root / "tenants" / tenant_key,
            STATE_MIGRATIONS,
            supported_floor=SUPPORTED_STATE_FLOOR,
            current_revision=CURRENT_STATE_FORMAT_REVISION,
        )
        reports.append({"tenant_key": tenant_key, **health.as_dict()})
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
        return path.stat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise PermissionError(f"{label} is not accessible: {exc}") from exc


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
        if prefix.is_dir() and not prefix.is_symlink():
            _require_directory_access(
                prefix,
                label=f"tenant blob prefix {tenant_key} ({prefix.name})",
            )
            for blob in _iter_directory(
                prefix, label=f"tenant blob prefix {tenant_key}"
            ):
                if not blob.is_symlink():
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
