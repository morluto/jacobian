"""Read-only compatibility gate for tenant state selected by a deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def main() -> int:
    args = _parser().parse_args()
    tenant_ids = _tenant_ids(args)
    if not tenant_ids:
        raise SystemExit("no jacobian:use tenant is configured")
    reports = inspect_selected_state(args.state_root, tenant_ids)
    print(json.dumps({"state_preflight": reports}, indent=2, sort_keys=True))
    return 1 if any(bool(report["blocking"]) for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
