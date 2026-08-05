"""Read-only health checks for persisted migration metadata."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from jacobian.persistence.migrations import (
    CURRENT_STATE_FORMAT_REVISION,
    STATE_MIGRATIONS,
    SUPPORTED_STATE_FLOOR,
)
from jacobian.persistence.state_health import StateHealth, inspect_state_health
from jacobian.storage.repository import ArtifactRepository


def _inspect(root: Path) -> StateHealth:
    return inspect_state_health(
        root,
        STATE_MIGRATIONS,
        supported_floor=SUPPORTED_STATE_FLOOR,
        current_revision=CURRENT_STATE_FORMAT_REVISION,
    )


def test_state_health_reads_ledger_without_mutating_it(tmp_path: Path) -> None:
    missing = _inspect(tmp_path)
    assert missing.status == "MISSING"
    assert not (tmp_path / "metadata.sqlite3").exists()

    with ArtifactRepository(tmp_path):
        pass

    assert _inspect(tmp_path).status == "COMPATIBLE"


def test_state_health_reports_historical_checksum_mismatch(tmp_path: Path) -> None:
    with ArtifactRepository(tmp_path):
        pass

    connection = sqlite3.connect(tmp_path / "metadata.sqlite3")
    try:
        connection.execute(
            "UPDATE jacobian_schema_migrations SET checksum = ? WHERE revision = 3",
            ("sha256:historical-checksum",),
        )
        connection.commit()
    finally:
        connection.close()

    health = _inspect(tmp_path)
    assert health.status == "INCOMPATIBLE"
    assert health.blocking is True
    assert [(item.revision, item.name) for item in health.mismatches] == [
        (3, "runtime-service-schema-v1")
    ]
