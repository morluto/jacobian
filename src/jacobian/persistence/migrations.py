"""Ordered SQLite migrations for the shared Jacobian state database."""

from __future__ import annotations

import sqlite3

from jacobian.persistence.database import Migration

_ARTIFACT_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS artifacts (
        artifact_uri TEXT PRIMARY KEY,
        manifest_digest TEXT NOT NULL UNIQUE,
        object_digest TEXT NOT NULL,
        payload_digest TEXT NOT NULL,
        schema_uri TEXT NOT NULL,
        semantics_uri TEXT NOT NULL,
        canonicalizer_digest TEXT NOT NULL,
        summary TEXT NOT NULL,
        committed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS artifacts_object_digest ON artifacts(object_digest)",
    """
    CREATE TABLE IF NOT EXISTS artifact_parents (
        artifact_uri TEXT NOT NULL,
        position INTEGER NOT NULL,
        parent_uri TEXT NOT NULL,
        PRIMARY KEY (artifact_uri, position),
        FOREIGN KEY (artifact_uri)
            REFERENCES artifacts(artifact_uri)
            ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS blob_quota (
        id INTEGER PRIMARY KEY CHECK (id = 0),
        size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0)
    )
    """,
)
_ARTIFACT_SCHEMA = "\n-- statement boundary --\n".join(_ARTIFACT_SCHEMA_STATEMENTS)

_QUOTA_RECONCILIATION = """
Add blob_quota.reconciliation_required as a fail-closed durable marker with a
default of one for legacy rows. Fresh and legacy stores use the same column
contract after this revision.
"""

_CHECKER_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS checkers (
        checker_id TEXT PRIMARY KEY,
        registration_json BLOB NOT NULL,
        authorized INTEGER NOT NULL CHECK (authorized IN (0, 1)),
        executable_digest TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS checker_audit (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        checker_id TEXT NOT NULL,
        action TEXT NOT NULL CHECK (action IN ('AUTHORIZED', 'REVOKED')),
        reason TEXT NOT NULL,
        recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (checker_id)
            REFERENCES checkers(checker_id) ON DELETE RESTRICT
    )
    """,
)
_RUNTIME_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_uri TEXT PRIMARY KEY,
        state TEXT NOT NULL,
        snapshot_json BLOB NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS experiment_recovery_failures (
        experiment_uri TEXT PRIMARY KEY,
        detected_at TEXT NOT NULL,
        snapshot_digest TEXT NOT NULL,
        detail TEXT NOT NULL,
        FOREIGN KEY (experiment_uri)
            REFERENCES experiments(experiment_uri) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS search_experiments (
        experiment_uri TEXT PRIMARY KEY,
        state TEXT NOT NULL,
        snapshot_json BLOB NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS search_idempotency (
        idempotency_key TEXT PRIMARY KEY,
        request_digest TEXT NOT NULL,
        experiment_uri TEXT NOT NULL UNIQUE,
        FOREIGN KEY (experiment_uri)
            REFERENCES search_experiments(experiment_uri) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS search_events (
        experiment_uri TEXT NOT NULL,
        sequence INTEGER NOT NULL,
        event_json BLOB NOT NULL,
        event_digest TEXT NOT NULL UNIQUE,
        PRIMARY KEY (experiment_uri, sequence),
        FOREIGN KEY (experiment_uri)
            REFERENCES search_experiments(experiment_uri) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS search_recovery_failures (
        experiment_uri TEXT PRIMARY KEY,
        detected_at TEXT NOT NULL,
        snapshot_digest TEXT NOT NULL,
        detail TEXT NOT NULL,
        FOREIGN KEY (experiment_uri)
            REFERENCES search_experiments(experiment_uri) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS search_events_no_update
    BEFORE UPDATE ON search_events
    BEGIN
        SELECT RAISE(ABORT, 'search lifecycle events are append-only');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS search_events_no_delete
    BEFORE DELETE ON search_events
    BEGIN
        SELECT RAISE(ABORT, 'search lifecycle events are append-only');
    END
    """,
    """
    CREATE TABLE IF NOT EXISTS installed_plugins (
        plugin_id TEXT PRIMARY KEY,
        domain_id TEXT NOT NULL,
        domain_version TEXT NOT NULL,
        registry_snapshot_uri TEXT,
        installed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    *_CHECKER_SCHEMA_STATEMENTS,
    """
    CREATE TABLE IF NOT EXISTS workspaces (
        workspace_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        root_branch_id TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL,
        FOREIGN KEY (root_branch_id)
            REFERENCES workspace_branches(branch_id)
            ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS workspace_branches (
        branch_id TEXT PRIMARY KEY,
        workspace_id TEXT NOT NULL,
        alias TEXT NOT NULL,
        head_revision_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (workspace_id, alias),
        FOREIGN KEY (workspace_id)
            REFERENCES workspaces(workspace_id) ON DELETE RESTRICT,
        FOREIGN KEY (head_revision_id)
            REFERENCES workspace_revisions(revision_id)
            ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS workspace_revisions (
        revision_id TEXT PRIMARY KEY,
        revision_artifact_uri TEXT NOT NULL UNIQUE,
        workspace_id TEXT NOT NULL,
        branch_id TEXT NOT NULL,
        parent_revision_id TEXT,
        request_digest TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (revision_artifact_uri)
            REFERENCES artifacts(artifact_uri) ON DELETE RESTRICT,
        FOREIGN KEY (workspace_id)
            REFERENCES workspaces(workspace_id) ON DELETE RESTRICT,
        FOREIGN KEY (branch_id)
            REFERENCES workspace_branches(branch_id) ON DELETE RESTRICT,
        FOREIGN KEY (parent_revision_id)
            REFERENCES workspace_revisions(revision_id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS workspace_idempotency (
        idempotency_key TEXT PRIMARY KEY,
        operation TEXT NOT NULL,
        request_digest TEXT NOT NULL,
        response_json BLOB NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS workspace_scratch (
        scratch_id TEXT PRIMARY KEY,
        workspace_id TEXT NOT NULL,
        branch_id TEXT NOT NULL,
        created_revision_id TEXT NOT NULL,
        payload_json BLOB NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (workspace_id)
            REFERENCES workspaces(workspace_id) ON DELETE RESTRICT,
        FOREIGN KEY (branch_id)
            REFERENCES workspace_branches(branch_id) ON DELETE RESTRICT,
        FOREIGN KEY (created_revision_id)
            REFERENCES workspace_revisions(revision_id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS workspace_findings (
        card_id TEXT PRIMARY KEY,
        workspace_id TEXT NOT NULL,
        branch_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        created_revision_id TEXT NOT NULL,
        payload_json BLOB NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (workspace_id)
            REFERENCES workspaces(workspace_id) ON DELETE RESTRICT,
        FOREIGN KEY (branch_id)
            REFERENCES workspace_branches(branch_id) ON DELETE RESTRICT,
        FOREIGN KEY (created_revision_id)
            REFERENCES workspace_revisions(revision_id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS workspace_findings_lookup
    ON workspace_findings(workspace_id, branch_id, kind, created_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS workspace_attempts (
        attempt_id TEXT PRIMARY KEY,
        workspace_id TEXT NOT NULL,
        branch_id TEXT NOT NULL,
        target_card_id TEXT NOT NULL,
        outcome TEXT NOT NULL,
        created_revision_id TEXT NOT NULL,
        payload_json BLOB NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (workspace_id)
            REFERENCES workspaces(workspace_id) ON DELETE RESTRICT,
        FOREIGN KEY (branch_id)
            REFERENCES workspace_branches(branch_id) ON DELETE RESTRICT,
        FOREIGN KEY (target_card_id)
            REFERENCES workspace_findings(card_id) ON DELETE RESTRICT,
        FOREIGN KEY (created_revision_id)
            REFERENCES workspace_revisions(revision_id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS workspace_attempts_lookup
    ON workspace_attempts(workspace_id, branch_id, target_card_id, created_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS workspace_marks (
        mark_id TEXT PRIMARY KEY,
        workspace_id TEXT NOT NULL,
        branch_id TEXT NOT NULL,
        target_card_id TEXT NOT NULL,
        state TEXT NOT NULL,
        created_revision_id TEXT NOT NULL,
        payload_json BLOB NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (workspace_id)
            REFERENCES workspaces(workspace_id) ON DELETE RESTRICT,
        FOREIGN KEY (branch_id)
            REFERENCES workspace_branches(branch_id) ON DELETE RESTRICT,
        FOREIGN KEY (target_card_id)
            REFERENCES workspace_findings(card_id) ON DELETE RESTRICT,
        FOREIGN KEY (created_revision_id)
            REFERENCES workspace_revisions(revision_id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS workspace_marks_lookup
    ON workspace_marks(workspace_id, branch_id, target_card_id, created_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS workspace_focus (
        branch_id TEXT PRIMARY KEY,
        workspace_id TEXT NOT NULL,
        updated_revision_id TEXT NOT NULL,
        payload_json BLOB NOT NULL,
        FOREIGN KEY (branch_id)
            REFERENCES workspace_branches(branch_id) ON DELETE RESTRICT,
        FOREIGN KEY (workspace_id)
            REFERENCES workspaces(workspace_id) ON DELETE RESTRICT,
        FOREIGN KEY (updated_revision_id)
            REFERENCES workspace_revisions(revision_id) ON DELETE RESTRICT
    )
    """,
)
_RUNTIME_SCHEMA = "\n-- statement boundary --\n".join(_RUNTIME_SCHEMA_STATEMENTS)

_STATE_FORMAT_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS jacobian_state_format (
        id INTEGER PRIMARY KEY CHECK (id = 0),
        format_revision INTEGER NOT NULL,
        recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS jacobian_data_upgrades (
        upgrade_id TEXT PRIMARY KEY,
        completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    INSERT INTO jacobian_state_format(id, format_revision)
    VALUES (0, 4)
    ON CONFLICT(id) DO UPDATE SET format_revision = excluded.format_revision
    """,
)
_STATE_FORMAT_SCHEMA = "\n-- statement boundary --\n".join(
    _STATE_FORMAT_SCHEMA_STATEMENTS
)


def _install_artifact_schema(connection: sqlite3.Connection) -> None:
    for statement in _ARTIFACT_SCHEMA_STATEMENTS:
        connection.execute(statement)


def _install_quota_reconciliation(connection: sqlite3.Connection) -> None:
    columns = {
        str(row["name"]) for row in connection.execute("PRAGMA table_info(blob_quota)")
    }
    if "reconciliation_required" not in columns:
        connection.execute(
            """
            ALTER TABLE blob_quota
            ADD COLUMN reconciliation_required
                INTEGER NOT NULL DEFAULT 1
                CHECK (reconciliation_required IN (0, 1))
            """
        )


def _install_runtime_schema(connection: sqlite3.Connection) -> None:
    for statement in _RUNTIME_SCHEMA_STATEMENTS:
        connection.execute(statement)
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(installed_plugins)")
    }
    if "registry_snapshot_uri" not in columns:
        connection.execute(
            "ALTER TABLE installed_plugins ADD COLUMN registry_snapshot_uri TEXT"
        )


def _install_state_format_schema(connection: sqlite3.Connection) -> None:
    for statement in _STATE_FORMAT_SCHEMA_STATEMENTS:
        connection.execute(statement)


_REMOVED_WORKSPACE_TABLES = (
    "workspace_focus",
    "workspace_marks",
    "workspace_attempts",
    "workspace_scratch",
    "workspace_findings",
    "workspace_branches",
    "workspace_revisions",
    "workspace_idempotency",
    "workspaces",
)


def _remove_workspace_schema(connection: sqlite3.Connection) -> None:
    """Remove the retired workspace product surface from every store."""

    for table in _REMOVED_WORKSPACE_TABLES:
        connection.execute(f"DROP TABLE IF EXISTS {table}")
    connection.execute(
        """
        UPDATE jacobian_state_format
        SET format_revision = 5
        WHERE id = 0
        """
    )


def _retire_data_upgrade_ledger(connection: sqlite3.Connection) -> None:
    """Retire the completed research-index upgrade bookkeeping."""

    connection.execute("DROP TABLE IF EXISTS jacobian_data_upgrades")
    connection.execute(
        """
        UPDATE jacobian_state_format
        SET format_revision = 6
        WHERE id = 0
        """
    )


_REASONING_LOG_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS reasoning_runs (
        run_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS reasoning_events (
        run_id TEXT NOT NULL,
        sequence INTEGER NOT NULL CHECK (sequence >= 0),
        kind TEXT NOT NULL,
        call_id TEXT,
        event_json BLOB NOT NULL,
        PRIMARY KEY (run_id, sequence),
        FOREIGN KEY (run_id) REFERENCES reasoning_runs(run_id) ON DELETE RESTRICT
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS reasoning_event_kind_call ON reasoning_events(run_id, kind, call_id) WHERE call_id IS NOT NULL",
    "CREATE UNIQUE INDEX IF NOT EXISTS reasoning_one_plan ON reasoning_events(run_id) WHERE kind = 'PLAN'",
    "CREATE UNIQUE INDEX IF NOT EXISTS reasoning_one_final ON reasoning_events(run_id) WHERE kind = 'FINAL'",
    """
    CREATE TRIGGER IF NOT EXISTS reasoning_runs_no_update BEFORE UPDATE ON reasoning_runs
    BEGIN SELECT RAISE(ABORT, 'reasoning runs are append-only'); END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS reasoning_runs_no_delete BEFORE DELETE ON reasoning_runs
    BEGIN SELECT RAISE(ABORT, 'reasoning runs are append-only'); END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS reasoning_events_no_update BEFORE UPDATE ON reasoning_events
    BEGIN SELECT RAISE(ABORT, 'reasoning events are append-only'); END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS reasoning_events_no_delete BEFORE DELETE ON reasoning_events
    BEGIN SELECT RAISE(ABORT, 'reasoning events are append-only'); END
    """,
)
_REASONING_LOG_SCHEMA = "\n-- statement boundary --\n".join(
    _REASONING_LOG_SCHEMA_STATEMENTS
)


def _install_reasoning_log_schema(connection: sqlite3.Connection) -> None:
    for statement in _REASONING_LOG_SCHEMA_STATEMENTS:
        connection.execute(statement)
    connection.execute(
        "UPDATE jacobian_state_format SET format_revision = 7 WHERE id = 0"
    )


def _install_memoryless_state_boundary(connection: sqlite3.Connection) -> None:
    connection.execute(
        "UPDATE jacobian_state_format SET format_revision = 8 WHERE id = 0"
    )


def _install_checker_manifest_boundary(connection: sqlite3.Connection) -> None:
    """Cut over checker authorization to manifest-bound implementation identity."""

    checker_count = int(
        connection.execute("SELECT COUNT(*) FROM checkers").fetchone()[0]
    )
    if checker_count:
        raise RuntimeError(
            "checker manifest identity requires a fresh state directory; "
            "existing checker authorizations cannot be migrated safely"
        )
    columns = {
        str(row["name"]) for row in connection.execute("PRAGMA table_info(checkers)")
    }
    if "implementation_digest" in columns:
        # The immutable migration ledger may have been lost after this boundary
        # completed.  The current schema already has only manifest identity, so
        # recording revision 9 is safe; no legacy identity is accepted.
        connection.execute(
            "UPDATE jacobian_state_format SET format_revision = 9 WHERE id = 0"
        )
        return
    if "executable_digest" not in columns:
        raise RuntimeError("checker table does not match the v1 authorization schema")
    connection.execute(
        "ALTER TABLE checkers RENAME COLUMN executable_digest TO implementation_digest"
    )
    connection.execute(
        "UPDATE jacobian_state_format SET format_revision = 9 WHERE id = 0"
    )


def _install_checker_distribution_identity_boundary(
    connection: sqlite3.Connection,
) -> None:
    """Require a fresh authorization state for manifest-bound Python dependencies."""

    checker_count = int(
        connection.execute("SELECT COUNT(*) FROM checkers").fetchone()[0]
    )
    if checker_count:
        raise RuntimeError(
            "checker distribution identity requires a fresh state directory; "
            "existing checker authorizations cannot be migrated safely"
        )
    connection.execute(
        "UPDATE jacobian_state_format SET format_revision = 10 WHERE id = 0"
    )


def _install_verification_record_manifest_boundary(
    connection: sqlite3.Connection,
) -> None:
    """Cut over durable verification records to embedded checker manifests."""

    connection.execute(
        "UPDATE jacobian_state_format SET format_revision = 11 WHERE id = 0"
    )


STATE_MIGRATIONS = (
    Migration(
        revision=1,
        name="artifact-metadata-v1",
        definition=_ARTIFACT_SCHEMA,
        apply=_install_artifact_schema,
    ),
    Migration(
        revision=2,
        name="blob-quota-reconciliation-v1",
        definition=_QUOTA_RECONCILIATION,
        apply=_install_quota_reconciliation,
    ),
    Migration(
        revision=3,
        name="runtime-service-schema-v1",
        definition=_RUNTIME_SCHEMA,
        apply=_install_runtime_schema,
    ),
    Migration(
        revision=4,
        name="state-format-boundary-v1",
        definition=_STATE_FORMAT_SCHEMA,
        apply=_install_state_format_schema,
    ),
    Migration(
        revision=5,
        name="remove-workspace-surface-v1",
        definition=(
            "Drop the retired workspace tables in dependency order and advance "
            "the persisted state format to revision 5."
        ),
        apply=_remove_workspace_schema,
        requires_foreign_keys_off=True,
    ),
    Migration(
        revision=6,
        name="retire-state-data-upgrade-ledger-v1",
        definition=(
            "Drop the completed research-index data-upgrade ledger and advance "
            "the persisted state format to revision 6."
        ),
        apply=_retire_data_upgrade_ledger,
    ),
    Migration(
        revision=7,
        name="reasoning-log-events-v1",
        definition=_REASONING_LOG_SCHEMA,
        apply=_install_reasoning_log_schema,
    ),
    Migration(
        revision=8,
        name="memoryless-state-boundary-v1",
        definition=(
            "Establish the pre-stable state boundary after removing research-memory "
            "schema and advance the persisted state format to revision 8."
        ),
        apply=_install_memoryless_state_boundary,
    ),
    Migration(
        revision=9,
        name="checker-manifest-identity-boundary-v1",
        definition=(
            "Replace broad package checker digests with versioned per-checker "
            "manifests, rename the durable implementation digest column, and "
            "reject existing authorizations rather than silently rebinding them."
        ),
        apply=_install_checker_manifest_boundary,
    ),
    Migration(
        revision=10,
        name="checker-distribution-identity-boundary-v1",
        definition=(
            "Bind checker manifests to separate checker and worker source closures "
            "plus exact Python distribution RECORD digests; reject existing "
            "authorizations rather than silently rebinding them."
        ),
        apply=_install_checker_distribution_identity_boundary,
    ),
    Migration(
        revision=11,
        name="verification-record-manifest-boundary-v1",
        definition=(
            "Require verification record v4 payloads to snapshot their complete "
            "checker execution manifest; older record shapes remain owned by "
            "their matching state revision and checkout."
        ),
        apply=_install_verification_record_manifest_boundary,
    ),
)

SUPPORTED_STATE_FLOOR = 11
CURRENT_STATE_FORMAT_REVISION = 11
