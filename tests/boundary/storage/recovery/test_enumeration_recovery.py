"""Crash and corrupt-snapshot recovery tests for enumeration experiments."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from tests.boundary.storage.recovery.enumeration_experiments_support import _claim
from tests.support.services import open_reference_services

from jacobian.contracts.discovery import (
    EnumerationBudget,
    EnumerationStopReason,
    ExperimentState,
    SearchEnumerateRequest,
)


def test_interrupted_experiment_is_recovered_as_an_error(tmp_path: Path) -> None:
    script = """
import os
import sys
from jacobian.contracts.discovery import EnumerationBudget, SearchEnumerateRequest
from tests.support.services import open_reference_services

root = sys.argv[1]
with open_reference_services(root, "matrices") as runtime:
    reference = runtime.references["matrices"]
    claim = runtime.core.artifacts.put(
        schema_uri=reference.claim_schema_uri,
        semantics_uri=reference.semantics_uri,
        payload={
            "claim_schema_version": "1",
            "domain_id": "jacobian.integer-matrices",
            "domain_version": "1",
            "semantics_uri": reference.semantics_uri,
            "quantifiers": [],
            "predicate": {"name": "is_nonsingular", "parameters": {}},
            "bounds": {},
            "required_capabilities": ["CandidateEnumerator", "Evaluator"],
            "correspondence_status": "HUMAN_REVIEWED",
        },
    )
    handle = runtime.application.experiments.start_enumeration(
        SearchEnumerateRequest(
            claim_uri=claim.artifact_uri,
            plugin_id=reference.plugin_id,
            bounds={"rows": 3, "cols": 3, "entries": [-1, 1]},
            budget=EnumerationBudget(
                candidates_max=512,
                wall_seconds=120,
                page_size=1,
            ),
        )
    )
    print(handle.experiment_uri, flush=True)
    os._exit(0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    experiment_uri = completed.stdout.strip().splitlines()[-1]

    with open_reference_services(tmp_path, "matrices") as recovered_runtime:
        recovered = recovered_runtime.application.experiments.inspect(experiment_uri)

        assert recovered.state == ExperimentState.ERROR
        assert recovered.stop_reason == EnumerationStopReason.ERROR
        assert recovered.coverage.value == "BOUNDED"
        assert recovered.verification.value == "UNVERIFIED"
        assert "ended before completion" in recovered.detail


def test_corrupt_enumeration_snapshot_does_not_block_other_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with open_reference_services(tmp_path, "matrices") as runtime:
        claim_uri, plugin_id = _claim(
            runtime,
            reference_name="matrices",
            predicate="is_nonsingular",
            parameters={},
        )
        monkeypatch.setattr(
            runtime.application.experiments,
            "_run_enumeration",
            lambda _experiment_uri: None,
        )
        valid = runtime.application.experiments.start_enumeration(
            SearchEnumerateRequest(
                claim_uri=claim_uri,
                plugin_id=plugin_id,
                bounds={"rows": 1, "cols": 1, "entries": [0]},
                budget=EnumerationBudget(
                    candidates_max=1,
                    wall_seconds=30,
                    page_size=1,
                ),
            )
        )
        valid_snapshot = runtime.application.experiments.inspect(valid.experiment_uri)
        corrupt_uri = "experiment://ffffffffffffffffffffffffffffffff"
        mismatched_uri = "experiment://eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
        invalid_state_uri = "experiment://dddddddddddddddddddddddddddddddd"
        with sqlite3.connect(runtime.core.store.db_path) as connection:
            connection.execute(
                """
                INSERT INTO experiments (
                    experiment_uri, state, snapshot_json
                ) VALUES (?, 'RUNNING', ?)
                """,
                (corrupt_uri, b"{"),
            )
            connection.execute(
                """
                INSERT INTO experiments (
                    experiment_uri, state, snapshot_json
                ) VALUES (?, 'PENDING', ?)
                """,
                (
                    mismatched_uri,
                    valid_snapshot.model_dump_json().encode(),
                ),
            )
            invalid_state_snapshot = valid_snapshot.model_copy(
                update={"experiment_uri": invalid_state_uri}
            )
            connection.execute(
                """
                INSERT INTO experiments (
                    experiment_uri, state, snapshot_json
                ) VALUES (?, 'BROKEN', ?)
                """,
                (
                    invalid_state_uri,
                    invalid_state_snapshot.model_dump_json().encode(),
                ),
            )

    with open_reference_services(tmp_path, "matrices") as recovered:
        assert recovered.application.experiments.inspect(
            valid.experiment_uri
        ).state is (ExperimentState.ERROR)
        with sqlite3.connect(recovered.core.store.db_path) as connection:
            states = connection.execute(
                """
                SELECT experiment_uri, state
                FROM experiments
                WHERE experiment_uri IN (?, ?, ?)
                ORDER BY experiment_uri
                """,
                (corrupt_uri, mismatched_uri, invalid_state_uri),
            ).fetchall()
            failures = connection.execute(
                """
                SELECT experiment_uri, snapshot_digest, detail
                FROM experiment_recovery_failures
                WHERE experiment_uri IN (?, ?, ?)
                ORDER BY experiment_uri
                """,
                (corrupt_uri, mismatched_uri, invalid_state_uri),
            ).fetchall()
        assert states == [
            (invalid_state_uri, "ERROR"),
            (mismatched_uri, "ERROR"),
            (corrupt_uri, "ERROR"),
        ]
        assert len(failures) == 3
        assert all(str(failure[1]).startswith("sha256:") for failure in failures)
        assert all("invalid" in str(failure[2]) for failure in failures)
