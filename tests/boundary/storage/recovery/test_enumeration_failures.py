"""Enumeration timeout, validation, and archival failure policy tests."""

from __future__ import annotations

import pytest
from tests.boundary.storage.recovery.enumeration_experiments_support import (
    _claim,
    _install_matrix_enumerator_plugin,
    _matrix_claim_for_plugin,
)

from jacobian.contracts.discovery import (
    EnumerationBudget,
    EnumerationStopReason,
    ExperimentState,
    SearchEnumerateRequest,
)
from jacobian.contracts.evaluation import EvaluationBatchResult, EvaluationProfile
from jacobian.contracts.results import (
    Execution,
    ExecutionStatus,
    InputStatus,
    InputValidation,
)
from jacobian.storage.errors import StorageError


def test_enumerator_candidate_is_validated_before_archival(
    attached_complete_runtime,
) -> None:
    plugin_id = _install_matrix_enumerator_plugin(
        attached_complete_runtime,
        entrypoint="tests.support.search_entrypoints:enumerate_invalid_candidate",
    )
    claim_uri = _matrix_claim_for_plugin(
        attached_complete_runtime,
        plugin_id=plugin_id,
    )

    handle = attached_complete_runtime.services.experiments.start_enumeration(
        SearchEnumerateRequest(
            claim_uri=claim_uri,
            plugin_id=plugin_id,
            bounds={"fixture": True},
            budget=EnumerationBudget(
                candidates_max=1,
                wall_seconds=30,
                page_size=1,
            ),
        )
    )
    snapshot = attached_complete_runtime.services.experiments.wait(
        handle.experiment_uri,
        timeout_seconds=45,
    )

    assert snapshot.state == ExperimentState.ERROR
    assert snapshot.stop_reason == EnumerationStopReason.ERROR
    assert snapshot.coverage.value == "BOUNDED"
    assert snapshot.verification.value == "UNVERIFIED"
    assert snapshot.accounting.raw_candidates == 0
    assert snapshot.archive_page_uris == ()


def test_enumerator_timeout_remains_a_bounded_nonconclusion(
    attached_complete_runtime,
) -> None:
    plugin_id = _install_matrix_enumerator_plugin(
        attached_complete_runtime,
        entrypoint="tests.support.process_entrypoints:wait_forever",
    )
    claim_uri = _matrix_claim_for_plugin(
        attached_complete_runtime,
        plugin_id=plugin_id,
    )
    handle = attached_complete_runtime.services.experiments.start_enumeration(
        SearchEnumerateRequest(
            claim_uri=claim_uri,
            plugin_id=plugin_id,
            bounds={"fixture": "timeout"},
            budget=EnumerationBudget(
                candidates_max=1,
                wall_seconds=1,
                page_size=1,
            ),
        )
    )

    with pytest.raises(
        TimeoutError,
        match="Inspect it or wait again with a larger timeout",
    ):
        attached_complete_runtime.services.experiments.wait(
            handle.experiment_uri, timeout_seconds=0
        )

    snapshot = attached_complete_runtime.services.experiments.wait(
        handle.experiment_uri,
        timeout_seconds=15,
    )

    assert snapshot.state == ExperimentState.TIMEOUT
    assert snapshot.stop_reason == EnumerationStopReason.WALL_TIME_LIMIT
    assert snapshot.coverage.value == "BOUNDED"
    assert snapshot.verification.value == "UNVERIFIED"
    assert snapshot.accounting.raw_candidates == 0


def test_evaluator_timeout_prevents_complete_enumeration_result(
    attached_complete_runtime,
) -> None:
    plugin_id = _install_matrix_enumerator_plugin(
        attached_complete_runtime,
        entrypoint="jacobian.plugins.matrices:enumerate_candidates_capability",
        evaluator_entrypoint="tests.support.process_entrypoints:wait_forever",
    )
    claim_uri = _matrix_claim_for_plugin(
        attached_complete_runtime,
        plugin_id=plugin_id,
    )
    handle = attached_complete_runtime.services.experiments.start_enumeration(
        SearchEnumerateRequest(
            claim_uri=claim_uri,
            plugin_id=plugin_id,
            bounds={"rows": 1, "cols": 1, "entries": [0]},
            budget=EnumerationBudget(
                candidates_max=1,
                wall_seconds=1,
                page_size=1,
            ),
        )
    )

    snapshot = attached_complete_runtime.services.experiments.wait(
        handle.experiment_uri,
        timeout_seconds=15,
    )

    assert snapshot.state == ExperimentState.TIMEOUT
    assert snapshot.stop_reason == EnumerationStopReason.WALL_TIME_LIMIT
    assert snapshot.enumerator_reported_complete is False
    assert snapshot.coverage.value == "BOUNDED"
    assert snapshot.verification.value == "UNVERIFIED"


def test_rejected_evaluation_batch_fails_enumeration(
    attached_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim_uri, plugin_id = _claim(
        attached_complete_runtime,
        reference_name="matrices",
        predicate="is_nonsingular",
        parameters={},
    )

    def reject_batch(**_kwargs: object) -> EvaluationBatchResult:
        return EvaluationBatchResult(
            execution=Execution(status=ExecutionStatus.COMPLETED),
            input=InputValidation(
                status=InputStatus.REJECTED,
                errors=("simulated incomplete evaluation",),
            ),
            claim_uri=claim_uri,
            plugin_id=plugin_id,
            profile=EvaluationProfile.EXACT_CANDIDATE,
            seed=0,
        )

    monkeypatch.setattr(
        attached_complete_runtime.services.evaluation, "evaluate_batch", reject_batch
    )
    handle = attached_complete_runtime.services.experiments.start_enumeration(
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

    snapshot = attached_complete_runtime.services.experiments.wait(
        handle.experiment_uri, timeout_seconds=15
    )

    assert snapshot.state == ExperimentState.ERROR
    assert snapshot.stop_reason == EnumerationStopReason.ERROR
    assert snapshot.accounting.evaluated_candidates == 0
    assert snapshot.archive_page_uris == ()
    assert "artifact or plugin response was invalid" in snapshot.detail
    assert "reference contract" in snapshot.detail
    assert "simulated incomplete evaluation" not in snapshot.detail


def test_terminal_archive_failure_marks_enumeration_error(
    attached_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    claim_uri, plugin_id = _claim(
        attached_complete_runtime,
        reference_name="matrices",
        predicate="is_nonsingular",
        parameters={},
    )
    original_put = attached_complete_runtime.services.experiments._put_internal_artifact

    def fail_terminal_archive(**kwargs: object) -> object:
        if kwargs.get("summary") == "enumeration archive manifest":
            raise StorageError("fixture archive failure")
        return original_put(**kwargs)

    monkeypatch.setattr(
        attached_complete_runtime.services.experiments,
        "_put_internal_artifact",
        fail_terminal_archive,
    )
    handle = attached_complete_runtime.services.experiments.start_enumeration(
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

    snapshot = attached_complete_runtime.services.experiments.wait(
        handle.experiment_uri,
        timeout_seconds=15,
    )

    assert snapshot.state == ExperimentState.ERROR
    assert snapshot.stop_reason == EnumerationStopReason.ERROR
    assert snapshot.archive_uri is None
    assert "could not save the final experiment archive" in snapshot.detail
    assert "experiment remains unverified" in snapshot.detail
    assert "StorageError" not in snapshot.detail
    assert "runtime_ms" not in snapshot.detail
    assert "fixture archive failure" not in snapshot.detail
    assert "fixture archive failure" in caplog.text
