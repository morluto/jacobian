"""Search orchestration plugin fail-closed and policy boundary tests.

Covers: proposer timeout, malformed/partial proposals, declared and output-limit
plugin failures, archive write failure, operator batch-size policy, evaluator
batch limit, archive parent limit, refiner verification-claim rejection, verified
counterexample feedback to refiner, and supporting-checker result not counted as
counterexample.
"""

from __future__ import annotations

import pytest
from search_orchestration_support import _install_search_plugin, _request

from jacobian.contracts.discovery import ExperimentState
from jacobian.contracts.evidence import WitnessRole
from jacobian.contracts.search import SearchCheckpoint, SearchStopReason
from jacobian.search.errors import SearchError
from jacobian.storage.errors import StorageError
from jacobian.storage.models import StorageLimits


def test_proposer_timeout_fails_closed(search_services) -> None:
    claim_uri, plugin_id = _install_search_plugin(
        search_services,
        proposer_entrypoint=("tests.support.search_entrypoints:propose_search_forever"),
    )
    handle = search_services.application.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-timeout-001",
            wall_seconds=1,
        )
    )

    with pytest.raises(
        TimeoutError,
        match="Inspect the experiment or wait again with a larger timeout",
    ):
        search_services.application.search.wait(
            handle.experiment_uri, timeout_seconds=0
        )

    snapshot = search_services.application.search.wait(
        handle.experiment_uri, timeout_seconds=10
    )

    assert snapshot.state is ExperimentState.TIMEOUT
    assert snapshot.stop_reason is SearchStopReason.WALL_TIME_LIMIT
    assert snapshot.verification.value == "UNVERIFIED"
    assert snapshot.accounting.proposed_candidates == 0
    assert snapshot.accounting.wall_time_ms > 0
    assert snapshot.archive_page_uris == ()


def test_malformed_proposal_fails_without_evidence_promotion(
    search_services,
) -> None:
    claim_uri, plugin_id = _install_search_plugin(
        search_services,
        proposer_entrypoint=(
            "tests.support.search_entrypoints:propose_malformed_search"
        ),
    )
    handle = search_services.application.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-malformed-001",
        )
    )

    snapshot = search_services.application.search.wait(
        handle.experiment_uri, timeout_seconds=15
    )

    assert snapshot.state is ExperimentState.ERROR
    assert snapshot.verification.value == "UNVERIFIED"
    assert "artifact or plugin response was invalid" in snapshot.detail
    assert "reference contract" in snapshot.detail
    assert "input_value" not in snapshot.detail
    assert snapshot.archive_page_uris == ()


def test_partial_iteration_accounting_survives_malformed_candidate(
    search_services,
) -> None:
    claim_uri, plugin_id = _install_search_plugin(
        search_services,
        proposer_entrypoint=(
            "tests.support.search_entrypoints:propose_partially_invalid_search"
        ),
    )
    handle = search_services.application.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-partial-accounting-001",
            batch_size=2,
        )
    )

    snapshot = search_services.application.search.wait(
        handle.experiment_uri, timeout_seconds=15
    )

    assert snapshot.state is ExperimentState.ERROR
    assert snapshot.accounting.proposed_candidates == 1
    assert snapshot.accounting.unique_candidates == 1
    assert snapshot.accounting.evaluated_candidates == 0


@pytest.mark.parametrize(
    ("entrypoint", "detail", "case_id"),
    [
        (
            "tests.support.search_entrypoints:propose_declared_failure",
            (
                "The plugin stopped before returning a result. Retry once; "
                "if it happens again, inspect the local plugin log."
            ),
            "declared",
        ),
        (
            "tests.support.search_entrypoints:propose_large_search_output",
            "The plugin returned too much data. Retry with a smaller request.",
            "output",
        ),
    ],
)
def test_search_plugin_failures_remain_operational(
    search_services,
    entrypoint: str,
    detail: str,
    case_id: str,
) -> None:
    if entrypoint.endswith("propose_large_search_output"):
        search_services.application.plugin_executor.max_output_bytes = 1024
    claim_uri, plugin_id = _install_search_plugin(
        search_services,
        proposer_entrypoint=entrypoint,
    )
    handle = search_services.application.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key=f"search-failure-{case_id}",
        )
    )

    snapshot = search_services.application.search.wait(
        handle.experiment_uri, timeout_seconds=15
    )

    assert snapshot.state is ExperimentState.ERROR
    assert snapshot.verification.value == "UNVERIFIED"
    assert detail in snapshot.detail
    assert snapshot.archive_page_uris == ()


def test_terminal_archive_failure_marks_search_error(
    search_services,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    claim_uri, plugin_id = _install_search_plugin(search_services)

    def fail_archive(*_args: object, **_kwargs: object) -> object:
        raise StorageError("fixture archive failure")

    monkeypatch.setattr(
        search_services.application.search, "_store_archive", fail_archive
    )
    handle = search_services.application.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-terminal-archive-failure-001",
            batch_size=4,
        )
    )

    snapshot = search_services.application.search.wait(
        handle.experiment_uri, timeout_seconds=15
    )

    assert snapshot.state is ExperimentState.ERROR
    assert snapshot.stop_reason is SearchStopReason.ERROR
    assert snapshot.archive_uri is None
    assert "could not save the final experiment archive" in snapshot.detail
    assert "experiment remains unverified" in snapshot.detail
    assert "StorageError" not in snapshot.detail
    assert "fixture archive failure" not in snapshot.detail
    assert "fixture archive failure" in caplog.text


def test_plugin_cannot_widen_operator_batch_policy(search_services) -> None:
    search_services.application.search.max_batch_size = 1
    claim_uri, plugin_id = _install_search_plugin(
        search_services,
        proposer_entrypoint=(
            "tests.support.search_entrypoints:propose_beyond_authority"
        ),
    )
    handle = search_services.application.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-policy-001",
            batch_size=8,
        )
    )

    snapshot = search_services.application.search.wait(
        handle.experiment_uri, timeout_seconds=15
    )

    assert snapshot.effective_budget.batch_size == 1
    assert snapshot.state is ExperimentState.ERROR
    assert "more candidates than authorized" in snapshot.detail
    assert snapshot.accounting.proposed_candidates == 0


def test_search_batch_respects_evaluator_limit(search_services) -> None:
    search_services.application.evaluation.max_batch_size = 2
    search_services.application.search.max_batch_size = 3
    claim_uri, plugin_id = _install_search_plugin(search_services)
    handle = search_services.application.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-evaluator-batch-policy-001",
            batch_size=3,
        )
    )

    snapshot = search_services.application.search.wait(
        handle.experiment_uri, timeout_seconds=30
    )

    assert snapshot.state is ExperimentState.COMPLETED
    assert snapshot.effective_budget.batch_size == 2
    assert snapshot.accounting.unique_candidates == 4
    assert snapshot.accounting.iterations == 2


def test_search_batch_respects_archive_parent_limit(search_services) -> None:
    claim_uri, plugin_id = _install_search_plugin(search_services)
    search_services.core.store.limits = StorageLimits(max_parents=6)
    handle = search_services.application.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-archive-parent-policy-001",
            batch_size=4,
        )
    )

    snapshot = search_services.application.search.wait(
        handle.experiment_uri, timeout_seconds=30
    )

    assert snapshot.state is ExperimentState.COMPLETED
    assert snapshot.effective_budget.batch_size == 3
    assert snapshot.accounting.unique_candidates == 4
    assert snapshot.accounting.iterations == 2
    for page_uri in snapshot.archive_page_uris:
        assert len(search_services.core.store.get(page_uri).manifest.parents) <= 6


@pytest.mark.parametrize("max_parents", [4, 5])
def test_witness_search_requires_archive_parent_capacity(
    search_services,
    max_parents: int,
) -> None:
    claim_uri, plugin_id = _install_search_plugin(
        search_services,
        include_witness_oracle=True,
    )
    search_services.core.store.limits = StorageLimits(max_parents=max_parents)

    with pytest.raises(
        SearchError,
        match="must be at least 6 for one witness-enabled search archive record",
    ):
        search_services.application.search.start(
            _request(
                claim_uri,
                plugin_id,
                idempotency_key=f"search-witness-parent-capacity-{max_parents}",
                witness_role=WitnessRole.DEFEATS_CANDIDATE,
                counterexample_checker_id="checker://sha256/" + "0" * 64,
            )
        )


def test_refiner_can_fit_previous_nominations_to_archive_parent_limit(
    search_services,
) -> None:
    claim_uri, plugin_id = _install_search_plugin(
        search_services,
        proposer_entrypoint=(
            "tests.support.search_entrypoints:"
            "propose_fixture_values_with_strategy_state"
        ),
        refiner_entrypoint=(
            "tests.support.search_entrypoints:"
            "refine_with_bounded_previous_batch_nominations"
        ),
    )
    search_services.core.store.limits = StorageLimits(max_parents=6)
    handle = search_services.application.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-nomination-parent-policy-001",
            batch_size=4,
        )
    )

    snapshot = search_services.application.search.wait(
        handle.experiment_uri, timeout_seconds=30
    )

    assert snapshot.state is ExperimentState.COMPLETED
    assert snapshot.effective_budget.batch_size == 3
    assert snapshot.accounting.iterations == 2
    assert snapshot.accounting.nominations == 2
    assert snapshot.checkpoint_uri is not None
    checkpoint = SearchCheckpoint.model_validate(
        search_services.core.store.get(snapshot.checkpoint_uri).payload
    )
    assert checkpoint.state["observed_lineage_parent_limit"] == 2
    for page_uri in snapshot.archive_page_uris:
        assert len(search_services.core.store.get(page_uri).manifest.parents) <= 6


def test_refiner_cannot_exceed_archive_nomination_parent_limit(
    search_services,
) -> None:
    claim_uri, plugin_id = _install_search_plugin(
        search_services,
        proposer_entrypoint=(
            "tests.support.search_entrypoints:"
            "propose_fixture_values_with_strategy_state"
        ),
        refiner_entrypoint=(
            "tests.support.search_entrypoints:"
            "refine_ignoring_previous_batch_nomination_limit"
        ),
    )
    search_services.core.store.limits = StorageLimits(max_parents=6)
    handle = search_services.application.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-nomination-parent-policy-002",
            batch_size=4,
        )
    )

    snapshot = search_services.application.search.wait(
        handle.experiment_uri, timeout_seconds=30
    )

    assert snapshot.state is ExperimentState.ERROR
    assert snapshot.stop_reason is SearchStopReason.ERROR
    assert snapshot.accounting.nominations == 0
    assert len(snapshot.archive_page_uris) == 1
    assert (
        snapshot.detail
        == "refiner returned more nomination lineage than the archive can preserve"
    )


def test_refiner_cannot_claim_verification(search_services) -> None:
    claim_uri, plugin_id = _install_search_plugin(
        search_services,
        refiner_entrypoint=(
            "tests.support.search_entrypoints:refine_with_verification_claim"
        ),
    )
    handle = search_services.application.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-promotion-001",
            batch_size=4,
        )
    )

    snapshot = search_services.application.search.wait(
        handle.experiment_uri, timeout_seconds=15
    )

    assert snapshot.state is ExperimentState.ERROR
    assert snapshot.verification.value == "UNVERIFIED"
    assert "artifact or plugin response was invalid" in snapshot.detail
    assert "verification" not in snapshot.detail
    assert snapshot.archive_page_uris == ()


def test_verified_counterexample_feedback_reaches_refiner(
    search_services,
) -> None:
    claim_uri, plugin_id = _install_search_plugin(
        search_services,
        refiner_entrypoint=(
            "tests.support.search_entrypoints:refine_from_verified_counterexample"
        ),
        include_witness_oracle=True,
    )
    manifest = search_services.core.plugins.get(plugin_id)
    checker = search_services.core.checkers.authorize(
        name="fixture-value-v1",
        entrypoint="tests.component.checkers._fixture_checkers:check_fixture_value",
        evidence_kind="WITNESS",
        format_id="fixture.value",
        format_version="1",
        claim_schema_uris=(manifest.claim_schema_uri,),
        semantics_uris=(manifest.semantics_uri,),
        candidate_schema_uris=(manifest.candidate_schema_uri,),
        reason="search orchestration conformance fixture",
    )
    search_services.core.store.limits = StorageLimits(max_parents=9)
    handle = search_services.application.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-feedback-001",
            batch_size=4,
            witness_role=WitnessRole.DEFEATS_CANDIDATE,
            counterexample_checker_id=checker.checker_id,
        )
    )

    snapshot = search_services.application.search.wait(
        handle.experiment_uri, timeout_seconds=30
    )

    assert snapshot.state is ExperimentState.COMPLETED
    assert snapshot.effective_budget.batch_size == 2
    assert snapshot.accounting.iterations == 2
    assert snapshot.accounting.attacked_candidates == 4
    assert snapshot.accounting.verified_counterexamples == 4
    assert snapshot.checkpoint_uri is not None
    checkpoint = SearchCheckpoint.model_validate(
        search_services.core.store.get(snapshot.checkpoint_uri).payload
    )
    assert checkpoint.state["saw_verified_counterexample"] is True
    assert all(record.counterexample_verified for record in checkpoint.latest_records)
    assert all(
        record.verification_record_uri is not None
        for record in checkpoint.latest_records
    )


def test_supporting_checker_decision_is_not_counted_as_counterexample(
    search_services,
) -> None:
    claim_uri, plugin_id = _install_search_plugin(
        search_services,
        include_witness_oracle=True,
    )
    manifest = search_services.core.plugins.get(plugin_id)
    checker = search_services.core.checkers.authorize(
        name="fixture-value-true-v1",
        entrypoint=(
            "tests.component.checkers._fixture_checkers:check_fixture_value_as_true"
        ),
        evidence_kind="WITNESS",
        format_id="fixture.value",
        format_version="1",
        claim_schema_uris=(manifest.claim_schema_uri,),
        semantics_uris=(manifest.semantics_uri,),
        candidate_schema_uris=(manifest.candidate_schema_uri,),
        reason="counterexample conclusion boundary fixture",
    )
    handle = search_services.application.search.start(
        _request(
            claim_uri,
            plugin_id,
            idempotency_key="search-supporting-decision-001",
            batch_size=4,
            witness_role=WitnessRole.DEFEATS_CANDIDATE,
            counterexample_checker_id=checker.checker_id,
        )
    )

    snapshot = search_services.application.search.wait(
        handle.experiment_uri, timeout_seconds=30
    )

    assert snapshot.state is ExperimentState.COMPLETED
    assert snapshot.accounting.verified_counterexamples == 0
    assert snapshot.checkpoint_uri is not None
    checkpoint = SearchCheckpoint.model_validate(
        search_services.core.store.get(snapshot.checkpoint_uri).payload
    )
    assert all(
        not record.counterexample_verified for record in checkpoint.latest_records
    )
