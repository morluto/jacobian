"""Enumeration experiment lifecycle and contract boundary tests."""

from __future__ import annotations

import pytest
from tests.boundary.storage.recovery.enumeration_experiments_support import _claim

from jacobian.contracts.discovery import (
    EnumerationBudget,
    EnumerationStopReason,
    ExperimentState,
    SearchEnumerateRequest,
)
from jacobian.experiments import ExperimentError, ExperimentNotFoundError
from jacobian.storage.models import StorageLimits


def test_unknown_experiment_error_explains_recovery(
    fresh_complete_runtime,
    caplog: pytest.LogCaptureFixture,
) -> None:
    missing_uri = "experiment://missing"

    with pytest.raises(
        ExperimentNotFoundError,
        match=r"Check the URI returned by search\.run or search\.enumerate",
    ) as raised:
        fresh_complete_runtime.services.experiments.inspect(missing_uri)

    assert missing_uri not in str(raised.value)
    assert missing_uri in caplog.text


def test_graph_enumeration_deduplicates_isomorphic_candidates(
    authorized_complete_runtime,
) -> None:
    claim_uri, plugin_id = _claim(
        authorized_complete_runtime,
        reference_name="graph_paths",
        predicate="is_bipartite",
        parameters={},
    )

    handle = authorized_complete_runtime.services.experiments.start_enumeration(
        SearchEnumerateRequest(
            claim_uri=claim_uri,
            plugin_id=plugin_id,
            bounds={"vertices": 3},
            quotient_by_isomorphism=True,
            budget=EnumerationBudget(
                candidates_max=8,
                wall_seconds=60,
                page_size=8,
            ),
        )
    )
    snapshot = authorized_complete_runtime.services.experiments.wait(
        handle.experiment_uri,
        timeout_seconds=90,
    )

    assert snapshot.state == ExperimentState.COMPLETED
    assert snapshot.stop_reason == EnumerationStopReason.COMPLETE
    assert snapshot.enumerator_reported_complete is True
    assert snapshot.coverage.value == "EXHAUSTIVE"
    assert snapshot.verification.value == "UNVERIFIED"
    assert snapshot.accounting.raw_candidates == 8
    # Candidate identity is directed-graph identity. The upper-triangular
    # generator therefore has six directed isomorphism classes, even though
    # the same edge subsets have four underlying-undirected classes.
    assert snapshot.accounting.unique_candidates == 6
    assert snapshot.accounting.duplicate_candidates == 2
    assert snapshot.accounting.evaluated_candidates == 6
    assert snapshot.scope_uri is not None
    assert snapshot.archive_uri is not None
    scope = authorized_complete_runtime.core.store.get(snapshot.scope_uri)
    assert scope.payload["enumerator_scope"]["arc_rule"] == (
        "v_i_to_v_j_only_when_i_less_than_j"
    )
    archive = authorized_complete_runtime.core.store.get(snapshot.archive_uri)
    assert set(archive.manifest.parents) == {
        snapshot.scope_uri,
        *snapshot.archive_page_uris,
    }
    for page_uri in snapshot.archive_page_uris:
        page = authorized_complete_runtime.core.store.get(page_uri)
        assert set(page.manifest.parents) == {
            *page.payload["candidate_uris"],
            *page.payload["evaluation_uris"],
        }


def test_experiment_metadata_uses_registered_schema_validation(
    authorized_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim_uri, plugin_id = _claim(
        authorized_complete_runtime,
        reference_name="matrices",
        predicate="is_nonsingular",
        parameters={},
    )
    validated_schema_uris: list[str] = []
    validated_semantics_uris: list[str] = []
    original_validate = authorized_complete_runtime.core.schemas.validate
    original_get_descriptor = authorized_complete_runtime.core.store.get_descriptor

    def record_validation(schema_uri: str, payload: object) -> object:
        validated_schema_uris.append(schema_uri)
        return original_validate(schema_uri, payload)

    def record_descriptor_validation(
        artifact_uri: str,
        *,
        expected_kind: str | None = None,
    ) -> dict[str, object]:
        if expected_kind == "semantics":
            validated_semantics_uris.append(artifact_uri)
        return original_get_descriptor(
            artifact_uri,
            expected_kind=expected_kind,
        )

    monkeypatch.setattr(
        authorized_complete_runtime.core.schemas, "validate", record_validation
    )
    monkeypatch.setattr(
        authorized_complete_runtime.core.store,
        "get_descriptor",
        record_descriptor_validation,
    )
    handle = authorized_complete_runtime.services.experiments.start_enumeration(
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
    snapshot = authorized_complete_runtime.services.experiments.wait(
        handle.experiment_uri, timeout_seconds=30
    )

    assert snapshot.state == ExperimentState.COMPLETED
    assert (
        authorized_complete_runtime.services.experiments.scope_schema_uri
        in validated_schema_uris
    )
    assert (
        authorized_complete_runtime.services.experiments.evaluation_schema_uri
        in validated_schema_uris
    )
    assert (
        authorized_complete_runtime.services.experiments.archive_page_schema_uri
        in validated_schema_uris
    )
    assert (
        authorized_complete_runtime.services.experiments.archive_manifest_schema_uri
        in validated_schema_uris
    )
    assert (
        authorized_complete_runtime.portfolio.references["matrices"].semantics_uri
        in validated_semantics_uris
    )


def test_matrix_enumeration_uses_the_same_experiment_contract(
    authorized_complete_runtime,
) -> None:
    claim_uri, plugin_id = _claim(
        authorized_complete_runtime,
        reference_name="matrices",
        predicate="is_nonsingular",
        parameters={},
    )
    handle = authorized_complete_runtime.services.experiments.start_enumeration(
        SearchEnumerateRequest(
            claim_uri=claim_uri,
            plugin_id=plugin_id,
            bounds={"rows": 1, "cols": 1, "entries": [0, 1]},
            budget=EnumerationBudget(
                candidates_max=2,
                wall_seconds=30,
                page_size=2,
            ),
        )
    )

    snapshot = authorized_complete_runtime.services.experiments.wait(
        handle.experiment_uri,
        timeout_seconds=45,
    )

    assert snapshot.state == ExperimentState.COMPLETED
    assert snapshot.stop_reason == EnumerationStopReason.COMPLETE
    assert snapshot.accounting.raw_candidates == 2
    assert snapshot.accounting.unique_candidates == 2
    assert snapshot.accounting.evaluated_candidates == 2
    assert snapshot.verification.value == "UNVERIFIED"


def test_enumeration_pages_respect_evaluator_batch_limit(
    authorized_complete_runtime,
) -> None:
    authorized_complete_runtime.services.evaluation.max_batch_size = 2
    claim_uri, plugin_id = _claim(
        authorized_complete_runtime,
        reference_name="matrices",
        predicate="is_nonsingular",
        parameters={},
    )
    handle = authorized_complete_runtime.services.experiments.start_enumeration(
        SearchEnumerateRequest(
            claim_uri=claim_uri,
            plugin_id=plugin_id,
            bounds={"rows": 1, "cols": 1, "entries": [0, 1, 2]},
            budget=EnumerationBudget(
                candidates_max=3,
                wall_seconds=30,
                page_size=3,
            ),
        )
    )

    snapshot = authorized_complete_runtime.services.experiments.wait(
        handle.experiment_uri, timeout_seconds=30
    )

    assert snapshot.state is ExperimentState.COMPLETED
    assert snapshot.stop_reason is EnumerationStopReason.COMPLETE
    assert snapshot.accounting.raw_candidates == 3
    assert snapshot.accounting.evaluated_candidates == 3
    assert snapshot.accounting.pages == 2
    assert snapshot.scope_uri is not None
    assert snapshot.archive_uri is not None
    archive = authorized_complete_runtime.core.store.get(snapshot.archive_uri)
    assert set(archive.manifest.parents) == {
        snapshot.scope_uri,
        snapshot.archive_page_uris[-1],
    }
    second_page = authorized_complete_runtime.core.store.get(
        snapshot.archive_page_uris[-1]
    )
    assert snapshot.archive_page_uris[0] in second_page.manifest.parents


def test_enumeration_uses_available_parent_capacity_per_page(
    authorized_complete_runtime,
) -> None:
    authorized_complete_runtime.core.store.limits = StorageLimits(max_parents=3)
    claim_uri, plugin_id = _claim(
        authorized_complete_runtime,
        reference_name="matrices",
        predicate="is_nonsingular",
        parameters={},
    )
    handle = authorized_complete_runtime.services.experiments.start_enumeration(
        SearchEnumerateRequest(
            claim_uri=claim_uri,
            plugin_id=plugin_id,
            bounds={"rows": 1, "cols": 1, "entries": [0, 1, 2, 3]},
            budget=EnumerationBudget(
                candidates_max=4,
                wall_seconds=30,
                page_size=2,
            ),
        )
    )

    snapshot = authorized_complete_runtime.services.experiments.wait(
        handle.experiment_uri, timeout_seconds=30
    )

    assert snapshot.state is ExperimentState.COMPLETED
    assert snapshot.stop_reason is EnumerationStopReason.COMPLETE
    first_page_uri = snapshot.archive_page_uris[0]
    first_page = authorized_complete_runtime.core.store.get(first_page_uri)
    assert len(first_page.payload["candidate_uris"]) == 2
    assert set(first_page.manifest.parents) == {
        first_page.payload["evaluation_uris"][0],
        *first_page.payload["candidate_uris"],
    }
    assert len(snapshot.archive_page_uris) == 3
    for index, page_uri in enumerate(snapshot.archive_page_uris[1:], start=1):
        page = authorized_complete_runtime.core.store.get(page_uri)
        assert len(page.payload["candidate_uris"]) == 1
        assert set(page.manifest.parents) == {
            page.payload["evaluation_uris"][0],
            *page.payload["candidate_uris"],
            snapshot.archive_page_uris[index - 1],
        }


def test_cancellation_never_becomes_an_exhaustive_conclusion(
    authorized_complete_runtime,
) -> None:
    claim_uri, plugin_id = _claim(
        authorized_complete_runtime,
        reference_name="matrices",
        predicate="is_nonsingular",
        parameters={},
    )
    handle = authorized_complete_runtime.services.experiments.start_enumeration(
        SearchEnumerateRequest(
            claim_uri=claim_uri,
            plugin_id=plugin_id,
            bounds={"rows": 3, "cols": 3, "entries": [-1, 1]},
            budget=EnumerationBudget(
                candidates_max=512,
                wall_seconds=120,
                page_size=128,
            ),
        )
    )

    cancelled = authorized_complete_runtime.services.experiments.cancel(
        handle.experiment_uri
    )
    snapshot = authorized_complete_runtime.services.experiments.wait(
        handle.experiment_uri,
        timeout_seconds=30,
    )

    assert cancelled.accepted is True
    assert snapshot.state == ExperimentState.CANCELLED
    assert snapshot.stop_reason == EnumerationStopReason.CANCELLED
    assert snapshot.coverage.value == "BOUNDED"
    assert snapshot.verification.value == "UNVERIFIED"
    assert snapshot.archive_uri is not None


def test_candidate_limit_never_becomes_exhaustive_coverage(
    authorized_complete_runtime,
) -> None:
    claim_uri, plugin_id = _claim(
        authorized_complete_runtime,
        reference_name="matrices",
        predicate="is_nonsingular",
        parameters={},
    )
    handle = authorized_complete_runtime.services.experiments.start_enumeration(
        SearchEnumerateRequest(
            claim_uri=claim_uri,
            plugin_id=plugin_id,
            bounds={"rows": 1, "cols": 2, "entries": [0, 1]},
            budget=EnumerationBudget(
                candidates_max=2,
                wall_seconds=30,
                page_size=2,
            ),
        )
    )

    snapshot = authorized_complete_runtime.services.experiments.wait(
        handle.experiment_uri,
        timeout_seconds=45,
    )

    assert snapshot.state == ExperimentState.COMPLETED
    assert snapshot.stop_reason == EnumerationStopReason.CANDIDATE_LIMIT
    assert snapshot.enumerator_reported_complete is False
    assert snapshot.coverage.value == "BOUNDED"
    assert snapshot.verification.value == "UNVERIFIED"
    assert snapshot.accounting.raw_candidates == 2


def test_quotient_search_requires_a_domain_canonicalizer(
    authorized_complete_runtime,
) -> None:
    claim_uri, plugin_id = _claim(
        authorized_complete_runtime,
        reference_name="matrices",
        predicate="is_nonsingular",
        parameters={},
    )

    with pytest.raises(ExperimentError, match="Canonicalizer"):
        authorized_complete_runtime.services.experiments.start_enumeration(
            SearchEnumerateRequest(
                claim_uri=claim_uri,
                plugin_id=plugin_id,
                bounds={"rows": 1, "cols": 1, "entries": [0, 1]},
                quotient_by_isomorphism=True,
                budget=EnumerationBudget(
                    candidates_max=2,
                    wall_seconds=30,
                    page_size=2,
                ),
            )
        )


def test_cancelling_a_terminal_experiment_does_not_change_it(
    authorized_complete_runtime,
) -> None:
    claim_uri, plugin_id = _claim(
        authorized_complete_runtime,
        reference_name="matrices",
        predicate="is_nonsingular",
        parameters={},
    )
    handle = authorized_complete_runtime.services.experiments.start_enumeration(
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
    completed = authorized_complete_runtime.services.experiments.wait(
        handle.experiment_uri,
        timeout_seconds=45,
    )

    cancelled = authorized_complete_runtime.services.experiments.cancel(
        handle.experiment_uri
    )
    after = authorized_complete_runtime.services.experiments.inspect(
        handle.experiment_uri
    )

    assert completed.state == ExperimentState.COMPLETED
    assert cancelled.accepted is False
    assert after == completed
