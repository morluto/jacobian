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
    search_services,
    caplog: pytest.LogCaptureFixture,
) -> None:
    missing_uri = "experiment://missing"

    with pytest.raises(
        ExperimentNotFoundError,
        match=r"Check the URI returned by search\.run or search\.enumerate",
    ) as raised:
        search_services.application.experiments.inspect(missing_uri)

    assert missing_uri not in str(raised.value)
    assert missing_uri in caplog.text


def test_graph_enumeration_deduplicates_isomorphic_candidates(
    graph_reference_services,
) -> None:
    claim_uri, plugin_id = _claim(
        graph_reference_services,
        reference_name="graph_paths",
        predicate="is_bipartite",
        parameters={},
    )

    handle = graph_reference_services.application.experiments.start_enumeration(
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
    snapshot = graph_reference_services.application.experiments.wait(
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
    scope = graph_reference_services.core.store.get(snapshot.scope_uri)
    assert scope.payload["enumerator_scope"]["arc_rule"] == (
        "v_i_to_v_j_only_when_i_less_than_j"
    )
    archive = graph_reference_services.core.store.get(snapshot.archive_uri)
    assert set(archive.manifest.parents) == {
        snapshot.scope_uri,
        *snapshot.archive_page_uris,
    }
    for page_uri in snapshot.archive_page_uris:
        page = graph_reference_services.core.store.get(page_uri)
        assert set(page.manifest.parents) == {
            *page.payload["candidate_uris"],
            *page.payload["evaluation_uris"],
        }


def test_experiment_metadata_uses_registered_schema_validation(
    matrix_reference_services,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim_uri, plugin_id = _claim(
        matrix_reference_services,
        reference_name="matrices",
        predicate="is_nonsingular",
        parameters={},
    )
    validated_schema_uris: list[str] = []
    validated_semantics_uris: list[str] = []
    original_validate = matrix_reference_services.core.schemas.validate
    original_get_descriptor = matrix_reference_services.core.store.get_descriptor

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
        matrix_reference_services.core.schemas, "validate", record_validation
    )
    monkeypatch.setattr(
        matrix_reference_services.core.store,
        "get_descriptor",
        record_descriptor_validation,
    )
    handle = matrix_reference_services.application.experiments.start_enumeration(
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
    snapshot = matrix_reference_services.application.experiments.wait(
        handle.experiment_uri, timeout_seconds=30
    )

    assert snapshot.state == ExperimentState.COMPLETED
    assert (
        matrix_reference_services.application.experiments.scope_schema_uri
        in validated_schema_uris
    )
    assert (
        matrix_reference_services.application.experiments.evaluation_schema_uri
        in validated_schema_uris
    )
    assert (
        matrix_reference_services.application.experiments.archive_page_schema_uri
        in validated_schema_uris
    )
    assert (
        matrix_reference_services.application.experiments.archive_manifest_schema_uri
        in validated_schema_uris
    )
    assert (
        matrix_reference_services.references["matrices"].semantics_uri
        in validated_semantics_uris
    )


def test_matrix_enumeration_uses_the_same_experiment_contract(
    matrix_reference_services,
) -> None:
    claim_uri, plugin_id = _claim(
        matrix_reference_services,
        reference_name="matrices",
        predicate="is_nonsingular",
        parameters={},
    )
    handle = matrix_reference_services.application.experiments.start_enumeration(
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

    snapshot = matrix_reference_services.application.experiments.wait(
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
    matrix_reference_services,
) -> None:
    matrix_reference_services.application.evaluation.max_batch_size = 2
    claim_uri, plugin_id = _claim(
        matrix_reference_services,
        reference_name="matrices",
        predicate="is_nonsingular",
        parameters={},
    )
    handle = matrix_reference_services.application.experiments.start_enumeration(
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

    snapshot = matrix_reference_services.application.experiments.wait(
        handle.experiment_uri, timeout_seconds=30
    )

    assert snapshot.state is ExperimentState.COMPLETED
    assert snapshot.stop_reason is EnumerationStopReason.COMPLETE
    assert snapshot.accounting.raw_candidates == 3
    assert snapshot.accounting.evaluated_candidates == 3
    assert snapshot.accounting.pages == 2
    assert snapshot.scope_uri is not None
    assert snapshot.archive_uri is not None
    archive = matrix_reference_services.core.store.get(snapshot.archive_uri)
    assert set(archive.manifest.parents) == {
        snapshot.scope_uri,
        snapshot.archive_page_uris[-1],
    }
    second_page = matrix_reference_services.core.store.get(
        snapshot.archive_page_uris[-1]
    )
    assert snapshot.archive_page_uris[0] in second_page.manifest.parents


def test_enumeration_uses_available_parent_capacity_per_page(
    matrix_reference_services,
) -> None:
    matrix_reference_services.core.store.limits = StorageLimits(max_parents=4)
    claim_uri, plugin_id = _claim(
        matrix_reference_services,
        reference_name="matrices",
        predicate="is_nonsingular",
        parameters={},
    )
    handle = matrix_reference_services.application.experiments.start_enumeration(
        SearchEnumerateRequest(
            claim_uri=claim_uri,
            plugin_id=plugin_id,
            bounds={"rows": 1, "cols": 1, "entries": [0, 1, 2, 3, 4, 5]},
            budget=EnumerationBudget(
                candidates_max=6,
                wall_seconds=30,
                page_size=3,
            ),
        )
    )

    snapshot = matrix_reference_services.application.experiments.wait(
        handle.experiment_uri, timeout_seconds=30
    )

    assert snapshot.state is ExperimentState.COMPLETED
    assert snapshot.stop_reason is EnumerationStopReason.COMPLETE
    first_page_uri = snapshot.archive_page_uris[0]
    first_page = matrix_reference_services.core.store.get(first_page_uri)
    assert len(first_page.payload["candidate_uris"]) == 3
    assert set(first_page.manifest.parents) == {
        first_page.payload["evaluation_uris"][0],
        *first_page.payload["candidate_uris"],
    }
    assert len(snapshot.archive_page_uris) == 3
    for index, page_uri in enumerate(snapshot.archive_page_uris[1:], start=1):
        page = matrix_reference_services.core.store.get(page_uri)
        assert len(page.payload["candidate_uris"]) == 3 - index
        assert set(page.manifest.parents) == {
            page.payload["evaluation_uris"][0],
            *page.payload["candidate_uris"],
            snapshot.archive_page_uris[index - 1],
        }


def test_cancellation_never_becomes_an_exhaustive_conclusion(
    matrix_reference_services,
) -> None:
    claim_uri, plugin_id = _claim(
        matrix_reference_services,
        reference_name="matrices",
        predicate="is_nonsingular",
        parameters={},
    )
    handle = matrix_reference_services.application.experiments.start_enumeration(
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

    cancelled = matrix_reference_services.application.experiments.cancel(
        handle.experiment_uri
    )
    snapshot = matrix_reference_services.application.experiments.wait(
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
    matrix_reference_services,
) -> None:
    claim_uri, plugin_id = _claim(
        matrix_reference_services,
        reference_name="matrices",
        predicate="is_nonsingular",
        parameters={},
    )
    handle = matrix_reference_services.application.experiments.start_enumeration(
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

    snapshot = matrix_reference_services.application.experiments.wait(
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
    matrix_reference_services,
) -> None:
    claim_uri, plugin_id = _claim(
        matrix_reference_services,
        reference_name="matrices",
        predicate="is_nonsingular",
        parameters={},
    )

    with pytest.raises(ExperimentError, match="Canonicalizer"):
        matrix_reference_services.application.experiments.start_enumeration(
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
    matrix_reference_services,
) -> None:
    claim_uri, plugin_id = _claim(
        matrix_reference_services,
        reference_name="matrices",
        predicate="is_nonsingular",
        parameters={},
    )
    handle = matrix_reference_services.application.experiments.start_enumeration(
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
    completed = matrix_reference_services.application.experiments.wait(
        handle.experiment_uri,
        timeout_seconds=45,
    )

    cancelled = matrix_reference_services.application.experiments.cancel(
        handle.experiment_uri
    )
    after = matrix_reference_services.application.experiments.inspect(
        handle.experiment_uri
    )

    assert completed.state == ExperimentState.COMPLETED
    assert cancelled.accepted is False
    assert after == completed
