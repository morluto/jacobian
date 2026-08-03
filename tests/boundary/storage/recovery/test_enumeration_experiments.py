from __future__ import annotations

import pytest

from jacobian.contracts.discovery import (
    EnumerationBudget,
    EnumerationStopReason,
    ExperimentState,
    SearchEnumerateRequest,
)
from jacobian.contracts.evaluation import EvaluationBatchResult, EvaluationProfile
from jacobian.contracts.plugins import PluginManifest
from jacobian.contracts.results import (
    Execution,
    ExecutionStatus,
    InputStatus,
    InputValidation,
)
from jacobian.experiments import ExperimentError, ExperimentNotFoundError
from jacobian.runtime.model import JacobianRuntime
from jacobian.storage.errors import StorageError


def _claim(
    runtime: JacobianRuntime,
    *,
    reference_name: str,
    predicate: str,
    parameters: dict[str, object],
) -> tuple[str, str]:
    reference = runtime.portfolio.references[reference_name]
    claim = runtime.core.artifacts.put(
        schema_uri=reference.claim_schema_uri,
        semantics_uri=reference.semantics_uri,
        payload={
            "claim_schema_version": "1",
            "domain_id": (
                "jacobian.graph-paths"
                if reference_name == "graph_paths"
                else "jacobian.integer-matrices"
            ),
            "domain_version": "1",
            "semantics_uri": reference.semantics_uri,
            "quantifiers": [],
            "predicate": {"name": predicate, "parameters": parameters},
            "bounds": {},
            "required_capabilities": ["CandidateEnumerator", "Evaluator"],
            "correspondence_status": "HUMAN_REVIEWED",
        },
    )
    return claim.artifact_uri, reference.plugin_id


def _install_matrix_enumerator_plugin(
    runtime: JacobianRuntime,
    *,
    entrypoint: str,
    evaluator_entrypoint: str = "jacobian.plugins.matrices:evaluate_capability",
) -> str:
    matrix = runtime.portfolio.references["matrices"]
    enumerator = runtime.core.plugins.register_implementation(entrypoint)
    evaluator = runtime.core.plugins.register_implementation(evaluator_entrypoint)
    manifest = runtime.core.artifacts.put(
        schema_uri=runtime.services.reference_installer.manifest_schema_uri,
        semantics_uri=runtime.services.reference_installer.manifest_semantics_uri,
        payload=PluginManifest(
            domain_id="jacobian.integer-matrices",
            domain_version="1",
            semantics_uri=matrix.semantics_uri,
            claim_schema_uri=matrix.claim_schema_uri,
            candidate_schema_uri=matrix.candidate_schema_uri,
            capabilities={
                "CandidateEnumerator": {
                    "implementation_uri": enumerator,
                    "entrypoint": entrypoint,
                    "version": "1",
                },
                "Evaluator": {
                    "implementation_uri": evaluator,
                    "entrypoint": evaluator_entrypoint,
                    "version": "1",
                },
            },
        ).model_dump(mode="json"),
    )
    runtime.core.plugins.install(manifest.artifact_uri)
    return manifest.artifact_uri


def _matrix_claim_for_plugin(
    runtime: JacobianRuntime,
    *,
    plugin_id: str,
) -> str:
    matrix = runtime.portfolio.references["matrices"]
    claim = runtime.core.artifacts.put(
        schema_uri=matrix.claim_schema_uri,
        semantics_uri=matrix.semantics_uri,
        payload={
            "claim_schema_version": "1",
            "domain_id": "jacobian.integer-matrices",
            "domain_version": "1",
            "semantics_uri": matrix.semantics_uri,
            "quantifiers": [],
            "predicate": {"name": "is_nonsingular", "parameters": {}},
            "bounds": {},
            "required_capabilities": ["CandidateEnumerator", "Evaluator"],
            "correspondence_status": "HUMAN_REVIEWED",
        },
    )
    validation = runtime.services.claims.validate(
        claim_uri=claim.artifact_uri,
        plugin_id=plugin_id,
    )
    assert validation.valid
    return claim.artifact_uri


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


def test_enumerator_candidate_is_validated_before_archival(
    authorized_complete_runtime,
) -> None:
    plugin_id = _install_matrix_enumerator_plugin(
        authorized_complete_runtime,
        entrypoint="tests.component.plugins._fixture_plugins:enumerate_invalid_candidate",
    )
    claim_uri = _matrix_claim_for_plugin(
        authorized_complete_runtime,
        plugin_id=plugin_id,
    )

    handle = authorized_complete_runtime.services.experiments.start_enumeration(
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
    snapshot = authorized_complete_runtime.services.experiments.wait(
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
    authorized_complete_runtime,
) -> None:
    plugin_id = _install_matrix_enumerator_plugin(
        authorized_complete_runtime,
        entrypoint="tests.component.plugins._fixture_plugins:wait_forever",
    )
    claim_uri = _matrix_claim_for_plugin(
        authorized_complete_runtime,
        plugin_id=plugin_id,
    )
    handle = authorized_complete_runtime.services.experiments.start_enumeration(
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
        authorized_complete_runtime.services.experiments.wait(
            handle.experiment_uri, timeout_seconds=0
        )

    snapshot = authorized_complete_runtime.services.experiments.wait(
        handle.experiment_uri,
        timeout_seconds=15,
    )

    assert snapshot.state == ExperimentState.TIMEOUT
    assert snapshot.stop_reason == EnumerationStopReason.WALL_TIME_LIMIT
    assert snapshot.coverage.value == "BOUNDED"
    assert snapshot.verification.value == "UNVERIFIED"
    assert snapshot.accounting.raw_candidates == 0


def test_evaluator_timeout_prevents_complete_enumeration_result(
    authorized_complete_runtime,
) -> None:
    plugin_id = _install_matrix_enumerator_plugin(
        authorized_complete_runtime,
        entrypoint="jacobian.plugins.matrices:enumerate_candidates_capability",
        evaluator_entrypoint="tests.component.plugins._fixture_plugins:wait_forever",
    )
    claim_uri = _matrix_claim_for_plugin(
        authorized_complete_runtime,
        plugin_id=plugin_id,
    )
    handle = authorized_complete_runtime.services.experiments.start_enumeration(
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

    snapshot = authorized_complete_runtime.services.experiments.wait(
        handle.experiment_uri,
        timeout_seconds=15,
    )

    assert snapshot.state == ExperimentState.TIMEOUT
    assert snapshot.stop_reason == EnumerationStopReason.WALL_TIME_LIMIT
    assert snapshot.enumerator_reported_complete is False
    assert snapshot.coverage.value == "BOUNDED"
    assert snapshot.verification.value == "UNVERIFIED"


def test_rejected_evaluation_batch_fails_enumeration(
    authorized_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim_uri, plugin_id = _claim(
        authorized_complete_runtime,
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
        authorized_complete_runtime.services.evaluation, "evaluate_batch", reject_batch
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
    authorized_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    claim_uri, plugin_id = _claim(
        authorized_complete_runtime,
        reference_name="matrices",
        predicate="is_nonsingular",
        parameters={},
    )
    original_put = (
        authorized_complete_runtime.services.experiments._put_internal_artifact
    )

    def fail_terminal_archive(**kwargs: object) -> object:
        if kwargs.get("summary") == "enumeration archive manifest":
            raise StorageError("fixture archive failure")
        return original_put(**kwargs)

    monkeypatch.setattr(
        authorized_complete_runtime.services.experiments,
        "_put_internal_artifact",
        fail_terminal_archive,
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
