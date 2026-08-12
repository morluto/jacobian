from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from jacobian.canonical import canonicalize_json, loads_strict_json
from jacobian.contracts.checkers import CheckerDecision
from jacobian.contracts.evidence import EvidenceBindings, WitnessEnvelope
from jacobian.contracts.results import (
    Arithmetic,
    Conclusion,
    Coverage,
    ExecutionStatus,
    Method,
)
from jacobian.process_policy import ProcessResult, ProcessTermination
from jacobian.registry import CheckerRegistry
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification import VerificationService


def _graph_case(
    tmp_path: Path,
    *,
    candidate_schema_definition: dict[str, object] | None = None,
    checker_entrypoint: str = "jacobian_checkers.graph_paths:check_omitted_path",
) -> tuple[
    ArtifactRepository,
    VerificationService,
    str,
    str,
    str,
    str,
    str,
]:
    store = ArtifactRepository(tmp_path)
    claim_schema = store.register_descriptor(
        kind="schema",
        name="graph.path-closure.claim",
        version="1",
        definition={"type": "object"},
    )
    candidate_schema = store.register_descriptor(
        kind="schema",
        name="graph.candidate",
        version="1",
        definition=candidate_schema_definition or {"type": "object"},
    )
    witness_schema = store.register_descriptor(
        kind="schema",
        name="graph.omitted-path.witness",
        version="1",
        definition=WitnessEnvelope.model_json_schema(),
    )
    semantics = store.register_descriptor(
        kind="semantics",
        name="directed-graph.path-language",
        version="1",
        definition={"paths": "all simple source-terminal paths"},
    )
    claim = store.put(
        schema_uri=claim_schema,
        semantics_uri=semantics,
        payload={"predicate": "intended_paths_complete", "simple": True},
    )
    candidate = store.put(
        schema_uri=candidate_schema,
        semantics_uri=semantics,
        payload={
            "vertices": ["s", "a", "b", "x", "t1", "t2"],
            "arcs": [
                ["s", "a"],
                ["a", "x"],
                ["s", "b"],
                ["b", "x"],
                ["x", "t1"],
                ["x", "t2"],
            ],
            "source": "s",
            "terminals": ["t1", "t2"],
            "intended_paths": [
                ["s", "a", "x", "t1"],
                ["s", "b", "x", "t2"],
            ],
        },
    )
    semantics_digest = store.get(semantics).manifest.object_digest
    witness_payload = WitnessEnvelope(
        witness_format="graph.omitted_path",
        format_version="1",
        role="DEFEATS_CANDIDATE",
        bindings=EvidenceBindings(
            claim_digest=claim.object_digest,
            semantics_digest=semantics_digest,
            candidate_digest=candidate.object_digest,
        ),
        payload={"path": ["s", "a", "x", "t2"]},
    )
    witness = store.put(
        schema_uri=witness_schema,
        semantics_uri=semantics,
        payload=loads_strict_json(
            canonicalize_json(witness_payload.model_dump(mode="json"))
        ),
        parents=(claim.artifact_uri, candidate.artifact_uri),
    )
    registry = CheckerRegistry(store)
    checker = registry.authorize(
        name="graph-omitted-path-v1",
        entrypoint=checker_entrypoint,
        evidence_kind="WITNESS",
        format_id="graph.omitted_path",
        format_version="1",
        claim_schema_uris=(claim_schema,),
        semantics_uris=(semantics,),
        candidate_schema_uris=(candidate_schema,),
    )
    return (
        store,
        VerificationService(store, registry),
        checker.checker_id,
        claim.artifact_uri,
        candidate.artifact_uri,
        witness.artifact_uri,
        candidate_schema,
    )


def test_checker_timeout_and_cancellation_are_non_conclusions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, service, checker_id, claim_uri, candidate_uri, witness_uri, _ = _graph_case(
        tmp_path
    )
    cases = (
        (
            ProcessResult(
                termination=ProcessTermination.TIMED_OUT,
                returncode=-9,
                stdout=b"",
                stderr=b"",
                stdout_exceeded=False,
                stderr_exceeded=False,
            ),
            ExecutionStatus.TIMEOUT,
        ),
        (
            ProcessResult(
                termination=ProcessTermination.CANCELLED,
                returncode=-9,
                stdout=b"",
                stderr=b"",
                stdout_exceeded=False,
                stderr_exceeded=False,
            ),
            ExecutionStatus.CANCELLED,
        ),
    )
    for stopped, expected_status in cases:
        monkeypatch.setattr(
            "jacobian.verification.service.execute_process",
            lambda *_args, _stopped=stopped, **_kwargs: _stopped,
        )

        result = service.verify_witness(
            claim_uri=claim_uri,
            candidate_uri=candidate_uri,
            witness_uri=witness_uri,
            checker_id=checker_id,
        )

        assert result.execution.status is expected_status
        assert result.conclusion is Conclusion.UNKNOWN, expected_status
        assert result.verification_record_uri is None, expected_status
        assert result.verification_record_uri is None, expected_status


def test_checker_failure_does_not_expose_internal_exception_text(
    tmp_path: Path,
) -> None:
    _, service, checker_id, claim_uri, candidate_uri, witness_uri, _ = _graph_case(
        tmp_path,
        checker_entrypoint=(
            "tests.component.checkers._fixture_checkers:fail_with_internal_detail"
        ),
    )

    result = service.verify_witness(
        claim_uri=claim_uri,
        candidate_uri=candidate_uri,
        witness_uri=witness_uri,
        checker_id=checker_id,
    )

    assert result.execution.status.value == "ERROR"
    assert result.execution.detail == (
        "The checker stopped before returning a decision. Retry once; "
        "if it happens again, inspect the local checker log."
    )
    assert "fixture" not in result.execution.detail
    assert "secret" not in result.execution.detail


@pytest.mark.parametrize(
    ("checker_entrypoint", "expected_detail"),
    [
        (
            "tests.component.checkers._fixture_checkers:exit_without_response",
            "The checker returned an unreadable response. Retry once; "
            "if it happens again, inspect the local checker log.",
        ),
        (
            "tests.component.checkers._fixture_checkers:return_invalid_decision",
            "The checker returned an invalid decision. Inspect the local checker "
            "log before retrying.",
        ),
    ],
)
def test_invalid_checker_responses_explain_recovery(
    tmp_path: Path,
    checker_entrypoint: str,
    expected_detail: str,
) -> None:
    _, service, checker_id, claim_uri, candidate_uri, witness_uri, _ = _graph_case(
        tmp_path,
        checker_entrypoint=checker_entrypoint,
    )

    result = service.verify_witness(
        claim_uri=claim_uri,
        candidate_uri=candidate_uri,
        witness_uri=witness_uri,
        checker_id=checker_id,
    )

    assert result.execution.status.value == "ERROR"
    assert result.execution.detail == expected_detail


def test_checker_output_limit_explains_recovery(tmp_path: Path) -> None:
    _, service, checker_id, claim_uri, candidate_uri, witness_uri, _ = _graph_case(
        tmp_path
    )
    service.max_checker_output_bytes = 32

    result = service.verify_witness(
        claim_uri=claim_uri,
        candidate_uri=candidate_uri,
        witness_uri=witness_uri,
        checker_id=checker_id,
    )

    assert result.execution.status.value == "ERROR"
    assert result.execution.detail == (
        "The checker returned too much data. Retry with a smaller input "
        "and inspect the local checker log if the limit is reached again."
    )


def test_checker_diagnostic_limit_explains_recovery(tmp_path: Path) -> None:
    _, service, checker_id, claim_uri, candidate_uri, witness_uri, _ = _graph_case(
        tmp_path,
        checker_entrypoint="tests.component.checkers._fixture_checkers:emit_large_diagnostic",
    )
    service.max_checker_diagnostic_bytes = 32

    result = service.verify_witness(
        claim_uri=claim_uri,
        candidate_uri=candidate_uri,
        witness_uri=witness_uri,
        checker_id=checker_id,
    )

    assert result.execution.status.value == "ERROR"
    assert result.execution.detail == (
        "The checker produced too many diagnostics. Retry with a smaller input "
        "and inspect the local checker log if the limit is reached again."
    )


def test_missing_verification_artifact_is_rejected_with_recovery(
    tmp_path: Path,
) -> None:
    _, service, checker_id, _, candidate_uri, witness_uri, _ = _graph_case(tmp_path)
    missing_uri = "artifact://sha256/" + "f" * 64

    result = service.verify_witness(
        claim_uri=missing_uri,
        candidate_uri=candidate_uri,
        witness_uri=witness_uri,
        checker_id=checker_id,
    )

    assert result.execution.status.value == "COMPLETED"
    assert result.input.status.value == "REJECTED"
    assert result.input.errors == (
        "A required verification artifact is unavailable or invalid. "
        "Check the artifact URIs and retry.",
    )
    assert missing_uri not in result.input.errors[0]
    assert result.verification_record_uri is None


def test_corrupt_verification_artifact_is_an_operational_failure(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store, service, checker_id, claim_uri, candidate_uri, witness_uri, _ = _graph_case(
        tmp_path
    )
    claim = store.get(claim_uri)
    claim_digest = claim.manifest.object_digest
    store._blob_path(claim.manifest.payload_digest).write_bytes(b"corrupt")

    result = service.verify_witness(
        claim_uri=claim_uri,
        candidate_uri=candidate_uri,
        witness_uri=witness_uri,
        checker_id=checker_id,
    )

    assert result.execution.status.value == "ERROR"
    assert result.input.status.value == "ACCEPTED"
    assert result.execution.detail == (
        "Jacobian detected corrupted local verification data. Restore the state "
        "directory from a trusted backup, then retry."
    )
    assert claim_digest not in result.execution.detail
    assert "ArtifactIntegrityError" not in result.execution.detail
    assert "blob digest mismatch" in caplog.text


def test_omitted_path_witness_is_independently_verified(tmp_path: Path) -> None:
    _, service, checker_id, claim_uri, candidate_uri, witness_uri, _ = _graph_case(
        tmp_path
    )

    result = service.verify_witness(
        claim_uri=claim_uri,
        candidate_uri=candidate_uri,
        witness_uri=witness_uri,
        checker_id=checker_id,
    )

    assert result.conclusion is Conclusion.FALSE
    assert result.verification_record_uri is not None
    assert result.verification_record_uri is not None


def test_witness_checker_cannot_certify_artifacts_outside_its_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, service, checker_id, claim_uri, candidate_uri, witness_uri, _ = _graph_case(
        tmp_path
    )
    foreign_uri = "artifact://sha256/" + "f" * 64
    decisions = (
        CheckerDecision(
            accepted=True,
            conclusion=Conclusion.FALSE,
            arithmetic=Arithmetic.EXACT_INTEGER,
            method=Method.DIRECT_WITNESS,
            coverage=Coverage.NOT_APPLICABLE,
            relation_id="graph.relation.omitted-path",
            relationship_source_artifact_uris=(foreign_uri,),
            relationship_target_artifact_uris=(claim_uri,),
        ),
        CheckerDecision(
            accepted=True,
            conclusion=Conclusion.FALSE,
            arithmetic=Arithmetic.EXACT_INTEGER,
            method=Method.DIRECT_WITNESS,
            coverage=Coverage.NOT_APPLICABLE,
            obligation_uri=foreign_uri,
        ),
    )

    for decision in decisions:
        monkeypatch.setattr(
            service,
            "_run_checker",
            lambda decision=decision, **_kwargs: decision,
        )

        result = service.verify_witness(
            claim_uri=claim_uri,
            candidate_uri=candidate_uri,
            witness_uri=witness_uri,
            checker_id=checker_id,
        )

        assert result.conclusion is Conclusion.UNKNOWN
        assert result.verification_record_uri is None
        assert result.verification_record_uri is None
        assert result.input.status.value == "REJECTED"
        assert "outside its verification request" in result.input.errors[0]


def test_valid_witness_rebound_to_another_candidate_is_rejected(
    tmp_path: Path,
) -> None:
    (
        store,
        service,
        checker_id,
        claim_uri,
        _,
        witness_uri,
        candidate_schema,
    ) = _graph_case(tmp_path)
    claim = store.get(claim_uri)
    nearby_candidate = store.put(
        schema_uri=candidate_schema,
        semantics_uri=claim.manifest.semantics_uri,
        payload={
            "vertices": ["s", "a", "x", "t1"],
            "arcs": [["s", "a"], ["a", "x"], ["x", "t1"]],
            "source": "s",
            "terminals": ["t1"],
            "intended_paths": [["s", "a", "x", "t1"]],
        },
    )

    result = service.verify_witness(
        claim_uri=claim_uri,
        candidate_uri=nearby_candidate.artifact_uri,
        witness_uri=witness_uri,
        checker_id=checker_id,
    )

    assert result.conclusion is Conclusion.UNKNOWN
    assert result.verification_record_uri is None
    assert result.input.status.value == "REJECTED"
    assert result.input.errors == (
        "The witness does not match the supplied claim and candidate. "
        "Recreate the witness from those exact artifacts, then retry.",
    )


def test_witness_without_bound_artifact_parents_is_rejected(
    tmp_path: Path,
) -> None:
    (
        store,
        service,
        checker_id,
        claim_uri,
        candidate_uri,
        witness_uri,
        _,
    ) = _graph_case(tmp_path)
    original = store.get(witness_uri)
    detached = store.put(
        schema_uri=original.manifest.schema_uri,
        semantics_uri=original.manifest.semantics_uri,
        payload=original.payload,
        parents=(),
    )

    result = service.verify_witness(
        claim_uri=claim_uri,
        candidate_uri=candidate_uri,
        witness_uri=detached.artifact_uri,
        checker_id=checker_id,
    )

    assert result.conclusion is Conclusion.UNKNOWN
    assert result.verification_record_uri is None
    assert result.input.status.value == "REJECTED"


def test_schema_label_cannot_authorize_an_invalid_candidate(tmp_path: Path) -> None:
    (
        _,
        service,
        checker_id,
        claim_uri,
        candidate_uri,
        witness_uri,
        _,
    ) = _graph_case(
        tmp_path,
        candidate_schema_definition={
            "type": "object",
            "required": ["operator_reviewed"],
            "properties": {"operator_reviewed": {"const": True}},
        },
    )

    result = service.verify_witness(
        claim_uri=claim_uri,
        candidate_uri=candidate_uri,
        witness_uri=witness_uri,
        checker_id=checker_id,
    )

    assert result.conclusion is Conclusion.UNKNOWN
    assert result.verification_record_uri is None
    assert result.input.status.value == "REJECTED"


def test_revocation_during_checker_execution_prevents_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        store,
        service,
        checker_id,
        claim_uri,
        candidate_uri,
        witness_uri,
        _,
    ) = _graph_case(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    original = service._run_checker

    def delayed_checker(**kwargs: Any) -> Any:
        entered.set()
        assert release.wait(timeout=5)
        return original(**kwargs)

    monkeypatch.setattr(service, "_run_checker", delayed_checker)
    result_holder: list[Any] = []
    worker = threading.Thread(
        target=lambda: result_holder.append(
            service.verify_witness(
                claim_uri=claim_uri,
                candidate_uri=candidate_uri,
                witness_uri=witness_uri,
                checker_id=checker_id,
            )
        )
    )
    worker.start()
    assert entered.wait(timeout=5)
    CheckerRegistry(store).revoke(checker_id, reason="concurrent test")
    release.set()
    worker.join(timeout=10)

    assert not worker.is_alive()
    result = result_holder[0]
    assert result.conclusion is Conclusion.UNKNOWN
    assert result.verification_record_uri is None
    assert result.verification_record_uri is None
