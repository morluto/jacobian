from pathlib import Path

from pydantic import BaseModel

from jacobian.atomic_capabilities import AtomicServiceAdapter
from jacobian.contracts.capabilities import (
    CapabilityCompletenessStatus,
    CapabilityRequest,
)
from jacobian.contracts.results import (
    Arithmetic,
    Assurance,
    Conclusion,
    Coverage,
    Execution,
    ExecutionStatus,
    InputStatus,
    InputValidation,
    Method,
    ResultEnvelope,
    Verification,
)
from jacobian.storage.repository import ArtifactRepository

_ARTIFACT_URI = "artifact://sha256/" + "a" * 64


def test_failed_exhaustive_result_cannot_claim_complete_coverage(
    tmp_path: Path,
) -> None:
    failed = ResultEnvelope(
        execution=Execution(status=ExecutionStatus.ERROR, detail="checker failed"),
        input=InputValidation(status=InputStatus.ACCEPTED),
        conclusion=Conclusion.UNKNOWN,
        assurance=Assurance(
            arithmetic=Arithmetic.EXACT_INTEGER,
            method=Method.EXHAUSTIVE_FINITE,
            coverage=Coverage.EXHAUSTIVE,
            verification=Verification.UNVERIFIED,
        ),
        claim_digest="sha256:" + "a" * 64,
        semantics_digest="sha256:" + "b" * 64,
        candidate_digest="sha256:" + "c" * 64,
    )
    adapter = AtomicServiceAdapter(
        capability_id="test.verify.exhaustive",
        title="Test exhaustive verifier",
        description="Project one failed exhaustive verifier result.",
        input_schema={"type": "object", "additionalProperties": False},
        output_schema=ResultEnvelope.model_json_schema(),
        invoke=lambda _payload: failed,
        store=ArtifactRepository(tmp_path),
    )

    result = adapter.invoke(
        CapabilityRequest(
            capability_id="test.verify.exhaustive",
            input={},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.completeness.status is CapabilityCompletenessStatus.PARTIAL


def test_exhaustive_result_without_scope_cannot_claim_complete_coverage(
    tmp_path: Path,
) -> None:
    exhaustive = ResultEnvelope(
        execution=Execution(status=ExecutionStatus.COMPLETED),
        input=InputValidation(status=InputStatus.ACCEPTED),
        conclusion=Conclusion.UNKNOWN,
        assurance=Assurance(
            arithmetic=Arithmetic.EXACT_INTEGER,
            method=Method.EXHAUSTIVE_FINITE,
            coverage=Coverage.EXHAUSTIVE,
            verification=Verification.UNVERIFIED,
        ),
        claim_digest="sha256:" + "a" * 64,
        semantics_digest="sha256:" + "b" * 64,
        candidate_digest="sha256:" + "c" * 64,
    )
    adapter = AtomicServiceAdapter(
        capability_id="test.explore.exhaustive",
        title="Test exhaustive exploration",
        description="Project one exhaustive result without a declared scope.",
        input_schema={"type": "object", "additionalProperties": False},
        output_schema=ResultEnvelope.model_json_schema(),
        invoke=lambda _payload: exhaustive,
        store=ArtifactRepository(tmp_path),
    )

    result = adapter.invoke(
        CapabilityRequest(
            capability_id="test.explore.exhaustive",
            input={},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.scope is None
    assert result.completeness.status is CapabilityCompletenessStatus.PARTIAL


def test_artifact_uri_in_explanatory_field_is_not_promoted_to_provenance(
    tmp_path: Path,
) -> None:
    """A valid artifact URI in a debug/message field must not enter artifact_uris."""

    class FakeResult(BaseModel):
        artifact_uri: str = _ARTIFACT_URI
        debug_message: str = f"{_ARTIFACT_URI} was considered but not used"

    adapter = AtomicServiceAdapter(
        capability_id="test.explicit",
        title="Test explicit references",
        description="Only the declared artifact_uri should appear.",
        input_schema={"type": "object", "additionalProperties": False},
        output_schema={"type": "object"},
        invoke=lambda _payload: FakeResult(),
        store=ArtifactRepository(tmp_path),
        artifact_references=lambda v: (v.artifact_uri,),
    )

    result = adapter.invoke(CapabilityRequest(capability_id="test.explicit", input={}))

    assert result.artifact_uris == (_ARTIFACT_URI,)


def test_no_artifact_references_callback_yields_empty_artifact_uris(
    tmp_path: Path,
) -> None:
    """Without an artifact_references callback, artifact_uris must be empty."""

    class FakeResult(BaseModel):
        explanatory: str = f"see {_ARTIFACT_URI} for context"

    adapter = AtomicServiceAdapter(
        capability_id="test.no_refs",
        title="Test no references",
        description="No artifact references declared.",
        input_schema={"type": "object", "additionalProperties": False},
        output_schema={"type": "object"},
        invoke=lambda _payload: FakeResult(),
        store=ArtifactRepository(tmp_path),
    )

    result = adapter.invoke(CapabilityRequest(capability_id="test.no_refs", input={}))

    assert result.artifact_uris == ()


def test_rejected_candidate_uri_does_not_enter_artifact_uris(
    tmp_path: Path,
) -> None:
    """A rejected candidate URI must not become provenance."""

    class FakeResult(BaseModel):
        produced_uri: str = _ARTIFACT_URI
        rejected_candidates: tuple[str, ...] = (
            "artifact://sha256/" + "b" * 64,
            "artifact://sha256/" + "c" * 64,
        )

    adapter = AtomicServiceAdapter(
        capability_id="test.rejected",
        title="Test rejected candidates",
        description="Only produced_uri should appear.",
        input_schema={"type": "object", "additionalProperties": False},
        output_schema={"type": "object"},
        invoke=lambda _payload: FakeResult(),
        store=ArtifactRepository(tmp_path),
        artifact_references=lambda v: (v.produced_uri,),
    )

    result = adapter.invoke(CapabilityRequest(capability_id="test.rejected", input={}))

    assert result.artifact_uris == (_ARTIFACT_URI,)
