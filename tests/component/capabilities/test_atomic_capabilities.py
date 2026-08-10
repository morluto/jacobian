from pathlib import Path

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
