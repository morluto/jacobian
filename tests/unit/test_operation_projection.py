from __future__ import annotations

import pytest

from jacobian.contracts.number_theory import IntegerValueResult
from jacobian.contracts.operations import OperationDiagnostic
from jacobian.contracts.results import ExecutionStatus
from jacobian.operation_projection import OperationProjection, project_operation_result
from jacobian.operation_publication import PublishedOperation
from jacobian.operations import Completed, Failed, NonConclusion


def test_completed_projection_preserves_published_artifact_lineage() -> None:
    value = IntegerValueResult(value="7")
    artifact_uri = "artifact://sha256/" + "a" * 64

    result = project_operation_result(
        OperationProjection(
            operation_id="integer.example.compute",
            version="1",
            terminal=Completed(value=value, runtime_ms=12),
            publication=PublishedOperation(
                output=value,
                artifact_uris=(artifact_uri,),
            ),
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.execution.runtime_ms == 12
    assert result.output == value.model_dump(mode="json")
    assert result.artifact_uris == (artifact_uri,)


def test_failed_projection_rejects_typed_output() -> None:
    diagnostic = OperationDiagnostic(
        code="EXAMPLE_FAILED",
        stage="execution",
        message="The example operation failed.",
    )

    with pytest.raises(ValueError, match="non-completed operations"):
        OperationProjection(
            operation_id="integer.example.compute",
            version="1",
            terminal=Failed(ExecutionStatus.ERROR, diagnostic),
            publication=PublishedOperation(
                output=IntegerValueResult(value="7"),
            ),
        )


def test_nonconclusion_projection_rejects_typed_output() -> None:
    diagnostic = OperationDiagnostic(
        code="EXAMPLE_UNSUPPORTED",
        stage="preflight",
        message="The example operation is unsupported.",
    )

    with pytest.raises(ValueError, match="non-completed operations"):
        OperationProjection(
            operation_id="integer.example.compute",
            version="1",
            terminal=NonConclusion(diagnostic),
            publication=PublishedOperation(
                output=IntegerValueResult(value="7"),
            ),
        )


def test_projection_rejects_verification_record_without_completed_lineage() -> None:
    diagnostic = OperationDiagnostic(
        code="EXAMPLE_FAILED",
        stage="execution",
        message="The example operation failed.",
    )

    with pytest.raises(ValueError, match="completed execution"):
        OperationProjection(
            operation_id="integer.example.verify",
            version="1",
            terminal=Failed(ExecutionStatus.ERROR, diagnostic),
            verification_record_uri="artifact://sha256/" + "a" * 64,
        )


def test_failed_projection_preserves_retained_artifacts_without_output() -> None:
    diagnostic = OperationDiagnostic(
        code="EXAMPLE_TIMEOUT",
        stage="execution",
        message="The example operation timed out.",
    )
    artifact_uri = "artifact://sha256/" + "b" * 64

    result = project_operation_result(
        OperationProjection(
            operation_id="integer.example.compute",
            version="1",
            terminal=Failed(
                ExecutionStatus.TIMEOUT,
                diagnostic,
                runtime_ms=23,
            ),
            publication=PublishedOperation(artifact_uris=(artifact_uri,)),
        )
    )

    assert result.execution.status is ExecutionStatus.TIMEOUT
    assert result.execution.runtime_ms == 23
    assert result.output == {}
    assert result.diagnostics == (diagnostic,)
    assert result.artifact_uris == (artifact_uri,)
