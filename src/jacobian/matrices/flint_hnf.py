"""Bounded Python-FLINT row Hermite-normal-form producer."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from jacobian.canonical import canonicalize_json, loads_strict_json
from jacobian.capability_service import CapabilityAdapter, CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityMode,
    CapabilityProviderAvailability,
    CapabilityProviderRuntime,
    CapabilityRelationship,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.matrices import (
    PYTHON_FLINT_HNF_CONFIGURATION,
    ExactIntegerMatrix,
    MatrixHermiteNormalFormOutput,
    MatrixHermiteNormalFormRequest,
)
from jacobian.contracts.results import Execution, ExecutionStatus
from jacobian.domains._examples import example
from jacobian.matrices.flint_hnf_worker import FLINT_HNF_WORKER_PROTOCOL
from jacobian.matrices.normal_forms import MatrixNormalFormArtifactService
from jacobian.process_policy import (
    ProcessRequest,
    ProcessResult,
    ProcessTermination,
    execute_process,
)
from jacobian.provider_runtime import PYTHON_FLINT_VERSION
from jacobian.providers.flint_runtime import python_flint_hnf_provider_runtime
from jacobian.schema_registry import model_schema
from jacobian.worker_environment import worker_environment

FLINT_HNF_STDOUT_LIMIT = 1_000_000
FLINT_HNF_STDERR_LIMIT = 64_000


@dataclass(frozen=True, slots=True)
class _FlintHnfRun:
    execution_status: ExecutionStatus
    runtime_ms: int
    normal_form: ExactIntegerMatrix | None = None
    transformation: ExactIntegerMatrix | None = None
    detail: str | None = None


def install_python_flint_hnf_capability(
    matrices: MatrixNormalFormArtifactService,
    runtime: CapabilityProviderRuntime,
) -> CapabilityAdapter:
    """Install the producer only for the exact supported HNF profile."""

    if (
        runtime.availability is not CapabilityProviderAvailability.AVAILABLE
        or runtime.provider != "python-flint"
        or runtime.version != PYTHON_FLINT_VERSION
        or runtime.configuration != PYTHON_FLINT_HNF_CONFIGURATION
    ):
        raise ValueError("the pinned Python-FLINT HNF runtime is not available")
    return PythonFlintHermiteNormalFormAdapter(
        matrices=matrices,
        runtime=runtime,
    )


class _PythonFlintHnfBackend:
    def __init__(self, runtime: CapabilityProviderRuntime) -> None:
        self.runtime = runtime

    def run(self, request: MatrixHermiteNormalFormRequest) -> _FlintHnfRun:
        started = time.monotonic()
        if python_flint_hnf_provider_runtime(refresh=True) != self.runtime:
            return _failure(
                started,
                ExecutionStatus.ERROR,
                (
                    "The installed Python-FLINT HNF runtime no longer matches the "
                    "capability descriptor; no normal-form evidence was retained."
                ),
            )
        worker_request = {
            "protocol": FLINT_HNF_WORKER_PROTOCOL,
            "matrix": request.matrix.model_dump(mode="json"),
        }
        completed = execute_process(
            ProcessRequest(
                executable=sys.executable,
                arguments=(
                    "-I",
                    "-m",
                    "jacobian.matrices.flint_hnf_worker",
                ),
                stdin_bytes=canonicalize_json(worker_request),
                timeout_seconds=float(request.resource_budget.wall_seconds),
                environment=worker_environment(locale="C"),
                cwd=str(Path.cwd()),
                stdout_limit_bytes=FLINT_HNF_STDOUT_LIMIT,
                stderr_limit_bytes=FLINT_HNF_STDERR_LIMIT,
            )
        )
        if completed.termination is ProcessTermination.START_FAILED:
            return _failure(
                started,
                ExecutionStatus.ERROR,
                "The isolated Python-FLINT HNF worker could not be started.",
            )
        operational = _operational_failure(started, completed)
        if operational is not None:
            return operational
        try:
            normal_form, transformation = _parse_worker_output(
                completed.stdout,
                rows=len(request.matrix.entries),
                columns=len(request.matrix.entries[0]),
            )
        except (UnicodeDecodeError, ValueError, ValidationError):
            return _failure(
                started,
                ExecutionStatus.ERROR,
                (
                    "The Python-FLINT HNF worker returned output outside its exact "
                    "bounded protocol; no normal-form evidence was retained."
                ),
            )
        if python_flint_hnf_provider_runtime(refresh=True) != self.runtime:
            return _failure(
                started,
                ExecutionStatus.ERROR,
                (
                    "The installed Python-FLINT HNF runtime changed during execution; "
                    "no normal-form evidence was retained."
                ),
            )
        return _FlintHnfRun(
            execution_status=ExecutionStatus.COMPLETED,
            runtime_ms=_runtime_ms(started),
            normal_form=normal_form,
            transformation=transformation,
        )


class PythonFlintHermiteNormalFormAdapter:
    """Produce one row-HNF candidate and left transformation."""

    def __init__(
        self,
        *,
        matrices: MatrixNormalFormArtifactService,
        runtime: CapabilityProviderRuntime,
    ) -> None:
        self.matrices = matrices
        self.backend = _PythonFlintHnfBackend(runtime)
        self._descriptor = CapabilityDescriptor(
            capability_id="matrix.normal_form.hermite",
            version="1",
            title="Compute an exact row Hermite normal form",
            description=(
                "Use pinned Python-FLINT to return H and U for one integer matrix, "
                "with the proposed relation H = U A. Verify row-HNF conditions and "
                "unimodularity separately."
            ),
            provider="python-flint",
            provider_runtime=runtime,
            modes=(CapabilityMode.EXPLORE,),
            input_schema=model_schema(MatrixHermiteNormalFormRequest),
            output_schema=model_schema(MatrixHermiteNormalFormOutput),
            tags=(
                "linear-algebra",
                "integer",
                "matrix",
                "hermite-normal-form",
                "exact",
                "python-flint",
            ),
            invocation_examples=(
                example(
                    "unit_matrix",
                    "Compute the row HNF of the one-by-one unit matrix.",
                    {"matrix": {"entries": [["1"]]}},
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        try:
            validated = MatrixHermiteNormalFormRequest.model_validate(request.input)
            matrix_uri = self.matrices.put_matrix(validated.matrix).artifact_uri
            resolved = self.matrices.resolve_matrix(matrix_uri)
        except (ValidationError, ValueError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_EXACT_INTEGER_MATRIX",
                    stage="input_validation",
                    message=str(exc),
                    path="matrix",
                    schema_uri=self.matrices.installation.matrix_schema_uri,
                    expected=(
                        "one nonempty 1..32 by 1..32 exact integer matrix with "
                        "canonical strings of at most 256 decimal digits"
                    ),
                    hint=(
                        "Encode every entry as a canonical integer string such as "
                        '"0", "-3", or "42".'
                    ),
                )
            ) from exc

        run = self.backend.run(validated)
        normal_form_uri: str | None = None
        if (
            run.execution_status is ExecutionStatus.COMPLETED
            and run.normal_form is not None
            and run.transformation is not None
        ):
            normal_form_uri = self.matrices.put_hermite_normal_form(
                matrix_uri=matrix_uri,
                normal_form=run.normal_form.entries,
                transformation=run.transformation.entries,
                producer=self.backend.runtime,
                resource_budget=validated.resource_budget,
            ).artifact_uri
        output = MatrixHermiteNormalFormOutput(
            status=(
                "NORMAL_FORM_PRODUCED"
                if normal_form_uri is not None
                else "NO_NORMAL_FORM_PRODUCED"
            ),
            matrix_uri=matrix_uri,
            normal_form_uri=normal_form_uri,
            normal_form=run.normal_form if normal_form_uri is not None else None,
            transformation=(
                run.transformation if normal_form_uri is not None else None
            ),
            certificate_available=normal_form_uri is not None,
            detail=(
                "Python-FLINT produced exact H and U with the proposed relation "
                "H = U A; row-HNF conditions and unimodularity remain unverified."
                if normal_form_uri is not None
                else (
                    run.detail
                    or "No normal-form evidence was produced; no conclusion follows."
                )
            ),
        )
        artifact_uris = (
            (matrix_uri, normal_form_uri)
            if normal_form_uri is not None
            else (matrix_uri,)
        )
        relationships = (
            (
                CapabilityRelationship(
                    relation_id="matrix.relation.row-hermite-normal-form-of",
                    source_artifact_uris=(normal_form_uri,),
                    target_artifact_uris=(matrix_uri,),
                ),
            )
            if normal_form_uri is not None
            else ()
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=Execution(
                status=run.execution_status,
                runtime_ms=run.runtime_ms,
                detail=run.detail,
            ),
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="the full declared exact integer matrix",
                parameters={
                    "declared_scope": "FULL_MATRIX",
                    "row_count": resolved.binding.row_count,
                    "column_count": resolved.binding.column_count,
                    "normal_form_convention": "FLINT_ROW_HNF",
                    "relation": "H=U*A",
                    "wall_seconds": validated.resource_budget.wall_seconds,
                },
                artifact_uri=matrix_uri,
            ),
            completeness=CapabilityCompleteness(
                status=(
                    CapabilityCompletenessStatus.COMPLETE
                    if normal_form_uri is not None
                    else CapabilityCompletenessStatus.UNKNOWN
                ),
                basis=(
                    "one full-size H and U were produced; completeness of the "
                    "provider computation is not independent verification"
                    if normal_form_uri is not None
                    else "the bounded provider attempt produced no normal-form "
                    "evidence; no mathematical conclusion follows"
                ),
                assurance_level=(
                    CapabilityAssuranceLevel.COMPUTED
                    if run.execution_status is ExecutionStatus.COMPLETED
                    else CapabilityAssuranceLevel.HEURISTIC
                ),
            ),
            assurance=CapabilityAssurance(
                level=(
                    CapabilityAssuranceLevel.COMPUTED
                    if run.execution_status is ExecutionStatus.COMPLETED
                    else CapabilityAssuranceLevel.HEURISTIC
                ),
                basis=(
                    "the pinned exact-arithmetic provider produced bound H and U, "
                    "but provider success does not verify row equivalence or HNF"
                    if normal_form_uri is not None
                    else "provider execution did not complete; no mathematical "
                    "conclusion follows"
                ),
            ),
            artifact_uris=artifact_uris,
            relationships=relationships,
        )


def _parse_worker_output(
    stdout: bytes,
    *,
    rows: int,
    columns: int,
) -> tuple[ExactIntegerMatrix, ExactIntegerMatrix]:
    if not stdout.endswith(b"\n") or stdout.count(b"\n") != 1:
        raise ValueError("worker output is not exactly one line")
    payload: Any = loads_strict_json(stdout[:-1])
    if (
        not isinstance(payload, dict)
        or payload.get("protocol") != FLINT_HNF_WORKER_PROTOCOL
        or payload.get("status") != "NORMAL_FORM_PRODUCED"
        or payload.get("backend_version") != PYTHON_FLINT_VERSION
        or set(payload)
        != {
            "protocol",
            "status",
            "backend_version",
            "flint_library_version",
            "normal_form",
            "transformation",
        }
        or payload.get("flint_library_version") != "3.6.0"
    ):
        raise ValueError("worker protocol is invalid")
    normal_form = ExactIntegerMatrix(entries=payload["normal_form"])
    transformation = ExactIntegerMatrix(entries=payload["transformation"])
    if (
        len(normal_form.entries),
        len(normal_form.entries[0]),
    ) != (rows, columns):
        raise ValueError("worker normal-form shape is invalid")
    if (
        len(transformation.entries),
        len(transformation.entries[0]),
    ) != (rows, rows):
        raise ValueError("worker transformation shape is invalid")
    return normal_form, transformation


def _operational_failure(
    started: float,
    completed: ProcessResult,
) -> _FlintHnfRun | None:
    if completed.termination is ProcessTermination.TIMED_OUT:
        return _failure(
            started,
            ExecutionStatus.TIMEOUT,
            "The bounded Python-FLINT HNF attempt timed out; no conclusion follows.",
        )
    if completed.stdout_exceeded or completed.stderr_exceeded:
        return _failure(
            started,
            ExecutionStatus.ERROR,
            "The Python-FLINT HNF worker exceeded its output limit.",
        )
    if completed.returncode != 0 or completed.stderr or not completed.stdout:
        return _failure(
            started,
            ExecutionStatus.ERROR,
            "The Python-FLINT HNF worker failed; no evidence was retained.",
        )
    return None


def _failure(
    started: float,
    status: ExecutionStatus,
    detail: str,
) -> _FlintHnfRun:
    return _FlintHnfRun(
        execution_status=status,
        runtime_ms=_runtime_ms(started),
        detail=detail,
    )


def _runtime_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))
