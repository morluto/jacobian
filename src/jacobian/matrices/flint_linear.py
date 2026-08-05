"""Bounded Python-FLINT rational linear-system evidence producers."""

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
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.linear import (
    LinearRationalInconsistencyFindOutput,
    LinearRationalInconsistencyFindRequest,
    LinearRationalSolutionFindOutput,
    LinearRationalSolutionFindRequest,
)
from jacobian.contracts.results import Execution, ExecutionStatus
from jacobian.domains._examples import example
from jacobian.matrices.flint_linear_worker import (
    FLINT_LINEAR_INCONSISTENCY_WORKER_PROTOCOL,
    FLINT_LINEAR_WORKER_PROTOCOL,
)
from jacobian.matrices.linear import LinearArtifactService
from jacobian.process_policy import (
    ProcessRequest,
    ProcessResult,
    ProcessTermination,
    execute_process,
)
from jacobian.provider_runtime import PYTHON_FLINT_VERSION
from jacobian.providers.flint_runtime import python_flint_provider_runtime
from jacobian.schema_registry import model_schema
from jacobian.worker_environment import worker_environment

FLINT_LINEAR_STDOUT_LIMIT = 64_000
FLINT_LINEAR_STDERR_LIMIT = 64_000


@dataclass(frozen=True, slots=True)
class _FlintLinearRun:
    execution_status: ExecutionStatus
    runtime_ms: int
    values: tuple[CanonicalRational, ...] | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class _FlintLinearInconsistencyRun:
    execution_status: ExecutionStatus
    runtime_ms: int
    left_witness: tuple[CanonicalRational, ...] | None = None
    rhs_pairing: CanonicalRational | None = None
    detail: str | None = None


def install_python_flint_linear_capability(
    linear: LinearArtifactService,
    runtime: CapabilityProviderRuntime,
) -> CapabilityAdapter:
    """Install one producer only for the exact supported optional runtime."""

    if (
        runtime.availability is not CapabilityProviderAvailability.AVAILABLE
        or runtime.provider != "python-flint"
        or runtime.version != PYTHON_FLINT_VERSION
    ):
        raise ValueError("the pinned Python-FLINT runtime is not available")
    return PythonFlintRationalSolutionFindAdapter(linear=linear, runtime=runtime)


def install_python_flint_inconsistency_capability(
    linear: LinearArtifactService,
    runtime: CapabilityProviderRuntime,
) -> CapabilityAdapter:
    """Install the exact inconsistency-certificate producer."""

    if (
        runtime.availability is not CapabilityProviderAvailability.AVAILABLE
        or runtime.provider != "python-flint"
        or runtime.version != PYTHON_FLINT_VERSION
    ):
        raise ValueError("the pinned Python-FLINT runtime is not available")
    return PythonFlintRationalInconsistencyFindAdapter(
        linear=linear,
        runtime=runtime,
    )


class _PythonFlintBackend:
    def __init__(self, runtime: CapabilityProviderRuntime) -> None:
        self.runtime = runtime

    def run(self, request: LinearRationalSolutionFindRequest) -> _FlintLinearRun:
        started = time.monotonic()
        if python_flint_provider_runtime(refresh=True) != self.runtime:
            return _failure(
                started,
                ExecutionStatus.ERROR,
                (
                    "The installed Python-FLINT runtime no longer matches the "
                    "capability descriptor; no solution evidence was retained."
                ),
            )
        worker_request = {
            "protocol": FLINT_LINEAR_WORKER_PROTOCOL,
            "system": request.system.model_dump(mode="json"),
        }
        completed = execute_process(
            ProcessRequest(
                executable=sys.executable,
                arguments=(
                    "-I",
                    "-m",
                    "jacobian.matrices.flint_linear_worker",
                ),
                stdin_bytes=canonicalize_json(worker_request),
                timeout_seconds=float(request.resource_budget.wall_seconds),
                environment=worker_environment(locale="C"),
                cwd=str(Path.cwd()),
                stdout_limit_bytes=FLINT_LINEAR_STDOUT_LIMIT,
                stderr_limit_bytes=FLINT_LINEAR_STDERR_LIMIT,
            )
        )
        if completed.termination is ProcessTermination.START_FAILED:
            return _failure(
                started,
                ExecutionStatus.ERROR,
                "The isolated Python-FLINT worker could not be started.",
            )
        operational = _operational_failure(started, completed)
        if operational is not None:
            return operational
        try:
            values = _parse_worker_output(completed.stdout)
        except (UnicodeDecodeError, ValueError, ValidationError):
            return _failure(
                started,
                ExecutionStatus.ERROR,
                (
                    "The Python-FLINT worker returned output outside its exact "
                    "bounded protocol; no solution evidence was retained."
                ),
            )
        if python_flint_provider_runtime(refresh=True) != self.runtime:
            return _failure(
                started,
                ExecutionStatus.ERROR,
                (
                    "The installed Python-FLINT runtime changed during execution; "
                    "no solution evidence was retained."
                ),
            )
        return _FlintLinearRun(
            execution_status=ExecutionStatus.COMPLETED,
            runtime_ms=_runtime_ms(started),
            values=values,
        )


class PythonFlintRationalSolutionFindAdapter:
    """Produce one exact vector candidate without verifying it."""

    def __init__(
        self,
        *,
        linear: LinearArtifactService,
        runtime: CapabilityProviderRuntime,
    ) -> None:
        self.linear = linear
        self.backend = _PythonFlintBackend(runtime)
        self._descriptor = CapabilityDescriptor(
            capability_id="linear.rational_solution.find",
            version="1",
            title="Find one exact rational solution",
            description=(
                "Use pinned Python-FLINT to return one exact vector for a declared "
                "finite A x = b system over QQ. A not-found outcome makes no "
                "consistency conclusion; verify any returned vector separately."
            ),
            provider="python-flint",
            provider_runtime=runtime,
            modes=(CapabilityMode.EXPLORE,),
            input_schema=model_schema(LinearRationalSolutionFindRequest),
            output_schema=model_schema(LinearRationalSolutionFindOutput),
            tags=(
                "linear-algebra",
                "rational",
                "exact",
                "solution",
                "witness",
                "python-flint",
            ),
            invocation_examples=(
                example(
                    "one_by_one_system",
                    "Find the solution of x=1 over QQ.",
                    {
                        "system": {
                            "variables": ["x"],
                            "coefficients": {"entries": [[{"num": "1", "den": "1"}]]},
                            "rhs": [{"num": "1", "den": "1"}],
                        }
                    },
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        try:
            validated = LinearRationalSolutionFindRequest.model_validate(request.input)
            system_uri = self.linear.put_system(validated.system).artifact_uri
            resolved = self.linear.resolve_system(system_uri)
        except (ValidationError, ValueError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_RATIONAL_LINEAR_SYSTEM",
                    stage="input_validation",
                    message=str(exc),
                    path="system",
                    schema_uri=self.linear.installation.system_schema_uri,
                    expected=(
                        "one 1..32 by 1..32 exact rational A x = b system with "
                        "unique ordered variable names and canonical reduced entries"
                    ),
                    hint=(
                        'Encode each rational as {"num":"integer","den":"positive '
                        'integer"} and keep coefficient, variable, and RHS dimensions '
                        "aligned."
                    ),
                )
            ) from exc
        run = self.backend.run(validated)
        solution_uri: str | None = None
        if run.execution_status is ExecutionStatus.COMPLETED and run.values is not None:
            solution_uri = self.linear.put_solution(
                system_uri=system_uri,
                values=run.values,
                producer=self.backend.runtime,
                resource_budget=validated.resource_budget,
            ).artifact_uri
        output = LinearRationalSolutionFindOutput(
            status=(
                "SOLUTION_PRODUCED"
                if solution_uri is not None
                else "NO_SOLUTION_PRODUCED"
            ),
            system_uri=system_uri,
            solution_uri=solution_uri,
            solution=run.values if solution_uri is not None else None,
            certificate_available=solution_uri is not None,
            detail=(
                "Python-FLINT produced one exact vector with all free variables "
                "set to zero; the relation remains unverified until independent "
                "replay."
                if solution_uri is not None
                else (
                    run.detail
                    or "No solution witness was produced; no consistency conclusion "
                    "follows."
                )
            ),
        )
        artifact_uris = (
            (system_uri, solution_uri) if solution_uri is not None else (system_uri,)
        )
        relationships = (
            (
                CapabilityRelationship(
                    relation_id="linear.relation.satisfies",
                    source_artifact_uris=(solution_uri,),
                    target_artifact_uris=(system_uri,),
                ),
            )
            if solution_uri is not None
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
                description="the full declared exact rational A x = b system",
                parameters={
                    "declared_scope": "FULL_SYSTEM",
                    "row_count": resolved.binding.row_count,
                    "column_count": resolved.binding.column_count,
                    "variable_order_digest": resolved.binding.variable_order_digest,
                    "free_variable_policy": "ZERO",
                    "wall_seconds": validated.resource_budget.wall_seconds,
                },
                artifact_uri=system_uri,
            ),
            completeness=CapabilityCompleteness(
                status=(
                    CapabilityCompletenessStatus.NOT_APPLICABLE
                    if solution_uri is not None
                    else CapabilityCompletenessStatus.UNKNOWN
                ),
                basis=(
                    "one directly checkable witness was requested and produced; no "
                    "enumeration or uniqueness claim is made"
                    if solution_uri is not None
                    else "the attempt produced no witness; no consistency or "
                    "inconsistency conclusion follows"
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
                    "the pinned exact-arithmetic provider produced a bound vector, "
                    "but provider success does not verify A x = b"
                    if solution_uri is not None
                    else (
                        "the bounded provider attempt completed without witness "
                        "evidence; no opposite conclusion follows"
                        if run.execution_status is ExecutionStatus.COMPLETED
                        else "provider execution did not complete; no mathematical "
                        "conclusion follows"
                    )
                ),
            ),
            artifact_uris=artifact_uris,
            relationships=relationships,
        )


class _PythonFlintInconsistencyBackend:
    def __init__(self, runtime: CapabilityProviderRuntime) -> None:
        self.runtime = runtime

    def run(
        self,
        request: LinearRationalInconsistencyFindRequest,
    ) -> _FlintLinearInconsistencyRun:
        started = time.monotonic()
        if python_flint_provider_runtime(refresh=True) != self.runtime:
            return _inconsistency_failure(
                started,
                ExecutionStatus.ERROR,
                (
                    "The installed Python-FLINT runtime no longer matches the "
                    "capability descriptor; no inconsistency evidence was retained."
                ),
            )
        worker_request = {
            "protocol": FLINT_LINEAR_INCONSISTENCY_WORKER_PROTOCOL,
            "system": request.system.model_dump(mode="json"),
        }
        completed = execute_process(
            ProcessRequest(
                executable=sys.executable,
                arguments=(
                    "-I",
                    "-m",
                    "jacobian.matrices.flint_linear_worker",
                ),
                stdin_bytes=canonicalize_json(worker_request),
                timeout_seconds=float(request.resource_budget.wall_seconds),
                environment=worker_environment(locale="C"),
                cwd=str(Path.cwd()),
                stdout_limit_bytes=FLINT_LINEAR_STDOUT_LIMIT,
                stderr_limit_bytes=FLINT_LINEAR_STDERR_LIMIT,
            )
        )
        if completed.termination is ProcessTermination.START_FAILED:
            return _inconsistency_failure(
                started,
                ExecutionStatus.ERROR,
                "The isolated Python-FLINT worker could not be started.",
            )
        operational = _inconsistency_operational_failure(started, completed)
        if operational is not None:
            return operational
        try:
            left_witness, rhs_pairing = _parse_inconsistency_worker_output(
                completed.stdout
            )
        except (UnicodeDecodeError, ValueError, ValidationError):
            return _inconsistency_failure(
                started,
                ExecutionStatus.ERROR,
                (
                    "The Python-FLINT worker returned output outside its exact "
                    "bounded protocol; no inconsistency evidence was retained."
                ),
            )
        if python_flint_provider_runtime(refresh=True) != self.runtime:
            return _inconsistency_failure(
                started,
                ExecutionStatus.ERROR,
                (
                    "The installed Python-FLINT runtime changed during execution; "
                    "no inconsistency evidence was retained."
                ),
            )
        return _FlintLinearInconsistencyRun(
            execution_status=ExecutionStatus.COMPLETED,
            runtime_ms=_runtime_ms(started),
            left_witness=left_witness,
            rhs_pairing=rhs_pairing,
        )


class PythonFlintRationalInconsistencyFindAdapter:
    """Produce a normalized left witness without certifying it."""

    def __init__(
        self,
        *,
        linear: LinearArtifactService,
        runtime: CapabilityProviderRuntime,
    ) -> None:
        self.linear = linear
        self.backend = _PythonFlintInconsistencyBackend(runtime)
        self._descriptor = CapabilityDescriptor(
            capability_id="linear.rational_inconsistency.find",
            version="1",
            title="Find an exact rational inconsistency certificate",
            description=(
                "Use pinned Python-FLINT to seek a normalized left witness y for "
                "a declared A x = b system, with y^T A = 0 and y^T b = 1. "
                "A not-found outcome makes no consistency conclusion."
            ),
            provider="python-flint",
            provider_runtime=runtime,
            modes=(CapabilityMode.EXPLORE,),
            input_schema=model_schema(LinearRationalInconsistencyFindRequest),
            output_schema=model_schema(LinearRationalInconsistencyFindOutput),
            tags=(
                "linear-algebra",
                "rational",
                "exact",
                "inconsistency",
                "certificate",
                "python-flint",
            ),
            invocation_examples=(
                example(
                    "inconsistent_one_variable",
                    "Analyze x=1 and x=2 as an inconsistent system.",
                    {
                        "system": {
                            "variables": ["x"],
                            "coefficients": {
                                "entries": [
                                    [{"num": "1", "den": "1"}],
                                    [{"num": "1", "den": "1"}],
                                ]
                            },
                            "rhs": [{"num": "1", "den": "1"}, {"num": "2", "den": "1"}],
                        }
                    },
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        try:
            validated = LinearRationalInconsistencyFindRequest.model_validate(
                request.input
            )
            system_uri = self.linear.put_system(validated.system).artifact_uri
            resolved = self.linear.resolve_system(system_uri)
        except (ValidationError, ValueError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_RATIONAL_LINEAR_SYSTEM",
                    stage="input_validation",
                    message=str(exc),
                    path="system",
                    schema_uri=self.linear.installation.system_schema_uri,
                    expected=(
                        "one 1..32 by 1..32 exact rational A x = b system with "
                        "unique ordered variable names and canonical reduced entries"
                    ),
                    hint=(
                        'Encode each rational as {"num":"integer","den":"positive '
                        'integer"} and keep coefficient, variable, and RHS dimensions '
                        "aligned."
                    ),
                )
            ) from exc
        run = self.backend.run(validated)
        certificate_uri: str | None = None
        if (
            run.execution_status is ExecutionStatus.COMPLETED
            and run.left_witness is not None
            and run.rhs_pairing is not None
        ):
            certificate_uri = self.linear.put_inconsistency(
                system_uri=system_uri,
                left_witness=run.left_witness,
                rhs_pairing=run.rhs_pairing,
                producer=self.backend.runtime,
                resource_budget=validated.resource_budget,
            ).artifact_uri
        output = LinearRationalInconsistencyFindOutput(
            status=(
                "CERTIFICATE_PRODUCED"
                if certificate_uri is not None
                else "NO_CERTIFICATE_PRODUCED"
            ),
            system_uri=system_uri,
            certificate_uri=certificate_uri,
            left_witness=run.left_witness if certificate_uri is not None else None,
            rhs_pairing=run.rhs_pairing if certificate_uri is not None else None,
            verification_candidate_available=certificate_uri is not None,
            detail=(
                "Python-FLINT produced a normalized exact left witness; the "
                "inconsistency claim remains unverified until independent replay."
                if certificate_uri is not None
                else (
                    run.detail
                    or "No normalized left witness was produced; no consistency "
                    "conclusion follows."
                )
            ),
        )
        artifact_uris = (
            (system_uri, certificate_uri)
            if certificate_uri is not None
            else (system_uri,)
        )
        relationships = (
            (
                CapabilityRelationship(
                    relation_id="linear.relation.inconsistency-certificate-of",
                    source_artifact_uris=(certificate_uri,),
                    target_artifact_uris=(system_uri,),
                ),
            )
            if certificate_uri is not None
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
                description="the full declared exact rational A x = b system",
                parameters={
                    "declared_scope": "FULL_SYSTEM",
                    "row_count": resolved.binding.row_count,
                    "column_count": resolved.binding.column_count,
                    "variable_order_digest": resolved.binding.variable_order_digest,
                    "normalization": "Y_TRANSPOSE_B_EQUALS_ONE",
                    "wall_seconds": validated.resource_budget.wall_seconds,
                },
                artifact_uri=system_uri,
            ),
            completeness=CapabilityCompleteness(
                status=(
                    CapabilityCompletenessStatus.NOT_APPLICABLE
                    if certificate_uri is not None
                    else CapabilityCompletenessStatus.UNKNOWN
                ),
                basis=(
                    "one directly checkable inconsistency witness was requested and "
                    "produced"
                    if certificate_uri is not None
                    else "the attempt produced no witness; no consistency or "
                    "inconsistency conclusion follows"
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
                    "the pinned exact-arithmetic provider produced a bound left "
                    "witness, but provider success does not verify its equations"
                    if certificate_uri is not None
                    else (
                        "the bounded provider attempt completed without witness "
                        "evidence; no opposite conclusion follows"
                        if run.execution_status is ExecutionStatus.COMPLETED
                        else "provider execution did not complete; no mathematical "
                        "conclusion follows"
                    )
                ),
            ),
            artifact_uris=artifact_uris,
            relationships=relationships,
        )


def _parse_worker_output(
    stdout: bytes,
) -> tuple[CanonicalRational, ...] | None:
    if not stdout.endswith(b"\n") or stdout.count(b"\n") != 1:
        raise ValueError("worker output is not exactly one line")
    payload: Any = loads_strict_json(stdout[:-1])
    if (
        not isinstance(payload, dict)
        or payload.get("protocol") != FLINT_LINEAR_WORKER_PROTOCOL
    ):
        raise ValueError("worker protocol mismatch")
    status = payload.get("status")
    if payload.get("backend_version") != PYTHON_FLINT_VERSION:
        raise ValueError("worker backend version mismatch")
    if status == "NO_SOLUTION_PRODUCED":
        if set(payload) != {"protocol", "status", "backend_version"}:
            raise ValueError("not-found output carries unexpected fields")
        return None
    if status != "SOLUTION_PRODUCED" or set(payload) != {
        "protocol",
        "status",
        "backend_version",
        "values",
    }:
        raise ValueError("worker status is invalid")
    values = payload["values"]
    if not isinstance(values, list) or not 1 <= len(values) <= 32:
        raise ValueError("worker solution shape is invalid")
    return tuple(CanonicalRational.model_validate(value) for value in values)


def _parse_inconsistency_worker_output(
    stdout: bytes,
) -> tuple[tuple[CanonicalRational, ...] | None, CanonicalRational | None]:
    if not stdout.endswith(b"\n") or stdout.count(b"\n") != 1:
        raise ValueError("worker output is not exactly one line")
    payload: Any = loads_strict_json(stdout[:-1])
    if (
        not isinstance(payload, dict)
        or payload.get("protocol") != FLINT_LINEAR_INCONSISTENCY_WORKER_PROTOCOL
    ):
        raise ValueError("worker protocol mismatch")
    status = payload.get("status")
    if payload.get("backend_version") != PYTHON_FLINT_VERSION:
        raise ValueError("worker backend version mismatch")
    if status == "NO_CERTIFICATE_PRODUCED":
        if set(payload) != {"protocol", "status", "backend_version"}:
            raise ValueError("not-found output carries unexpected fields")
        return None, None
    if status != "CERTIFICATE_PRODUCED" or set(payload) != {
        "protocol",
        "status",
        "backend_version",
        "left_witness",
        "rhs_pairing",
    }:
        raise ValueError("worker status is invalid")
    values = payload["left_witness"]
    if not isinstance(values, list) or not 1 <= len(values) <= 32:
        raise ValueError("worker inconsistency-witness shape is invalid")
    witness = tuple(CanonicalRational.model_validate(value) for value in values)
    pairing = CanonicalRational.model_validate(payload["rhs_pairing"])
    if pairing.as_fraction() != 1:
        raise ValueError("worker witness is not normalized")
    return witness, pairing


def _operational_failure(
    started: float,
    completed: ProcessResult,
) -> _FlintLinearRun | None:
    if completed.termination is ProcessTermination.TIMED_OUT:
        return _failure(
            started,
            ExecutionStatus.TIMEOUT,
            "The bounded Python-FLINT attempt timed out; no conclusion follows.",
        )
    if completed.stdout_exceeded or completed.stderr_exceeded:
        return _failure(
            started,
            ExecutionStatus.ERROR,
            "The Python-FLINT worker exceeded its output limit.",
        )
    if completed.returncode != 0 or completed.stderr or not completed.stdout:
        return _failure(
            started,
            ExecutionStatus.ERROR,
            "The Python-FLINT worker failed; no solution evidence was retained.",
        )
    return None


def _failure(
    started: float,
    status: ExecutionStatus,
    detail: str,
) -> _FlintLinearRun:
    return _FlintLinearRun(
        execution_status=status,
        runtime_ms=_runtime_ms(started),
        detail=detail,
    )


def _inconsistency_operational_failure(
    started: float,
    completed: ProcessResult,
) -> _FlintLinearInconsistencyRun | None:
    if completed.termination is ProcessTermination.TIMED_OUT:
        return _inconsistency_failure(
            started,
            ExecutionStatus.TIMEOUT,
            "The bounded Python-FLINT attempt timed out; no conclusion follows.",
        )
    if completed.stdout_exceeded or completed.stderr_exceeded:
        return _inconsistency_failure(
            started,
            ExecutionStatus.ERROR,
            "The Python-FLINT worker exceeded its output limit.",
        )
    if completed.returncode != 0 or completed.stderr or not completed.stdout:
        return _inconsistency_failure(
            started,
            ExecutionStatus.ERROR,
            ("The Python-FLINT worker failed; no inconsistency evidence was retained."),
        )
    return None


def _inconsistency_failure(
    started: float,
    status: ExecutionStatus,
    detail: str,
) -> _FlintLinearInconsistencyRun:
    return _FlintLinearInconsistencyRun(
        execution_status=status,
        runtime_ms=_runtime_ms(started),
        detail=detail,
    )


def _runtime_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))
