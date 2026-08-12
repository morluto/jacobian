"""Bounded cvc5 adapter for unverified quantifier-free Alethe production."""

from __future__ import annotations

import os
import stat
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from jacobian.bounded_process import bounded_process_cancelled
from jacobian.canonical import loads_strict_json
from jacobian.capability_service import CapabilityAdapter, CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityInvocationExample,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.contracts.results import Execution, ExecutionStatus
from jacobian.contracts.smt import (
    SmtUnsatProofFindOutput,
    SmtUnsatProofFindRequest,
)
from jacobian.process_policy import (
    ProcessRequest,
    ProcessResult,
    ProcessTermination,
    execute_process,
)
from jacobian.providers.external_solver_runtime import CVC5_VERSION
from jacobian.sat_smt.cvc5_protocol import (
    Cvc5UnsatisfiableWorkerResult,
    Cvc5WorkerResult,
    parse_cvc5_worker_result,
)
from jacobian.sat_smt.cvc5_worker import CVC5_PROOF_LIMIT
from jacobian.sat_smt.smt import ResolvedSmtProblem, SmtArtifactService
from jacobian.schema_registry import model_schema
from jacobian.worker_environment import worker_environment

CVC5_STDOUT_LIMIT = 4_096
CVC5_STDERR_LIMIT = 64_000
_SolverStatus = Literal["SATISFIABLE", "UNSATISFIABLE", "UNKNOWN"]


@dataclass(frozen=True, slots=True)
class _Cvc5Run:
    execution_status: ExecutionStatus
    runtime_ms: int
    solver_status: _SolverStatus | None = None
    proof: bytes | None = None
    alethe_hole_count: int | None = None
    diagnostic: CapabilityDiagnostic | None = None


def install_cvc5_capability(
    smt: SmtArtifactService,
    runtime: CapabilityProviderRuntime,
) -> CapabilityAdapter:
    """Install the Alethe producer for one exact available cvc5 runtime."""

    if (
        runtime.provider != "cvc5"
        or runtime.availability is not CapabilityProviderAvailability.AVAILABLE
        or runtime.version != CVC5_VERSION
        or runtime.digest is None
        or runtime.digest_kind
        is not CapabilityProviderDigestKind.PYTHON_DISTRIBUTION_RECORD
    ):
        raise ValueError("cvc5 capability requires the pinned available runtime")
    return Cvc5UnsatProofFindAdapter(
        smt=smt,
        backend=_Cvc5Backend(runtime=runtime),
    )


class _Cvc5Backend:
    def __init__(self, *, runtime: CapabilityProviderRuntime) -> None:
        self.runtime = runtime

    def run(
        self,
        resolved: ResolvedSmtProblem,
        request: SmtUnsatProofFindRequest,
    ) -> _Cvc5Run:
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="jacobian-cvc5-") as directory:
            root = Path(directory)
            input_path = root / "input.smt2"
            proof_path = root / "proof.alethe"
            input_path.write_bytes(resolved.problem.raw_bytes())
            command = [
                sys.executable,
                "-I",
                "-m",
                "jacobian.sat_smt.cvc5_worker",
                str(input_path),
                str(proof_path),
                request.logic,
                str(request.resource_budget.wall_seconds * 1000),
            ]
            completed = execute_process(
                ProcessRequest(
                    executable=command[0],
                    arguments=tuple(command[1:]),
                    environment=worker_environment(locale="C"),
                    cwd=str(root),
                    timeout_seconds=float(request.resource_budget.wall_seconds),
                    stdin_bytes=b"",
                    stdout_limit_bytes=CVC5_STDOUT_LIMIT,
                    stderr_limit_bytes=CVC5_STDERR_LIMIT,
                )
            )
            operational = _operational_failure(started, completed)
            if operational is not None:
                return operational
            try:
                worker_result = _parse_worker_output(completed.stdout)
            except ValueError:
                return _run_failure(
                    started,
                    code="INVALID_CVC5_WORKER_OUTPUT",
                    stage="solver_output",
                    message=(
                        "The cvc5 worker returned output outside its exact bounded "
                        "protocol; no solver evidence was retained."
                    ),
                )
            if isinstance(worker_result, Cvc5UnsatisfiableWorkerResult):
                try:
                    proof = _read_proof_file(proof_path)
                except (FileNotFoundError, OSError, OverflowError):
                    return _run_failure(
                        started,
                        code="INVALID_CVC5_PROOF_FILE",
                        stage="proof_capture",
                        message=(
                            "The cvc5 Alethe output was missing, unsafe, or exceeded "
                            "the durable artifact limit."
                        ),
                    )
                if proof.count(b":rule hole") != worker_result.alethe_hole_count:
                    return _run_failure(
                        started,
                        code="CVC5_PROOF_METADATA_MISMATCH",
                        stage="proof_capture",
                        message=(
                            "The cvc5 worker proof bytes disagreed with its hole "
                            "metadata; no evidence was retained."
                        ),
                    )
                return _Cvc5Run(
                    execution_status=ExecutionStatus.COMPLETED,
                    runtime_ms=_runtime_ms(started),
                    solver_status=worker_result.solver_status,
                    proof=proof,
                    alethe_hole_count=worker_result.alethe_hole_count,
                )
            if proof_path.exists():
                return _run_failure(
                    started,
                    code="UNEXPECTED_CVC5_PROOF",
                    stage="proof_capture",
                    message=(
                        "The cvc5 worker attached proof material to a non-UNSAT "
                        "report; no evidence was retained."
                    ),
                )
            return _Cvc5Run(
                execution_status=ExecutionStatus.COMPLETED,
                runtime_ms=_runtime_ms(started),
                solver_status=worker_result.solver_status,
            )


class Cvc5UnsatProofFindAdapter:
    """Preserve raw cvc5 Alethe evidence without validating it."""

    typed_input = True

    def __init__(
        self,
        *,
        smt: SmtArtifactService,
        backend: _Cvc5Backend,
    ) -> None:
        self.smt = smt
        self.backend = backend
        self._descriptor = CapabilityDescriptor(
            capability_id="smt.unsat_proof.find",
            version="1",
            title="Find a quantifier-free SMT UNSAT proof",
            description=(
                "Ask pinned cvc5 to emit raw Alethe for one QF_UF, QF_LIA, or "
                "QF_LRA query; preserving bytes or observing no holes does not "
                "establish UNSAT. The synchronous budget is at most 150 seconds; "
                "named Boolean CNF is generally better for finite colorings."
            ),
            provider="cvc5",
            provider_runtime=backend.runtime,
            input_schema=model_schema(SmtUnsatProofFindRequest),
            output_schema=model_schema(SmtUnsatProofFindOutput),
            tags=(
                "smt",
                "unsat",
                "proof",
                "alethe",
                "exploration",
                "cvc5",
                "qf-uf",
                "qf-lia",
                "qf-lra",
            ),
            invocation_examples=(
                CapabilityInvocationExample(
                    name="qf-lia-contradiction",
                    description=(
                        "Minimal valid request shape for a bounded arithmetic "
                        "contradiction."
                    ),
                    input={
                        "logic": "QF_LIA",
                        "smtlib_text": (
                            "(set-logic QF_LIA)\n"
                            "(declare-fun x () Int)\n"
                            "(assert (= x 0))\n"
                            "(assert (= x 1))\n"
                            "(check-sat)\n"
                        ),
                        "resource_budget": {"wall_seconds": 30},
                    },
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        try:
            validated = SmtUnsatProofFindRequest.model_validate(request.input)
            problem_uri = self.smt.put_problem(
                logic=validated.logic,
                smtlib_text=validated.smtlib_text,
            ).artifact_uri
            resolved = self.smt.resolve_problem(problem_uri)
        except (ValidationError, ValueError) as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_SMT_UNSAT_PROOF_REQUEST",
                    stage="input_validation",
                    message=str(exc),
                    path="smtlib_text",
                    schema_uri=self.smt.installation.problem_schema_uri,
                    expected=(
                        "one exact QF_UF, QF_LIA, or QF_LRA SMT-LIB 2.6 query "
                        "within an enforceable wall-time budget"
                    ),
                    hint=(
                        "Use one set-logic, declarations and assertions, then one "
                        "final check-sat; incremental and result commands are excluded."
                    ),
                )
            ) from exc
        run = self.backend.run(resolved, validated)
        if (
            run.execution_status is ExecutionStatus.COMPLETED
            and bounded_process_cancelled()
        ):
            run = _Cvc5Run(
                execution_status=ExecutionStatus.CANCELLED,
                runtime_ms=run.runtime_ms,
                diagnostic=CapabilityDiagnostic(
                    code="CVC5_CANCELLED",
                    stage="solver_execution",
                    message=(
                        "The client cancelled after cvc5 stopped and before proof "
                        "materialization; no solver proof evidence was retained."
                    ),
                ),
            )
        if run.execution_status is not ExecutionStatus.COMPLETED:
            return _failed_result(self.descriptor, resolved, run)
        if run.solver_status is None:
            return _failed_result(self.descriptor, resolved, run)
        proof_uri: str | None = None
        holes: int | None = None
        if run.solver_status == "UNSATISFIABLE":
            if run.proof is None:
                return _failed_result(self.descriptor, resolved, run)
            proof_uri = self.smt.put_proof(
                problem_uri=problem_uri,
                proof=run.proof,
                producer=self.backend.runtime,
                resource_budget=validated.resource_budget.artifact_budget(),
            ).artifact_uri
            holes = run.alethe_hole_count
            if holes is None:
                return _failed_result(self.descriptor, resolved, run)
        output = SmtUnsatProofFindOutput(
            status="PROOF_PRODUCED" if proof_uri is not None else "NO_PROOF_PRODUCED",
            solver_status=run.solver_status,
            problem_uri=problem_uri,
            proof_uri=proof_uri,
            contains_holes=(holes > 0) if holes is not None else None,
            alethe_hole_count=holes,
            detail=(
                (
                    f"cvc5 produced raw Alethe with {holes} lexical hole marker(s); "
                    "the proof remains unverified until a separate compatible "
                    "checker accepts it."
                )
                if proof_uri is not None
                else (
                    f"cvc5 reported {run.solver_status} without producing an "
                    "UNSAT proof; no SAT or UNSAT conclusion follows."
                )
            ),
        )
        artifacts = [problem_uri]
        if proof_uri is not None:
            artifacts.append(proof_uri)
        return _completed_result(
            self.descriptor,
            run,
            output.model_dump(mode="json"),
            tuple(artifacts),
        )


def _completed_result(
    descriptor: CapabilityDescriptor,
    run: _Cvc5Run,
    output: dict[str, object],
    artifact_uris: tuple[str, ...],
) -> CapabilityResult:
    return CapabilityResult(
        capability_id=descriptor.capability_id,
        capability_version=descriptor.version,
        execution=Execution(
            status=ExecutionStatus.COMPLETED,
            runtime_ms=run.runtime_ms,
        ),
        output=output,
        artifact_uris=artifact_uris,
    )


def _failed_result(
    descriptor: CapabilityDescriptor,
    resolved: ResolvedSmtProblem,
    run: _Cvc5Run,
) -> CapabilityResult:
    if run.diagnostic is None:
        raise RuntimeError("run diagnostic is unexpectedly None")
    return CapabilityResult(
        capability_id=descriptor.capability_id,
        capability_version=descriptor.version,
        execution=Execution(
            status=run.execution_status,
            runtime_ms=run.runtime_ms,
            detail=run.diagnostic.message,
        ),
        diagnostics=(run.diagnostic,),
        artifact_uris=(resolved.artifact.artifact_uri,),
    )


def _operational_failure(
    started: float,
    completed: ProcessResult,
) -> _Cvc5Run | None:
    if completed.termination is ProcessTermination.START_FAILED:
        return _run_failure(
            started,
            code="CVC5_EXECUTION_FAILED",
            stage="solver_execution",
            message="The isolated pinned cvc5 worker could not be started.",
        )
    if completed.termination is ProcessTermination.CANCELLED:
        return _run_failure(
            started,
            status=ExecutionStatus.CANCELLED,
            code="CVC5_CANCELLED",
            stage="solver_execution",
            message=(
                "The client cancelled the cvc5 operation; the worker was "
                "terminated and no mathematical conclusion or solver evidence "
                "was retained."
            ),
        )
    if completed.termination is ProcessTermination.OUTPUT_LIMIT_EXCEEDED:
        return _run_failure(
            started,
            code="CVC5_OUTPUT_LIMIT_EXCEEDED",
            stage="solver_output",
            message=(
                "The cvc5 worker exceeded a bounded output stream limit; no "
                "solver evidence was retained."
            ),
        )
    if completed.termination is ProcessTermination.TIMED_OUT:
        return _run_failure(
            started,
            status=ExecutionStatus.TIMEOUT,
            code="CVC5_TIMEOUT",
            stage="solver_execution",
            message=(
                "cvc5 exceeded the declared wall-time budget; no mathematical "
                "conclusion follows."
            ),
        )
    if completed.returncode != 0 or completed.stderr:
        return _run_failure(
            started,
            code="CVC5_EXECUTION_FAILED",
            stage="solver_execution",
            message=(
                "The isolated cvc5 worker failed or emitted an unexpected "
                "diagnostic stream; no solver evidence was retained."
            ),
        )
    return None


def _parse_worker_output(
    stdout: bytes,
) -> Cvc5WorkerResult:
    return parse_cvc5_worker_result(loads_strict_json(stdout))


def _read_proof_file(path: Path) -> bytes:
    path_metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(path_metadata.st_mode):
        raise OSError("proof output is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("proof output is not a regular file")
        if metadata.st_size > CVC5_PROOF_LIMIT:
            raise OverflowError("proof output exceeds its durable artifact limit")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            proof = stream.read(CVC5_PROOF_LIMIT + 1)
        if len(proof) > CVC5_PROOF_LIMIT:
            raise OverflowError("proof output grew beyond its durable artifact limit")
        return proof
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _run_failure(
    started: float,
    *,
    code: str,
    stage: str,
    message: str,
    status: ExecutionStatus = ExecutionStatus.ERROR,
) -> _Cvc5Run:
    return _Cvc5Run(
        execution_status=status,
        runtime_ms=_runtime_ms(started),
        diagnostic=CapabilityDiagnostic(
            code=code,
            stage=stage,
            message=message,
        ),
    )


def _runtime_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))
