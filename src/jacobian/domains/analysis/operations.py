"""Validated real-function operations backed by Arb."""

from __future__ import annotations

import sys
from pathlib import Path

from jacobian.bounded_process import ProcessResourceLimits
from jacobian.canonical import (
    canonicalize_json,
    loads_strict_json,
)
from jacobian.contracts.operations import OperationDiagnostic
from jacobian.contracts.results import ExecutionStatus
from jacobian.contracts.validated_analysis import (
    ArbPointEnclosureRequest,
    ArbPointEnclosureResult,
)
from jacobian.domains._examples import example
from jacobian.domains.analysis.protocol import (
    PROTOCOL,
    ArbEnclosedWorkerResponse,
    ArbNonfiniteWorkerResponse,
    ArbPointEnclosureWorkerRequest,
    ArbPointEnclosureWorkerResponse,
    parse_arb_worker_response,
)
from jacobian.operation_bindings import inline_operation
from jacobian.operation_declarations import OperationDeclaration
from jacobian.operations import OperationAbortError
from jacobian.process_policy import (
    ProcessRequest,
    ProcessTermination,
    execute_process,
)
from jacobian.worker_environment import worker_environment

_WORKER_MODULE = "jacobian.domains.analysis.worker"


def _run_worker(
    request: ArbPointEnclosureRequest,
) -> ArbPointEnclosureWorkerResponse:
    worker_request = ArbPointEnclosureWorkerRequest(
        protocol=PROTOCOL,
        request=request,
    )
    completed = execute_process(
        ProcessRequest(
            executable=sys.executable,
            arguments=("-I", "-m", _WORKER_MODULE),
            stdin_bytes=canonicalize_json(worker_request.model_dump(mode="json")),
            timeout_seconds=float(request.wall_seconds),
            environment=worker_environment(locale="C"),
            cwd=str(Path.cwd()),
            stdout_limit_bytes=2_000_000,
            stderr_limit_bytes=64_000,
            resource_limits=ProcessResourceLimits(
                cpu_seconds=request.wall_seconds + 1,
                address_space_bytes=1024 * 1024 * 1024,
            ),
        )
    )
    if completed.termination is ProcessTermination.TIMED_OUT:
        raise TimeoutError
    if (
        completed.returncode != 0
        or completed.termination is not ProcessTermination.EXITED
    ):
        raise RuntimeError("Arb point-enclosure worker failed")
    value = loads_strict_json(completed.stdout)
    return parse_arb_worker_response(value)


def _diagnostic(code: str, message: str) -> OperationDiagnostic:
    return OperationDiagnostic(
        code=code,
        stage="validated_analysis_backend",
        message=message,
    )


def _point_enclosure(
    request: ArbPointEnclosureRequest,
) -> ArbPointEnclosureResult:
    try:
        response = _run_worker(request)
        if isinstance(response, ArbEnclosedWorkerResponse):
            return ArbPointEnclosureResult(
                status="ENCLOSED",
                function=request.function,
                argument=request.argument,
                precision_bits=request.precision_bits,
                lower=response.lower,
                upper=response.upper,
                relative_accuracy_bits=response.relative_accuracy_bits,
                exact=response.exact,
                detail=(
                    "Pinned Arb ball arithmetic returned an outward-rounded "
                    "enclosure with exact dyadic endpoints."
                ),
            )
        if isinstance(response, ArbNonfiniteWorkerResponse):
            return ArbPointEnclosureResult(
                status="NONFINITE",
                function=request.function,
                argument=request.argument,
                precision_bits=request.precision_bits,
                detail=(
                    "Arb returned a non-finite ball; no enclosure conclusion "
                    "is available."
                ),
            )
        raise AssertionError("unreachable Arb worker response")
    except TimeoutError:
        detail = (
            "The Arb worker exceeded the declared wall-clock budget; "
            "no enclosure conclusion is available."
        )
        raise OperationAbortError(
            ExecutionStatus.TIMEOUT,
            _diagnostic("ARB_POINT_ENCLOSURE_TIMEOUT", detail),
        ) from None
    except (OSError, RuntimeError, ValueError):
        detail = (
            "The Arb worker failed or returned malformed output; "
            "no enclosure conclusion is available."
        )
        raise OperationAbortError(
            ExecutionStatus.ERROR,
            _diagnostic("ARB_POINT_ENCLOSURE_BACKEND_ERROR", detail),
        ) from None


POINT_ENCLOSURE_OPERATIONS = (
    inline_operation(
        OperationDeclaration(
            operation_id="analysis.real_function.point_enclosure.compute",
            version="1",
            title="Enclose a real function at a rational point",
            description=(
                "Use pinned Arb ball arithmetic to enclose one supported real "
                "function (square root, logarithm, exponential, sine, or cosine) "
                "at one exact rational point within a wall-clock budget."
            ),
            request_type=ArbPointEnclosureRequest,
            result_type=ArbPointEnclosureResult,
            execute=_point_enclosure,
            tags=(
                "analysis",
                "validated",
                "arb",
                "enclosure",
                "bounded",
                "square-root",
                "sqrt",
                "logarithm",
                "log",
                "exponential",
                "exp",
                "sine",
                "sin",
                "cosine",
                "cos",
            ),
            examples=(
                example(
                    "sqrt_zero",
                    "Enclose sqrt(0) at 32-bit precision.",
                    {
                        "function": "SQRT",
                        "argument": {"num": "0", "den": "1"},
                        "precision_bits": 32,
                        "wall_seconds": 10,
                    },
                ),
            ),
        )
    ),
)

__all__ = ["POINT_ENCLOSURE_OPERATIONS"]
