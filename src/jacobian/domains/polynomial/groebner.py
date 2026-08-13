"""Bounded isolated Gröbner-basis operation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Never

from pydantic import ValidationError

from jacobian.bounded_process import ProcessResourceLimits
from jacobian.canonical import canonicalize_json, loads_strict_json
from jacobian.contracts.operations import OperationDiagnostic
from jacobian.contracts.polynomial_operations import (
    PolynomialGroebnerBasisRequest,
    PolynomialGroebnerBasisResult,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains._examples import example
from jacobian.domains.polynomial.groebner_protocol import (
    GroebnerWorkerRequest,
    GroebnerWorkerResultLimitExceeded,
    parse_groebner_worker_response,
)
from jacobian.operation_bindings import durable_operation
from jacobian.operation_declarations import OperationDeclaration
from jacobian.operations import OperationAbortError
from jacobian.process_policy import ProcessRequest, ProcessTermination, execute_process
from jacobian.worker_environment import worker_environment

_STDOUT_LIMIT = 2_000_000
_STDERR_LIMIT = 64_000


def _failure(
    status: ExecutionStatus,
    code: str,
    message: str,
) -> Never:
    raise OperationAbortError(
        status,
        OperationDiagnostic(
            code=code,
            stage="polynomial_groebner_computation",
            message=message,
            hint="Reduce the ideal size or degree, or increase the bounded wall time.",
        ),
    )


def _compute(
    request: PolynomialGroebnerBasisRequest,
) -> PolynomialGroebnerBasisResult:
    completed = execute_process(
        ProcessRequest(
            executable=sys.executable,
            arguments=(
                "-I",
                "-m",
                "jacobian.domains.polynomial.groebner_worker",
            ),
            stdin_bytes=canonicalize_json(
                GroebnerWorkerRequest(
                    protocol="jacobian.polynomial.groebner.sympy.v1",
                    request=request,
                ).model_dump(mode="json")
            ),
            timeout_seconds=float(request.resource_budget.wall_seconds),
            environment=worker_environment(locale="C"),
            cwd=str(Path.cwd()),
            stdout_limit_bytes=_STDOUT_LIMIT,
            stderr_limit_bytes=_STDERR_LIMIT,
            resource_limits=ProcessResourceLimits(
                cpu_seconds=request.resource_budget.wall_seconds + 1,
                address_space_bytes=1024 * 1024 * 1024,
            ),
        )
    )
    if completed.termination is ProcessTermination.START_FAILED:
        return _failure(
            ExecutionStatus.ERROR,
            "POLYNOMIAL_GROEBNER_WORKER_START_FAILED",
            "The isolated SymPy Gröbner worker could not be started.",
        )
    if completed.termination is ProcessTermination.TIMED_OUT:
        return _failure(
            ExecutionStatus.TIMEOUT,
            "POLYNOMIAL_GROEBNER_TIMEOUT",
            "The Gröbner wall-clock budget expired; no partial basis was retained.",
        )
    if completed.stdout_exceeded or completed.stderr_exceeded:
        return _failure(
            ExecutionStatus.ERROR,
            "POLYNOMIAL_GROEBNER_OUTPUT_LIMIT_EXCEEDED",
            "The isolated worker exceeded its bounded output protocol.",
        )
    if completed.returncode != 0:
        return _failure(
            ExecutionStatus.ERROR,
            "POLYNOMIAL_GROEBNER_WORKER_FAILED",
            "The isolated SymPy Gröbner computation did not complete successfully.",
        )
    try:
        response = parse_groebner_worker_response(loads_strict_json(completed.stdout))
        if isinstance(response, GroebnerWorkerResultLimitExceeded):
            return _failure(
                ExecutionStatus.ERROR,
                "POLYNOMIAL_GROEBNER_RESULT_LIMIT_EXCEEDED",
                response.error.message,
            )
        result = response.result
    except (TypeError, ValueError, ValidationError):
        return _failure(
            ExecutionStatus.ERROR,
            "POLYNOMIAL_GROEBNER_PROTOCOL_INVALID",
            "The worker returned a result outside the bounded exact contract.",
        )
    return result


POLYNOMIAL_GROEBNER_OPERATION = durable_operation(
    OperationDeclaration(
        operation_id="polynomial.groebner_basis.compute",
        version="1",
        title="Compute a bounded Gröbner basis",
        description=(
            "Compute a complete reduced monic Gröbner basis in the commutative "
            "polynomial ring QQ[x_1,...,x_n] in an isolated SymPy worker under "
            "declared input, output, and wall-clock limits. Free-associative and "
            "other noncommutative polynomial algebras are not supported."
        ),
        request_type=PolynomialGroebnerBasisRequest,
        result_type=PolynomialGroebnerBasisResult,
        execute=_compute,
        tags=("polynomial", "groebner", "ideal", "bounded", "exact"),
        examples=(
            example(
                "unit_ideal",
                "Compute a Groebner basis for the unit ideal in one variable.",
                {
                    "generators": [
                        {
                            "variables": ["x"],
                            "polynomial": {
                                "terms": [
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "exponents": [0],
                                    }
                                ]
                            },
                        }
                    ],
                    "monomial_order": "lex",
                    "resource_budget": {"wall_seconds": 10},
                },
            ),
        ),
    ),
    resource_reason="a Gröbner basis may exceed the bounded inline response budget",
)

__all__ = ["POLYNOMIAL_GROEBNER_OPERATION"]
