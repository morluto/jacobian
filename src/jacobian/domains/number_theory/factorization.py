"""Budgeted declarations for complete factorization-derived operations."""

from __future__ import annotations

import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, overload

from jacobian.bounded_process import ProcessResourceLimits
from jacobian.canonical import canonicalize_json, loads_strict_json
from jacobian.contracts.number_theory import (
    ArithmeticFunctionRequest,
    BooleanResult,
    DivisorListResult,
    FactorizationRequest,
    IntegerValueResult,
    PowerfulNumberRequest,
    PowerfulNumberResult,
    PrimeFactorizationResult,
)
from jacobian.contracts.operations import (
    OperationDiagnostic,
    OperationExample,
)
from jacobian.contracts.results import ContractModel, ExecutionStatus
from jacobian.domains._examples import example
from jacobian.domains.number_theory.factorization_protocol import (
    PROTOCOL,
    DivisorsWorkerRequest,
    DivisorsWorkerResponse,
    FactorizationWorkerRequest,
    FactorizationWorkerResponse,
    PowerfulWorkerRequest,
    PowerfulWorkerResponse,
    PrimeFactorizationWorkerRequest,
    PrimeFactorizationWorkerResponse,
    ProperDivisorsWorkerRequest,
    ProperDivisorsWorkerResponse,
    RadicalWorkerRequest,
    RadicalWorkerResponse,
    SquarefreeWorkerRequest,
    SquarefreeWorkerResponse,
    parse_factorization_worker_response,
)
from jacobian.operation_bindings import inline_operation
from jacobian.operation_declarations import OperationDeclaration
from jacobian.operations import (
    OperationAbortError,
    OperationRefusalError,
)
from jacobian.process_policy import ProcessRequest, ProcessTermination, execute_process
from jacobian.worker_environment import worker_environment

_STDOUT_LIMIT = 2_000_000
_STDERR_LIMIT = 64_000
_ADDRESS_SPACE_LIMIT = 512 * 1024 * 1024


def _diagnostic(code: str, message: str) -> OperationDiagnostic:
    return OperationDiagnostic(
        code=code,
        stage="integer_factorization",
        message=message,
        hint="Reduce the integer size or increase the bounded wall time.",
    )


def _classify_termination(
    completed: Any,
) -> None:
    """Raise a typed execution signal for an unsuccessful worker termination."""

    if completed.termination is ProcessTermination.START_FAILED:
        raise OperationAbortError(
            ExecutionStatus.ERROR,
            _diagnostic(
                "INTEGER_FACTORIZATION_WORKER_START_FAILED",
                "The isolated SymPy worker could not be started safely.",
            ),
        )
    if completed.termination is ProcessTermination.TIMED_OUT:
        raise OperationAbortError(
            ExecutionStatus.TIMEOUT,
            _diagnostic(
                "INTEGER_FACTORIZATION_TIMEOUT",
                "The factorization budget expired; no complete result is available.",
            ),
        )
    if completed.stdout_exceeded or completed.stderr_exceeded:
        raise OperationAbortError(
            ExecutionStatus.ERROR,
            _diagnostic(
                "INTEGER_FACTORIZATION_OUTPUT_LIMIT_EXCEEDED",
                "The worker exceeded its bounded output protocol.",
            ),
        )
    resource_signals = {-signal.SIGKILL}
    if (cpu_signal := getattr(signal, "SIGXCPU", None)) is not None:
        resource_signals.add(-cpu_signal)
    if completed.returncode in resource_signals:
        raise OperationAbortError(
            ExecutionStatus.TIMEOUT,
            _diagnostic(
                "INTEGER_FACTORIZATION_RESOURCE_LIMIT_EXCEEDED",
                "The worker exhausted its CPU or process resource budget; no complete result is available.",
            ),
        )
    if completed.returncode != 0:
        raise OperationAbortError(
            ExecutionStatus.ERROR,
            _diagnostic(
                "INTEGER_FACTORIZATION_WORKER_FAILED",
                "The isolated SymPy computation failed without a conclusion.",
            ),
        )
    return None


@overload
def _run_worker(
    request: DivisorsWorkerRequest,
) -> DivisorsWorkerResponse: ...


@overload
def _run_worker(
    request: ProperDivisorsWorkerRequest,
) -> ProperDivisorsWorkerResponse: ...


@overload
def _run_worker(
    request: PrimeFactorizationWorkerRequest,
) -> PrimeFactorizationWorkerResponse: ...


@overload
def _run_worker(
    request: PowerfulWorkerRequest,
) -> PowerfulWorkerResponse: ...


@overload
def _run_worker(
    request: SquarefreeWorkerRequest,
) -> SquarefreeWorkerResponse: ...


@overload
def _run_worker(
    request: RadicalWorkerRequest,
) -> RadicalWorkerResponse: ...


def _run_worker(
    request: FactorizationWorkerRequest,
) -> FactorizationWorkerResponse:
    wall_seconds = int(request.request.resource_budget.wall_seconds)
    completed = execute_process(
        ProcessRequest(
            executable=sys.executable,
            arguments=(
                "-I",
                "-m",
                "jacobian.domains.number_theory.factorization_worker",
            ),
            stdin_bytes=canonicalize_json(request.model_dump(mode="json")),
            timeout_seconds=float(wall_seconds),
            environment=worker_environment(locale="C"),
            cwd=str(Path.cwd()),
            stdout_limit_bytes=_STDOUT_LIMIT,
            stderr_limit_bytes=_STDERR_LIMIT,
            resource_limits=ProcessResourceLimits(
                cpu_seconds=wall_seconds + 1,
                address_space_bytes=_ADDRESS_SPACE_LIMIT,
            ),
        )
    )
    _classify_termination(completed)
    try:
        payload = loads_strict_json(completed.stdout)
        return parse_factorization_worker_response(
            payload,
            expected_operation=request.operation,
        )
    except (TypeError, ValueError) as exc:
        raise OperationAbortError(
            ExecutionStatus.ERROR,
            _diagnostic(
                "INTEGER_FACTORIZATION_PROTOCOL_INVALID",
                "The worker returned data outside the exact result contract.",
            ),
        ) from exc


def _reject_zero(request: FactorizationRequest) -> None:
    if int(request.value) == 0:
        raise OperationRefusalError(
            _diagnostic(
                "INTEGER_FACTORIZATION_NOT_APPLICABLE",
                "Zero has no finite factorization or divisor enumeration.",
            )
        )


def _compute_divisors(
    request: FactorizationRequest,
) -> DivisorListResult:
    _reject_zero(request)
    response = _run_worker(
        DivisorsWorkerRequest(
            protocol=PROTOCOL,
            operation="divisors",
            request=request,
        )
    )
    return response.result


def _compute_proper_divisors(
    request: FactorizationRequest,
) -> DivisorListResult:
    _reject_zero(request)
    response = _run_worker(
        ProperDivisorsWorkerRequest(
            protocol=PROTOCOL,
            operation="proper_divisors",
            request=request,
        )
    )
    return response.result


def _compute_prime_factorization(
    request: FactorizationRequest,
) -> PrimeFactorizationResult:
    _reject_zero(request)
    response = _run_worker(
        PrimeFactorizationWorkerRequest(
            protocol=PROTOCOL,
            operation="prime_factorization",
            request=request,
        )
    )
    return response.result


def _compute_powerful(
    request: PowerfulNumberRequest,
) -> PowerfulNumberResult:
    response = _run_worker(
        PowerfulWorkerRequest(
            protocol=PROTOCOL,
            operation="powerful",
            request=request,
        )
    )
    return response.result


def _compute_squarefree(
    request: ArithmeticFunctionRequest,
) -> BooleanResult:
    response = _run_worker(
        SquarefreeWorkerRequest(
            protocol=PROTOCOL,
            operation="squarefree",
            request=request,
        )
    )
    return response.result


def _compute_radical(
    request: ArithmeticFunctionRequest,
) -> IntegerValueResult:
    response = _run_worker(
        RadicalWorkerRequest(
            protocol=PROTOCOL,
            operation="radical",
            request=request,
        )
    )
    return response.result


def _operation[RequestT: ContractModel, ResultT: ContractModel](
    *,
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    implementation: Callable[[RequestT], ResultT],
    tags: tuple[str, ...],
    examples: tuple[OperationExample, ...] = (),
) -> OperationDeclaration[RequestT, ResultT]:
    return inline_operation(
        OperationDeclaration(
            operation_id=operation_id,
            version="2",
            title=title,
            description=description,
            request_type=request_model,
            result_type=result_model,
            execute=implementation,
            tags=tags,
            examples=examples,
        )
    )


FACTORIZATION_OPERATIONS = (
    _operation(
        operation_id="integer.compute.divisors",
        title="Enumerate positive divisors",
        description=(
            "Enumerate every positive divisor in an isolated, resource-bounded "
            "SymPy worker. Timeout is a non-conclusion."
        ),
        request_model=FactorizationRequest,
        result_model=DivisorListResult,
        implementation=_compute_divisors,
        tags=("number-theory", "enumeration"),
        examples=(
            example(
                "divisors_12", "Enumerate the positive divisors of 12.", {"value": "12"}
            ),
        ),
    ),
    _operation(
        operation_id="integer.compute.proper_divisors",
        title="Enumerate proper divisors",
        description=(
            "Enumerate every positive proper divisor in an isolated, "
            "resource-bounded SymPy worker. Timeout is a non-conclusion."
        ),
        request_model=FactorizationRequest,
        result_model=DivisorListResult,
        implementation=_compute_proper_divisors,
        tags=("number-theory", "enumeration"),
        examples=(
            example(
                "proper_divisors_12",
                "Enumerate the proper divisors of 12.",
                {"value": "12"},
            ),
        ),
    ),
    _operation(
        operation_id="integer.compute.prime_factorization",
        title="Factor an integer",
        description=(
            "Compute a complete prime-power factorization in an isolated, "
            "resource-bounded SymPy worker. Timeout is a non-conclusion."
        ),
        request_model=FactorizationRequest,
        result_model=PrimeFactorizationResult,
        implementation=_compute_prime_factorization,
        tags=("number-theory", "factorization"),
        examples=(
            example(
                "prime_factorization_360",
                "Factor 360 into prime powers.",
                {"value": "360"},
            ),
        ),
    ),
    _operation(
        operation_id="integer.decide.powerful",
        title="Decide powerful-number status",
        description=(
            "Decide whether every prime exponent of one positive integer is at "
            "least two, preserving the complete factor witness and every "
            "violating prime from an isolated, resource-bounded SymPy worker."
        ),
        request_model=PowerfulNumberRequest,
        result_model=PowerfulNumberResult,
        implementation=_compute_powerful,
        tags=("number-theory", "factorization", "predicate"),
        examples=(
            example(
                "powerful_72",
                "Decide whether 72 is powerful and inspect its factor witness.",
                {"value": "72"},
            ),
        ),
    ),
    _operation(
        operation_id="integer.decide.squarefree",
        title="Decide squarefreeness",
        description=(
            "Decide whether a bounded nonnegative integer is square-free in an "
            "isolated, resource-bounded SymPy worker."
        ),
        request_model=ArithmeticFunctionRequest,
        result_model=BooleanResult,
        implementation=_compute_squarefree,
        tags=("number-theory", "predicate"),
        examples=(
            example("squarefree_30", "Check whether 30 is square-free.", {"n": 30}),
        ),
    ),
    _operation(
        operation_id="integer.compute.radical",
        title="Compute integer radical",
        description=(
            "Compute the product of distinct prime divisors in an isolated, "
            "resource-bounded SymPy worker."
        ),
        request_model=ArithmeticFunctionRequest,
        result_model=IntegerValueResult,
        implementation=_compute_radical,
        tags=("number-theory", "arithmetic-function"),
        examples=(example("radical_360", "Compute the radical of 360.", {"n": 360}),),
    ),
)
