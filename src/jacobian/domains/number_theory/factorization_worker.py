"""Isolated SymPy worker for complete factorization-derived operations."""

from __future__ import annotations

import sys

from pydantic import ValidationError

from jacobian.canonical import canonicalize_json, loads_strict_json
from jacobian.domains.number_theory.factorization_kernels import (
    compute_radical,
    decide_powerful,
    decide_squarefree,
    enumerate_divisors,
    enumerate_proper_divisors,
    factorize_primes,
)
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
    parse_factorization_worker_request,
)


def _run(worker_request: FactorizationWorkerRequest) -> FactorizationWorkerResponse:
    if isinstance(worker_request, DivisorsWorkerRequest):
        return DivisorsWorkerResponse(
            protocol=PROTOCOL,
            operation="divisors",
            result=enumerate_divisors(worker_request.request),
        )
    if isinstance(worker_request, ProperDivisorsWorkerRequest):
        return ProperDivisorsWorkerResponse(
            protocol=PROTOCOL,
            operation="proper_divisors",
            result=enumerate_proper_divisors(worker_request.request),
        )
    if isinstance(worker_request, PrimeFactorizationWorkerRequest):
        return PrimeFactorizationWorkerResponse(
            protocol=PROTOCOL,
            operation="prime_factorization",
            result=factorize_primes(worker_request.request),
        )
    if isinstance(worker_request, PowerfulWorkerRequest):
        return PowerfulWorkerResponse(
            protocol=PROTOCOL,
            operation="powerful",
            result=decide_powerful(worker_request.request),
        )
    if isinstance(worker_request, SquarefreeWorkerRequest):
        return SquarefreeWorkerResponse(
            protocol=PROTOCOL,
            operation="squarefree",
            result=decide_squarefree(worker_request.request),
        )
    if isinstance(worker_request, RadicalWorkerRequest):
        return RadicalWorkerResponse(
            protocol=PROTOCOL,
            operation="radical",
            result=compute_radical(worker_request.request),
        )
    raise AssertionError("unreachable factorization worker request")


def main() -> int:
    try:
        worker_request = parse_factorization_worker_request(
            loads_strict_json(sys.stdin.buffer.read())
        )
        response = _run(worker_request)
        sys.stdout.buffer.write(canonicalize_json(response.model_dump(mode="json")))
        return 0
    except (TypeError, ValueError, ValidationError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
