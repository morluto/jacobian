"""Bounded deterministic dense finite-field factorization through SymPy."""

import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.number_theory.galois._models import (
    MAX_FACTOR_DEGREE,
    MAX_FIELD_ORDER,
)
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

MAX_FACTOR_WORK = 1_000_000_000
FACTOR_WALL_SECONDS = 60.0
_FACTOR_WORKER = Path(__file__).resolve().with_name("_factor_worker.py")


def factor_mod_prime(
    prime: int, coefficients: tuple[int, ...]
) -> tuple[int, tuple[tuple[tuple[int, ...], int], ...]]:
    """Admit once, then retain units and square-free multiplicities.

    SymPy's deterministic Berlekamp implementation scans at most d kernel
    vectors and p residues. Across the current disjoint factors, reduction
    and Euclidean gcd cost O(d^2) scalar updates per vector/residue. Matrix
    construction costs O(p*d^2), elimination O(d^3); square-free stages have
    total degree at most d. 32*(p+1)*d^3 reserves these classical updates,
    including square-free decomposition and conversion. Residues and scalar
    products stay below p and p^2; storage is O(d^2), and factor output has
    at most 2*d coefficients (multiplicities do not expand the output).

    FLINT's faster splitter uses unbounded random retries. The maintained
    deterministic path retains a finite work bound. A supervised worker
    makes both square-free decomposition and Berlekamp splitting killable;
    the shared 60-second deadline is a safety limit, not the work bound.
    """
    degree = len(coefficients) - 1
    work = 32 * (prime + 1) * degree**3
    if (
        not 1 <= degree <= MAX_FACTOR_DEGREE
        or not 2 <= prime <= MAX_FIELD_ORDER
        or work > MAX_FACTOR_WORK
    ):
        raise OperationResourceAdmissionError(
            location=("coefficients",),
            code="galois_theory.factor_work_bound",
            message=f"factor degree={degree}, prime={prime}, work={work}; limits: degree {MAX_FACTOR_DEGREE}, prime {MAX_FIELD_ORDER}, work {MAX_FACTOR_WORK}",
        )
    execution = current_request_execution()
    with request_execution(time.monotonic()) if execution is None else nullcontext():
        execution = current_request_execution()
        assert execution is not None
        deadline = execution.started_at + FACTOR_WALL_SECONDS
        if execution.deadline is not None:
            deadline = min(deadline, execution.deadline)
        bind_request_deadline(deadline)
        request_checkpoint("before finite-field factorization")
        result = _factor_polynomial(prime, coefficients)
        request_checkpoint("after finite-field factorization")
        return result


def _factor_polynomial(
    prime: int, coefficients: tuple[int, ...]
) -> tuple[int, tuple[tuple[tuple[int, ...], int], ...]]:
    execution = current_request_execution()
    assert execution is not None and execution.deadline is not None
    input_bytes = json.dumps([prime, coefficients]).encode("utf-8")
    try:
        with TemporaryDirectory(prefix="jacobian-finite-field-") as directory:
            request_checkpoint("before finite-field worker startup")
            remaining = execution.deadline - time.monotonic()
            if remaining <= 0:
                raise OperationExecutionTimeoutError(
                    "finite-field factorization expired"
                )
            completed = run_bounded_process(
                [sys.executable, str(_FACTOR_WORKER)],
                input_bytes=input_bytes,
                timeout_seconds=remaining,
                environment=worker_environment(locale="C.UTF-8"),
                # At most 2*d residue coefficients plus d multiplicities.
                stdout_limit=16_384,
                stderr_limit=16_384,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=math.ceil(FACTOR_WALL_SECONDS) + 1,
                    address_space_bytes=1024 * 1024 * 1024,
                    file_size_bytes=1024 * 1024,
                ),
                cwd=directory,
            )
    except (OperationExecutionTimeoutError, OperationExecutionCancelledError):
        raise
    except OSError as error:
        raise RuntimeError(
            "finite-field factorization worker could not start"
        ) from error
    request_checkpoint("after finite-field factorization worker")
    if completed.cancelled:
        raise OperationExecutionCancelledError("finite-field factorization cancelled")
    if completed.timed_out:
        raise OperationExecutionTimeoutError("finite-field factorization expired")
    if (
        completed.returncode != 0
        or completed.stdout_exceeded
        or completed.stderr_exceeded
    ):
        raise RuntimeError(
            "finite-field factorization worker did not establish an outcome"
        )
    try:
        payload = json.loads(completed.stdout)
        if not isinstance(payload, list) or len(payload) != 2:
            raise ValueError("expected unit and factors")
        unit, entries = payload
        degree = len(coefficients) - 1
        if type(unit) is not int or not 0 < unit < prime:
            raise ValueError("invalid factorization unit")
        if not isinstance(entries, list) or not 1 <= len(entries) <= degree:
            raise ValueError("invalid factor count")
        factors = []
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 2:
                raise ValueError("invalid factor entry")
            factor, multiplicity = entry
            if (
                not isinstance(factor, list)
                or not 2 <= len(factor) <= degree + 1
                or any(type(c) is not int or not 0 <= c < prime for c in factor)
                or factor[-1] != 1
                or type(multiplicity) is not int
                or not 1 <= multiplicity <= degree
            ):
                raise ValueError("invalid factor representation")
            factors.append((tuple(factor), multiplicity))
        if (
            sum((len(factor) - 1) * multiplicity for factor, multiplicity in factors)
            != degree
        ):
            raise ValueError("factor degrees do not fit the source degree")
    except (TypeError, ValueError, UnicodeDecodeError) as error:
        request_checkpoint("during finite-field response decoding")
        raise RuntimeError(
            "finite-field factorization worker returned malformed output"
        ) from error
    return unit, tuple(factors)
