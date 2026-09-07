"""Bounded deterministic dense finite-field factorization through SymPy."""

import time
from contextlib import nullcontext

from jacobian._execution import (
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

MAX_FACTOR_WORK = 1_000_000_000
FACTOR_WALL_SECONDS = 60.0


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
    deterministic path avoids that extra execution policy here. The four
    degree-27..108 motivating packets took 0.02..0.28 seconds locally; the
    shared 60-second deadline is a safety margin, not the work bound.
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
    from sympy.polys.domains import ZZ
    from sympy.polys.galoistools import gf_berlekamp, gf_sqf_list

    unit, squarefree = gf_sqf_list(list(reversed(coefficients)), prime, ZZ)
    factors: list[tuple[tuple[int, ...], int]] = []
    for polynomial, multiplicity in squarefree:
        request_checkpoint("during finite-field factorization")
        for factor in gf_berlekamp(polynomial, prime, ZZ):
            factors.append((tuple(int(c) for c in reversed(factor)), int(multiplicity)))
    return int(unit), tuple(sorted(factors, key=lambda item: (len(item[0]), item[0])))
