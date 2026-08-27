"""Worker-safe kernels for bounded factorization-derived operations."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.number_theory._certification_models import (
    CertifiedFactor,
    CertifiedFactorizationRequest,
    CertifiedFactorizationResult,
    PrattCertificateNode,
    PrimalityCertificateRequest,
    PrimalityCertificateResult,
)
from jacobian.math.number_theory._direct_factorization_models import (
    DivisorListResult,
    FactorizationRequest,
    PrimeFactorizationResult,
    RadicalResult,
    SquarefreeResult,
)
from jacobian.math.number_theory._integer_models import (
    ArithmeticFunctionRequest,
    PrimePower,
)

# ---------------------------------------------------------------------------
# Pratt certificate construction and verification
# ---------------------------------------------------------------------------

_PRATT_BASE_PRIME = 2
_CERTIFIED_FACTORIZATION_WORKER = (
    Path(__file__).resolve().with_name("_certified_factorization_worker.py")
)
_DIRECT_FACTORIZATION_WORKER = (
    Path(__file__).resolve().with_name("_direct_factorization_worker.py")
)
_FACTORIZATION_WORKER_TIMEOUT_SECONDS = 60.0
_FACTORIZATION_WORKER_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_FACTORIZATION_WORKER_FILE_SIZE_BYTES = 1024 * 1024


def _build_pratt_certificate(prime: int) -> PrattCertificateNode:
    """Construct a Pratt certificate for one known prime.

    The base case is ``prime == 2``.  For ``prime > 2`` we search for a witness
    ``a`` such that ``a^(prime-1) ≡ 1 (mod prime)`` and ``a^((prime-1)/q) ≢ 1
    (mod prime)`` for every prime factor ``q`` of ``prime - 1``.  Each such
    ``q`` is then recursively certified.
    """
    if prime == _PRATT_BASE_PRIME:
        return PrattCertificateNode(prime="2")

    from sympy import factorint, primitive_root

    prime_minus_one = prime - 1
    factors_of_pmo = sorted(factorint(prime_minus_one).items())

    # A primitive root modulo ``prime`` is guaranteed to satisfy the Pratt
    # witness condition: its multiplicative order is exactly ``prime - 1``,
    # so ``a^((prime-1)/q) ≢ 1 (mod prime)`` for every prime ``q | prime-1``.
    witness = int(primitive_root(prime))

    sub_certificates = tuple(
        _build_pratt_certificate(int(q)) for q, _ in factors_of_pmo
    )
    return PrattCertificateNode(
        prime=format_canonical_integer(prime),
        witness=format_canonical_integer(witness),
        sub_certificates=sub_certificates,
    )


def compute_pratt_certificate(
    request: PrimalityCertificateRequest,
) -> PrimalityCertificateResult:
    """Produce a Pratt primality certificate for one declared candidate.

    Returns ``COMPOSITE`` (no certificate) when the candidate is not prime.
    """
    from sympy import isprime

    value = parse_canonical_integer(request.value)
    if not isprime(value):
        return PrimalityCertificateResult._from_kernel(
            status="COMPOSITE", value=request.value
        )
    return PrimalityCertificateResult._from_kernel(
        status="CERTIFIED",
        value=request.value,
        certificate=_build_pratt_certificate(value),
    )


# ---------------------------------------------------------------------------
# Subexponential certified factorization
# ---------------------------------------------------------------------------


def _factorize_certified_in_process(
    request: CertifiedFactorizationRequest,
) -> CertifiedFactorizationResult:
    """Factor one bounded integer using subexponential methods.

    Backed by ``sympy.ntheory.factorint`` without a trial-division limit, so
    Pollard rho, Pollard p-1, and ECM are all available.  Each prime factor
    carries an independent Pratt primality certificate.
    """
    from sympy import factorint

    value = parse_canonical_integer(request.value)
    decomposition = sorted(factorint(value).items())
    factors = tuple(
        CertifiedFactor(
            prime=format_canonical_integer(int(prime)),
            exponent=int(exponent),
            certificate=_build_pratt_certificate(int(prime)),
        )
        for prime, exponent in decomposition
    )
    return CertifiedFactorizationResult._from_kernel(
        value=request.value,
        factors=factors,
    )


def factorize_certified(
    request: CertifiedFactorizationRequest,
) -> CertifiedFactorizationResult:
    """Factor through a killable worker; a stop establishes no factor claim."""

    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    try:
        with TemporaryDirectory(
            prefix="jacobian-certified-factor-"
        ) as worker_directory:
            completed = run_bounded_process(
                [sys.executable, str(_CERTIFIED_FACTORIZATION_WORKER)],
                input_bytes=json.dumps(
                    {"value": request.value}, separators=(",", ":")
                ).encode(),
                timeout_seconds=_FACTORIZATION_WORKER_TIMEOUT_SECONDS,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=1024 * 1024,
                stderr_limit=64 * 1024,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=math.ceil(_FACTORIZATION_WORKER_TIMEOUT_SECONDS),
                    address_space_bytes=_FACTORIZATION_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_FACTORIZATION_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=worker_directory,
            )
    except OSError:
        return CertifiedFactorizationResult._unknown(
            value=request.value,
            detail="the bounded factorization worker could not be started",
        )
    if (
        completed.timed_out
        or completed.cancelled
        or completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        return CertifiedFactorizationResult._unknown(
            value=request.value,
            detail="the bounded factorization worker did not establish a complete result",
        )
    try:
        response = json.loads(completed.stdout.decode("utf-8"))
        if response.get("ok") is not True:
            raise ValueError("worker failure")
        return CertifiedFactorizationResult.model_validate(response["result"])
    except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return CertifiedFactorizationResult._unknown(
            value=request.value,
            detail="the bounded factorization worker returned malformed output",
        )


# ---------------------------------------------------------------------------
# Divisor and factorization-derived operations
# ---------------------------------------------------------------------------


def _bounded_direct_factorization(value: int) -> tuple[PrimePower, ...] | None:
    """Factor one admitted nonzero integer in a killable worker.

    ``None`` is an operational non-conclusion, never a factor claim.  The
    caller owns projection into its operation-specific typed result.
    """

    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    try:
        with TemporaryDirectory(prefix="jacobian-direct-factor-") as worker_directory:
            completed = run_bounded_process(
                [sys.executable, str(_DIRECT_FACTORIZATION_WORKER)],
                input_bytes=json.dumps(
                    {"value": str(value)}, separators=(",", ":")
                ).encode(),
                timeout_seconds=_FACTORIZATION_WORKER_TIMEOUT_SECONDS,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=64 * 1024,
                stderr_limit=64 * 1024,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=math.ceil(_FACTORIZATION_WORKER_TIMEOUT_SECONDS),
                    address_space_bytes=_FACTORIZATION_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_FACTORIZATION_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=worker_directory,
            )
    except OSError:
        return None
    if (
        completed.timed_out
        or completed.cancelled
        or completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        return None
    try:
        response = json.loads(completed.stdout.decode("utf-8"))
        raw_factors = response["factors"]
        if not isinstance(raw_factors, list):
            raise ValueError("factors must be a list")
        factors = tuple(
            PrimePower.model_validate({"prime": pair[0], "power": pair[1]})
            for pair in raw_factors
        )
        if len(factors) > 256:
            raise ValueError("too many factors")
        if [factor.prime for factor in factors] != sorted(
            (factor.prime for factor in factors), key=int
        ) or len({factor.prime for factor in factors}) != len(factors):
            raise ValueError("noncanonical factors")
        from sympy import isprime

        if not all(isprime(int(factor.prime)) for factor in factors) or math.prod(
            int(factor.prime) ** factor.power for factor in factors
        ) != abs(value):
            raise ValueError("factorization does not reconstruct the input")
        return factors
    except (
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return None


def _divisors_from_factors(
    factors: tuple[PrimePower, ...], *, proper: bool, value: int
) -> tuple[str, ...]:
    divisor_count = math.prod(factor.power + 1 for factor in factors)
    if divisor_count > 4_096:
        raise ValueError("divisor output exceeds admitted bound")
    divisors = [1]
    for factor in factors:
        prime = int(factor.prime)
        power_values = [1]
        for _ in range(factor.power):
            power_values.append(power_values[-1] * prime)
        divisors = [base * power for base in divisors for power in power_values]
    ordered = tuple(str(divisor) for divisor in sorted(divisors))
    return ordered[:-1] if proper else ordered


def verify_divisor_list_result(result: DivisorListResult) -> bool:
    """Replay one bounded divisor claim through the owner kernel."""

    if result.status != "COMPLETE":
        return False
    factors = _bounded_direct_factorization(int(result.value))
    if factors is None:
        return False
    try:
        divisors = _divisors_from_factors(
            factors,
            proper=result.convention == "PROPER_DIVISORS",
            value=int(result.value),
        )
    except ValueError:
        return False
    return result.divisors == divisors


def verify_prime_factorization_result(result: PrimeFactorizationResult) -> bool:
    """Replay one complete prime-factorization claim through the owner kernel."""

    if result.status != "COMPLETE":
        return False
    factors = _bounded_direct_factorization(int(result.value))
    return factors is not None and result.factors == factors


def enumerate_divisors(request: FactorizationRequest) -> DivisorListResult:
    value = int(request.value)
    if value == 0:
        raise ValueError("zero has infinitely many divisors")
    factors = _bounded_direct_factorization(value)
    if factors is None:
        return DivisorListResult._unknown(
            value=request.value,
            convention="ALL_POSITIVE_DIVISORS",
            detail="the bounded factorization worker did not establish every divisor",
        )
    try:
        divisors = _divisors_from_factors(factors, proper=False, value=value)
    except ValueError:
        return DivisorListResult._unknown(
            value=request.value,
            convention="ALL_POSITIVE_DIVISORS",
            detail="the complete divisor family exceeds the admitted output bound",
        )
    return DivisorListResult(
        value=request.value, divisors=divisors, convention="ALL_POSITIVE_DIVISORS"
    )


def enumerate_proper_divisors(request: FactorizationRequest) -> DivisorListResult:
    value = int(request.value)
    if value == 0:
        raise ValueError("zero has infinitely many divisors")
    factors = _bounded_direct_factorization(value)
    if factors is None:
        return DivisorListResult._unknown(
            value=request.value,
            convention="PROPER_DIVISORS",
            detail="the bounded factorization worker did not establish every proper divisor",
        )
    try:
        divisors = _divisors_from_factors(factors, proper=True, value=value)
    except ValueError:
        return DivisorListResult._unknown(
            value=request.value,
            convention="PROPER_DIVISORS",
            detail="the complete divisor family exceeds the admitted output bound",
        )
    return DivisorListResult(
        value=request.value, divisors=divisors, convention="PROPER_DIVISORS"
    )


def factorize_primes(request: FactorizationRequest) -> PrimeFactorizationResult:
    value = int(request.value)
    if value == 0:
        raise ValueError("zero has no finite prime factorization")
    factors = _bounded_direct_factorization(value)
    if factors is None:
        return PrimeFactorizationResult._unknown(
            value=request.value,
            detail="the bounded factorization worker did not establish a complete result",
        )
    return PrimeFactorizationResult(value=request.value, factors=factors)


def decide_squarefree(request: ArithmeticFunctionRequest) -> SquarefreeResult:
    if request.n == 0:
        return SquarefreeResult(status="NOT_SQUAREFREE", n=request.n)
    factors = _bounded_direct_factorization(request.n)
    if factors is None:
        return SquarefreeResult._unknown(
            n=request.n,
            detail="the bounded factorization worker did not establish squarefreeness",
        )
    return SquarefreeResult(
        status="SQUAREFREE"
        if all(factor.power == 1 for factor in factors)
        else "NOT_SQUAREFREE",
        n=request.n,
    )


def compute_radical(request: ArithmeticFunctionRequest) -> RadicalResult:
    if request.n == 0:
        return RadicalResult(n=0, value="0")
    factors = _bounded_direct_factorization(request.n)
    if factors is None:
        return RadicalResult._unknown(
            n=request.n,
            detail="the bounded factorization worker did not establish the radical",
        )
    radical = 1
    for factor in factors:
        radical *= int(factor.prime)
    return RadicalResult(n=request.n, value=str(radical))
