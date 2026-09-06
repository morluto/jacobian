"""Worker-safe kernels for bounded factorization-derived operations."""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Literal

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
    loads_strict_json,
    parse_canonical_integer,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._certification_models import (
    CertifiedFactor,
    CertifiedFactorizationRequest,
    CertifiedFactorizationResult,
    PrattCertificateFactor,
    PrattCertificateNode,
    PrimalityCertificateRequest,
    PrimalityCertificateResult,
)
from jacobian.math.number_theory._direct_factorization_models import (
    MAX_DIRECT_DIVISORS,
    MAX_DIRECT_FACTOR_ENTRIES,
    DivisorListResult,
    FactorizationRequest,
    PrimeFactorizationResult,
)
from jacobian.math.number_theory._integer_models import PrimePower

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

BoundedFactorizationFailureKind = Literal[
    "WORKER_START_FAILED",
    "WORKER_TIMEOUT",
    "WORKER_CANCELLED",
    "STDOUT_LIMIT_EXCEEDED",
    "STDERR_LIMIT_EXCEEDED",
    "WORKER_RESOURCE_LIMIT",
    "WORKER_EXITED",
    "MALFORMED_OUTPUT",
    "REQUEST_DEADLINE_EXPIRED",
]
BoundedFactorizationTimeoutLayer = Literal[
    "WORKER_START",
    "WORKER_WALL",
    "REQUEST_CANCELLATION",
    "OUTPUT_LIMIT",
    "PROCESS_RESOURCE",
    "WORKER_EXIT",
    "RESULT_VALIDATION",
    "REQUEST_DEADLINE",
]


def _bounded_prime_power(base: int, exponent: int, limit: int) -> int | None:
    """Compute ``base**exponent`` only while it can fit below ``limit``."""

    result = 1
    factor = base
    remaining_exponent = exponent
    while remaining_exponent:
        if remaining_exponent & 1:
            if result > limit // factor:
                return None
            result *= factor
        remaining_exponent //= 2
        if remaining_exponent:
            if factor > limit // factor:
                factor = limit + 1
            else:
                factor *= factor
    return result


@dataclass(frozen=True, slots=True)
class BoundedFactorizationFailure:
    """Bounded evidence for one unsuccessful isolated factorization attempt."""

    kind: BoundedFactorizationFailureKind
    timeout_layer: BoundedFactorizationTimeoutLayer
    elapsed_seconds: float
    timeout_seconds: float
    returncode: int | None = None


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

    factors = tuple(
        PrattCertificateFactor(
            prime=format_canonical_integer(int(prime_factor)),
            exponent=int(exponent),
            certificate=_build_pratt_certificate(int(prime_factor)),
        )
        for prime_factor, exponent in factors_of_pmo
    )
    return PrattCertificateNode(
        prime=format_canonical_integer(prime),
        witness=format_canonical_integer(witness),
        factors=factors,
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


def verify_pratt_certificate(claim: PrattCertificateNode) -> bool:
    """Return whether a Pratt node proves the primality it claims."""
    prime = parse_canonical_integer(claim.prime)
    if prime == _PRATT_BASE_PRIME:
        return claim.witness is None and not claim.factors
    if prime < 2 or claim.witness is None:
        return False
    witness = parse_canonical_integer(claim.witness)
    if not 2 <= witness < prime or pow(witness, prime - 1, prime) != 1:
        return False
    factor_primes = tuple(
        parse_canonical_integer(factor.prime) for factor in claim.factors
    )
    if factor_primes != tuple(sorted(factor_primes)) or len(set(factor_primes)) != len(
        factor_primes
    ):
        return False
    if (
        math.prod(
            factor_prime**factor.exponent
            for factor_prime, factor in zip(factor_primes, claim.factors, strict=True)
        )
        != prime - 1
    ):
        return False
    return all(
        parse_canonical_integer(factor.certificate.prime) == factor_prime
        and pow(witness, (prime - 1) // factor_prime, prime) != 1
        and verify_pratt_certificate(factor.certificate)
        for factor_prime, factor in zip(factor_primes, claim.factors, strict=True)
    )


def verify_certified_factorization(claim: CertifiedFactorizationResult) -> bool:
    """Return whether factor claims reconstruct their source with valid proofs."""
    value = parse_canonical_integer(claim.value)
    primes = tuple(parse_canonical_integer(factor.prime) for factor in claim.factors)
    if value < 2 or primes != tuple(sorted(primes)) or len(set(primes)) != len(primes):
        return False
    if any(
        parse_canonical_integer(factor.certificate.prime)
        != parse_canonical_integer(factor.prime)
        or not verify_pratt_certificate(factor.certificate)
        for factor in claim.factors
    ):
        return False
    reconstructed = math.prod(
        prime**factor.exponent
        for prime, factor in zip(primes, claim.factors, strict=True)
    )
    return bool(reconstructed == value)


def verify_primality_certificate(claim: PrimalityCertificateResult) -> bool:
    """Return whether a primality status is supported by its declared evidence."""
    from sympy import isprime

    value = parse_canonical_integer(claim.value)
    if value < 2:
        return False
    if claim.status == "COMPOSITE":
        return claim.certificate is None and not isprime(value)
    return (
        claim.certificate is not None
        and parse_canonical_integer(claim.certificate.prime) == value
        and verify_pratt_certificate(claim.certificate)
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
            input_bytes = encode_strict_json({"value": request.value})
            completed = run_bounded_process(
                [sys.executable, str(_CERTIFIED_FACTORIZATION_WORKER)],
                input_bytes=input_bytes,
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
    except OSError as exc:
        raise RuntimeError("bounded factorization worker could not be started") from exc
    if completed.cancelled:
        raise OperationExecutionCancelledError("factorization worker was cancelled")
    if completed.timed_out:
        raise OperationExecutionTimeoutError("factorization worker timed out")
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError("bounded factorization worker failed")
    try:
        response = loads_strict_json(
            completed.stdout,
            limits=CanonicalLimits(max_input_bytes=1024 * 1024),
        )
        if (
            not isinstance(response, dict)
            or set(response) != {"ok", "result", "request_digest"}
            or response.get("ok") is not True
        ):
            raise ValueError("worker failure")
        if response["request_digest"] != hashlib.sha256(input_bytes).hexdigest():
            raise ValueError("worker response is not bound to its request")
        return CertifiedFactorizationResult.model_validate(response["result"])
    except (KeyError, TypeError, ValueError, CanonicalizationError) as exc:
        raise RuntimeError(
            "bounded factorization worker returned malformed output"
        ) from exc


# ---------------------------------------------------------------------------
# Divisor and factorization-derived operations
# ---------------------------------------------------------------------------


def _admit_nonzero(request: FactorizationRequest) -> None:
    if int(request.value) == 0:
        raise OperationDomainValidationError(
            location=("value",),
            code="number_theory.zero_has_no_finite_factorization_or_divisor_enumeration",
            message="zero has no finite factorization or divisor enumeration",
        )


def _bounded_direct_factorization(  # noqa: C901
    value: int,
    *,
    timeout_seconds: float = _FACTORIZATION_WORKER_TIMEOUT_SECONDS,
    failure: list[BoundedFactorizationFailure] | None = None,
) -> tuple[PrimePower, ...] | None:
    """Factor one admitted nonzero integer in a killable worker.

    ``None`` is an operational non-conclusion, never a factor claim.  The
    caller owns projection into its operation-specific typed result.
    """

    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    started = monotonic()

    def failed(
        kind: BoundedFactorizationFailureKind,
        timeout_layer: BoundedFactorizationTimeoutLayer,
        returncode: int | None = None,
    ) -> None:
        if failure is not None:
            failure.append(
                BoundedFactorizationFailure(
                    kind=kind,
                    timeout_layer=timeout_layer,
                    elapsed_seconds=max(0.0, monotonic() - started),
                    timeout_seconds=timeout_seconds,
                    returncode=returncode,
                )
            )

    try:
        with TemporaryDirectory(prefix="jacobian-direct-factor-") as worker_directory:
            completed = run_bounded_process(
                [sys.executable, str(_DIRECT_FACTORIZATION_WORKER)],
                input_bytes=encode_strict_json(
                    {"value": format_canonical_integer(value)}
                ),
                timeout_seconds=timeout_seconds,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=64 * 1024,
                stderr_limit=64 * 1024,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=math.ceil(timeout_seconds),
                    address_space_bytes=_FACTORIZATION_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_FACTORIZATION_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=worker_directory,
            )
    except OSError as exc:
        failed("WORKER_START_FAILED", "WORKER_START")
        if failure is None:
            raise RuntimeError(
                "bounded factorization worker could not be started"
            ) from exc
        return None
    if completed.cancelled:
        failed("WORKER_CANCELLED", "REQUEST_CANCELLATION", completed.returncode)
        if failure is None:
            raise OperationExecutionCancelledError("factorization worker was cancelled")
        return None
    if completed.timed_out:
        failed("WORKER_TIMEOUT", "WORKER_WALL", completed.returncode)
        if failure is None:
            raise OperationExecutionTimeoutError("factorization worker timed out")
        return None
    if completed.stdout_exceeded:
        failed("STDOUT_LIMIT_EXCEEDED", "OUTPUT_LIMIT", completed.returncode)
        if failure is None:
            raise RuntimeError("factorization worker exceeded its output channel")
        return None
    if completed.stderr_exceeded:
        failed("STDERR_LIMIT_EXCEEDED", "OUTPUT_LIMIT", completed.returncode)
        if failure is None:
            raise RuntimeError("factorization worker exceeded its error channel")
        return None
    if completed.returncode != 0:
        if completed.returncode is not None and completed.returncode < 0:
            failed("WORKER_RESOURCE_LIMIT", "PROCESS_RESOURCE", completed.returncode)
        else:
            failed("WORKER_EXITED", "WORKER_EXIT", completed.returncode)
        if failure is None:
            raise RuntimeError("bounded factorization worker failed")
        return None
    try:
        response = loads_strict_json(
            completed.stdout,
            limits=CanonicalLimits(max_input_bytes=64 * 1024),
        )
        raw_factors = response["factors"]
        if not isinstance(raw_factors, list):
            raise ValueError("factors must be a list")
        factors = tuple(
            PrimePower.model_validate({"prime": pair[0], "power": pair[1]})
            for pair in raw_factors
        )
        if len(factors) > MAX_DIRECT_FACTOR_ENTRIES:
            raise ValueError("too many factors")
        if [factor.prime for factor in factors] != sorted(
            (factor.prime for factor in factors), key=int
        ) or len({factor.prime for factor in factors}) != len(factors):
            raise ValueError("noncanonical factors")
        from sympy import isprime

        quotient = abs(value)
        for factor in factors:
            base = int(factor.prime)
            if base > quotient:
                raise ValueError("factor base exceeds the remaining quotient")
            prime_power = _bounded_prime_power(base, factor.power, quotient)
            if prime_power is None or quotient % prime_power != 0 or not isprime(base):
                raise ValueError("factorization contains an invalid prime power")
            quotient //= prime_power
        if quotient != 1:
            raise ValueError("factorization does not reconstruct the input")
        return factors
    except (
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        CanonicalizationError,
    ) as exc:
        failed("MALFORMED_OUTPUT", "RESULT_VALIDATION", completed.returncode)
        if failure is None:
            raise RuntimeError(
                "bounded factorization worker returned malformed output"
            ) from exc
        return None


def _divisors_from_factors(
    factors: tuple[PrimePower, ...], *, proper: bool
) -> tuple[str, ...]:
    divisor_count = math.prod(factor.power + 1 for factor in factors)
    output_count = divisor_count - int(proper)
    if output_count > MAX_DIRECT_DIVISORS:
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


def _bounded_least_prime_factor(
    value: int,
    *,
    timeout_seconds: float = _FACTORIZATION_WORKER_TIMEOUT_SECONDS,
    failure: list[BoundedFactorizationFailure] | None = None,
) -> int | None:
    """Return the least prime factor from one validated worker factorization.

    The direct-factorization decoder validates every returned prime, exponent,
    ordering, and the complete product before this projection takes the first
    factor.  Reusing that decoder prevents a well-shaped but non-minimal first
    pair from becoming an LPF claim and keeps exponentiation inside its cheap
    bounded validation path.
    """
    factors = _bounded_direct_factorization(
        value,
        timeout_seconds=timeout_seconds,
        failure=failure,
    )
    if not factors:
        return None
    return int(factors[0].prime)


def enumerate_divisors(request: FactorizationRequest) -> DivisorListResult:
    _admit_nonzero(request)
    value = int(request.value)
    factors = _bounded_direct_factorization(value)
    if factors is None:
        raise RuntimeError(
            "bounded factorization worker did not establish every divisor"
        )
    divisors = _divisors_from_factors(factors, proper=False)
    return DivisorListResult._from_kernel(
        value=request.value,
        divisors=divisors,
        convention="ALL_POSITIVE_DIVISORS",
    )


def enumerate_proper_divisors(request: FactorizationRequest) -> DivisorListResult:
    _admit_nonzero(request)
    value = int(request.value)
    factors = _bounded_direct_factorization(value)
    if factors is None:
        raise RuntimeError(
            "bounded factorization worker did not establish every proper divisor"
        )
    divisors = _divisors_from_factors(factors, proper=True)
    return DivisorListResult._from_kernel(
        value=request.value,
        divisors=divisors,
        convention="PROPER_DIVISORS",
    )


def factorize_primes(request: FactorizationRequest) -> PrimeFactorizationResult:
    _admit_nonzero(request)
    value = int(request.value)
    factors = _bounded_direct_factorization(value)
    if factors is None:
        raise RuntimeError(
            "bounded factorization worker did not establish a complete result"
        )
    return PrimeFactorizationResult._from_kernel(value=request.value, factors=factors)
