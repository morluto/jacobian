"""Budgeted declarations for complete factorization-derived operations."""

from __future__ import annotations

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory._certification_models import (
    CertifiedFactorizationRequest,
    CertifiedFactorizationResult,
    PrimalityCertificateRequest,
    PrimalityCertificateResult,
)
from jacobian.math.number_theory._direct_factorization_models import (
    DivisorListResult,
    FactorizationRequest,
    PrimeFactorizationResult,
)
from jacobian.math.number_theory._factorization_kernels import (
    compute_pratt_certificate,
    enumerate_divisors,
    factorize_certified,
    factorize_primes,
)

FACTORIZATION_OPERATIONS = (
    MathTool(
        operation_id="integer.factor.certified_compute",
        title="Compute a certified integer factorization",
        description="Factor one bounded 30-digit integer (~100 bits) in an isolated subexponential worker (Pollard rho, Pollard p-1, ECM via sympy.ntheory.factorint), returning a complete prime-power factorization with per-factor Pratt certificates. Worker timeout, cancellation, or failure raises an execution error without a partial factorization.",
        request_type=CertifiedFactorizationRequest,
        result_type=CertifiedFactorizationResult,
        run=factorize_certified,
        tags=("number-theory", "factorization", "bounded", "prime", "certificate"),
        examples=(
            OperationExample(
                name="semiprime_10403",
                description="Factor 10403 with subexponential methods and per-factor Pratt certificates. The input must be a canonical integer of at least 2 and at most 30 digits.",
                input={"value": "10403"},
            ),
        ),
    ),
    MathTool(
        operation_id="integer.primality.certificate.compute",
        title="Compute a Pratt primality certificate",
        description="Produce a Pratt primality certificate for one declared prime, or report COMPOSITE when the candidate is not prime.",
        request_type=PrimalityCertificateRequest,
        result_type=PrimalityCertificateResult,
        run=compute_pratt_certificate,
        tags=("number-theory", "primality", "certificate"),
        examples=(
            OperationExample(
                name="pratt_101",
                description="Produce a Pratt certificate for the prime 101. The input must be a canonical integer of at least 2 and at most 30 digits.",
                input={"value": "101"},
            ),
        ),
    ),
    MathTool(
        operation_id="integer.compute.divisors",
        title="Enumerate positive divisors",
        description="Enumerate and return every positive divisor exactly, including the input's absolute value. Worker timeout, cancellation, or failure raises an execution error without a partial enumeration.",
        request_type=FactorizationRequest,
        result_type=DivisorListResult,
        run=enumerate_divisors,
        tags=("number-theory", "enumeration"),
        discovery_terms=("positive divisors",),
        examples=(
            OperationExample(
                name="divisors_12",
                description="Enumerate the positive divisors of 12.",
                input={"value": "12"},
            ),
        ),
    ),
    MathTool(
        operation_id="integer.compute.prime_factorization",
        title="Factor an integer",
        description="Factor an integer into prime powers and return the complete prime-power factorization. Worker timeout, cancellation, or failure raises an execution error without a partial factorization.",
        request_type=FactorizationRequest,
        result_type=PrimeFactorizationResult,
        run=factorize_primes,
        tags=("number-theory", "factorization"),
        examples=(
            OperationExample(
                name="prime_factorization_360",
                description="Factor 360 into prime powers.",
                input={"value": "360"},
            ),
        ),
    ),
)
