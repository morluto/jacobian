"""Certified factoring operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.contracts.base import ContractModel
from jacobian.contracts.certified_factoring import (
    CertifiedFactorRequest,
    CertifiedFactorResult,
    PrimalityCertificateRequest,
    PrimalityCertificateResult,
)
from jacobian.contracts.operations import OperationExample
from jacobian.domains._examples import example
from jacobian.domains.certified_factoring.operations import (
    compute_certified_factor,
    compute_primality_certificate,
)
from jacobian.math_tools import MathTool


def cf_operation[RequestT: ContractModel, ResultT: ContractModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


CERTIFIED_FACTORING_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    cf_operation(
        "integer.factor.certified_compute",
        "Certified integer factorization",
        "Factor a positive integer using SymPy's factorint (Pollard rho, p-1, ECM) and return prime factors with multiplicities and Pratt primality certificates.",
        CertifiedFactorRequest,
        CertifiedFactorResult,
        compute_certified_factor,
        "number-theory",
        "factoring",
        "certified",
        "exact",
        examples=(
            example(
                "factor_60",
                "Factor 60 into primes with Pratt certificates.",
                {"n": "60"},
            ),
        ),
    ),
    cf_operation(
        "integer.primality.certificate.compute",
        "Compute a Pratt primality certificate",
        "Produce a recursive Pratt certificate proving that a declared integer is prime.",
        PrimalityCertificateRequest,
        PrimalityCertificateResult,
        compute_primality_certificate,
        "number-theory",
        "primality",
        "certificate",
        "exact",
        examples=(
            example(
                "prime_17",
                "Produce a Pratt certificate for the prime 17.",
                {"p": "17"},
            ),
        ),
    ),
)
