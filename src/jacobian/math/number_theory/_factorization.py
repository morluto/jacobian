"""Budgeted declarations for complete factorization-derived operations."""

from __future__ import annotations

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory._factorization_kernels import (
    compute_radical,
    decide_powerful,
    decide_squarefree,
    enumerate_divisors,
    enumerate_proper_divisors,
    factorize_primes,
)
from jacobian.math.number_theory._models import (
    ArithmeticFunctionRequest,
    BooleanResult,
    DivisorListResult,
    IntegerValueResult,
    NonzeroFactorizationRequest,
    PowerfulNumberRequest,
    PowerfulNumberResult,
    PrimeFactorizationResult,
)


def _compute_divisors(
    request: NonzeroFactorizationRequest,
) -> DivisorListResult:
    return enumerate_divisors(request)


def _compute_proper_divisors(
    request: NonzeroFactorizationRequest,
) -> DivisorListResult:
    return enumerate_proper_divisors(request)


def _compute_prime_factorization(
    request: NonzeroFactorizationRequest,
) -> PrimeFactorizationResult:
    return factorize_primes(request)


def _compute_powerful(
    request: PowerfulNumberRequest,
) -> PowerfulNumberResult:
    return decide_powerful(request)


def _compute_squarefree(
    request: ArithmeticFunctionRequest,
) -> BooleanResult:
    return decide_squarefree(request)


def _compute_radical(
    request: ArithmeticFunctionRequest,
) -> IntegerValueResult:
    return compute_radical(request)


def _operation[RequestT: StrictModel, ResultT: StrictModel](
    *,
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    implementation: Callable[[RequestT], ResultT],
    tags: tuple[str, ...],
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version="2",
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=implementation,
        tags=tags,
        examples=examples,
    )


FACTORIZATION_OPERATIONS = (
    _operation(
        operation_id="integer.compute.divisors",
        title="Enumerate positive divisors",
        description="Enumerate every positive divisor exactly.",
        request_model=NonzeroFactorizationRequest,
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
        description="Enumerate every positive proper divisor exactly.",
        request_model=NonzeroFactorizationRequest,
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
        description="Compute a complete prime-power factorization.",
        request_model=NonzeroFactorizationRequest,
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
            "violating prime."
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
        description="Decide whether a bounded nonnegative integer is square-free.",
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
        description="Compute the product of distinct prime divisors exactly.",
        request_model=ArithmeticFunctionRequest,
        result_model=IntegerValueResult,
        implementation=_compute_radical,
        tags=("number-theory", "arithmetic-function"),
        examples=(example("radical_360", "Compute the radical of 360.", {"n": 360}),),
    ),
)
