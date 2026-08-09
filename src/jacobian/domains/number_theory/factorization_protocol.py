"""Closed protocol for isolated factorization-derived operations."""

from __future__ import annotations

from typing import Annotated, Literal, overload

from pydantic import Field, TypeAdapter, ValidationError

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
from jacobian.contracts.results import ContractModel

PROTOCOL: Literal["jacobian.number-theory.factorization.sympy.v1"] = (
    "jacobian.number-theory.factorization.sympy.v1"
)


class DivisorsWorkerRequest(ContractModel):
    protocol: Literal["jacobian.number-theory.factorization.sympy.v1"]
    operation: Literal["divisors"]
    request: FactorizationRequest


class ProperDivisorsWorkerRequest(ContractModel):
    protocol: Literal["jacobian.number-theory.factorization.sympy.v1"]
    operation: Literal["proper_divisors"]
    request: FactorizationRequest


class PrimeFactorizationWorkerRequest(ContractModel):
    protocol: Literal["jacobian.number-theory.factorization.sympy.v1"]
    operation: Literal["prime_factorization"]
    request: FactorizationRequest


class PowerfulWorkerRequest(ContractModel):
    protocol: Literal["jacobian.number-theory.factorization.sympy.v1"]
    operation: Literal["powerful"]
    request: PowerfulNumberRequest


class SquarefreeWorkerRequest(ContractModel):
    protocol: Literal["jacobian.number-theory.factorization.sympy.v1"]
    operation: Literal["squarefree"]
    request: ArithmeticFunctionRequest


class RadicalWorkerRequest(ContractModel):
    protocol: Literal["jacobian.number-theory.factorization.sympy.v1"]
    operation: Literal["radical"]
    request: ArithmeticFunctionRequest


type FactorizationWorkerRequest = Annotated[
    DivisorsWorkerRequest
    | ProperDivisorsWorkerRequest
    | PrimeFactorizationWorkerRequest
    | PowerfulWorkerRequest
    | SquarefreeWorkerRequest
    | RadicalWorkerRequest,
    Field(discriminator="operation"),
]


class DivisorsWorkerResponse(ContractModel):
    protocol: Literal["jacobian.number-theory.factorization.sympy.v1"]
    operation: Literal["divisors"]
    result: DivisorListResult


class ProperDivisorsWorkerResponse(ContractModel):
    protocol: Literal["jacobian.number-theory.factorization.sympy.v1"]
    operation: Literal["proper_divisors"]
    result: DivisorListResult


class PrimeFactorizationWorkerResponse(ContractModel):
    protocol: Literal["jacobian.number-theory.factorization.sympy.v1"]
    operation: Literal["prime_factorization"]
    result: PrimeFactorizationResult


class PowerfulWorkerResponse(ContractModel):
    protocol: Literal["jacobian.number-theory.factorization.sympy.v1"]
    operation: Literal["powerful"]
    result: PowerfulNumberResult


class SquarefreeWorkerResponse(ContractModel):
    protocol: Literal["jacobian.number-theory.factorization.sympy.v1"]
    operation: Literal["squarefree"]
    result: BooleanResult


class RadicalWorkerResponse(ContractModel):
    protocol: Literal["jacobian.number-theory.factorization.sympy.v1"]
    operation: Literal["radical"]
    result: IntegerValueResult


type FactorizationWorkerResponse = Annotated[
    DivisorsWorkerResponse
    | ProperDivisorsWorkerResponse
    | PrimeFactorizationWorkerResponse
    | PowerfulWorkerResponse
    | SquarefreeWorkerResponse
    | RadicalWorkerResponse,
    Field(discriminator="operation"),
]

_REQUEST_ADAPTER: TypeAdapter[FactorizationWorkerRequest] = TypeAdapter(
    FactorizationWorkerRequest
)
_RESPONSE_ADAPTER: TypeAdapter[FactorizationWorkerResponse] = TypeAdapter(
    FactorizationWorkerResponse
)


def parse_factorization_worker_request(value: object) -> FactorizationWorkerRequest:
    return _REQUEST_ADAPTER.validate_python(value)


@overload
def parse_factorization_worker_response(
    value: object, *, expected_operation: Literal["divisors"]
) -> DivisorsWorkerResponse: ...


@overload
def parse_factorization_worker_response(
    value: object, *, expected_operation: Literal["proper_divisors"]
) -> ProperDivisorsWorkerResponse: ...


@overload
def parse_factorization_worker_response(
    value: object, *, expected_operation: Literal["prime_factorization"]
) -> PrimeFactorizationWorkerResponse: ...


@overload
def parse_factorization_worker_response(
    value: object, *, expected_operation: Literal["powerful"]
) -> PowerfulWorkerResponse: ...


@overload
def parse_factorization_worker_response(
    value: object, *, expected_operation: Literal["squarefree"]
) -> SquarefreeWorkerResponse: ...


@overload
def parse_factorization_worker_response(
    value: object, *, expected_operation: Literal["radical"]
) -> RadicalWorkerResponse: ...


@overload
def parse_factorization_worker_response(
    value: object,
    *,
    expected_operation: Literal[
        "divisors",
        "proper_divisors",
        "prime_factorization",
        "powerful",
        "squarefree",
        "radical",
    ],
) -> FactorizationWorkerResponse: ...


def parse_factorization_worker_response(
    value: object,
    *,
    expected_operation: Literal[
        "divisors",
        "proper_divisors",
        "prime_factorization",
        "powerful",
        "squarefree",
        "radical",
    ],
) -> FactorizationWorkerResponse:
    try:
        response = _RESPONSE_ADAPTER.validate_python(value)
    except ValidationError as exc:
        raise ValueError("invalid factorization worker response") from exc
    if response.operation != expected_operation:
        raise ValueError("factorization worker operation does not match the request")
    return response
