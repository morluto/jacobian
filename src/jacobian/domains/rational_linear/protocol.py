"""Closed, request-refined protocol for rational-linear FLINT workers."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, ValidationError

from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.linear import (
    MAX_LINEAR_DIMENSION,
    LinearRationalInconsistencyFindRequest,
    LinearRationalSolutionFindRequest,
)
from jacobian.contracts.results import ContractModel

SOLUTION_PROTOCOL = "jacobian.rational-linear-solution-worker/v1"
INCONSISTENCY_PROTOCOL = "jacobian.rational-linear-inconsistency-worker/v1"


class RationalLinearSolutionWorkerRequest(ContractModel):
    protocol: Literal["jacobian.rational-linear-solution-worker/v1"]
    request: LinearRationalSolutionFindRequest


class RationalLinearInconsistencyWorkerRequest(ContractModel):
    protocol: Literal["jacobian.rational-linear-inconsistency-worker/v1"]
    request: LinearRationalInconsistencyFindRequest


type RationalLinearWorkerRequest = Annotated[
    RationalLinearSolutionWorkerRequest | RationalLinearInconsistencyWorkerRequest,
    Field(discriminator="protocol"),
]


class RationalLinearSolutionProduced(ContractModel):
    protocol: Literal["jacobian.rational-linear-solution-worker/v1"]
    status: Literal["SOLUTION_PRODUCED"]
    values: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_DIMENSION,
    )


class RationalLinearNoSolutionProduced(ContractModel):
    protocol: Literal["jacobian.rational-linear-solution-worker/v1"]
    status: Literal["NO_SOLUTION_PRODUCED"]


type RationalLinearSolutionWorkerResponse = Annotated[
    RationalLinearSolutionProduced | RationalLinearNoSolutionProduced,
    Field(discriminator="status"),
]


class RationalLinearCertificateProduced(ContractModel):
    protocol: Literal["jacobian.rational-linear-inconsistency-worker/v1"]
    status: Literal["CERTIFICATE_PRODUCED"]
    left_witness: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_DIMENSION,
    )
    rhs_pairing: CanonicalRational


class RationalLinearNoCertificateProduced(ContractModel):
    protocol: Literal["jacobian.rational-linear-inconsistency-worker/v1"]
    status: Literal["NO_CERTIFICATE_PRODUCED"]


type RationalLinearInconsistencyWorkerResponse = Annotated[
    RationalLinearCertificateProduced | RationalLinearNoCertificateProduced,
    Field(discriminator="status"),
]


class RationalLinearWorkerFailure(ContractModel):
    status: Literal["ERROR"]
    error_code: Literal["INVALID_REQUEST", "EXECUTION_FAILED"]


_REQUEST_ADAPTER: TypeAdapter[RationalLinearWorkerRequest] = TypeAdapter(
    RationalLinearWorkerRequest
)
_SOLUTION_RESPONSE_ADAPTER: TypeAdapter[RationalLinearSolutionWorkerResponse] = (
    TypeAdapter(RationalLinearSolutionWorkerResponse)
)
_INCONSISTENCY_RESPONSE_ADAPTER: TypeAdapter[
    RationalLinearInconsistencyWorkerResponse
] = TypeAdapter(RationalLinearInconsistencyWorkerResponse)


def parse_rational_linear_worker_request(value: object) -> RationalLinearWorkerRequest:
    return _REQUEST_ADAPTER.validate_python(value)


def parse_solution_worker_response(
    value: object,
    *,
    expected_value_count: int,
) -> RationalLinearSolutionWorkerResponse:
    try:
        response = _SOLUTION_RESPONSE_ADAPTER.validate_python(value)
    except ValidationError as exc:
        raise ValueError("invalid rational-linear solution response") from exc
    if (
        isinstance(response, RationalLinearSolutionProduced)
        and len(response.values) != expected_value_count
    ):
        raise ValueError("solution dimensions do not match the source system")
    return response


def parse_inconsistency_worker_response(
    value: object,
    *,
    expected_witness_count: int,
) -> RationalLinearInconsistencyWorkerResponse:
    try:
        response = _INCONSISTENCY_RESPONSE_ADAPTER.validate_python(value)
    except ValidationError as exc:
        raise ValueError("invalid rational-linear inconsistency response") from exc
    if (
        isinstance(response, RationalLinearCertificateProduced)
        and len(response.left_witness) != expected_witness_count
    ):
        raise ValueError(
            "inconsistency witness dimensions do not match the source system"
        )
    return response
