"""Closed protocol for isolated exact rational optimization."""

from __future__ import annotations

from typing import Literal

from pydantic import ValidationError

from jacobian.contracts.results import ContractModel
from jacobian.contracts.validated_analysis import (
    RationalLinearProgramRequest,
    RationalLinearProgramResult,
)

PROTOCOL: Literal["jacobian.optimization.rational-linear/v1"] = (
    "jacobian.optimization.rational-linear/v1"
)


class RationalOptimizationWorkerRequest(ContractModel):
    protocol: Literal["jacobian.optimization.rational-linear/v1"]
    request: RationalLinearProgramRequest


class RationalOptimizationWorkerResponse(ContractModel):
    protocol: Literal["jacobian.optimization.rational-linear/v1"]
    result: RationalLinearProgramResult


def parse_optimization_worker_request(
    value: object,
) -> RationalOptimizationWorkerRequest:
    return RationalOptimizationWorkerRequest.model_validate(value)


def parse_optimization_worker_response(
    value: object,
) -> RationalOptimizationWorkerResponse:
    try:
        return RationalOptimizationWorkerResponse.model_validate(value)
    except ValidationError as exc:
        raise ValueError("invalid rational optimization worker response") from exc
