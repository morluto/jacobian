"""Closed protocol for isolated exact-gram LLL reduction."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import ValidationError

from jacobian.contracts.matrix_operations import (
    LatticeReductionRequest,
    LatticeReductionResult,
)
from jacobian.contracts.results import ContractModel

PROTOCOL: Literal["jacobian.flint-lll-worker/v1"] = "jacobian.flint-lll-worker/v1"


class LllWorkerErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    INPUT_LIMIT_EXCEEDED = "FLINT_LLL_INPUT_LIMIT_EXCEEDED"
    MATRIX_INVALID = "FLINT_LLL_MATRIX_INVALID"
    VERSION_MISMATCH = "FLINT_LLL_VERSION_MISMATCH"
    RELATION_INVALID = "FLINT_LLL_RELATION_INVALID"
    EXECUTION_FAILED = "FLINT_LLL_EXECUTION_FAILED"
    OUTPUT_LIMIT_EXCEEDED = "FLINT_LLL_OUTPUT_LIMIT_EXCEEDED"


class LllWorkerRequest(ContractModel):
    protocol: Literal["jacobian.flint-lll-worker/v1"]
    request: LatticeReductionRequest


class LllWorkerResponse(ContractModel):
    protocol: Literal["jacobian.flint-lll-worker/v1"]
    result: LatticeReductionResult


class LllWorkerFailure(ContractModel):
    protocol: Literal["jacobian.flint-lll-worker/v1"]
    error_code: LllWorkerErrorCode


def parse_lll_worker_request(value: object) -> LllWorkerRequest:
    return LllWorkerRequest.model_validate(value)


def parse_lll_worker_response(
    value: object,
    *,
    request: LatticeReductionRequest,
) -> LllWorkerResponse:
    try:
        response = LllWorkerResponse.model_validate(value)
    except ValidationError as exc:
        raise ValueError("invalid LLL worker response") from exc
    source_rows = len(request.basis.entries)
    source_columns = len(request.basis.entries[0])
    result = response.result
    if len(result.reduced_basis.entries) != source_rows or any(
        len(row) != source_columns for row in result.reduced_basis.entries
    ):
        raise ValueError("LLL reduced basis dimensions do not match the source")
    if len(result.transformation.entries) != source_rows or any(
        len(row) != source_rows for row in result.transformation.entries
    ):
        raise ValueError("LLL transformation dimensions do not match the source")
    if result.rank > min(source_rows, source_columns):
        raise ValueError("LLL rank exceeds the source matrix dimensions")
    return response
