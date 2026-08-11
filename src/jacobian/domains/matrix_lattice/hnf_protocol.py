"""Closed protocol for isolated Python-FLINT row-HNF execution."""

from __future__ import annotations

from typing import Literal

from pydantic import ValidationError

from jacobian.contracts.matrix_lattice import (
    HermiteNormalFormRequest,
    HermiteNormalFormResult,
)
from jacobian.contracts.results import ContractModel

PROTOCOL: Literal["jacobian.matrix-lattice-hnf-worker/v1"] = (
    "jacobian.matrix-lattice-hnf-worker/v1"
)


class HermiteNormalFormWorkerRequest(ContractModel):
    protocol: Literal["jacobian.matrix-lattice-hnf-worker/v1"]
    request: HermiteNormalFormRequest


class HermiteNormalFormWorkerResponse(ContractModel):
    protocol: Literal["jacobian.matrix-lattice-hnf-worker/v1"]
    status: Literal["NORMAL_FORM_PRODUCED"]
    result: HermiteNormalFormResult


class HermiteNormalFormWorkerFailure(ContractModel):
    protocol: Literal["jacobian.matrix-lattice-hnf-worker/v1"]
    status: Literal["ERROR"]
    error_code: Literal["INVALID_REQUEST", "EXECUTION_FAILED"]


def parse_hnf_worker_request(value: object) -> HermiteNormalFormWorkerRequest:
    return HermiteNormalFormWorkerRequest.model_validate(value)


def parse_hnf_worker_response(
    value: object,
    *,
    request: HermiteNormalFormRequest,
) -> HermiteNormalFormWorkerResponse:
    try:
        response = HermiteNormalFormWorkerResponse.model_validate(value)
    except ValidationError as exc:
        raise ValueError("invalid HNF worker response") from exc
    source_rows = len(request.matrix.entries)
    source_columns = len(request.matrix.entries[0])
    result = response.result
    if len(result.normal_form.entries) != source_rows or any(
        len(row) != source_columns for row in result.normal_form.entries
    ):
        raise ValueError("HNF normal form dimensions do not match the source")
    if len(result.transformation.entries) != source_rows or any(
        len(row) != source_rows for row in result.transformation.entries
    ):
        raise ValueError("HNF transformation dimensions do not match the source")
    return response
