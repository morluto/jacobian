"""Closed request and result envelopes for the bounded Gröbner worker."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Discriminator, StrictStr, Tag, TypeAdapter, ValidationError

from jacobian.contracts.polynomial_operations import (
    PolynomialGroebnerBasisRequest,
    PolynomialGroebnerBasisResult,
)
from jacobian.contracts.results import ContractModel

PROTOCOL = "jacobian.polynomial.groebner.sympy.v1"


class GroebnerWorkerRequest(ContractModel):
    protocol: Literal["jacobian.polynomial.groebner.sympy.v1"]
    request: PolynomialGroebnerBasisRequest


class GroebnerWorkerSuccess(ContractModel):
    protocol: Literal["jacobian.polynomial.groebner.sympy.v1"]
    result: PolynomialGroebnerBasisResult


class GroebnerWorkerResultLimitError(ContractModel):
    code: Literal["POLYNOMIAL_GROEBNER_RESULT_LIMIT_EXCEEDED"]
    message: StrictStr


class GroebnerWorkerResultLimitExceeded(ContractModel):
    protocol: Literal["jacobian.polynomial.groebner.sympy.v1"]
    error: GroebnerWorkerResultLimitError


def _response_kind(value: Any) -> str | None:
    if isinstance(value, dict):
        if "result" in value:
            return "result"
        if "error" in value:
            return "result_limit_exceeded"
        return None
    if isinstance(value, GroebnerWorkerSuccess):
        return "result"
    if isinstance(value, GroebnerWorkerResultLimitExceeded):
        return "result_limit_exceeded"
    return None


type GroebnerWorkerResponse = Annotated[
    Annotated[GroebnerWorkerSuccess, Tag("result")]
    | Annotated[GroebnerWorkerResultLimitExceeded, Tag("result_limit_exceeded")],
    Discriminator(_response_kind),
]

_RESPONSE_ADAPTER: TypeAdapter[GroebnerWorkerResponse] = TypeAdapter(
    GroebnerWorkerResponse
)


def parse_groebner_worker_response(value: object) -> GroebnerWorkerResponse:
    """Parse one response whose success and limit states cannot overlap."""

    try:
        return _RESPONSE_ADAPTER.validate_python(value)
    except ValidationError as exc:
        raise ValueError("invalid Gröbner worker response") from exc
