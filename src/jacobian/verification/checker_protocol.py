"""Closed subprocess protocol for operator-authorized checker execution."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Discriminator, Tag, TypeAdapter, ValidationError

from jacobian.contracts.checkers import CheckerDecision
from jacobian.contracts.common import Sha256Digest
from jacobian.contracts.results import ContractModel

type CheckerWorkerErrorCode = Literal[
    "EXECUTION_FAILED",
    "INVALID_REQUEST",
    "SOURCE_CHANGED",
    "UNDECLARED_IMPORT",
    "MALFORMED_RUNTIME",
    "RESPONSE_INVALID",
]


class CheckerWorkerSuccess(ContractModel):
    decision: CheckerDecision
    measured_implementation_digest: Sha256Digest
    measured_runtime_digest: Sha256Digest | None


class CheckerWorkerFailure(ContractModel):
    error_code: CheckerWorkerErrorCode


def _response_kind(value: Any) -> str | None:
    if isinstance(value, dict):
        if "decision" in value:
            return "success"
        if "error_code" in value:
            return "failure"
        return None
    if isinstance(value, CheckerWorkerSuccess):
        return "success"
    if isinstance(value, CheckerWorkerFailure):
        return "failure"
    return None


type CheckerWorkerResponse = Annotated[
    Annotated[CheckerWorkerSuccess, Tag("success")]
    | Annotated[CheckerWorkerFailure, Tag("failure")],
    Discriminator(_response_kind),
]

_RESPONSE_ADAPTER: TypeAdapter[CheckerWorkerResponse] = TypeAdapter(
    CheckerWorkerResponse
)


class CheckerWorkerProtocolError(ValueError):
    """The checker worker returned no recognized closed envelope."""


class CheckerWorkerDecisionError(CheckerWorkerProtocolError):
    """The checker worker returned a success envelope with an invalid decision."""


def parse_checker_worker_response(value: object) -> CheckerWorkerResponse:
    """Parse one checker response without admitting mixed or unknown states."""

    try:
        return _RESPONSE_ADAPTER.validate_python(value)
    except ValidationError as exc:
        if any("decision" in error["loc"] for error in exc.errors()):
            raise CheckerWorkerDecisionError("invalid checker decision") from exc
        raise CheckerWorkerProtocolError("invalid checker worker response") from exc
