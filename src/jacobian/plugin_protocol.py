"""Closed subprocess protocol for operator-installed plugin execution."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Discriminator, StrictStr, Tag, TypeAdapter, ValidationError

from jacobian.contracts.common import Sha256Digest
from jacobian.contracts.results import ContractModel


class PluginWorkerFailureCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    SOURCE_CHANGED = "SOURCE_CHANGED"
    RESPONSE_INVALID = "RESPONSE_INVALID"
    EXECUTION_FAILED = "EXECUTION_FAILED"


class PluginWorkerSuccess(ContractModel):
    response: dict[str, Any]
    measured_implementation_digest: Sha256Digest


class PluginWorkerOperationalFailure(ContractModel):
    error_code: Literal["SOURCE_CHANGED", "EXECUTION_FAILED"]


class PluginWorkerContractFailure(ContractModel):
    error_code: Literal["INVALID_REQUEST", "RESPONSE_INVALID"]
    path: StrictStr
    expected: StrictStr
    actual_type: StrictStr


type PluginWorkerFailure = PluginWorkerOperationalFailure | PluginWorkerContractFailure


def _response_kind(value: Any) -> str | None:
    if isinstance(value, dict):
        if "response" in value:
            return "success"
        if "error_code" in value and "path" in value:
            return "contract_failure"
        if "error_code" in value:
            return "operational_failure"
        return None
    if isinstance(value, PluginWorkerSuccess):
        return "success"
    if isinstance(value, PluginWorkerContractFailure):
        return "contract_failure"
    if isinstance(value, PluginWorkerOperationalFailure):
        return "operational_failure"
    return None


type PluginWorkerResponse = Annotated[
    Annotated[PluginWorkerSuccess, Tag("success")]
    | Annotated[PluginWorkerContractFailure, Tag("contract_failure")]
    | Annotated[PluginWorkerOperationalFailure, Tag("operational_failure")],
    Discriminator(_response_kind),
]

_RESPONSE_ADAPTER: TypeAdapter[PluginWorkerResponse] = TypeAdapter(PluginWorkerResponse)


class PluginWorkerProtocolError(ValueError):
    """The plugin worker returned no recognized closed envelope."""


def parse_plugin_worker_response(value: object) -> PluginWorkerResponse:
    """Parse one plugin response without admitting partial failure metadata."""

    try:
        return _RESPONSE_ADAPTER.validate_python(value)
    except ValidationError as exc:
        raise PluginWorkerProtocolError("invalid plugin worker response") from exc
