"""Closed protocol for isolated Arb point-enclosure execution."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from jacobian.contracts.results import ContractModel
from jacobian.contracts.validated_analysis import (
    ArbPointEnclosureRequest,
    ExactDyadic,
)

PROTOCOL: Literal["jacobian.analysis.arb-point-enclosure/v1"] = (
    "jacobian.analysis.arb-point-enclosure/v1"
)


class ArbPointEnclosureWorkerRequest(ContractModel):
    protocol: Literal["jacobian.analysis.arb-point-enclosure/v1"]
    request: ArbPointEnclosureRequest


class ArbEnclosedWorkerResponse(ContractModel):
    protocol: Literal["jacobian.analysis.arb-point-enclosure/v1"]
    status: Literal["ENCLOSED"]
    lower: ExactDyadic
    upper: ExactDyadic
    relative_accuracy_bits: StrictInt | None = None
    exact: StrictBool

    @model_validator(mode="after")
    def bind_exactness_and_interval(self) -> Self:
        if self.lower.as_fraction() > self.upper.as_fraction():
            raise ValueError("worker enclosure lower endpoint exceeds upper endpoint")
        if self.exact != (self.relative_accuracy_bits is None):
            raise ValueError(
                "exact worker enclosures omit relative accuracy; inexact ones report it"
            )
        return self


class ArbNonfiniteWorkerResponse(ContractModel):
    protocol: Literal["jacobian.analysis.arb-point-enclosure/v1"]
    status: Literal["NONFINITE"]


type ArbPointEnclosureWorkerResponse = Annotated[
    ArbEnclosedWorkerResponse | ArbNonfiniteWorkerResponse,
    Field(discriminator="status"),
]

_RESPONSE_ADAPTER: TypeAdapter[ArbPointEnclosureWorkerResponse] = TypeAdapter(
    ArbPointEnclosureWorkerResponse
)


def parse_arb_worker_request(value: object) -> ArbPointEnclosureWorkerRequest:
    return ArbPointEnclosureWorkerRequest.model_validate(value)


def parse_arb_worker_response(value: object) -> ArbPointEnclosureWorkerResponse:
    try:
        return _RESPONSE_ADAPTER.validate_python(value)
    except ValidationError as exc:
        raise ValueError("invalid Arb point-enclosure worker response") from exc
