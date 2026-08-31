"""Strict private protocol for the one-shot selected-image isolation worker."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, StrictInt, TypeAdapter

from jacobian._models import StrictModel
from jacobian.math.number_theory.algebraic_numbers.real import (
    RationalIsolatingInterval,
)
from jacobian.math.number_theory.number_fields.values import (
    SimpleNumberFieldPresentation,
)


class SelectedImageWorkerRequest(StrictModel):
    """The admitted source and value to isolate in a killable worker."""

    field: SimpleNumberFieldPresentation
    real_root_index: StrictInt = Field(ge=0)
    value_coefficients_descending: tuple[str, ...] = Field(
        min_length=1,
        description=(
            "Rational coefficients of the backend value in descending degree, "
            "as canonical fraction strings (numerator/denominator)."
        ),
    )
    minimal_polynomial_coefficient_bound: str = Field(
        description="Canonical integer bound on the selected-image polynomial height."
    )


class SelectedImageWorkerComplete(StrictModel):
    kind: Literal["complete"]
    order: Literal["LT", "EQ", "GT"]
    isolating_interval: RationalIsolatingInterval


class SelectedImageWorkerError(StrictModel):
    kind: Literal["error"]
    reason: str
    message: str


SelectedImageWorkerResponse = SelectedImageWorkerComplete | SelectedImageWorkerError

SELECTED_IMAGE_WORKER_RESPONSE_ADAPTER: TypeAdapter[SelectedImageWorkerResponse] = (
    TypeAdapter(SelectedImageWorkerResponse)
)

__all__ = [
    "SELECTED_IMAGE_WORKER_RESPONSE_ADAPTER",
    "SelectedImageWorkerComplete",
    "SelectedImageWorkerError",
    "SelectedImageWorkerRequest",
    "SelectedImageWorkerResponse",
]
