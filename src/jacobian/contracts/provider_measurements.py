"""Operator-facing measurements for installed operation providers."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.operations import (
    ProviderAvailability,
    ProviderObservation,
)
from jacobian.contracts.results import ContractModel


class ProviderMeasurementStatus(StrEnum):
    COMPLETED = "COMPLETED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


class ProviderMeasurementSample(ContractModel):
    status: ProviderMeasurementStatus
    seconds: float | None = Field(default=None, ge=0)
    peak_rss_bytes: int | None = Field(default=None, ge=0)
    output_bytes: int | None = Field(default=None, ge=0)
    detail: str | None = Field(default=None, min_length=1, max_length=512)

    @model_validator(mode="after")
    def bind_measurement_status(self) -> Self:
        if self.status is ProviderMeasurementStatus.COMPLETED and self.seconds is None:
            raise ValueError("completed provider measurement requires elapsed seconds")
        if (
            self.status is not ProviderMeasurementStatus.COMPLETED
            and self.detail is None
        ):
            raise ValueError("incomplete provider measurement requires a detail")
        return self


class ProviderInstalledSize(ContractModel):
    """The installed footprint measurement for one provider runtime."""

    status: ProviderMeasurementStatus
    bytes: int | None = Field(default=None, ge=0)
    detail: str | None = Field(default=None, min_length=1, max_length=512)

    @model_validator(mode="after")
    def bind_installed_size_status(self) -> Self:
        if self.status is ProviderMeasurementStatus.COMPLETED and self.bytes is None:
            raise ValueError("completed installed-size measurement requires bytes")
        if (
            self.status is not ProviderMeasurementStatus.COMPLETED
            and self.bytes is not None
        ):
            raise ValueError(
                "incomplete installed-size measurement must not include bytes"
            )
        if (
            self.status is not ProviderMeasurementStatus.COMPLETED
            and self.detail is None
        ):
            raise ValueError("incomplete installed-size measurement requires a detail")
        return self


class ProviderMeasurement(ContractModel):
    measurement_version: Literal["2"] = "2"
    provider_runtime: ProviderObservation
    installed_size: ProviderInstalledSize
    cold_install: ProviderMeasurementSample
    cold_start: ProviderMeasurementSample
    reproduction_case: ProviderMeasurementSample

    @model_validator(mode="after")
    def require_available_provider(self) -> Self:
        if self.provider_runtime.availability is not ProviderAvailability.AVAILABLE:
            raise ValueError("only an available provider runtime can be measured")
        return self
