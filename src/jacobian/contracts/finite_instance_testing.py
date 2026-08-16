"""Typed contracts for bounded finite-instance claim testing."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian.contracts.base import ContractModel


class ClaimInstance(ContractModel):
    """One instance in a finite-instance test set."""

    key: str = Field(min_length=1, max_length=128)
    payload: str = Field(default="", max_length=4096)


class FiniteInstanceTestRequest(ContractModel):
    """Request to test a quantified claim over a finite instance set."""

    claim_name: str = Field(min_length=1, max_length=128)
    instances: tuple[ClaimInstance, ...] = Field(min_length=0, max_length=1024)
    timeout_ms: StrictInt = Field(default=5000, ge=100, le=30000)

    @model_validator(mode="after")
    def require_unique_keys(self) -> Self:
        keys = [inst.key for inst in self.instances]
        if len(set(keys)) != len(keys):
            raise ValueError("instance keys must be unique")
        return self


class InstanceTestResult(ContractModel):
    """The result of evaluating one claim on one instance."""

    key: str = Field(min_length=1, max_length=128)
    holds: bool
    detail: str = Field(min_length=1, max_length=1024)


class FiniteInstanceTestResult(ContractModel):
    """Result of a bounded finite-instance claim test."""

    status: Literal["COMPUTED", "VIOLATED", "EMPTY", "INVALID"]
    claim_name: str = Field(min_length=1, max_length=128)
    instance_count: StrictInt = Field(ge=0)
    passed_count: StrictInt = Field(ge=0)
    results: tuple[InstanceTestResult, ...] = Field(default=())
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_counts(self) -> Self:
        if self.instance_count != len(self.results):
            raise ValueError("instance_count must equal the number of results")
        if self.passed_count != sum(1 for r in self.results if r.holds):
            raise ValueError("passed_count must equal the number of holding results")
        if self.status == "EMPTY" and self.instance_count != 0:
            raise ValueError("an EMPTY result requires zero instances")
        if self.status == "COMPUTED" and self.passed_count != self.instance_count:
            raise ValueError("a COMPUTED result requires all instances to hold")
        if self.status == "VIOLATED" and self.passed_count == self.instance_count:
            raise ValueError("a VIOLATED result requires at least one failure")
        return self
