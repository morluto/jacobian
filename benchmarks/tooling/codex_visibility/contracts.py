"""Passive experiment contracts for the Codex visibility experiment."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CueLevel(StrEnum):
    """How directly a case exposes the existence of a specialized tool."""

    EXPLICIT = "EXPLICIT"
    AFFORDANCE = "AFFORDANCE"
    LATENT = "LATENT"


class AdoptionExpectation(StrEnum):
    """Whether the prompt should use or abstain from Jacobian MCP tools."""

    USE = "USE"
    ABSTAIN = "ABSTAIN"


class ToolMode(StrEnum):
    """How Codex receives and dispatches tools during one visibility run."""

    DIRECT = "direct"
    UNIFIED_EXEC = "unified_exec"


class VisibilityOutputOutcome(BaseModel):
    """One acceptable completed-operation output shape for a USE case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str = Field(min_length=1)
    required_output_fields: tuple[str, ...] = Field(min_length=1)
    expected_output_values: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid_output_fields(self) -> VisibilityOutputOutcome:
        if len(set(self.required_output_fields)) != len(self.required_output_fields):
            raise ValueError("required_output_fields must be unique")
        if not set(self.expected_output_values).issubset(self.required_output_fields):
            raise ValueError("expected output values must name required output fields")
        return self


class VisibilityCase(BaseModel):
    """One agent-visible prompt plus hidden trajectory expectations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    cue_level: CueLevel
    prompt: str = Field(min_length=1)
    expectation: AdoptionExpectation = AdoptionExpectation.USE
    expected_operation_ids: tuple[str, ...] = ()
    diagnostic_operation_ids: tuple[str, ...] = ()
    acceptable_output_outcomes: tuple[VisibilityOutputOutcome, ...] = ()

    @model_validator(mode="after")
    def _valid_expectation(self) -> VisibilityCase:
        if len(set(self.expected_operation_ids)) != len(self.expected_operation_ids):
            raise ValueError("expected_operation_ids must be unique")
        if len(set(self.diagnostic_operation_ids)) != len(
            self.diagnostic_operation_ids
        ):
            raise ValueError("diagnostic_operation_ids must be unique")
        if set(self.expected_operation_ids) & set(self.diagnostic_operation_ids):
            raise ValueError("required and diagnostic operation IDs must be disjoint")
        outcome_ids = {
            outcome.operation_id for outcome in self.acceptable_output_outcomes
        }
        if not outcome_ids.issubset(
            set(self.expected_operation_ids) | set(self.diagnostic_operation_ids)
        ):
            raise ValueError("output-outcome operation IDs must be tracked")
        if (
            self.expectation is AdoptionExpectation.USE
            and not self.expected_operation_ids
            and not self.acceptable_output_outcomes
        ):
            raise ValueError("USE cases require an operation or output outcome")
        if self.expectation is AdoptionExpectation.ABSTAIN and (
            self.expected_operation_ids
            or self.diagnostic_operation_ids
            or self.acceptable_output_outcomes
        ):
            raise ValueError("ABSTAIN cases cannot declare operations or outcomes")
        return self


class VisibilitySuite(BaseModel):
    """Versioned visibility prompt suite."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1", "2"]
    suite_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    cases: tuple[VisibilityCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_cases(self) -> VisibilitySuite:
        case_ids = [case.case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("case_id values must be unique")
        if self.schema_version == "1" and any(
            case.expectation is not AdoptionExpectation.USE for case in self.cases
        ):
            raise ValueError("schema version 1 supports only USE cases")
        return self


def load_suite(path: Path) -> VisibilitySuite:
    """Load and fully validate a visibility suite."""
    return VisibilitySuite.model_validate(json.loads(path.read_text(encoding="utf-8")))


__all__ = [
    "AdoptionExpectation",
    "CueLevel",
    "ToolMode",
    "VisibilityCase",
    "VisibilityOutputOutcome",
    "VisibilitySuite",
    "load_suite",
]
