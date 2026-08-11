"""Closed request and response models for the Jacobian Lean helper."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, StrictInt, StrictStr

from jacobian.contracts.lean_exploration import LeanTypedGoal
from jacobian.contracts.lean_metavariable_fields import (
    LeanElaborationContext,
    LeanStructuredMetavariable,
)
from jacobian.contracts.results import ContractModel


class _LeanHelperRequest(ContractModel):
    pickle_path: StrictStr = Field(min_length=1)
    request_id: StrictStr = Field(min_length=1)
    max_goals: StrictInt = Field(ge=1, le=64)
    max_local_declarations: StrictInt = Field(ge=1, le=256)
    max_rendered_bytes: StrictInt = Field(ge=1_024, le=262_144)


class LeanTypedGoalsHelperRequest(_LeanHelperRequest):
    mode: Literal["typed_goals"] = "typed_goals"


class LeanMetavariableFieldsHelperRequest(_LeanHelperRequest):
    mode: Literal["metavariable_fields"] = "metavariable_fields"


type LeanHelperRequest = (
    LeanTypedGoalsHelperRequest | LeanMetavariableFieldsHelperRequest
)


class LeanTypedGoalsHelperPayload(ContractModel):
    expression_serialization: Literal["LEAN_PRETTY_PRINTED_EXPR"]
    typed_goals: tuple[LeanTypedGoal, ...] = Field(max_length=64)


class LeanMetavariableFieldsHelperPayload(ContractModel):
    expression_serialization: Literal["LEAN_PRETTY_PRINTED_EXPR"]
    structured_metavariables: tuple[LeanStructuredMetavariable, ...] = Field(
        max_length=64
    )
    elaboration_context: LeanElaborationContext
    coercion_provenance: Literal["UNAVAILABLE"]
    coercion_provenance_basis: StrictStr = Field(min_length=1, max_length=2_000)


type LeanHelperPayload = (
    LeanTypedGoalsHelperPayload | LeanMetavariableFieldsHelperPayload
)


class LeanTypedGoalsHelperResult(ContractModel):
    request_id: StrictStr = Field(min_length=1)
    payload: LeanTypedGoalsHelperPayload


class LeanMetavariableFieldsHelperResult(ContractModel):
    request_id: StrictStr = Field(min_length=1)
    payload: LeanMetavariableFieldsHelperPayload


class LeanHelperErrorEnvelope(ContractModel):
    request_id: StrictStr = Field(min_length=1)
    code: Literal[
        "LEAN_PROOF_STATE_GOAL_LIMIT",
        "LEAN_PROOF_STATE_LOCAL_LIMIT",
        "LEAN_PROOF_STATE_OUTPUT_LIMIT",
        "LEAN_PROOF_STATE_UNKNOWN_MODE",
        "LEAN_PROOF_STATE_QUERY_FAILED",
    ]
    message: StrictStr = Field(min_length=1)


__all__ = [
    "LeanHelperErrorEnvelope",
    "LeanHelperPayload",
    "LeanHelperRequest",
    "LeanMetavariableFieldsHelperPayload",
    "LeanMetavariableFieldsHelperRequest",
    "LeanMetavariableFieldsHelperResult",
    "LeanTypedGoalsHelperPayload",
    "LeanTypedGoalsHelperRequest",
    "LeanTypedGoalsHelperResult",
]
