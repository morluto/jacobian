"""Closed wire models for the pinned ``leanprover-community/repl`` protocol."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, JsonValue, StrictInt, StrictStr

from jacobian.contracts.results import ContractModel

LeanReplIndex = Annotated[StrictInt, Field(ge=0)]


class LeanReplWireModel(ContractModel):
    """Closed model whose Python names map explicitly to Lean's JSON names."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class LeanReplCommandRequest(LeanReplWireModel):
    cmd: StrictStr
    env: LeanReplIndex | None = None


class LeanReplProofStepRequest(LeanReplWireModel):
    proof_state: LeanReplIndex = Field(
        validation_alias="proofState",
        serialization_alias="proofState",
    )
    tactic: StrictStr


class LeanReplPickleProofStateRequest(LeanReplWireModel):
    proof_state: LeanReplIndex = Field(
        validation_alias="proofState",
        serialization_alias="proofState",
    )
    pickle_to: StrictStr = Field(
        validation_alias="pickleTo",
        serialization_alias="pickleTo",
    )


type LeanReplRequest = (
    LeanReplCommandRequest | LeanReplProofStepRequest | LeanReplPickleProofStateRequest
)


class LeanReplPosition(LeanReplWireModel):
    line: LeanReplIndex
    column: LeanReplIndex


class LeanReplMessage(LeanReplWireModel):
    pos: LeanReplPosition
    end_pos: LeanReplPosition | None = Field(
        default=None,
        validation_alias="endPos",
        serialization_alias="endPos",
    )
    severity: Literal["trace", "info", "warning", "error"]
    data: StrictStr


class LeanReplSorry(LeanReplWireModel):
    goal: StrictStr
    proof_state: LeanReplIndex | None = Field(
        validation_alias="proofState",
        serialization_alias="proofState",
    )
    # The pinned REPL deliberately omits its synthetic zero positions.
    pos: LeanReplPosition | None = None
    end_pos: LeanReplPosition | None = Field(
        default=None,
        validation_alias="endPos",
        serialization_alias="endPos",
    )


class LeanReplTactic(LeanReplWireModel):
    pos: LeanReplPosition
    end_pos: LeanReplPosition = Field(
        validation_alias="endPos",
        serialization_alias="endPos",
    )
    goals: StrictStr
    tactic: StrictStr
    proof_state: LeanReplIndex | None = Field(
        validation_alias="proofState",
        serialization_alias="proofState",
    )
    used_constants: tuple[StrictStr, ...] = Field(
        validation_alias="usedConstants",
        serialization_alias="usedConstants",
    )


class LeanReplCommandResponse(LeanReplWireModel):
    env: LeanReplIndex
    messages: tuple[LeanReplMessage, ...] = ()
    sorries: tuple[LeanReplSorry, ...] = ()
    tactics: tuple[LeanReplTactic, ...] = ()
    infotree: JsonValue | None = None


class LeanReplProofStepResponse(LeanReplWireModel):
    proof_state: LeanReplIndex = Field(
        validation_alias="proofState",
        serialization_alias="proofState",
    )
    goals: tuple[StrictStr, ...]
    messages: tuple[LeanReplMessage, ...] = ()
    sorries: tuple[LeanReplSorry, ...] = ()
    traces: tuple[StrictStr, ...] = ()
    proof_status: StrictStr = Field(
        validation_alias="proofStatus",
        serialization_alias="proofStatus",
    )


class LeanReplPickleProofStateResponse(LeanReplProofStepResponse):
    """Proof snapshot returned after a successful ``pickleTo`` request."""


class LeanReplErrorResponse(LeanReplWireModel):
    message: StrictStr


type LeanReplResponse = (
    LeanReplCommandResponse | LeanReplProofStepResponse | LeanReplErrorResponse
)
type LeanReplProofResponse = LeanReplProofStepResponse | LeanReplErrorResponse
type LeanReplPickleResponse = LeanReplPickleProofStateResponse | LeanReplErrorResponse
type LeanReplExecution = tuple[LeanReplCommandResponse, LeanReplProofResponse]
type LeanReplValidatedExecution = tuple[
    LeanReplCommandResponse,
    LeanReplProofStepResponse,
    LeanReplProofResponse,
]


__all__ = [
    "LeanReplCommandRequest",
    "LeanReplCommandResponse",
    "LeanReplErrorResponse",
    "LeanReplExecution",
    "LeanReplMessage",
    "LeanReplPickleProofStateRequest",
    "LeanReplPickleProofStateResponse",
    "LeanReplPickleResponse",
    "LeanReplProofResponse",
    "LeanReplProofStepRequest",
    "LeanReplProofStepResponse",
    "LeanReplRequest",
    "LeanReplResponse",
    "LeanReplValidatedExecution",
]
