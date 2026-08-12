"""Contracts for structured metavariable, local-instance, and elaboration fields.

``lean.proof_state.metavariable_fields`` exposes typed fields from Lean
``MetaM`` / ``MetavarContext`` / ``LocalInstances`` and the elaboration
``Term.Context`` for an immutable proof-state artifact. Fields are extracted
by the pinned Lean helper through maintained accessors
(``MVarId.getDecl``, ``MVarId.isAssigned``, ``MVarId.isDelayedAssigned``,
``LocalContext.find?``) rather than by parsing pretty-printed output.

Coercion provenance is reported as ``UNAVAILABLE``: the maintained
``Lean.Meta.Coe`` APIs (``expandCoe``, ``getCoeFnInfo?``) operate on
expressions during elaboration and do not retain a per-metavariable coercion
log on a pickled proof state. Inferring coercions by string parsing is
forbidden, so the honest contract reports the limitation instead of
fabricating provenance.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictBool, StrictInt, model_validator

from jacobian.contracts.common import ArtifactUri, Sha256Digest
from jacobian.contracts.lean import LeanEnvironment
from jacobian.contracts.results import ContractModel

LeanMetavarKind = Literal["NATURAL", "SYNTHETIC", "SYNTHETIC_OPAQUE"]


class LeanMetavariableFieldsRequest(ContractModel):
    state_uri: ArtifactUri
    environment: LeanEnvironment = LeanEnvironment.CORE
    max_goals: StrictInt = Field(default=32, ge=1, le=64)
    max_local_declarations: StrictInt = Field(default=128, ge=1, le=256)
    max_rendered_bytes: StrictInt = Field(default=65_536, ge=1_024, le=262_144)


class LeanLocalInstanceField(ContractModel):
    class_name: str = Field(min_length=1, max_length=512)
    fvar_user_name: str = Field(min_length=0, max_length=512)
    fvar_type: str = Field(min_length=0, max_length=20_000)


class LeanStructuredMetavariable(ContractModel):
    goal_index: StrictInt = Field(ge=0, le=63)
    user_name: str = Field(min_length=0, max_length=512)
    is_user_name_anonymous: StrictBool
    kind: LeanMetavarKind
    is_assigned: StrictBool
    is_delayed_assigned: StrictBool
    depth: StrictInt = Field(ge=0)
    num_scope_args: StrictInt = Field(ge=0)
    target_type: str = Field(min_length=1, max_length=20_000)
    local_instances: tuple[LeanLocalInstanceField, ...] = Field(max_length=256)


class LeanElaborationContext(ContractModel):
    decl_name: str = Field(min_length=0, max_length=512)
    may_postpone: StrictBool
    err_to_sorry: StrictBool
    auto_bound_implicit: StrictBool
    implicit_lambda: StrictBool
    is_noncomputable_section: StrictBool
    ignore_tc_failures: StrictBool
    in_pattern: StrictBool
    save_rec_app_syntax: StrictBool
    holes_as_synthetic_opaque: StrictBool


class LeanMetavariableFieldsArtifact(ContractModel):
    metavariable_schema_version: Literal["1"] = "1"
    environment: LeanEnvironment
    environment_digest: Sha256Digest
    source_digest: Sha256Digest
    state_uri: ArtifactUri
    state_digest: Sha256Digest
    structured_metavariables: tuple[LeanStructuredMetavariable, ...] = Field(
        max_length=64
    )
    elaboration_context: LeanElaborationContext
    coercion_provenance: Literal["UNAVAILABLE"] = "UNAVAILABLE"
    coercion_provenance_basis: str = Field(min_length=1, max_length=2_000)
    expression_serialization: Literal["LEAN_PRETTY_PRINTED_EXPR"] = (
        "LEAN_PRETTY_PRINTED_EXPR"
    )
    lean_version: str
    lean_commit: str
    mathlib_commit: str | None = None

    @model_validator(mode="after")
    def require_contiguous_metavariable_indices(self) -> Self:
        if tuple(m.goal_index for m in self.structured_metavariables) != tuple(
            range(len(self.structured_metavariables))
        ):
            raise ValueError("structured metavariable indices must be contiguous")
        return self


class LeanMetavariableFieldsOutput(LeanMetavariableFieldsArtifact):
    metavariable_fields_uri: ArtifactUri


__all__ = [
    "LeanElaborationContext",
    "LeanLocalInstanceField",
    "LeanMetavariableFieldsArtifact",
    "LeanMetavariableFieldsOutput",
    "LeanMetavariableFieldsRequest",
    "LeanStructuredMetavariable",
]
