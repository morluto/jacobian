"""Typed subprocess protocol for pinned Lean declaration discovery."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictInt, StrictStr, model_validator

from jacobian.contracts.common import Sha256Digest
from jacobian.contracts.lean import (
    LeanDeclarationKind,
    LeanDeclarationRecord,
    LeanDeclarationSearchStopReason,
    LeanDependencyEdge,
    LeanDependencyNode,
)
from jacobian.contracts.results import ContractModel


class LeanDeclarationSearchQuery(ContractModel):
    operation: Literal["search"] = "search"
    declaration_name: None = None
    name_contains: StrictStr | None
    type_constants: tuple[StrictStr, ...]
    namespace_prefixes: tuple[StrictStr, ...]
    target_module_prefixes: tuple[StrictStr, ...]
    kinds: tuple[LeanDeclarationKind, ...]
    limit: StrictInt = Field(ge=1, le=50)
    candidate_names: tuple[StrictStr, ...] = ()
    candidate_scan_positions: tuple[StrictInt, ...] = ()
    scanned_declarations_total: StrictInt | None = Field(default=None, ge=0)
    max_depth: Literal[0] = 0
    max_nodes: Literal[1] = 1

    @model_validator(mode="after")
    def bind_search_and_catalog(self) -> Self:
        if self.name_contains is None and not self.type_constants:
            raise ValueError("a declaration search requires a name or type constraint")
        if self.scanned_declarations_total is None:
            if self.candidate_names or self.candidate_scan_positions:
                raise ValueError("catalog candidates require a scanned total")
            return self
        if len(self.candidate_names) != len(self.candidate_scan_positions):
            raise ValueError("catalog names and scan positions must have equal length")
        if len(set(self.candidate_names)) != len(self.candidate_names):
            raise ValueError("catalog candidate names must be unique")
        previous = 0
        for position in self.candidate_scan_positions:
            if position <= previous or position > self.scanned_declarations_total:
                raise ValueError("catalog scan positions must be ordered and in range")
            previous = position
        return self


class LeanDeclarationInspectQuery(ContractModel):
    operation: Literal["inspect"] = "inspect"
    declaration_name: StrictStr
    name_contains: None = None
    type_constants: tuple[()] = ()
    namespace_prefixes: tuple[()] = ()
    target_module_prefixes: tuple[StrictStr, ...]
    kinds: tuple[()] = ()
    limit: Literal[1] = 1
    candidate_names: tuple[()] = ()
    candidate_scan_positions: tuple[()] = ()
    scanned_declarations_total: None = None
    max_depth: Literal[0] = 0
    max_nodes: Literal[1] = 1


class LeanDeclarationDependenciesQuery(ContractModel):
    operation: Literal["dependencies"] = "dependencies"
    declaration_name: StrictStr
    name_contains: None = None
    type_constants: tuple[()] = ()
    namespace_prefixes: tuple[()] = ()
    target_module_prefixes: tuple[StrictStr, ...]
    kinds: tuple[()] = ()
    limit: Literal[1] = 1
    candidate_names: tuple[()] = ()
    candidate_scan_positions: tuple[()] = ()
    scanned_declarations_total: None = None
    max_depth: StrictInt = Field(ge=0, le=8)
    max_nodes: StrictInt = Field(ge=1, le=500)


type LeanDeclarationQuery = (
    LeanDeclarationSearchQuery
    | LeanDeclarationInspectQuery
    | LeanDeclarationDependenciesQuery
)


class LeanDeclarationSearchPayload(ContractModel):
    operation: Literal["search"]
    declarations: tuple[LeanDeclarationRecord, ...]
    scanned_declarations: StrictInt = Field(ge=0)
    stop_reason: LeanDeclarationSearchStopReason


class LeanDeclarationInspectPayload(ContractModel):
    operation: Literal["inspect"]
    declaration: LeanDeclarationRecord


class LeanDeclarationDependenciesPayload(ContractModel):
    operation: Literal["dependencies"]
    nodes: tuple[LeanDependencyNode, ...]
    edges: tuple[LeanDependencyEdge, ...]
    frontier: tuple[StrictStr, ...]
    node_budget_exhausted: StrictBool
    closure_complete: StrictBool


type LeanDeclarationPayload = (
    LeanDeclarationSearchPayload
    | LeanDeclarationInspectPayload
    | LeanDeclarationDependenciesPayload
)
LeanDeclarationDiscriminatedPayload = Annotated[
    LeanDeclarationPayload,
    Field(discriminator="operation"),
]


class LeanDeclarationResultEnvelope(ContractModel):
    request_id: StrictStr = Field(min_length=1)
    payload: LeanDeclarationDiscriminatedPayload


class LeanDeclarationErrorEnvelope(ContractModel):
    request_id: StrictStr = Field(min_length=1)
    code: Literal["LEAN_DECLARATION_NOT_FOUND", "LEAN_QUERY_FAILED"]
    message: StrictStr = Field(min_length=1)


class LeanDeclarationBackendResult(ContractModel):
    environment_digest: Sha256Digest
    lean_version: StrictStr = Field(min_length=1, max_length=64)
    lean_commit: StrictStr = Field(min_length=7, max_length=64)
    mathlib_commit: StrictStr | None = Field(default=None, min_length=7, max_length=64)
    payload: LeanDeclarationDiscriminatedPayload


__all__ = [
    "LeanDeclarationBackendResult",
    "LeanDeclarationDependenciesPayload",
    "LeanDeclarationDependenciesQuery",
    "LeanDeclarationErrorEnvelope",
    "LeanDeclarationInspectPayload",
    "LeanDeclarationInspectQuery",
    "LeanDeclarationPayload",
    "LeanDeclarationQuery",
    "LeanDeclarationResultEnvelope",
    "LeanDeclarationSearchPayload",
    "LeanDeclarationSearchQuery",
]
