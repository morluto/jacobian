"""Contracts for pinned Lean certificate checkers."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, StrictBool, StrictInt, model_validator

from jacobian.contracts.common import ArtifactUri, Sha256Digest
from jacobian.contracts.results import (
    Conclusion,
    ContractModel,
    Execution,
    InputValidation,
    ResultEnvelope,
)


class LeanEnvironment(StrEnum):
    CORE = "CORE"
    MATHLIB = "MATHLIB"


class LeanDiagnosticPhase(StrEnum):
    RUNTIME_SETUP = "RUNTIME_SETUP"
    SOURCE_ELABORATION = "SOURCE_ELABORATION"
    STATE_RECONSTRUCTION = "STATE_RECONSTRUCTION"
    TACTIC_EXECUTION = "TACTIC_EXECUTION"
    TERM_ELABORATION = "TERM_ELABORATION"
    KERNEL_CHECK = "KERNEL_CHECK"


class LeanDiagnosticSource(StrEnum):
    STATEMENT = "STATEMENT"
    PROOF = "PROOF"
    TACTIC = "TACTIC"
    TERM = "TERM"


class LeanDiagnosticPosition(ContractModel):
    """Zero-based position relative to one agent-supplied Lean payload field."""

    line: StrictInt = Field(ge=0)
    column: StrictInt = Field(ge=0)


class LeanDiagnosticSourceSpan(ContractModel):
    """A payload-relative source range, never a generated-file coordinate."""

    source: LeanDiagnosticSource
    start: LeanDiagnosticPosition
    end: LeanDiagnosticPosition

    @model_validator(mode="after")
    def require_forward_source_range(self) -> Self:
        if (self.end.line, self.end.column) < (
            self.start.line,
            self.start.column,
        ):
            raise ValueError("Lean diagnostic source span must not run backwards")
        return self


class LeanDiagnostic(ContractModel):
    """Stable Lean-owned diagnostic adjacent to, but distinct from, a verdict."""

    code: str = Field(pattern=r"^LEAN_[A-Z0-9_]+$", max_length=128)
    phase: LeanDiagnosticPhase
    severity: Literal["ERROR", "WARNING", "INFO"]
    message: str = Field(min_length=1, max_length=2_000)
    source_span: LeanDiagnosticSourceSpan | None = None
    goal_index: StrictInt | None = Field(default=None, ge=0, le=63)
    metavariable: str | None = Field(default=None, min_length=1, max_length=512)
    raw_backend_message: str = Field(min_length=1, max_length=20_000)


class LeanClaim(ContractModel):
    lean_contract_version: Literal["1"] = "1"
    environment: LeanEnvironment = LeanEnvironment.CORE
    statement: str = Field(min_length=1, max_length=2_000)
    allowed_axioms: tuple[str, ...] = ()


class LeanCandidate(ContractModel):
    lean_contract_version: Literal["1"] = "1"
    environment: LeanEnvironment = LeanEnvironment.CORE
    statement: str = Field(min_length=1, max_length=2_000)
    proof: str = Field(min_length=1, max_length=20_000)


class LeanVerifyResult(ContractModel):
    claim_uri: ArtifactUri
    candidate_uri: ArtifactUri
    certificate_uri: ArtifactUri
    result: ResultEnvelope
    diagnostics: tuple[LeanDiagnostic, ...] = ()
    cache_hit: bool = False


class LeanCheckOutput(ContractModel):
    conclusion: Conclusion
    execution: Execution
    input: InputValidation
    diagnostics: tuple[LeanDiagnostic, ...]
    claim_uri: ArtifactUri
    candidate_uri: ArtifactUri
    certificate_uri: ArtifactUri
    verification_record_uri: ArtifactUri | None = None
    cache_hit: bool = False


class LeanDeclarationKind(StrEnum):
    AXIOM = "AXIOM"
    DEFINITION = "DEFINITION"
    THEOREM = "THEOREM"
    OPAQUE = "OPAQUE"
    QUOTIENT = "QUOTIENT"
    INDUCTIVE = "INDUCTIVE"
    CONSTRUCTOR = "CONSTRUCTOR"
    RECURSOR = "RECURSOR"


class LeanDeclarationMatchReason(StrEnum):
    NAME_SUBSTRING = "NAME_SUBSTRING"
    TYPE_CONSTANTS = "TYPE_CONSTANTS"


class LeanDeclarationSearchStopReason(StrEnum):
    RESULT_LIMIT = "RESULT_LIMIT"
    EXHAUSTED = "EXHAUSTED"


class LeanDeclarationSource(ContractModel):
    module: str | None = Field(default=None, min_length=1, max_length=512)
    line: StrictInt = Field(ge=1)
    column: StrictInt = Field(ge=0)
    end_line: StrictInt = Field(ge=1)
    end_column: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def require_forward_source_range(self) -> Self:
        if (self.end_line, self.end_column) < (self.line, self.column):
            raise ValueError("declaration source range must not run backwards")
        return self


class LeanDeclarationRecord(ContractModel):
    name: str = Field(min_length=1, max_length=512)
    type: str = Field(min_length=1, max_length=20_000)
    kind: LeanDeclarationKind
    namespace: str | None = Field(default=None, min_length=1, max_length=512)
    docstring: str | None = Field(default=None, max_length=20_000)
    source: LeanDeclarationSource | None = None
    match_reasons: tuple[LeanDeclarationMatchReason, ...] = ()

    @model_validator(mode="after")
    def require_unique_match_reasons(self) -> Self:
        if len(set(self.match_reasons)) != len(self.match_reasons):
            raise ValueError("declaration match reasons must be unique")
        return self


class LeanDeclarationTypePattern(ContractModel):
    """All named constants must occur in the elaborated declaration type."""

    constants: tuple[str, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def require_distinct_constant_names(self) -> Self:
        _require_distinct_lean_names(
            self.constants, field_name="type pattern constants"
        )
        return self


class LeanDeclarationSearchRequest(ContractModel):
    environment: LeanEnvironment = Field(
        default=LeanEnvironment.CORE,
        description=(
            "Pinned declaration environment to search; use MATHLIB for Mathlib "
            "theorems and CORE for Lean's core declarations."
        ),
    )
    name_contains: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description=(
            "Case-sensitive substring of the fully qualified declaration name."
        ),
    )
    type_pattern: LeanDeclarationTypePattern | None = Field(
        default=None,
        description=(
            "Optional exact named constants that must all occur in the elaborated type."
        ),
    )
    namespace_prefixes: tuple[str, ...] = Field(
        default=(),
        max_length=16,
        description="Optional fully qualified namespace prefixes used as filters.",
    )
    kinds: tuple[LeanDeclarationKind, ...] = Field(
        default=(),
        description="Optional declaration-kind filters such as THEOREM or DEFINITION.",
    )
    result_limit: StrictInt = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum declarations returned by the deterministic bounded scan.",
    )

    @model_validator(mode="after")
    def require_query_and_distinct_filters(self) -> Self:
        if self.name_contains is None and self.type_pattern is None:
            raise ValueError("name_contains or type_pattern is required")
        if self.name_contains is not None:
            _require_lean_text(self.name_contains, field_name="name_contains")
        _require_distinct_lean_names(
            self.namespace_prefixes,
            field_name="namespace prefixes",
        )
        if len(set(self.kinds)) != len(self.kinds):
            raise ValueError("declaration kinds must be unique")
        return self


class LeanDeclarationEnvironmentIdentity(ContractModel):
    """Portable pinned runtime identity attached to declaration results."""

    environment: LeanEnvironment
    environment_digest: Sha256Digest
    lean_version: str = Field(min_length=1, max_length=64)
    lean_commit: str = Field(min_length=7, max_length=64)
    mathlib_commit: str | None = Field(default=None, min_length=7, max_length=64)

    @model_validator(mode="after")
    def bind_mathlib_commit_to_environment(self) -> Self:
        if self.environment is LeanEnvironment.MATHLIB and self.mathlib_commit is None:
            raise ValueError("MATHLIB declaration identity requires mathlib_commit")
        if self.environment is LeanEnvironment.CORE and self.mathlib_commit is not None:
            raise ValueError("CORE declaration identity cannot include mathlib_commit")
        return self


class LeanDeclarationSearchOutput(LeanDeclarationEnvironmentIdentity):
    query: LeanDeclarationSearchRequest
    declarations: tuple[LeanDeclarationRecord, ...]
    scanned_declarations: StrictInt = Field(ge=0)
    stop_reason: LeanDeclarationSearchStopReason

    @model_validator(mode="after")
    def bind_result_budget(self) -> Self:
        if len(self.declarations) > self.query.result_limit:
            raise ValueError("declaration results exceed the requested result limit")
        names = tuple(declaration.name for declaration in self.declarations)
        if len(set(names)) != len(names):
            raise ValueError("declaration search results must have unique names")
        expected_reasons = (
            *(
                (LeanDeclarationMatchReason.NAME_SUBSTRING,)
                if self.query.name_contains is not None
                else ()
            ),
            *(
                (LeanDeclarationMatchReason.TYPE_CONSTANTS,)
                if self.query.type_pattern is not None
                else ()
            ),
        )
        if any(
            declaration.match_reasons != expected_reasons
            for declaration in self.declarations
        ):
            raise ValueError("declaration match reasons must bind the search query")
        if (
            self.stop_reason is LeanDeclarationSearchStopReason.RESULT_LIMIT
            and len(self.declarations) != self.query.result_limit
        ):
            raise ValueError(
                "a result-limit stop must fill the requested result budget"
            )
        return self


class LeanDeclarationInspectRequest(ContractModel):
    environment: LeanEnvironment = Field(
        default=LeanEnvironment.CORE,
        description=(
            "Pinned declaration environment containing the exact declaration name."
        ),
    )
    declaration_name: str = Field(
        min_length=1,
        max_length=512,
        description="Exact fully qualified Lean declaration name to inspect.",
    )

    @model_validator(mode="after")
    def require_exact_name_text(self) -> Self:
        _require_lean_text(self.declaration_name, field_name="declaration_name")
        return self


class LeanDeclarationInspectOutput(LeanDeclarationEnvironmentIdentity):
    query: LeanDeclarationInspectRequest
    declaration: LeanDeclarationRecord

    @model_validator(mode="after")
    def bind_exact_name(self) -> Self:
        if self.declaration.name != self.query.declaration_name:
            raise ValueError(
                "inspected declaration differs from the requested exact name"
            )
        if self.declaration.match_reasons:
            raise ValueError("exact declaration inspection has no search match reasons")
        return self


class LeanDependencyEdgeKind(StrEnum):
    TYPE = "TYPE"
    VALUE = "VALUE"


class LeanDependencyGraphRequest(ContractModel):
    environment: LeanEnvironment = LeanEnvironment.CORE
    root_declaration: str = Field(min_length=1, max_length=512)
    max_depth: StrictInt = Field(default=2, ge=0, le=8)
    max_nodes: StrictInt = Field(default=100, ge=1, le=500)

    @model_validator(mode="after")
    def require_exact_root_name(self) -> Self:
        _require_lean_text(self.root_declaration, field_name="root declaration")
        return self


class LeanDependencyNode(ContractModel):
    name: str = Field(min_length=1, max_length=512)
    kind: LeanDeclarationKind
    depth: StrictInt = Field(ge=0, le=8)


class LeanDependencyEdge(ContractModel):
    source: str = Field(min_length=1, max_length=512)
    target: str = Field(min_length=1, max_length=512)
    kinds: tuple[LeanDependencyEdgeKind, ...] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def require_canonical_distinct_kinds(self) -> Self:
        if tuple(sorted(set(self.kinds), key=str)) != self.kinds:
            raise ValueError("dependency edge kinds must be unique and sorted")
        return self


class LeanDependencyGraphArtifact(ContractModel):
    dependency_graph_schema_version: Literal["1"] = "1"
    environment: LeanEnvironment
    environment_digest: Sha256Digest
    query: LeanDependencyGraphRequest
    nodes: tuple[LeanDependencyNode, ...]
    edges: tuple[LeanDependencyEdge, ...]
    frontier: tuple[str, ...]
    node_budget_exhausted: StrictBool
    closure_complete: StrictBool

    @model_validator(mode="after")
    def require_consistent_bounded_graph(self) -> Self:
        depths = _validate_dependency_nodes(self)
        _validate_dependency_edges(self, depths)
        _validate_dependency_frontier(self, depths)
        return self


def _validate_dependency_nodes(
    graph: LeanDependencyGraphArtifact,
) -> dict[str, int]:
    if not graph.nodes or graph.nodes[0].name != graph.query.root_declaration:
        raise ValueError("dependency graph must begin with its requested root")
    if graph.nodes[0].depth != 0:
        raise ValueError("dependency graph root must have depth zero")
    if len(graph.nodes) > graph.query.max_nodes:
        raise ValueError("dependency graph exceeds its node budget")
    if any(node.depth > graph.query.max_depth for node in graph.nodes):
        raise ValueError("dependency graph exceeds its depth budget")
    depths = {node.name: node.depth for node in graph.nodes}
    if len(depths) != len(graph.nodes):
        raise ValueError("dependency graph node names must be unique")
    return depths


def _validate_dependency_edges(
    graph: LeanDependencyGraphArtifact,
    depths: dict[str, int],
) -> None:
    for edge in graph.edges:
        if edge.source not in depths or edge.target not in depths:
            raise ValueError("dependency edge endpoint is absent from nodes")
        if depths[edge.target] > depths[edge.source] + 1:
            raise ValueError("dependency edge skips a traversal depth")


def _validate_dependency_frontier(
    graph: LeanDependencyGraphArtifact,
    depths: dict[str, int],
) -> None:
    if len(set(graph.frontier)) != len(graph.frontier):
        raise ValueError("dependency frontier names must be unique")
    if any(name not in depths for name in graph.frontier):
        raise ValueError("dependency frontier must refer to returned nodes")
    if graph.closure_complete != (not graph.frontier):
        raise ValueError("dependency closure completeness must match its frontier")
    if graph.node_budget_exhausted and not graph.frontier:
        raise ValueError("an exhausted node budget must identify a frontier")


class LeanDependencyGraphOutput(LeanDependencyGraphArtifact):
    dependency_graph_uri: ArtifactUri


def _require_lean_text(value: str, *, field_name: str) -> None:
    if not value.strip() or "\x00" in value or any(char in "\r\n" for char in value):
        raise ValueError(f"{field_name} must be one non-empty Lean name fragment")


def _require_distinct_lean_names(
    values: tuple[str, ...],
    *,
    field_name: str,
) -> None:
    for value in values:
        _require_lean_text(value, field_name=field_name)
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must be unique")
