"""Contracts for atomic Lean statement proposal and comparison.

Each contract exposes exactly one inspectable artifact. None of these
capabilities certify that a formal statement matches an informal claim,
or that two statements are semantically equivalent. These results create no
verification record; theorem verification remains the responsibility of ``lean.check``.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.common import ArtifactUri, Sha256Digest
from jacobian.contracts.lean import LeanEnvironment
from jacobian.contracts.results import ContractModel

LeanCoreEnvironment = Literal[LeanEnvironment.CORE]

# ---------------------------------------------------------------------------
# lean.statement.propose
# ---------------------------------------------------------------------------


class LeanStatementProposalRequest(ContractModel):
    """Type-check one proposed Lean statement against an informal claim."""

    operation: Literal["PROPOSE", "ELABORATE_PROPOSITION"] = "PROPOSE"
    environment: LeanCoreEnvironment = LeanEnvironment.CORE
    informal_claim: str | None = Field(default=None, min_length=1, max_length=4_000)
    proposed_statement: str = Field(min_length=1, max_length=2_000)
    source_locator: str | None = Field(default=None, max_length=512)

    @model_validator(mode="after")
    def require_single_line_statement(self) -> Self:
        if "\n" in self.proposed_statement or "\r" in self.proposed_statement:
            raise ValueError("proposed_statement must be one Lean expression")
        if ":=" in self.proposed_statement:
            raise ValueError("proposed_statement must not contain ':='")
        if self.operation == "PROPOSE" and self.informal_claim is None:
            raise ValueError("informal_claim is required for PROPOSE")
        if (
            self.operation == "ELABORATE_PROPOSITION"
            and self.informal_claim is not None
        ):
            raise ValueError("informal_claim must be omitted for ELABORATE_PROPOSITION")
        return self


class LeanElaborationDiagnostic(ContractModel):
    severity: Literal["ERROR", "WARNING", "INFO"]
    message: str = Field(min_length=1, max_length=20_000)


class LeanElaborationOption(ContractModel):
    name: str = Field(min_length=1, max_length=128)
    value: str = Field(min_length=1, max_length=128)


class LeanStatementProposalArtifact(ContractModel):
    """One type-checked Lean statement proposal with elaboration status."""

    proposal_schema_version: Literal["2"] = "2"
    operation: Literal["PROPOSE", "ELABORATE_PROPOSITION"] = "PROPOSE"
    environment: LeanEnvironment
    environment_digest: Sha256Digest
    informal_claim: str | None = None
    proposed_statement: str
    elaborates: bool
    elaborated_expression: str | None = Field(default=None, max_length=20_000)
    sorry_count: int = Field(ge=0)
    goals: tuple[str, ...]
    messages: tuple[str, ...]
    diagnostics: tuple[LeanElaborationDiagnostic, ...] = ()
    used_imports: tuple[str, ...] = ()
    used_declarations: tuple[str, ...] = ()
    options: tuple[LeanElaborationOption, ...] = ()
    lean_version: str
    lean_commit: str
    mathlib_commit: str | None = None
    source_locator: str | None = None
    semantic_scope: Literal["ELABORATION_ONLY"] = "ELABORATION_ONLY"

    @model_validator(mode="after")
    def require_operation_specific_shape(self) -> Self:
        if self.operation == "PROPOSE" and self.informal_claim is None:
            raise ValueError("a proposal artifact requires an informal claim")
        if self.operation == "PROPOSE" and self.elaborates and self.sorry_count < 1:
            raise ValueError(
                "an elaborating proposal must report at least one sorry "
                "because the type-check proof uses sorry"
            )
        if self.operation == "ELABORATE_PROPOSITION":
            if self.informal_claim is not None:
                raise ValueError("direct elaboration cannot bind an informal claim")
            if self.sorry_count != 0:
                raise ValueError("direct proposition elaboration does not use sorry")
            if self.elaborates != (self.elaborated_expression is not None):
                raise ValueError(
                    "direct elaboration must return an expression exactly on success"
                )
        if len(set(self.used_imports)) != len(self.used_imports):
            raise ValueError("used imports must be unique")
        if len(set(self.used_declarations)) != len(self.used_declarations):
            raise ValueError("used declarations must be unique")
        if len({option.name for option in self.options}) != len(self.options):
            raise ValueError("elaboration option names must be unique")
        return self


class LeanStatementProposalOutput(LeanStatementProposalArtifact):
    """Statement-proposal output with its immutable artifact URI."""

    proposal_uri: ArtifactUri


# ---------------------------------------------------------------------------
# lean.statement.compare
# ---------------------------------------------------------------------------


class LeanStatementComparisonRequest(ContractModel):
    """Compare two Lean statements and their axiom sets (fail-closed)."""

    environment: LeanCoreEnvironment = LeanEnvironment.CORE
    statement_a: str = Field(min_length=1, max_length=2_000)
    statement_b: str = Field(min_length=1, max_length=2_000)
    axiom_set_a: tuple[str, ...] = Field(default=(), max_length=64)
    axiom_set_b: tuple[str, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def require_single_line_statements(self) -> Self:
        for value, field_name in (
            (self.statement_a, "statement_a"),
            (self.statement_b, "statement_b"),
        ):
            if "\n" in value or "\r" in value:
                raise ValueError(f"{field_name} must be one Lean expression")
            if ":=" in value:
                raise ValueError(f"{field_name} must not contain ':='")
        return self

    @model_validator(mode="after")
    def require_valid_axiom_names(self) -> Self:
        for axioms, field_name in (
            (self.axiom_set_a, "axiom_set_a"),
            (self.axiom_set_b, "axiom_set_b"),
        ):
            for axiom in axioms:
                if not axiom.strip() or "\x00" in axiom or "\n" in axiom:
                    raise ValueError(f"{field_name} contains an invalid axiom name")
        return self


class LeanStatementComparisonArtifact(ContractModel):
    """Syntactic and axiom-set comparison result (no semantic equivalence)."""

    comparison_schema_version: Literal["1"] = "1"
    environment: LeanEnvironment
    statement_a: str
    statement_b: str
    axiom_set_a: tuple[str, ...]
    axiom_set_b: tuple[str, ...]
    statements_identical: bool
    axiom_sets_identical: bool
    both_elaborate: bool
    elaboration_checked: bool
    elaboration_messages_a: tuple[str, ...]
    elaboration_messages_b: tuple[str, ...]
    lean_version: str
    lean_commit: str

    @model_validator(mode="after")
    def require_elaboration_checked_when_both_elaborate(self) -> Self:
        if self.both_elaborate and not self.elaboration_checked:
            raise ValueError(
                "both_elaborate cannot be True when elaboration was not checked"
            )
        return self


class LeanStatementComparisonOutput(LeanStatementComparisonArtifact):
    """Statement-comparison output with its immutable artifact URI."""

    comparison_uri: ArtifactUri
