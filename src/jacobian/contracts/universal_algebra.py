"""Typed contracts for exact finite universal-algebra operations."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from jacobian.contracts.common import ArtifactUri
from jacobian.contracts.results import ContractModel

Identifier = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$",
        strict=True,
    ),
]


class MagmaTerm(ContractModel):
    kind: Literal["VARIABLE", "PRODUCT"]
    variable: Identifier | None = None
    left: MagmaTerm | None = None
    right: MagmaTerm | None = None

    @model_validator(mode="after")
    def require_well_formed_term(self) -> Self:
        if self.kind == "VARIABLE":
            if self.variable is None or self.left is not None or self.right is not None:
                raise ValueError("variable terms require only a variable name")
        elif self.variable is not None or self.left is None or self.right is None:
            raise ValueError("product terms require exactly two child terms")
        if self.node_count() > 31 or self.depth() > 16:
            raise ValueError("magma term exceeds the exact evaluator budget")
        return self

    def node_count(self) -> int:
        if self.kind == "VARIABLE":
            return 1
        if self.left is None or self.right is None:
            raise ValueError("product terms require exactly two child terms")
        return 1 + self.left.node_count() + self.right.node_count()

    def depth(self) -> int:
        if self.kind == "VARIABLE":
            return 1
        if self.left is None or self.right is None:
            raise ValueError("product terms require exactly two child terms")
        return 1 + max(self.left.depth(), self.right.depth())

    def variable_names(self) -> frozenset[str]:
        if self.kind == "VARIABLE":
            if self.variable is None:
                raise ValueError("variable terms require only a variable name")
            return frozenset((self.variable,))
        if self.left is None or self.right is None:
            raise ValueError("product terms require exactly two child terms")
        return self.left.variable_names() | self.right.variable_names()


class FiniteMagma(ContractModel):
    structure_schema_version: Literal["1"] = "1"
    operation: Literal["binary"] = "binary"
    order: int = Field(ge=1, le=8)
    table: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def require_closed_square_table(self) -> Self:
        if len(self.table) != self.order or any(
            len(row) != self.order for row in self.table
        ):
            raise ValueError("magma table must be square with the declared order")
        if any(value < 0 or value >= self.order for row in self.table for value in row):
            raise ValueError("magma table values must lie in the carrier")
        return self


class MagmaLaw(ContractModel):
    law_id: Identifier
    variables: tuple[Identifier, ...] = Field(min_length=1, max_length=4)
    left: MagmaTerm
    right: MagmaTerm

    @model_validator(mode="after")
    def require_exact_declared_variables(self) -> Self:
        if self.variables != tuple(sorted(set(self.variables))):
            raise ValueError("law variables must be unique and sorted")
        used = self.left.variable_names() | self.right.variable_names()
        if used != frozenset(self.variables):
            raise ValueError("law variables must exactly match variables used in terms")
        return self


class FiniteMagmaLawProblem(ContractModel):
    problem_schema_version: Literal["1"] = "1"
    structure: FiniteMagma
    laws: tuple[MagmaLaw, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def require_unique_laws_and_bounded_valuations(self) -> Self:
        law_ids = tuple(law.law_id for law in self.laws)
        if len(set(law_ids)) != len(law_ids):
            raise ValueError("law identifiers must be unique")
        valuation_budget = sum(
            self.structure.order ** len(law.variables) for law in self.laws
        )
        if valuation_budget > 1_000_000:
            raise ValueError("finite law evaluation exceeds the valuation budget")
        return self


class UniversalAlgebraEvaluationRequest(ContractModel):
    problem: FiniteMagmaLawProblem


class UniversalAlgebraCountermodelSearchRequest(ContractModel):
    order: int = Field(ge=1, le=4)
    source_laws: tuple[MagmaLaw, ...] = Field(min_length=1, max_length=8)
    target_law: MagmaLaw

    @model_validator(mode="after")
    def require_distinct_laws_and_bounded_encoding(self) -> Self:
        law_ids = (
            *(law.law_id for law in self.source_laws),
            self.target_law.law_id,
        )
        if len(set(law_ids)) != len(law_ids):
            raise ValueError("source and target law identifiers must be unique")
        valuation_budget = sum(
            self.order ** len(law.variables)
            for law in (*self.source_laws, self.target_law)
        )
        if valuation_budget > 16_384:
            raise ValueError("countermodel encoding exceeds the valuation budget")
        return self


class FiniteMagmaTableEnumerationRequest(ContractModel):
    order: int = Field(ge=1, le=2)


class FiniteMagmaTableEnumerationArtifact(ContractModel):
    enumeration_schema_version: Literal["1"] = "1"
    order: int = Field(ge=1, le=2)
    table_uris: tuple[ArtifactUri, ...] = Field(min_length=1, max_length=16)
    enumerated_count: int = Field(ge=1, le=16)
    total_count: int = Field(ge=1, le=16)
    ordering: Literal["LEXICOGRAPHIC_ROW_MAJOR"] = "LEXICOGRAPHIC_ROW_MAJOR"
    complete: Literal[True] = True

    @model_validator(mode="after")
    def require_exact_complete_enumeration(self) -> Self:
        expected = self.order ** (self.order * self.order)
        if (
            self.total_count != expected
            or self.enumerated_count != expected
            or len(self.table_uris) != expected
            or len(set(self.table_uris)) != expected
        ):
            raise ValueError(
                "complete magma-table enumeration must contain every table once"
            )
        return self


class FiniteMagmaTableEnumerationOutput(ContractModel):
    enumeration_uri: ArtifactUri
    order: int = Field(ge=1, le=2)
    table_uris: tuple[ArtifactUri, ...] = Field(min_length=1, max_length=16)
    enumerated_count: int = Field(ge=1, le=16)
    total_count: int = Field(ge=1, le=16)
    ordering: Literal["LEXICOGRAPHIC_ROW_MAJOR"] = "LEXICOGRAPHIC_ROW_MAJOR"
    exactness: Literal["EXACT_FINITE"] = "EXACT_FINITE"
    completeness: Literal["COMPLETE"] = "COMPLETE"

    @model_validator(mode="after")
    def require_exact_output_count(self) -> Self:
        expected = self.order ** (self.order * self.order)
        if (
            self.total_count != expected
            or self.enumerated_count != expected
            or len(self.table_uris) != expected
            or len(set(self.table_uris)) != expected
        ):
            raise ValueError(
                "complete magma-table output must contain every table once"
            )
        return self


class MagmaAssignmentValue(ContractModel):
    variable: Identifier
    value: int = Field(ge=0, le=7)


class MagmaLawCounterexample(ContractModel):
    assignment: tuple[MagmaAssignmentValue, ...] = Field(min_length=1, max_length=4)
    left_value: int = Field(ge=0, le=7)
    right_value: int = Field(ge=0, le=7)

    @model_validator(mode="after")
    def require_canonical_distinguishing_assignment(self) -> Self:
        variables = tuple(item.variable for item in self.assignment)
        if variables != tuple(sorted(set(variables))):
            raise ValueError("counterexample assignment must be unique and sorted")
        if self.left_value == self.right_value:
            raise ValueError("counterexample term values must differ")
        return self


class MagmaLawCoverage(StrEnum):
    EXHAUSTIVE = "EXHAUSTIVE"
    COUNTEREXAMPLE_FOUND = "COUNTEREXAMPLE_FOUND"


class MagmaLawEvaluationRecord(ContractModel):
    law_id: Identifier
    holds: bool
    coverage: MagmaLawCoverage
    checked_valuations: int = Field(ge=1, le=1_000_000)
    counterexample: MagmaLawCounterexample | None = None

    @model_validator(mode="after")
    def require_truth_evidence_shape(self) -> Self:
        if self.holds:
            if (
                self.coverage is not MagmaLawCoverage.EXHAUSTIVE
                or self.counterexample is not None
            ):
                raise ValueError("true law records require exhaustive coverage")
        elif (
            self.coverage is not MagmaLawCoverage.COUNTEREXAMPLE_FOUND
            or self.counterexample is None
        ):
            raise ValueError("false law records require a counterexample")
        return self


class FiniteMagmaLawEvaluationArtifact(ContractModel):
    evaluation_schema_version: Literal["1"] = "1"
    problem_uri: ArtifactUri
    records: tuple[MagmaLawEvaluationRecord, ...] = Field(
        min_length=1,
        max_length=16,
    )
    arithmetic: Literal["EXACT_FINITE"] = "EXACT_FINITE"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"


class FiniteMagmaLawEvaluationClaim(ContractModel):
    claim_schema_version: Literal["1"] = "1"
    predicate: Literal["EXACT_FINITE_MAGMA_LAW_EVALUATION"] = (
        "EXACT_FINITE_MAGMA_LAW_EVALUATION"
    )
    problem_uri: ArtifactUri


class FiniteMagmaLawReplayPayload(ContractModel):
    method: Literal["EXHAUSTIVE_LEXICOGRAPHIC_REPLAY"] = (
        "EXHAUSTIVE_LEXICOGRAPHIC_REPLAY"
    )
    problem_uri: ArtifactUri
    evaluation_uri: ArtifactUri


class UniversalAlgebraEvaluationOutput(ContractModel):
    problem_uri: ArtifactUri
    evaluation_uri: ArtifactUri
    claim_uri: ArtifactUri
    certificate_uri: ArtifactUri
    records: tuple[MagmaLawEvaluationRecord, ...]
    completeness: Literal["COMPLETE"] = "COMPLETE"


class CountermodelSearchStatus(StrEnum):
    WITNESS_FOUND = "WITNESS_FOUND"
    NO_WITNESS_FOUND = "NO_WITNESS_FOUND"
    INDETERMINATE = "INDETERMINATE"


class FiniteMagmaCountermodelArtifact(ContractModel):
    search_schema_version: Literal["1"] = "1"
    order: int = Field(ge=1, le=4)
    source_laws: tuple[MagmaLaw, ...] = Field(min_length=1, max_length=8)
    target_law: MagmaLaw
    status: CountermodelSearchStatus
    structure: FiniteMagma | None = None
    source_records: tuple[MagmaLawEvaluationRecord, ...] | None = None
    target_record: MagmaLawEvaluationRecord | None = None
    backend: Literal["z3"] = "z3"
    backend_version: str = Field(min_length=1, max_length=64)
    encoding: Literal["COMPLETE_FIXED_ORDER_FINITE_TABLE"] = (
        "COMPLETE_FIXED_ORDER_FINITE_TABLE"
    )

    @model_validator(mode="after")
    def require_status_evidence_shape(self) -> Self:
        evidence = (self.structure, self.source_records, self.target_record)
        if self.status is CountermodelSearchStatus.WITNESS_FOUND:
            if (
                self.structure is None
                or self.source_records is None
                or self.target_record is None
            ):
                raise ValueError(
                    "found countermodels require complete witness evidence"
                )
            if self.structure.order != self.order:
                raise ValueError(
                    "countermodel carrier order must match the search order"
                )
            if len(self.source_records) != len(self.source_laws):
                raise ValueError(
                    "source evaluation records must cover every source law"
                )
            if tuple(record.law_id for record in self.source_records) != tuple(
                law.law_id for law in self.source_laws
            ):
                raise ValueError(
                    "source evaluation records must preserve source law order"
                )
            if any(not record.holds for record in self.source_records):
                raise ValueError("a countermodel must satisfy every source law")
            if (
                self.target_record.law_id != self.target_law.law_id
                or self.target_record.holds
            ):
                raise ValueError("a countermodel must falsify the target law")
        elif any(value is not None for value in evidence):
            raise ValueError("non-witness search results cannot carry a candidate")
        return self


class UniversalAlgebraCountermodelSearchOutput(ContractModel):
    search_uri: ArtifactUri
    status: CountermodelSearchStatus
    structure: FiniteMagma | None = None
    source_records: tuple[MagmaLawEvaluationRecord, ...] | None = None
    target_record: MagmaLawEvaluationRecord | None = None
    scope: Literal["ONE_FIXED_CARRIER_ORDER"] = "ONE_FIXED_CARRIER_ORDER"

    @model_validator(mode="after")
    def require_status_evidence_shape(self) -> Self:
        evidence = (self.structure, self.source_records, self.target_record)
        if self.status is CountermodelSearchStatus.WITNESS_FOUND:
            if any(value is None for value in evidence):
                raise ValueError(
                    "found countermodels require complete witness evidence"
                )
        elif any(value is not None for value in evidence):
            raise ValueError("non-witness search outputs cannot carry a candidate")
        return self
