"""Canonical CNF values and direct bounded predicates."""

from __future__ import annotations

from typing import Self

from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

_MAX_VARIABLES = 1_024
_MAX_CLAUSES = 8_192
_MAX_LITERALS = 32_768


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


class CanonicalCnf(StrictModel):
    """A bounded canonical propositional formula value."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"variables": ["a", "b"], "clauses": [[-1, 2], [1]]}]
        }
    )

    variables: tuple[str, ...] = Field(
        max_length=_MAX_VARIABLES,
        description="Distinct variable names in ascending lexicographic order.",
        examples=[["a", "b"]],
    )
    clauses: tuple[tuple[StrictInt, ...], ...] = Field(
        max_length=_MAX_CLAUSES,
        description=(
            "Unique non-tautological clauses in canonical order. Literals are signed "
            "one-based indexes into variables and each clause is ordered by variable."
        ),
        examples=[[[-1, 2], [1]]],
    )

    @field_validator("variables")
    @classmethod
    def require_canonical_variables(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not name or len(name) > 128 for name in value):
            raise _validation_error(
                "logic.cnf_variable_name",
                "CNF variables must be nonempty names of at most 128 characters",
            )
        if len(set(value)) != len(value) or value != tuple(sorted(value)):
            raise _validation_error(
                "logic.cnf_variables_not_canonical",
                "CNF variables must be distinct and sorted",
            )
        return value

    @model_validator(mode="after")
    def require_canonical_clauses(self) -> Self:
        if sum(len(clause) for clause in self.clauses) > _MAX_LITERALS:
            raise _validation_error(
                "logic.cnf_literal_budget",
                f"CNF may contain at most {_MAX_LITERALS} literals",
            )
        try:
            normalized = tuple(
                _canonical_clause(clause, len(self.variables))
                for clause in self.clauses
            )
        except _TautologicalClauseError as exc:
            raise _validation_error(
                "logic.cnf_tautological_clause", "CNF clauses must be non-tautological"
            ) from exc
        if self.clauses != tuple(sorted(set(normalized), key=_clause_sort_key)):
            raise _validation_error(
                "logic.cnf_clauses_not_canonical",
                "CNF clauses must be unique, non-tautological, and sorted",
            )
        return self


class CnfCanonicalizeRequest(StrictModel):
    """Named clauses whose literals refer to the supplied variable order."""

    variable_names: tuple[str, ...] = Field(max_length=_MAX_VARIABLES)
    clauses: tuple[tuple[StrictInt, ...], ...] = Field(max_length=_MAX_CLAUSES)

    @model_validator(mode="after")
    def require_bounded_input(self) -> Self:
        if any(not name or len(name) > 128 for name in self.variable_names):
            raise _validation_error(
                "logic.cnf_variable_name",
                "CNF variable names must be nonempty and at most 128 characters",
            )
        if len(set(self.variable_names)) != len(self.variable_names):
            raise _validation_error(
                "logic.cnf_variable_names_not_unique",
                "CNF variable names must be unique",
            )
        if sum(len(clause) for clause in self.clauses) > _MAX_LITERALS:
            raise _validation_error(
                "logic.cnf_literal_budget",
                f"CNF may contain at most {_MAX_LITERALS} literals",
            )
        for clause in self.clauses:
            for literal in clause:
                if literal == 0 or abs(literal) > len(self.variable_names):
                    raise _validation_error(
                        "logic.cnf_literal_out_of_range",
                        "CNF literal references an undeclared variable",
                    )
        return self


class CnfCanonicalizeResult(StrictModel):
    cnf: CanonicalCnf


class SatAssignmentCheckRequest(StrictModel):
    cnf: CanonicalCnf
    assignment: tuple[StrictBool, ...] = Field(max_length=_MAX_VARIABLES)

    @model_validator(mode="after")
    def require_total_assignment(self) -> Self:
        if len(self.assignment) != len(self.cnf.variables):
            raise _validation_error(
                "logic.assignment_length",
                "assignment must contain one Boolean per CNF variable",
            )
        return self


class SatAssignmentCheckResult(StrictModel):
    satisfies: StrictBool
    first_unsatisfied_clause: StrictInt | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def bind_failure_index(self) -> Self:
        if self.satisfies != (self.first_unsatisfied_clause is None):
            raise _validation_error(
                "logic.assignment_failure_index",
                "an assignment result must carry an index exactly when it fails",
            )
        return self


def canonicalize_cnf(request: CnfCanonicalizeRequest) -> CnfCanonicalizeResult:
    """Return the unique canonical CNF for one named clause collection."""

    indexed_names = tuple(enumerate(request.variable_names, start=1))
    sorted_names = tuple(sorted(indexed_names, key=lambda item: item[1]))
    old_to_new = {old: new for new, (old, _name) in enumerate(sorted_names, start=1)}
    clauses: set[tuple[int, ...]] = set()
    for clause in request.clauses:
        remapped = tuple(
            old_to_new[abs(literal)] if literal > 0 else -old_to_new[abs(literal)]
            for literal in clause
        )
        try:
            clauses.add(_canonical_clause(remapped, len(sorted_names)))
        except _TautologicalClauseError:
            continue
    return CnfCanonicalizeResult(
        cnf=CanonicalCnf(
            variables=tuple(name for _old, name in sorted_names),
            clauses=tuple(sorted(clauses, key=_clause_sort_key)),
        )
    )


def check_sat_assignment(
    request: SatAssignmentCheckRequest,
) -> SatAssignmentCheckResult:
    """Evaluate a total Boolean assignment directly against one canonical CNF."""

    for index, clause in enumerate(request.cnf.clauses):
        if not any(
            request.assignment[abs(literal) - 1]
            if literal > 0
            else not request.assignment[abs(literal) - 1]
            for literal in clause
        ):
            return SatAssignmentCheckResult(
                satisfies=False, first_unsatisfied_clause=index
            )
    return SatAssignmentCheckResult(satisfies=True)


class _TautologicalClauseError(Exception):
    pass


def _canonical_clause(clause: tuple[int, ...], variable_count: int) -> tuple[int, ...]:
    literals: set[int] = set()
    for literal in clause:
        if literal == 0 or abs(literal) > variable_count:
            raise ValueError("CNF literal references an undeclared variable")
        if -literal in literals:
            raise _TautologicalClauseError
        literals.add(literal)
    return tuple(sorted(literals, key=lambda literal: (abs(literal), literal > 0)))


def _clause_sort_key(clause: tuple[int, ...]) -> tuple[tuple[int, bool], ...]:
    return tuple((abs(literal), literal > 0) for literal in clause)


__all__ = [
    "_MAX_VARIABLES",
    "CanonicalCnf",
    "CnfCanonicalizeRequest",
    "CnfCanonicalizeResult",
    "SatAssignmentCheckRequest",
    "SatAssignmentCheckResult",
    "canonicalize_cnf",
    "check_sat_assignment",
]
