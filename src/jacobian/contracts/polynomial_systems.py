"""Exact rational polynomial-system verification contracts."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.common import ArtifactUri, CheckerUri
from jacobian.contracts.exact import (
    RATIONAL_SEARCH_GRID_LIMIT,
    CanonicalRational,
    bounded_rational_grid_size,
)
from jacobian.contracts.polynomials import PolynomialVariable, SparseRationalPolynomial
from jacobian.contracts.results import ContractModel


class RationalPolynomialSystem(ContractModel):
    system_schema_version: Literal["1"] = "1"
    domain: Literal["QQ"] = "QQ"
    variables: tuple[PolynomialVariable, ...] = Field(min_length=1, max_length=4)
    equations: tuple[SparseRationalPolynomial, ...] = Field(
        min_length=1,
        max_length=64,
    )
    inequations: tuple[SparseRationalPolynomial, ...] = Field(
        default=(),
        max_length=64,
    )

    @model_validator(mode="after")
    def require_one_declared_ring(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("polynomial-system variables must be unique")
        dimension = len(self.variables)
        if any(
            len(term.exponents) != dimension
            for polynomial in (*self.equations, *self.inequations)
            for term in polynomial.terms
        ):
            raise ValueError("every system monomial must match the variable order")
        return self


class RationalPolynomialAssignment(ContractModel):
    assignment_schema_version: Literal["1"] = "1"
    values: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=4)


class PolynomialSystemSolutionRequest(ContractModel):
    system: RationalPolynomialSystem
    assignment: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def require_assignment_dimension(self) -> Self:
        if len(self.assignment) != len(self.system.variables):
            raise ValueError("assignment dimension must match the variable order")
        return self


class PolynomialSystemRationalSearchRequest(ContractModel):
    system: RationalPolynomialSystem
    max_abs_numerator: int = Field(ge=0, le=8)
    max_denominator: int = Field(ge=1, le=8)

    @model_validator(mode="after")
    def require_bounded_grid(self) -> Self:
        if (
            bounded_rational_grid_size(
                self.max_abs_numerator,
                self.max_denominator,
                len(self.system.variables),
            )
            > RATIONAL_SEARCH_GRID_LIMIT
        ):
            raise ValueError("declared rational grid exceeds 10,000 points")
        return self


class PolynomialSystemRationalSearchOutput(ContractModel):
    found: bool
    system_uri: ArtifactUri
    assignment_uri: ArtifactUri | None = None
    assignment: tuple[CanonicalRational, ...] | None = None
    examined_assignment_count: int = Field(ge=0, le=RATIONAL_SEARCH_GRID_LIMIT)
    grid_assignment_count: int = Field(ge=1, le=RATIONAL_SEARCH_GRID_LIMIT)
    coverage: Literal["COMPLETE_SEARCH_OBJECTIVE"] = "COMPLETE_SEARCH_OBJECTIVE"

    @model_validator(mode="after")
    def bind_candidate(self) -> Self:
        if self.found != (
            self.assignment_uri is not None and self.assignment is not None
        ):
            raise ValueError("found status must match the assignment candidate")
        if self.examined_assignment_count > self.grid_assignment_count:
            raise ValueError("examined count cannot exceed the grid size")
        return self


class PolynomialSystemSolutionClaim(ContractModel):
    claim_schema_version: Literal["1"] = "1"
    predicate: Literal["ASSIGNMENT_SATISFIES_POLYNOMIAL_SYSTEM"] = (
        "ASSIGNMENT_SATISFIES_POLYNOMIAL_SYSTEM"
    )
    domain: Literal["QQ"] = "QQ"
    system_uri: ArtifactUri
    assignment_uri: ArtifactUri


class PolynomialSystemSolutionReplay(ContractModel):
    method: Literal["DIRECT_EXACT_EVALUATION"] = "DIRECT_EXACT_EVALUATION"
    system_uri: ArtifactUri
    assignment_uri: ArtifactUri
    equation_residuals: tuple[CanonicalRational, ...]
    inequation_values: tuple[CanonicalRational, ...]


class PolynomialSystemSolutionOutput(ContractModel):
    satisfies: bool | None
    conclusion: Literal["TRUE", "FALSE", "UNKNOWN"]
    equation_residuals: tuple[CanonicalRational, ...]
    inequation_values: tuple[CanonicalRational, ...]
    system_uri: ArtifactUri
    assignment_uri: ArtifactUri
    claim_uri: ArtifactUri
    certificate_uri: ArtifactUri
    verification_record_uri: ArtifactUri | None = None
    checker_id: CheckerUri | None = None

    @model_validator(mode="after")
    def preserve_truth_and_verification(self) -> Self:
        expected_satisfies = {
            "TRUE": True,
            "FALSE": False,
            "UNKNOWN": None,
        }[self.conclusion]
        if self.satisfies is not expected_satisfies:
            raise ValueError("satisfies must preserve TRUE, FALSE, and UNKNOWN")
        if self.verification_record_uri is not None and (
            self.conclusion == "UNKNOWN" or self.checker_id is None
        ):
            raise ValueError(
                "a residual verification record requires a decisive conclusion "
                "and checker identity"
            )
        return self
