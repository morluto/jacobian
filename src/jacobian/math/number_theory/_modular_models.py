"""Typed contracts for bounded modular polynomial residue images."""

from __future__ import annotations

import math
from collections import Counter
from itertools import product
from typing import Annotated, Literal, Self

from pydantic import (
    Field,
    StrictInt,
    StringConstraints,
    WithJsonSchema,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue

from jacobian._models import StrictModel
from jacobian.math.modular_polynomials import _INTEGER as _TERM_INTEGER
from jacobian.math.modular_polynomials import (
    ModularPolynomialTerm as _ModularPolynomialTerm,
)
from jacobian.math.modular_polynomials import (
    NormalizedModularPolynomialTerm as _NormalizedModularPolynomialTerm,
)
from jacobian.math.number_theory._models import (
    _MAX_INTEGER_LENGTH,
    _validation_error,
)

_MAX_RESIDUE_VARIABLES = 6
_MAX_RESIDUE_DOMAIN_SIZE = 32
_MAX_RESIDUE_TERMS = 64
_MAX_RESIDUE_EXPONENT = 32
_MAX_RESIDUE_ASSIGNMENTS = 4_096
_MAX_POLYNOMIAL_RESIDUE_MODULUS = 1_000_000

ResidueVariableName = Annotated[
    str,
    StringConstraints(
        pattern=r"^[a-z][a-z0-9_]{0,31}$",
        max_length=32,
        strict=True,
    ),
]
ResidueDomain = Annotated[
    tuple[StrictInt, ...],
    Field(min_length=1, max_length=_MAX_RESIDUE_DOMAIN_SIZE),
]
ResidueAssignment = Annotated[
    tuple[StrictInt, ...],
    Field(min_length=1, max_length=_MAX_RESIDUE_VARIABLES),
]
CanonicalResidue = Annotated[
    StrictInt,
    Field(ge=0, lt=_MAX_POLYNOMIAL_RESIDUE_MODULUS),
]


class ModularPolynomialVariable(StrictModel):
    """One named variable and its canonical finite residue domain."""

    name: ResidueVariableName
    residues: ResidueDomain

    @model_validator(mode="after")
    def require_canonical_domain(self) -> Self:
        if any(residue < 0 for residue in self.residues):
            raise _validation_error(
                "variable_residues_must_be_nonnegative",
                "variable residues must be nonnegative",
            )
        if self.residues != tuple(sorted(set(self.residues))):
            raise _validation_error(
                "variable_residues_must_be_strictly_increasing",
                "variable residues must be strictly increasing",
            )
        return self


def _residue_image_term_schema() -> JsonSchemaValue:
    """Project the shared term schema onto residue-image admission."""

    schema = _ModularPolynomialTerm.model_json_schema()
    coefficient = schema["properties"]["coefficient"]
    coefficient["maxLength"] = _MAX_INTEGER_LENGTH
    coefficient["pattern"] = _TERM_INTEGER.pattern
    exponents = schema["properties"]["exponents"]
    exponents["maxItems"] = _MAX_RESIDUE_VARIABLES
    exponents["items"]["maximum"] = _MAX_RESIDUE_EXPONENT
    return schema


ResidueImagePolynomialTerm = Annotated[
    _ModularPolynomialTerm,
    WithJsonSchema(_residue_image_term_schema()),
]


class ModularPolynomialResidueImageRequest(StrictModel):
    """A bounded sparse polynomial over declared finite residue domains."""

    modulus: StrictInt = Field(ge=2, le=_MAX_POLYNOMIAL_RESIDUE_MODULUS)
    variables: tuple[ModularPolynomialVariable, ...] = Field(
        min_length=1,
        max_length=_MAX_RESIDUE_VARIABLES,
    )
    terms: tuple[ResidueImagePolynomialTerm, ...] = Field(
        min_length=0,
        max_length=_MAX_RESIDUE_TERMS,
    )

    @model_validator(mode="after")
    def require_canonical_bounded_polynomial(self) -> Self:
        variable_names = [variable.name for variable in self.variables]
        if len(variable_names) != len(set(variable_names)):
            raise _validation_error(
                "polynomial_variable_names_must_be_unique",
                "polynomial variable names must be unique",
            )
        if any(
            residue >= self.modulus
            for variable in self.variables
            for residue in variable.residues
        ):
            raise _validation_error(
                "every_variable_residue_must_be_less_than_the_modulus",
                "every variable residue must be less than the modulus",
            )
        assignment_count = math.prod(
            len(variable.residues) for variable in self.variables
        )
        if assignment_count > _MAX_RESIDUE_ASSIGNMENTS:
            raise _validation_error(
                "declared_residue_domains_exceed_the_4_096_assignment_bound",
                "declared residue domains exceed the 4,096-assignment bound",
            )
        if any(len(term.exponents) != len(self.variables) for term in self.terms):
            raise _validation_error(
                "every_term_exponent_vector_must_match_the_variable_count",
                "every term exponent vector must match the variable count",
            )
        if any(
            len(term.coefficient) > _MAX_INTEGER_LENGTH
            or any(
                exponent < 0 or exponent > _MAX_RESIDUE_EXPONENT
                for exponent in term.exponents
            )
            for term in self.terms
        ):
            raise _validation_error(
                "term_outside_residue_image_admission",
                "term coefficient or exponents exceed the residue-image admission",
            )
        exponent_vectors = [term.exponents for term in self.terms]
        if exponent_vectors != sorted(set(exponent_vectors)):
            raise _validation_error(
                "term_exponent_vectors_must_be_unique_and_lexicographically_increasing",
                "term exponent vectors must be unique and lexicographically increasing",
            )
        if any(int(term.coefficient) % self.modulus == 0 for term in self.terms):
            raise _validation_error(
                "sparse_polynomial_terms_must_have_nonzero_coefficient_modulo_m",
                "sparse polynomial terms must have nonzero coefficient modulo m",
            )
        return self


def _residue_image_normalized_term_schema() -> JsonSchemaValue:
    """Project the shared normalized term schema onto residue-image results."""

    schema = _NormalizedModularPolynomialTerm.model_json_schema()
    exponents = schema["properties"]["exponents"]
    exponents["maxItems"] = _MAX_RESIDUE_VARIABLES
    exponents["items"]["minimum"] = 0
    exponents["items"]["maximum"] = _MAX_RESIDUE_EXPONENT
    return schema


ResidueImageNormalizedPolynomialTerm = Annotated[
    _NormalizedModularPolynomialTerm,
    WithJsonSchema(_residue_image_normalized_term_schema()),
]


class ModularPolynomialResidueCount(StrictModel):
    """Multiplicity of one reachable residue in the declared assignment table."""

    residue: CanonicalResidue
    count: StrictInt = Field(ge=1, le=_MAX_RESIDUE_ASSIGNMENTS)


class ModularPolynomialResidueWitness(StrictModel):
    """The first lexicographic assignment reaching one residue."""

    residue: CanonicalResidue
    assignment: ResidueAssignment


class ModularPolynomialResidueTableRow(StrictModel):
    """One exact assignment-to-residue evaluation."""

    assignment: ResidueAssignment
    residue: CanonicalResidue


class ModularPolynomialResidueImageResult(StrictModel):
    """Exact residue-image summary with an optional complete assignment table."""

    modulus: StrictInt = Field(ge=2, le=_MAX_POLYNOMIAL_RESIDUE_MODULUS)
    variable_order: tuple[ResidueVariableName, ...] = Field(
        min_length=1,
        max_length=_MAX_RESIDUE_VARIABLES,
    )
    domains: tuple[ResidueDomain, ...] = Field(
        min_length=1,
        max_length=_MAX_RESIDUE_VARIABLES,
    )
    normalized_terms: tuple[ResidueImageNormalizedPolynomialTerm, ...] = Field(
        min_length=0,
        max_length=_MAX_RESIDUE_TERMS,
    )
    enumeration_scope: Literal["COMPLETE_DECLARED_CARTESIAN_PRODUCT"]
    total_assignments: StrictInt = Field(ge=1, le=_MAX_RESIDUE_ASSIGNMENTS)
    image: tuple[CanonicalResidue, ...] = Field(
        min_length=1,
        max_length=_MAX_RESIDUE_ASSIGNMENTS,
    )
    residue_counts: tuple[ModularPolynomialResidueCount, ...] = Field(
        min_length=1,
        max_length=_MAX_RESIDUE_ASSIGNMENTS,
    )
    witnesses: tuple[ModularPolynomialResidueWitness, ...] = Field(
        min_length=1,
        max_length=_MAX_RESIDUE_ASSIGNMENTS,
    )
    table: tuple[ModularPolynomialResidueTableRow, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=_MAX_RESIDUE_ASSIGNMENTS,
    )

    @model_validator(mode="after")
    def bind_complete_residue_image(self) -> Self:
        assignments = _validate_residue_image_shape(self)
        residues = _validate_residue_image_table(self, assignments)
        _validate_residue_image_summaries(self, assignments, residues)
        return self


def _evaluate_normalized_modular_polynomial(
    terms: tuple[_NormalizedModularPolynomialTerm, ...],
    assignment: tuple[int, ...],
    modulus: int,
) -> int:
    value = 0
    for term in terms:
        monomial = term.coefficient
        for coordinate, exponent in zip(
            assignment,
            term.exponents,
            strict=True,
        ):
            monomial = monomial * pow(coordinate, exponent, modulus) % modulus
        value = (value + monomial) % modulus
    return value


def _validate_residue_image_shape(
    result: ModularPolynomialResidueImageResult,
) -> tuple[tuple[int, ...], ...]:
    if len(set(result.variable_order)) != len(result.variable_order):
        raise _validation_error(
            "result_variable_names_must_be_unique",
            "result variable names must be unique",
        )
    if len(result.domains) != len(result.variable_order):
        raise _validation_error(
            "result_domains_must_match_the_variable_count",
            "result domains must match the variable count",
        )
    if any(
        domain != tuple(sorted(set(domain)))
        or any(residue < 0 or residue >= result.modulus for residue in domain)
        for domain in result.domains
    ):
        raise _validation_error(
            "result_domains_must_contain_canonical_increasing_residues",
            "result domains must contain canonical increasing residues",
        )
    if any(
        len(term.exponents) != len(result.variable_order)
        or term.coefficient >= result.modulus
        or any(
            exponent < 0 or exponent > _MAX_RESIDUE_EXPONENT
            for exponent in term.exponents
        )
        for term in result.normalized_terms
    ):
        raise _validation_error(
            "normalized_terms_do_not_match_the_result_scope",
            "normalized terms do not match the result scope",
        )
    exponent_vectors = [term.exponents for term in result.normalized_terms]
    if exponent_vectors != sorted(set(exponent_vectors)):
        raise _validation_error(
            "normalized_term_exponents_must_be_canonical",
            "normalized term exponents must be canonical",
        )
    assignment_count = math.prod(len(domain) for domain in result.domains)
    if assignment_count > _MAX_RESIDUE_ASSIGNMENTS:
        raise _validation_error(
            "result_domains_exceed_the_4_096_assignment_bound",
            "result domains exceed the 4,096-assignment bound",
        )
    if result.total_assignments != assignment_count:
        raise _validation_error(
            "total_assignments_do_not_match_the_declared_domains",
            "total assignments do not match the declared domains",
        )
    if result.table is not None and len(result.table) != assignment_count:
        raise _validation_error(
            "complete_table_length_does_not_match_the_declared_domains",
            "complete table length does not match the declared domains",
        )
    return tuple(product(*result.domains))


def _validate_residue_image_table(
    result: ModularPolynomialResidueImageResult,
    assignments: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    expected_residues = tuple(
        _evaluate_normalized_modular_polynomial(
            result.normalized_terms,
            assignment,
            result.modulus,
        )
        for assignment in assignments
    )
    if result.table is not None:
        if tuple(row.assignment for row in result.table) != assignments:
            raise _validation_error(
                "complete_table_must_enumerate_the_declared_cartesian_product_in_order",
                "complete table must enumerate the declared Cartesian product in order",
            )
        if tuple(row.residue for row in result.table) != expected_residues:
            raise _validation_error(
                "complete_table_contains_an_incorrect_polynomial_evaluation",
                "complete table contains an incorrect polynomial evaluation",
            )
    return expected_residues


def _validate_residue_image_summaries(
    result: ModularPolynomialResidueImageResult,
    assignments: tuple[tuple[int, ...], ...],
    residues: tuple[int, ...],
) -> None:
    image = tuple(sorted(set(residues)))
    if result.image != image:
        raise _validation_error(
            "residue_image_does_not_match_the_complete_table",
            "residue image does not match the complete table",
        )
    counts = Counter(residues)
    expected_counts = tuple(
        ModularPolynomialResidueCount(residue=residue, count=counts[residue])
        for residue in image
    )
    if result.residue_counts != expected_counts:
        raise _validation_error(
            "residue_counts_do_not_match_the_complete_table",
            "residue counts do not match the complete table",
        )
    first_assignments: dict[int, tuple[int, ...]] = {}
    for assignment, residue in zip(assignments, residues, strict=True):
        first_assignments.setdefault(residue, assignment)
    expected_witnesses = tuple(
        ModularPolynomialResidueWitness(
            residue=residue,
            assignment=first_assignments[residue],
        )
        for residue in image
    )
    if result.witnesses != expected_witnesses:
        raise _validation_error(
            "residue_witnesses_must_be_the_first_table_assignments",
            "residue witnesses must be the first table assignments",
        )
