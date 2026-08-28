"""Typed contracts for bounded modular polynomial residue images."""

from __future__ import annotations

import math
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
from jacobian.math.number_theory._models import (
    MAX_INTEGER_DIGITS,
    _validation_error,
)
from jacobian.math.number_theory.modular_polynomials import (
    _INTEGER as _TERM_INTEGER,
)
from jacobian.math.number_theory.modular_polynomials import (
    ModularPolynomialTerm as _ModularPolynomialTerm,
)
from jacobian.math.number_theory.modular_polynomials import (
    NormalizedModularPolynomialTerm as _NormalizedModularPolynomialTerm,
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
    coefficient["maxLength"] = MAX_INTEGER_DIGITS
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
    def require_structural_shape(self) -> Self:
        _validate_residue_image_shape(self)
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        modulus: int,
        variable_order: tuple[str, ...],
        domains: tuple[tuple[int, ...], ...],
        normalized_terms: tuple[_NormalizedModularPolynomialTerm, ...],
        total_assignments: int,
        image: tuple[int, ...],
        residue_counts: tuple[ModularPolynomialResidueCount, ...],
        witnesses: tuple[ModularPolynomialResidueWitness, ...],
        table: tuple[ModularPolynomialResidueTableRow, ...] | None,
    ) -> Self:
        """Build a result after the complete image kernel establishes it."""

        return cls.model_construct(
            modulus=modulus,
            variable_order=variable_order,
            domains=domains,
            normalized_terms=normalized_terms,
            enumeration_scope="COMPLETE_DECLARED_CARTESIAN_PRODUCT",
            total_assignments=total_assignments,
            image=image,
            residue_counts=residue_counts,
            witnesses=witnesses,
            table=table,
        )


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
        or term.coefficient < 0
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
    assignments = tuple(product(*result.domains))
    if result.table is not None:
        if len(result.table) != assignment_count:
            raise _validation_error(
                "complete_table_length_does_not_match_the_declared_domains",
                "complete table length does not match the declared domains",
            )
        if tuple(row.assignment for row in result.table) != assignments:
            raise _validation_error(
                "complete_table_must_enumerate_the_declared_cartesian_product_in_order",
                "complete table must enumerate the declared Cartesian product in order",
            )
    if result.image != tuple(sorted(set(result.image))):
        raise _validation_error(
            "residue_image_must_be_strictly_increasing",
            "residue image must be strictly increasing",
        )
    if tuple(item.residue for item in result.residue_counts) != result.image:
        raise _validation_error(
            "residue_counts_must_follow_the_image_order",
            "residue counts must follow the image order",
        )
    if tuple(item.residue for item in result.witnesses) != result.image:
        raise _validation_error(
            "residue_witnesses_must_follow_the_image_order",
            "residue witnesses must follow the image order",
        )
    if any(witness.assignment not in assignments for witness in result.witnesses):
        raise _validation_error(
            "residue_witnesses_must_be_declared_assignments",
            "residue witnesses must be declared assignments",
        )
    return assignments
