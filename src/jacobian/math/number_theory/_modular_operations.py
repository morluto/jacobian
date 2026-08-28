"""Exact modular-arithmetic operation kernels."""

from __future__ import annotations

import math
from itertools import product
from typing import Literal, cast

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._models import MAX_INTEGER_DIGITS
from jacobian.math.number_theory._modular_basic_models import (
    MAX_CRT_COMBINED_MODULUS,
    MAX_MODULUS,
    ChineseRemainderRequest,
    ChineseRemainderResult,
    JacobiSymbolRequest,
    JacobiSymbolResult,
    ModularUnitRequest,
    ModulusRequest,
    QuadraticResiduesResult,
)
from jacobian.math.number_theory._modular_models import (
    _MAX_RESIDUE_ASSIGNMENTS,
    _MAX_RESIDUE_EXPONENT,
    ModularPolynomialResidueCount,
    ModularPolynomialResidueImageRequest,
    ModularPolynomialResidueImageResult,
    ModularPolynomialResidueTableRow,
    ModularPolynomialResidueWitness,
)
from jacobian.math.number_theory.arithmetic.values import IntegerValue
from jacobian.math.number_theory.modular_polynomials import (
    NormalizedModularPolynomialTerm,
)


def _domain_error(location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(
        location=location, code=f"number_theory.{code}", message=message
    )


def _admit_unit(request: ModularUnitRequest) -> None:
    if math.gcd(int(request.value), request.modulus) != 1:
        _domain_error(
            ("value",),
            "value_must_be_coprime_to_the_modulus",
            "value must be coprime to the modulus",
        )


def _admit_crt(request: ChineseRemainderRequest) -> None:
    if len(request.residues) != len(request.moduli):
        _domain_error(
            ("residues",),
            "residues_and_moduli_must_have_equal_length",
            "residues and moduli must have equal length",
        )
    combined = 1
    for index, modulus in enumerate(request.moduli):
        if not 2 <= modulus <= MAX_MODULUS:
            _domain_error(
                ("moduli", index),
                "modulus_out_of_range",
                f"every modulus must be between 2 and {MAX_MODULUS:,}",
            )
        combined = combined // math.gcd(combined, modulus) * modulus
        if combined > MAX_CRT_COMBINED_MODULUS:
            _domain_error(
                ("moduli", index),
                "combined_modulus_exceeds_bound",
                "the system's combined modulus must have fewer than "
                f"{len(str(MAX_CRT_COMBINED_MODULUS))} digits; split the "
                "congruence system into narrower subsystems",
            )
    for index, (residue, modulus) in enumerate(
        zip(request.residues, request.moduli, strict=True)
    ):
        if not 0 <= residue < modulus:
            _domain_error(
                ("residues", index),
                "every_residue_must_be_canonical_for_its_modulus",
                "every residue must be canonical for its modulus",
            )
        for other_index in range(index):
            if (residue - request.residues[other_index]) % math.gcd(
                modulus, request.moduli[other_index]
            ):
                _domain_error(
                    ("residues", index),
                    "congruence_system_is_inconsistent",
                    "congruence system is inconsistent",
                )


def _admit_residue_image(request: ModularPolynomialResidueImageRequest) -> None:
    variable_names = [variable.name for variable in request.variables]
    if len(variable_names) != len(set(variable_names)):
        _domain_error(
            ("variables",),
            "polynomial_variable_names_must_be_unique",
            "polynomial variable names must be unique",
        )
    if any(
        residue >= request.modulus
        for variable in request.variables
        for residue in variable.residues
    ):
        _domain_error(
            ("variables",),
            "every_variable_residue_must_be_less_than_the_modulus",
            "every variable residue must be less than the modulus",
        )
    assignment_count = math.prod(
        len(variable.residues) for variable in request.variables
    )
    if assignment_count > _MAX_RESIDUE_ASSIGNMENTS:
        _domain_error(
            ("variables",),
            "residue_assignment_count_exceeds_bound",
            "declared residue domains exceed the "
            f"{_MAX_RESIDUE_ASSIGNMENTS:,}-assignment bound",
        )
    if any(len(term.exponents) != len(request.variables) for term in request.terms):
        _domain_error(
            ("terms",),
            "every_term_exponent_vector_must_match_the_variable_count",
            "every term exponent vector must match the variable count",
        )
    if any(
        len(term.coefficient) > MAX_INTEGER_DIGITS
        or any(
            exponent < 0 or exponent > _MAX_RESIDUE_EXPONENT
            for exponent in term.exponents
        )
        for term in request.terms
    ):
        _domain_error(
            ("terms",),
            "term_outside_residue_image_admission",
            "term coefficient or exponents exceed the residue-image admission",
        )
    exponent_vectors = [term.exponents for term in request.terms]
    if exponent_vectors != sorted(set(exponent_vectors)):
        _domain_error(
            ("terms",),
            "term_exponent_vectors_must_be_unique_and_lexicographically_increasing",
            "term exponent vectors must be unique and lexicographically increasing",
        )
    if any(int(term.coefficient) % request.modulus == 0 for term in request.terms):
        _domain_error(
            ("terms",),
            "sparse_polynomial_terms_must_have_nonzero_coefficient_modulo_m",
            "sparse polynomial terms must have nonzero coefficient modulo m",
        )


def compute_jacobi_symbol(request: JacobiSymbolRequest) -> JacobiSymbolResult:
    from sympy import jacobi_symbol

    if request.n % 2 == 0:
        _domain_error(
            ("n",),
            "jacobi_symbol_denominator_must_be_odd",
            "Jacobi symbol denominator must be odd",
        )
    return JacobiSymbolResult(
        a=request.a,
        n=request.n,
        jacobi=cast(Literal[-1, 0, 1], int(jacobi_symbol(int(request.a), request.n))),
    )


def compute_modular_inverse(request: ModularUnitRequest) -> IntegerValue:
    _admit_unit(request)
    return IntegerValue(value=str(pow(int(request.value), -1, request.modulus)))


def compute_multiplicative_order(request: ModularUnitRequest) -> IntegerValue:
    from sympy import n_order

    _admit_unit(request)
    value, modulus = int(request.value), request.modulus
    return IntegerValue(value=str(int(n_order(value, modulus))))


def enumerate_quadratic_residues(request: ModulusRequest) -> QuadraticResiduesResult:
    from sympy.ntheory.residue_ntheory import quadratic_residues

    return QuadraticResiduesResult(
        residues=tuple(str(int(value)) for value in quadratic_residues(request.modulus))
    )


def compute_modular_polynomial_residue_image(
    request: ModularPolynomialResidueImageRequest,
) -> ModularPolynomialResidueImageResult:
    return _residue_image(request, table=None)


def compute_modular_polynomial_residue_assignments(
    request: ModularPolynomialResidueImageRequest,
) -> ModularPolynomialResidueImageResult:
    return _residue_image(request, table=[])


def _residue_image(
    request: ModularPolynomialResidueImageRequest,
    *,
    table: list[ModularPolynomialResidueTableRow] | None,
) -> ModularPolynomialResidueImageResult:
    _admit_residue_image(request)
    normalized_terms = tuple(
        NormalizedModularPolynomialTerm(
            coefficient=int(term.coefficient) % request.modulus,
            exponents=term.exponents,
        )
        for term in request.terms
    )
    counts: dict[int, int] = {}
    first_assignments: dict[int, tuple[int, ...]] = {}
    total_assignments = 0
    for assignment in product(*(variable.residues for variable in request.variables)):
        residue = _evaluate_modular_polynomial(
            normalized_terms, assignment, request.modulus
        )
        total_assignments += 1
        if table is not None:
            table.append(
                ModularPolynomialResidueTableRow(assignment=assignment, residue=residue)
            )
        counts[residue] = counts.get(residue, 0) + 1
        first_assignments.setdefault(residue, assignment)
    image = tuple(sorted(counts))
    return ModularPolynomialResidueImageResult._from_kernel(
        modulus=request.modulus,
        variable_order=tuple(variable.name for variable in request.variables),
        domains=tuple(variable.residues for variable in request.variables),
        normalized_terms=normalized_terms,
        total_assignments=total_assignments,
        image=image,
        residue_counts=tuple(
            ModularPolynomialResidueCount(residue=residue, count=counts[residue])
            for residue in image
        ),
        witnesses=tuple(
            ModularPolynomialResidueWitness(
                residue=residue, assignment=first_assignments[residue]
            )
            for residue in image
        ),
        table=tuple(table) if table is not None else None,
    )


def _evaluate_modular_polynomial(
    terms: tuple[NormalizedModularPolynomialTerm, ...],
    assignment: tuple[int, ...],
    modulus: int,
) -> int:
    value = 0
    for term in terms:
        monomial = term.coefficient
        for coordinate, exponent in zip(assignment, term.exponents, strict=True):
            monomial = monomial * pow(coordinate, exponent, modulus) % modulus
        value = (value + monomial) % modulus
    return value


def solve_chinese_remainder(request: ChineseRemainderRequest) -> ChineseRemainderResult:
    from sympy.ntheory.modular import solve_congruence

    _admit_crt(request)
    result = solve_congruence(
        *zip(request.residues, request.moduli, strict=True), check=True
    )
    if result is None or result[0] is None:
        raise ValueError("congruence system is inconsistent")
    residue, modulus = result
    return ChineseRemainderResult(residue=str(int(residue)), modulus=str(int(modulus)))
