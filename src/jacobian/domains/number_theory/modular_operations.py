"""Exact modular-arithmetic operation kernels."""

from __future__ import annotations

import math
from collections import Counter
from itertools import product
from typing import Literal, cast

from jacobian.contracts.number_theory import (
    ChineseRemainderRequest,
    ChineseRemainderResult,
    FiniteAbelianGroupFactorizationRequest,
    FiniteAbelianGroupFactorizationResult,
    FiniteAbelianRepresentationCount,
    FiniteAbelianRepresentationWitness,
    IntegerValueResult,
    JacobiSymbolRequest,
    JacobiSymbolResult,
    ModularPolynomialResidueCount,
    ModularPolynomialResidueImageRequest,
    ModularPolynomialResidueImageResult,
    ModularPolynomialResidueTableRow,
    ModularPolynomialResidueWitness,
    ModularValueRequest,
    ModulusRequest,
    NormalizedModularPolynomialTerm,
    QuadraticResiduesResult,
)


def compute_jacobi_symbol(request: JacobiSymbolRequest) -> JacobiSymbolResult:
    from sympy import jacobi_symbol

    return JacobiSymbolResult(
        a=request.a,
        n=request.n,
        jacobi=cast(Literal[-1, 0, 1], int(jacobi_symbol(int(request.a), request.n))),
    )


def compute_modular_inverse(request: ModularValueRequest) -> IntegerValueResult:
    return IntegerValueResult(value=str(pow(int(request.value), -1, request.modulus)))


def compute_multiplicative_order(request: ModularValueRequest) -> IntegerValueResult:
    from sympy import n_order

    value, modulus = int(request.value), request.modulus
    if math.gcd(value, modulus) != 1:
        raise ValueError("multiplicative order requires coprime value and modulus")
    return IntegerValueResult(value=str(int(n_order(value, modulus))))


def enumerate_quadratic_residues(request: ModulusRequest) -> QuadraticResiduesResult:
    from sympy.ntheory.residue_ntheory import quadratic_residues

    return QuadraticResiduesResult(
        residues=tuple(str(int(value)) for value in quadratic_residues(request.modulus))
    )


def compute_modular_polynomial_residue_image(
    request: ModularPolynomialResidueImageRequest,
) -> ModularPolynomialResidueImageResult:
    return _residue_image(request, table=None)


def compute_finite_abelian_group_factorization(
    request: FiniteAbelianGroupFactorizationRequest,
) -> FiniteAbelianGroupFactorizationResult:
    """Exhaustively test unique representation in a product of cyclic groups."""
    moduli = request.moduli

    def normalize(element: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            coordinate % modulus
            for coordinate, modulus in zip(element, moduli, strict=True)
        )

    left = tuple(normalize(element) for element in request.left)
    right = tuple(normalize(element) for element in request.right)
    representations: dict[
        tuple[int, ...], list[tuple[tuple[int, ...], tuple[int, ...]]]
    ] = {}
    for left_element in left:
        for right_element in right:
            total = tuple(
                (left_coordinate + right_coordinate) % modulus
                for left_coordinate, right_coordinate, modulus in zip(
                    left_element, right_element, moduli, strict=True
                )
            )
            representations.setdefault(total, []).append((left_element, right_element))
    group = tuple(product(*(range(modulus) for modulus in moduli)))
    histogram = Counter(len(representations.get(element, ())) for element in group)
    first_missing = next(
        (element for element in group if element not in representations), None
    )
    duplicate_element = next(
        (element for element in group if len(representations.get(element, ())) > 1),
        None,
    )
    duplicate = None
    if duplicate_element is not None:
        first, second = representations[duplicate_element][:2]
        duplicate = FiniteAbelianRepresentationWitness(
            element=duplicate_element,
            left=first[0],
            right=first[1],
            other_left=second[0],
            other_right=second[1],
        )
    group_order = math.prod(moduli)
    exact = len(left) * len(right) == group_order and histogram == {1: group_order}
    return FiniteAbelianGroupFactorizationResult(
        semantics_version="finite-abelian-group-factorization.v1",
        moduli=moduli,
        normalized_left=left,
        normalized_right=right,
        group_order=group_order,
        pair_count=len(left) * len(right),
        distinct_sum_count=len(representations),
        representation_histogram=tuple(
            FiniteAbelianRepresentationCount(
                representation_count=count,
                element_count=histogram[count],
            )
            for count in sorted(histogram)
        ),
        is_exact_factorization=exact,
        first_missing=None if exact else first_missing,
        first_duplicate=None if exact else duplicate,
    )


def materialize_modular_polynomial_residue_assignments(
    request: ModularPolynomialResidueImageRequest,
) -> ModularPolynomialResidueImageResult:
    return _residue_image(request, table=[])


def _residue_image(
    request: ModularPolynomialResidueImageRequest,
    *,
    table: list[ModularPolynomialResidueTableRow] | None,
) -> ModularPolynomialResidueImageResult:
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
    return ModularPolynomialResidueImageResult(
        semantics_version="modular-polynomial-residue-image.v1",
        modulus=request.modulus,
        variable_order=tuple(variable.name for variable in request.variables),
        domains=tuple(variable.residues for variable in request.variables),
        normalized_terms=normalized_terms,
        enumeration_scope="COMPLETE_DECLARED_CARTESIAN_PRODUCT",
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

    result = solve_congruence(
        *zip(request.residues, request.moduli, strict=True), check=True
    )
    if result is None or result[0] is None:
        raise ValueError("congruence system is inconsistent")
    residue, modulus = result
    return ChineseRemainderResult(residue=str(int(residue)), modulus=str(int(modulus)))
