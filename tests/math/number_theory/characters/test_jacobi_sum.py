from __future__ import annotations

import math

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
    dirichlet_character_jacobi_sum,
)


def _quadratic_character(prime: int):
    group = character_group(prime)
    # In a cyclic group of even order, the order-two character is the half-order
    # dual coordinate along the canonical generator axis.
    return dirichlet_character(group, (group.generator_orders[0] // 2,))


def test_quadratic_jacobi_sum_matches_independent_legendre_oracle() -> None:
    character = _quadratic_character(5)
    result = dirichlet_character_jacobi_sum(character, character)

    def legendre(value: int) -> int:
        residue = value % 5
        if math.gcd(residue, 5) != 1:
            return 0
        return 1 if pow(residue, 2, 5) == 1 else -1

    oracle = sum(legendre(a) * legendre(1 - a) for a in range(5))
    assert oracle == -1
    assert result.value.field.order == 4
    assert tuple((c.num, c.den) for c in result.value.coefficients_ascending) == (
        (-1, 1),
        (0, 1),
    )


def test_quadratic_jacobi_sum_for_modulus_three_is_one() -> None:
    character = _quadratic_character(3)
    result = dirichlet_character_jacobi_sum(character, character)
    assert result.value.field.order == 2
    assert tuple((c.num, c.den) for c in result.value.coefficients_ascending) == (
        (1, 1),
    )


def test_order_four_jacobi_sum_uses_the_canonical_positive_root() -> None:
    group = character_group(5)
    character = dirichlet_character(group, (1,))
    result = dirichlet_character_jacobi_sum(character, character)

    # Independent Gaussian-integer enumeration from chi(2)=i and 2 generating
    # (Z/5Z)^*: chi(1)=1, chi(2)=i, chi(3)=-i, chi(4)=-1.
    values = {1: (1, 0), 2: (0, 1), 3: (0, -1), 4: (-1, 0)}
    oracle_real = 0
    oracle_imaginary = 0
    for residue in range(5):
        other = (1 - residue) % 5
        if residue not in values or other not in values:
            continue
        first = values[residue]
        second = values[other]
        oracle_real += first[0] * second[0] - first[1] * second[1]
        oracle_imaginary += first[0] * second[1] + first[1] * second[0]

    assert (oracle_real, oracle_imaginary) == (-1, -2)
    assert result.value.field.order == 4
    assert tuple((c.num, c.den) for c in result.value.coefficients_ascending) == (
        (-1, 1),
        (-2, 1),
    )


def test_jacobi_sum_requires_identical_character_group_parent() -> None:
    left = _quadratic_character(5)
    right = _quadratic_character(3)
    with pytest.raises(OperationDomainValidationError) as error:
        dirichlet_character_jacobi_sum(left, right)
    assert (
        error.value.errors()[0]["type"]
        == "dirichlet_character.jacobi_sum.parent_mismatch"
    )


def test_jacobi_sum_bounds_cyclotomic_target_order() -> None:
    character = _quadratic_character(509)
    # The group exponent is 508, beyond the existing canonical cyclotomic value
    # carrier's order bound; reject before constructing the cyclotomic field.
    with pytest.raises(OperationResourceAdmissionError) as error:
        dirichlet_character_jacobi_sum(character, character)
    assert (
        error.value.errors()[0]["type"]
        == "dirichlet_character.jacobi_sum.cyclotomic_order_bound"
    )
