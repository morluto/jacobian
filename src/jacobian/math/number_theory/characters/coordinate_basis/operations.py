"""Exact transport of Dirichlet characters between supplied unit bases."""

from __future__ import annotations

from itertools import product
from math import gcd, lcm, prod

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.characters.coordinate_basis._models import (
    DirichletCharacterBasisChangeResult,
    DirichletCharacterCoordinateIsomorphism,
)
from jacobian.math.number_theory.characters.values import (
    MAX_CHARACTER_GROUP_MODULUS,
    CyclotomicValue,
    DirichletCharacter,
    DirichletCharacterGroup,
)

_MAX_WORK = 131_072


def _domain_error(code: str, message: str) -> None:
    raise OperationDomainValidationError(
        location=("coordinate_isomorphism",),
        code=f"dirichlet_character.coordinate_basis.{code}",
        message=message,
    )


def _resource_error(code: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("coordinate_isomorphism",),
        code=f"dirichlet_character.coordinate_basis.{code}",
        message=message,
    )


def _prime_exponents(value: int) -> dict[int, int]:
    factors: dict[int, int] = {}
    prime = 2
    remaining = value
    while prime * prime <= remaining:
        while remaining % prime == 0:
            factors[prime] = factors.get(prime, 0) + 1
            remaining //= prime
        prime += 1 if prime == 2 else 2
    if remaining > 1:
        factors[remaining] = factors.get(remaining, 0) + 1
    return factors


def _invariant_factors(generator_orders: tuple[int, ...]) -> tuple[int, ...]:
    factors_by_prime = tuple(_prime_exponents(order) for order in generator_orders)
    primes = {prime for factors in factors_by_prime for prime in factors}
    rank = len(generator_orders)
    result = [1] * rank
    for prime in primes:
        exponents = sorted(factors.get(prime, 0) for factors in factors_by_prime)
        for index, exponent in enumerate(exponents):
            result[index] *= prime**exponent
    return tuple(value for value in result if value > 1)


def _validate_group(group: DirichletCharacterGroup) -> None:
    modulus = group.modulus
    if not 1 <= modulus <= MAX_CHARACTER_GROUP_MODULUS:
        _resource_error("modulus_bound", "modulus exceeds the character table bound")
    units = tuple(residue for residue in range(modulus) if gcd(residue, modulus) == 1)
    if group.unit_residues != units or group.character_count != len(units):
        _domain_error("group_unit_table", "group must list exactly the canonical units")

    orders = group.generator_orders
    coordinates = group.unit_coordinates
    if any(order > len(units) for order in orders):
        _domain_error(
            "group_coordinate_order",
            "generator orders exceed the finite unit-group size",
        )
    if prod(orders) != len(units):
        _domain_error(
            "group_coordinate_order",
            "generator orders must multiply to the number of units",
        )
    if group.invariant_factors != _invariant_factors(orders):
        _domain_error(
            "group_invariant_factors",
            "invariant factors must describe the supplied cyclic coordinate axes",
        )
    expected_exponent = lcm(*orders) if orders else 1
    if group.exponent != expected_exponent or group.exponent > len(units):
        _domain_error(
            "group_exponent", "group exponent must be the lcm of generator orders"
        )

    expected_coordinates = set(product(*(range(order) for order in orders)))
    if (
        len(expected_coordinates) != len(units)
        or set(coordinates) != expected_coordinates
    ):
        _domain_error(
            "group_coordinate_bijection",
            "unit-coordinate rows must cover the full direct product exactly once",
        )
    for residue, row in zip(units, coordinates, strict=True):
        reconstructed = 1 % modulus
        for generator, exponent in zip(group.generators, row, strict=True):
            reconstructed = reconstructed * pow(generator, exponent, modulus) % modulus
        if reconstructed != residue:
            _domain_error(
                "group_coordinate_reconstruction",
                "each unit coordinate row must reconstruct its canonical residue",
            )


def _character_exponent(character: DirichletCharacter, row: tuple[int, ...]) -> int:
    group = character.group
    return (
        sum(
            coordinate * (group.exponent // order) * unit_coordinate
            for coordinate, order, unit_coordinate in zip(
                character.coordinates, group.generator_orders, row, strict=True
            )
        )
        % group.exponent
    )


def change_dirichlet_character_coordinate_basis(
    character: DirichletCharacter,
    coordinate_isomorphism: DirichletCharacterCoordinateIsomorphism,
) -> DirichletCharacterBasisChangeResult:
    """Express the same residue character in a supplied target unit basis."""

    source = coordinate_isomorphism.source_group
    target = coordinate_isomorphism.target_group
    if character.group != source:
        _domain_error(
            "source_mismatch", "isomorphism source group must equal character group"
        )
    if source.modulus != target.modulus:
        _domain_error("modulus_mismatch", "coordinate bases must use one modulus")
    if source.exponent != target.exponent:
        _domain_error(
            "cyclotomic_parent_mismatch",
            "source and target bases must use the same exact root-of-unity parent",
        )

    work = (
        len(source.unit_residues)
        * (len(source.generator_orders) + len(target.generator_orders))
        + source.modulus
    )
    if work > _MAX_WORK:
        _resource_error("work_bound", "coordinate transport exceeds its admitted work")
    if source.modulus > MAX_CHARACTER_GROUP_MODULUS:
        _resource_error("modulus_bound", "modulus exceeds the character table bound")

    _validate_group(source)
    _validate_group(target)

    source_rows = dict(zip(source.unit_residues, source.unit_coordinates, strict=True))
    for index, generator in enumerate(target.generators):
        supplied_image = (
            coordinate_isomorphism.target_generator_images_in_source_coordinates[index]
        )
        if supplied_image != source_rows[generator]:
            _domain_error(
                "isomorphism_image_mismatch",
                "each target generator image must name that same unit in source coordinates",
            )

    target_coordinates: list[int] = []
    for row in coordinate_isomorphism.target_generator_images_in_source_coordinates:
        exponent = _character_exponent(character, row)
        order_factor = (
            source.exponent // target.generator_orders[len(target_coordinates)]
        )
        if exponent % order_factor:
            _domain_error(
                "character_image_order",
                "character value on a target generator does not have its declared order",
            )
        target_coordinates.append(exponent // order_factor)

    transported = DirichletCharacter(
        group=target, coordinates=tuple(target_coordinates)
    )
    residues = tuple(range(source.modulus))
    target_rows = dict(zip(target.unit_residues, target.unit_coordinates, strict=True))
    values = tuple(
        CyclotomicValue(
            order=target.exponent,
            exponent=_character_exponent(transported, target_rows[residue]),
        )
        if residue in target_rows
        else None
        for residue in residues
    )
    return DirichletCharacterBasisChangeResult(
        source_character=character,
        transported_character=transported,
        residues=residues,
        values=values,
    )


__all__ = ["change_dirichlet_character_coordinate_basis"]
