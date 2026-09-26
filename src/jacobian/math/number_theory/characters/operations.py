"""Native exact operations for bounded principal Dirichlet characters."""

from __future__ import annotations

import math
from itertools import product
from math import gcd
from typing import cast

from pydantic import ValidationError
from pydantic_core import PydanticCustomError
from sympy import factorint

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.characters._models import (
    DirichletCharacterTableResult,
    DirichletCharacterValueResult,
    PrincipalDirichletCharacterValueResult,
    _require_bounded_digits,
)
from jacobian.math.number_theory.characters.values import (
    MAX_CHARACTER_GROUP_MODULUS,
    MAX_PRINCIPAL_CHARACTER_MODULUS,
    CyclotomicValue,
    DirichletCharacter,
    DirichletCharacterGroup,
    PrincipalDirichletCharacter,
)

__all__ = [
    "character_group",
    "dirichlet_character",
    "dirichlet_character_conjugate",
    "dirichlet_character_product",
    "dirichlet_character_table",
    "dirichlet_character_value",
    "principal_dirichlet_character",
    "principal_dirichlet_character_value",
    "require_complete_character_group",
    "require_complete_principal_dirichlet_character",
    "require_principal_dirichlet_character_value_result",
]

MAX_CHARACTER_GROUP_WORK = 500_000


def _admit_character_integer(
    integer: int, *, type_code: str = "dirichlet_character.integer_type"
) -> None:
    """Admit the exact integer envelope shared by native value operations."""

    if type(integer) is not int:
        raise OperationDomainValidationError(
            location=("integer",),
            code=type_code,
            message="integer must be an exact integer",
        )
    try:
        _require_bounded_digits(integer)
    except PydanticCustomError as exc:
        raise OperationResourceAdmissionError(
            location=("integer",), code=exc.type, message=exc.message()
        ) from exc


def _admit_character_group(modulus: int) -> None:
    """Admit one character-group modulus, shared by native and catalog paths."""

    if type(modulus) is not int:
        raise OperationDomainValidationError(
            location=("modulus",),
            code="dirichlet_character.group.modulus_type",
            message="character-group modulus must be an integer",
        )
    if modulus < 1:
        raise OperationDomainValidationError(
            location=("modulus",),
            code="dirichlet_character.group.modulus_sign",
            message="character-group modulus must be positive",
        )
    if modulus > MAX_CHARACTER_GROUP_MODULUS:
        raise OperationResourceAdmissionError(
            location=("modulus",),
            code="dirichlet_character.group.modulus_bound",
            message=(
                "character-group modulus exceeds the bound "
                f"{MAX_CHARACTER_GROUP_MODULUS}"
            ),
        )
    phi = _euler_phi(modulus)
    table_work = modulus + phi * max(1, len(f"{modulus}"))
    if phi > MAX_CHARACTER_GROUP_MODULUS or table_work > MAX_CHARACTER_GROUP_WORK:
        raise OperationResourceAdmissionError(
            location=("modulus",),
            code="dirichlet_character.group.table_bound",
            message="character-group unit table exceeds the admitted work envelope",
        )


def _euler_phi(modulus: int) -> int:
    """Return Euler's totient using exact prime factorization."""

    if modulus <= 1:
        return 1
    result = 1
    for prime, exponent in factorint(modulus).items():
        result *= (prime - 1) * prime ** (exponent - 1)
    return result


def _multiplicative_order(residue: int, order_group: int, modulus: int) -> int:
    """Return the exact multiplicative order of a unit modulo ``modulus``."""

    order = order_group
    for prime, _ in factorint(order).items():
        while order % prime == 0 and pow(residue, order // prime, modulus) == 1:
            order //= prime
    return order


def _primitive_root_prime_power(prime: int, exponent: int) -> tuple[int, int]:
    """Return the canonical generator and order of an odd prime-power group."""

    modulus = prime**exponent
    group_order = (prime - 1) * prime ** (exponent - 1)
    for candidate in range(2, modulus):
        if gcd(candidate, modulus) != 1:
            continue
        if _multiplicative_order(candidate, group_order, modulus) == group_order:
            return candidate, group_order
    raise RuntimeError("odd prime-power unit group has no generator")


def _admit_principal_modulus(modulus: int) -> None:
    if type(modulus) is not int:
        raise OperationDomainValidationError(
            location=("modulus",),
            code="dirichlet_character.principal.modulus_type",
            message="principal-character modulus must be an integer",
        )
    if not 1 <= modulus <= MAX_PRINCIPAL_CHARACTER_MODULUS:
        raise OperationDomainValidationError(
            location=("modulus",),
            code="dirichlet_character.principal.modulus_bound",
            message=(
                "principal-character modulus must be between 1 and "
                f"{MAX_PRINCIPAL_CHARACTER_MODULUS}"
            ),
        )


def _require_character(character: DirichletCharacter) -> DirichletCharacter:
    if not isinstance(character, DirichletCharacter):
        raise OperationDomainValidationError(
            location=("character",),
            code="dirichlet_character.character_type",
            message="character must be a Dirichlet character value",
        )
    group = require_complete_character_group(
        cast(DirichletCharacterGroup, getattr(character, "group", None))
    )
    orders = group.generator_orders
    coordinates = getattr(character, "coordinates", None)
    if (
        type(coordinates) is not tuple
        or len(coordinates) != len(orders)
        or any(
            type(coordinate) is not int or coordinate < 0 or coordinate >= order
            for coordinate, order in zip(coordinates, orders, strict=True)
        )
    ):
        raise OperationDomainValidationError(
            location=("character", "coordinates"),
            code="dirichlet_character.coordinates_invalid",
            message="character coordinates must lie on the complete dual-group axes",
        )
    return DirichletCharacter.model_construct(group=group, coordinates=coordinates)


def require_complete_principal_dirichlet_character(
    character: PrincipalDirichletCharacter,
) -> PrincipalDirichletCharacter:
    """Check the mathematical table claimed by a caller-supplied character."""
    if not isinstance(character, PrincipalDirichletCharacter):
        raise OperationDomainValidationError(
            location=("character",),
            code="dirichlet_character.principal.character_type",
            message="character must be a principal Dirichlet character value",
        )
    try:
        character = PrincipalDirichletCharacter.model_validate(character.model_dump())
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("character",),
            code="dirichlet_character.principal.invalid_character",
            message="principal character has malformed authored fields",
        ) from exc
    _admit_principal_modulus(character.modulus)

    expected_units = tuple(
        residue
        for residue in range(character.modulus)
        if math.gcd(residue, character.modulus) == 1
    )
    if character.unit_residues != expected_units:
        raise OperationDomainValidationError(
            location=("character", "unit_residues"),
            code="dirichlet_character.unit_residues_mismatch",
            message=(
                "unit residues must be the complete canonical unit group modulo modulus"
            ),
        )
    units = frozenset(expected_units)
    expected_values = tuple(
        1 if residue in units else 0 for residue in range(character.modulus)
    )
    if character.values != expected_values:
        raise OperationDomainValidationError(
            location=("character", "values"),
            code="dirichlet_character.values_table_mismatch",
            message=(
                "values must be the complete extension-by-zero principal character table"
            ),
        )
    return character


def require_principal_dirichlet_character_value_result(
    result: PrincipalDirichletCharacterValueResult,
) -> None:
    """Check the character and source-evaluation relation of a claimed result."""

    if not isinstance(result, PrincipalDirichletCharacterValueResult):
        raise OperationDomainValidationError(
            location=("result",),
            code="dirichlet_character.principal.result_type",
            message="result must be a principal-character evaluation value",
        )
    # This admission is deliberately performed before int() so bools, lists,
    # and oversized authored carriers cannot become a successful source claim.
    _admit_character_integer(
        cast(int, getattr(result, "integer", None)),
        type_code="dirichlet_character.principal.integer_type",
    )
    try:
        canonical = PrincipalDirichletCharacterValueResult.model_validate(
            result.model_dump()
        )
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("result",),
            code="dirichlet_character.principal.invalid_result",
            message="principal-character result has malformed authored fields",
        ) from exc
    result = canonical
    require_complete_principal_dirichlet_character(result.character)
    residue = int(result.integer) % result.character.modulus
    if result.canonical_residue != residue:
        raise ValueError("canonical residue does not match the source integer")
    expected_is_unit = math.gcd(residue, result.character.modulus) == 1
    if result.is_unit != expected_is_unit:
        raise ValueError("unit status does not match the source character modulus")
    if result.value != result.character.values[residue]:
        raise ValueError("value does not match the source principal-character table")


def dirichlet_character(
    group: DirichletCharacterGroup, coordinates: tuple[int, ...]
) -> DirichletCharacter:
    group = require_complete_character_group(group)
    if (
        type(coordinates) is not tuple
        or len(coordinates) != len(group.generator_orders)
        or any(
            type(c) is not int or c < 0 or c >= order
            for c, order in zip(coordinates, group.generator_orders, strict=False)
        )
    ):
        raise OperationDomainValidationError(
            location=("coordinates",),
            code="dirichlet_character.coordinates_shape",
            message="character coordinates must match the group's dual axes",
        )
    return DirichletCharacter(group=group, coordinates=coordinates)


def _character_value(
    character: DirichletCharacter, integer: int
) -> CyclotomicValue | None:
    """Evaluate after the caller has performed one character admission."""
    residue = integer % character.group.modulus
    if residue not in character.group.unit_residues:
        return None
    row = character.group.unit_coordinates[character.group.unit_residues.index(residue)]
    order = character.group.exponent
    exponent = (
        sum(
            c * (order // o) * r
            for c, o, r in zip(
                character.coordinates,
                character.group.generator_orders,
                row,
                strict=True,
            )
        )
        % order
    )
    return CyclotomicValue(order=order, exponent=exponent)


def dirichlet_character_value(
    character: DirichletCharacter, integer: int
) -> DirichletCharacterValueResult:
    _admit_character_integer(integer)
    character = _require_character(character)
    value = _character_value(character, integer)
    residue = integer % character.group.modulus
    return DirichletCharacterValueResult(
        character=character,
        integer=integer,
        canonical_residue=residue,
        is_unit=value is not None,
        value=value,
    )


def dirichlet_character_product(
    left: DirichletCharacter, right: DirichletCharacter
) -> DirichletCharacter:
    left = _require_character(left)
    right = _require_character(right)
    if left.group != right.group:
        raise OperationDomainValidationError(
            location=("right", "group"),
            code="dirichlet_character.parent_mismatch",
            message="characters must use the identical group parent",
        )
    coords = tuple(
        (a + b) % order
        for a, b, order in zip(
            left.coordinates,
            right.coordinates,
            left.group.generator_orders,
            strict=True,
        )
    )
    return DirichletCharacter(group=left.group, coordinates=coords)


def dirichlet_character_table(
    character: DirichletCharacter,
) -> DirichletCharacterTableResult:
    character = _require_character(character)
    residues = tuple(range(character.group.modulus))
    return DirichletCharacterTableResult(
        character=character,
        residues=residues,
        values=tuple(_character_value(character, residue) for residue in residues),
    )


def dirichlet_character_conjugate(character: DirichletCharacter) -> DirichletCharacter:
    character = _require_character(character)
    return DirichletCharacter(
        group=character.group,
        coordinates=tuple(
            (-c) % order
            for c, order in zip(
                character.coordinates, character.group.generator_orders, strict=True
            )
        ),
    )


def principal_dirichlet_character(modulus: int) -> PrincipalDirichletCharacter:
    """Return the complete extension-by-zero principal character modulo ``modulus``."""

    _admit_principal_modulus(modulus)
    unit_residues = tuple(
        residue for residue in range(modulus) if math.gcd(residue, modulus) == 1
    )
    units = frozenset(unit_residues)
    return PrincipalDirichletCharacter._from_kernel(
        modulus=modulus,
        unit_residues=unit_residues,
        values=tuple(1 if residue in units else 0 for residue in range(modulus)),
    )


def principal_dirichlet_character_value(
    character: PrincipalDirichletCharacter, integer: int
) -> int:
    """Return the exact value of ``character`` at one integer."""

    _admit_character_integer(
        integer, type_code="dirichlet_character.principal.integer_type"
    )
    character = require_complete_principal_dirichlet_character(character)
    return character.values[integer % character.modulus]


def _canonical_blocks(modulus: int) -> tuple[tuple[int, int, int], ...]:
    """Return the nontrivial canonical prime-power generator blocks.

    Blocks of order one generate nothing and carry no coordinates; they are
    dropped so the generator tuple is a genuine generating set.
    """

    if modulus <= 1:
        return ()
    return tuple(
        block
        for prime, exponent in sorted(factorint(modulus).items())
        for block in _prime_power_blocks(prime, exponent)
        if block[2] > 1
    )


def _prime_power_blocks(prime: int, exponent: int) -> tuple[tuple[int, int, int], ...]:
    """Return canonical ``(block modulus, generator, order)`` rows.

    Odd prime powers are cyclic with the smallest primitive-root generator.
    Powers of two use the standard ``{-1, 5}`` presentation.
    """

    modulus = prime**exponent
    if modulus <= 2:
        return ((modulus, 1 % modulus, 1),)
    if prime == 2 and modulus == 4:
        return ((4, 3, 2),)
    if prime == 2:
        return ((modulus, modulus - 1, 2), (modulus, 5, 2 ** (exponent - 2)))
    generator, order = _primitive_root_prime_power(prime, exponent)
    return ((modulus, generator, order),)


def _crt_lift(modulus: int, block_modulus: int, block_residue: int) -> int:
    """Lift one block residue to ``modulus`` as 1 outside its block."""

    cofactor = modulus // block_modulus
    first = block_residue * cofactor * pow(cofactor % block_modulus, -1, block_modulus)
    if cofactor == 1:
        return first % modulus
    second = block_modulus * pow(block_modulus % cofactor, -1, cofactor)
    return (first + second) % modulus


def _invariant_factors(component_orders: tuple[int, ...]) -> tuple[int, ...]:
    """Combine cyclic component orders into the divisibility chain."""

    if not component_orders or all(order == 1 for order in component_orders):
        return ()
    prime_exponents: dict[int, list[int]] = {}
    for order in component_orders:
        if order == 1:
            continue
        for prime, exponent in factorint(order).items():
            prime_exponents.setdefault(prime, []).append(prime**exponent)
    rank = max(len(powers) for powers in prime_exponents.values())
    for powers in prime_exponents.values():
        powers.sort()
        while len(powers) < rank:
            powers.insert(0, 1)
    factors = tuple(
        math.prod(powers[index] for powers in prime_exponents.values())
        for index in range(rank)
    )
    return tuple(sorted(factors))


def _discrete_log(generator: int, target: int, order: int, modulus: int) -> int:
    """Return the bounded discrete logarithm of ``target`` to ``generator``."""

    power = 1 % modulus
    for exponent in range(order):
        if power == target:
            return exponent
        power = (power * generator) % modulus
    raise ValueError("target is not in the cyclic subgroup")


def _component_log_table(
    block_modulus: int, generators_orders: tuple[tuple[int, int], ...]
) -> dict[int, tuple[int, ...]]:
    """Tabulate joint discrete logarithms over one prime-power block.

    Blocks with several generators (powers of two) are solved jointly by
    enumerating the bounded product exactly once; first enumeration order is
    canonical.
    """

    table: dict[int, tuple[int, ...]] = {}
    ranges = [range(order) for _, order in generators_orders]
    for coordinates in product(*ranges):
        value = 1 % block_modulus
        for (generator, _), coordinate in zip(
            generators_orders, coordinates, strict=True
        ):
            value = (value * pow(generator, coordinate, block_modulus)) % block_modulus
        table.setdefault(value, coordinates)
    return table


def require_complete_character_group(
    group: DirichletCharacterGroup,
) -> DirichletCharacterGroup:
    """Check the mathematical decomposition claimed by a caller-supplied group."""

    if not isinstance(group, DirichletCharacterGroup):
        raise OperationDomainValidationError(
            location=("group",),
            code="dirichlet_character.group.type",
            message="group must be a Dirichlet-character group value",
        )
    modulus = cast(int, getattr(group, "modulus", None))
    _admit_character_group(modulus)
    expected_units = tuple(
        residue for residue in range(modulus) if math.gcd(residue, modulus) == 1
    )
    if getattr(group, "unit_residues", None) != expected_units:
        raise OperationDomainValidationError(
            location=("group", "unit_residues"),
            code="dirichlet_character.group.unit_residues_mismatch",
            message=(
                "unit residues must be the complete canonical unit group modulo modulus"
            ),
        )
    structural_fields = (
        getattr(group, "generator_orders", None),
        getattr(group, "generators", None),
        getattr(group, "invariant_factors", None),
        getattr(group, "unit_coordinates", None),
    )
    if any(type(value) is not tuple for value in structural_fields):
        raise OperationDomainValidationError(
            location=("group",),
            code="dirichlet_character.group.invalid_group",
            message="character group has malformed authored fields",
        )
    declared_orders = cast(tuple[object, ...], structural_fields[0])
    declared_generators = cast(tuple[object, ...], structural_fields[1])
    invariant_factors = cast(tuple[object, ...], structural_fields[2])
    unit_coordinates = cast(tuple[object, ...], structural_fields[3])
    if (
        type(getattr(group, "character_count", None)) is not int
        or type(getattr(group, "exponent", None)) is not int
        or any(type(value) is not int or value <= 0 for value in declared_orders)
        or any(type(value) is not int for value in declared_generators)
        or any(type(value) is not int or value <= 0 for value in invariant_factors)
        or len(unit_coordinates) != len(expected_units)
        or any(
            type(row) is not tuple or any(type(value) is not int for value in row)
            for row in unit_coordinates
        )
    ):
        raise OperationDomainValidationError(
            location=("group",),
            code="dirichlet_character.group.invalid_group",
            message="character group has malformed authored fields",
        )
    if group.character_count != len(expected_units):
        raise OperationDomainValidationError(
            location=("group", "character_count"),
            code="dirichlet_character.group.character_count_mismatch",
            message="character count must equal phi(modulus)",
        )
    orders = cast(tuple[int, ...], declared_orders)
    generators = cast(tuple[int, ...], declared_generators)
    if math.prod(orders) != group.character_count:
        raise OperationDomainValidationError(
            location=("group", "generator_orders"),
            code="dirichlet_character.group.generator_order_product",
            message="generator orders must multiply to the unit-group size",
        )
    for generator, order in zip(generators, orders, strict=True):
        if (
            math.gcd(generator, modulus) != 1
            or pow(generator, order, modulus) != 1 % modulus
            or _multiplicative_order(generator, order, modulus) != order
        ):
            raise OperationDomainValidationError(
                location=("group", "generators"),
                code="dirichlet_character.group.generator_order",
                message=(
                    "each generator must be a unit with its declared exact "
                    "multiplicative order"
                ),
            )
    if tuple(group.invariant_factors) != _invariant_factors(orders):
        raise OperationDomainValidationError(
            location=("group", "invariant_factors"),
            code="dirichlet_character.group.invariant_factor_mismatch",
            message=(
                "invariant factors must be the divisibility chain of the "
                "supplied coordinate axes"
            ),
        )
    expected_exponent = math.lcm(*orders) if orders else 1
    if group.exponent != expected_exponent:
        raise OperationDomainValidationError(
            location=("group", "exponent"),
            code="dirichlet_character.group.exponent_mismatch",
            message="the common exponent must be the least common multiple of orders",
        )
    expected_coordinates = set(product(*(range(order) for order in orders)))
    if (
        len(expected_coordinates) != len(expected_units)
        or set(unit_coordinates) != expected_coordinates
    ):
        raise OperationDomainValidationError(
            location=("group", "unit_coordinates"),
            code="dirichlet_character.group.coordinate_bijection",
            message=(
                "unit coordinates must cover the full coordinate product exactly once"
            ),
        )
    _require_generator_coordinate_round_trip(group)
    return group


def _require_generator_coordinate_round_trip(group: DirichletCharacterGroup) -> None:
    """Replay the generator-coordinate identity on every claimed unit."""

    unit_index = dict(zip(group.unit_residues, group.unit_coordinates, strict=True))
    for residue, row in zip(group.unit_residues, group.unit_coordinates, strict=True):
        rebuilt = 1 % group.modulus
        for generator, coordinate, _order in zip(
            group.generators, row, group.generator_orders, strict=True
        ):
            rebuilt = (
                rebuilt * pow(generator, coordinate, group.modulus)
            ) % group.modulus
        if rebuilt != residue or unit_index[residue] != row:
            raise OperationDomainValidationError(
                location=("group", "unit_coordinates"),
                code="dirichlet_character.group.coordinate_mismatch",
                message="generator coordinates must reconstruct every unit residue",
            )


def character_group(modulus: int) -> DirichletCharacterGroup:
    """Return the finite unit-group decomposition and character coordinates."""

    _admit_character_group(modulus)
    if modulus == 1:
        return DirichletCharacterGroup._from_kernel(
            modulus=1,
            unit_residues=(0,),
            character_count=1,
            invariant_factors=(),
            generators=(),
            generator_orders=(),
            unit_coordinates=(((),)),
            exponent=1,
        )
    units = tuple(
        residue for residue in range(modulus) if math.gcd(residue, modulus) == 1
    )
    blocks = _canonical_blocks(modulus)
    generators = tuple(
        _crt_lift(modulus, block_modulus, generator)
        for block_modulus, generator, _ in blocks
    )
    orders = tuple(block[2] for block in blocks)
    for generator, order in zip(generators, orders, strict=True):
        if math.gcd(generator, modulus) != 1:
            raise RuntimeError("lifted generator is not a unit")
        if _multiplicative_order(generator, order, modulus) != order:
            raise RuntimeError("lifted generator order failed its defining check")
    grouped: dict[int, list[tuple[int, int]]] = {}
    for block_modulus, block_generator, order in blocks:
        grouped.setdefault(block_modulus, []).append((block_generator, order))
    tables = {}
    for block_modulus, generators_orders in grouped.items():
        table = _component_log_table(block_modulus, tuple(generators_orders))
        if len(table) != _euler_phi(block_modulus):
            raise RuntimeError("block generators failed their completeness check")
        tables[block_modulus] = table
    coordinates = tuple(
        tuple(
            coordinate
            for block_modulus in grouped
            for coordinate in tables[block_modulus][residue % block_modulus]
        )
        for residue in units
    )
    group = DirichletCharacterGroup._from_kernel(
        modulus=modulus,
        unit_residues=units,
        character_count=len(units),
        invariant_factors=_invariant_factors(orders),
        generators=generators,
        generator_orders=orders,
        unit_coordinates=coordinates,
        exponent=math.lcm(*orders) if orders else 1,
    )
    _require_generator_coordinate_round_trip(group)
    if group.character_count != _euler_phi(modulus):
        raise RuntimeError("character count failed its phi(modulus) identity")
    return group
