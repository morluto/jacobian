"""Bounded exact Cohen--Oesterle dimensions for conductor-thirteen characters."""

from __future__ import annotations

from fractions import Fraction
from math import gcd

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.cyclic_linear._models import (
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.characters.operations import (
    dirichlet_character_value,
)
from jacobian.math.number_theory.characters.values import (
    CyclotomicValue,
    DirichletCharacter,
)
from jacobian.math.number_theory.modular_forms import cyclotomic
from jacobian.math.number_theory.modular_forms.cyclotomic import _reduce


def _element(
    field: RationalCyclotomicField, coordinates: tuple[Fraction, ...]
) -> RationalCyclotomicElement:
    return RationalCyclotomicElement(
        field=field,
        coefficients_ascending=tuple(
            CanonicalRational(num=item.numerator, den=item.denominator)
            for item in coordinates
        ),
    )


def _root_in_field(
    value: CyclotomicValue, field: RationalCyclotomicField
) -> RationalCyclotomicElement:
    embedded_exponent = Fraction(value.exponent * field.order, value.order)
    if embedded_exponent.denominator != 1:
        raise ValueError("character root is outside the declared coefficient field")
    # Map ζ_value.order to ζ_field.order ** embedded_exponent, then reduce
    # in the declared field rather than relabelling the source power.
    exponent = embedded_exponent.numerator
    power = [Fraction(0)] * (exponent + 1)
    power[exponent] = Fraction(1)
    return _element(field, _reduce(power, field))


def _validated_character_table(
    level: int, character: DirichletCharacter
) -> tuple[dict[int, CyclotomicValue], int]:
    values = {
        residue: dirichlet_character_value(character, residue).value
        for residue in range(level)
        if gcd(residue, level) == 1
    }
    if any(value is None for value in values.values()):
        raise RuntimeError("character table omitted a unit value")
    table = {residue: value for residue, value in values.items() if value is not None}
    value_order = 1
    for value in table.values():
        root_order = value.order // gcd(value.order, value.exponent)
        value_order = value_order * root_order // gcd(value_order, root_order)
    if value_order > 6 or 6 % value_order:
        raise OperationDomainValidationError(
            location=("space", "character"),
            code="modular_form.character_dimension_order",
            message=(
                "the bounded character dimension slice requires character values "
                "in Q(zeta_6)"
            ),
        )
    if table[(-1) % level].exponent != 0:
        raise OperationDomainValidationError(
            location=("space", "character"),
            code="modular_form.character_dimension_parity",
            message="the bounded character dimension slice requires even parity",
        )
    conductor = None
    for candidate in range(1, level + 1):
        if level % candidate == 0 and all(
            left == right
            for left_residue, left in table.items()
            for right_residue, right in table.items()
            if left_residue % candidate == right_residue % candidate
        ):
            conductor = candidate
            break
    if conductor != 13:
        raise OperationDomainValidationError(
            location=("space", "character"),
            code="modular_form.character_dimension_conductor",
            message="the bounded character dimension slice requires conductor 13",
        )
    return table, conductor


def _character_sum(
    level: int,
    polynomial: tuple[int, ...],
    values: dict[int, CyclotomicValue],
    field: RationalCyclotomicField,
) -> RationalCyclotomicElement:
    total = _element(field, (Fraction(0),) * field.degree)
    for residue in range(level):
        if (
            gcd(residue, level) == 1
            and sum(
                coefficient * residue**power
                for power, coefficient in enumerate(polynomial)
            )
            % level
            == 0
        ):
            total = cyclotomic.add(total, _root_in_field(values[residue], field))
    return total


def _nu_infinity(level: int, conductor: int) -> int:
    result = 0
    for divisor in range(1, level + 1):
        if level % divisor:
            continue
        common = gcd(divisor, level // divisor)
        if (level // conductor) % common == 0:
            result += sum(gcd(residue, common) == 1 for residue in range(common))
    return result


def _integral_rational_part(value: RationalCyclotomicElement) -> int:
    rational = tuple(
        Fraction(int(item.num), int(item.den)) for item in value.coefficients_ascending
    )
    if any(rational[1:]) or rational[0].denominator != 1:
        raise RuntimeError("Cohen--Oesterle dimension expression is not integral")
    return rational[0].numerator


def character_space_dimensions(
    level: int,
    weight: int,
    character: DirichletCharacter,
    field: RationalCyclotomicField,
) -> tuple[int, int]:
    """Return (cusp, full) dimensions for the explicit conductor-13 slice.

    Admission is intentionally limited to weight two and levels 13, 26, 39,
    with an even character of conductor 13 and values in Q(zeta_6). The
    Cohen--Oesterle character sums are evaluated directly in that exact field;
    no backend dimension claim is used as the mathematical expectation.
    """
    if (
        type(level) is not int
        or level not in (13, 26, 39)
        or weight != 2
        or type(character) is not DirichletCharacter
        or character.group.modulus != level
        or field.order != 6
    ):
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.character_dimension_unsupported",
            message="character dimensions support weight 2, levels 13/26/39, and Q(zeta_6)",
        )
    values, conductor = _validated_character_table(level, character)

    nu0 = level
    for prime in (2, 3, 13):
        if level % prime == 0:
            nu0 = nu0 * (prime + 1) // prime
    nu_infinity = _nu_infinity(level, conductor)
    nu2 = _character_sum(level, (1, 0, 1), values, field)
    nu3 = _character_sum(level, (1, 1, 1), values, field)
    gamma4 = Fraction(-1, 4) if weight % 4 == 2 else Fraction(1, 4)
    gamma3 = {
        0: Fraction(1, 3),
        1: Fraction(0),
        2: Fraction(-1, 3),
    }[weight % 3]
    total = cyclotomic.add(
        _element(
            field,
            (Fraction((weight - 1) * nu0, 12),) + (Fraction(0),) * (field.degree - 1),
        ),
        cyclotomic.add(
            cyclotomic.multiply(
                _element(field, (gamma4,) + (Fraction(0),) * (field.degree - 1)), nu2
            ),
            cyclotomic.multiply(
                _element(field, (gamma3,) + (Fraction(0),) * (field.degree - 1)), nu3
            ),
        ),
    )
    total = cyclotomic.subtract(
        total,
        _element(
            field, (Fraction(nu_infinity, 2),) + (Fraction(0),) * (field.degree - 1)
        ),
    )
    cusp = _integral_rational_part(total)
    if cusp < 0:
        raise RuntimeError("Cohen--Oesterle formula returned a negative dimension")
    return cusp, cusp + nu_infinity


__all__ = ["character_space_dimensions"]
