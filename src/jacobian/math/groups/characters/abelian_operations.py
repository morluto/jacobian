"""Exact character tables for bounded finite Abelian products."""

from __future__ import annotations

from itertools import product
from math import lcm

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.groups.characters._abelian_models import (
    MAX_ABELIAN_CHARACTER_TABLE_CELLS,
    MAX_ABELIAN_CHARACTER_TABLE_ORDER,
    FiniteAbelianCharacterRow,
    FiniteAbelianCharacterTableRequest,
    FiniteAbelianCharacterTableResult,
)
from jacobian.math.groups.characters._cyclotomic import (
    MAX_ARITHMETIC_ORDER,
    value_from_power,
)
from jacobian.math.groups.characters._models import CyclotomicValue


def finite_abelian_character_table(
    request: FiniteAbelianCharacterTableRequest,
) -> FiniteAbelianCharacterTableResult:
    """Return the complete exact Fourier table of an admitted product group."""
    group = request.group
    order = group.order
    exponent = lcm(*group.moduli)
    if order > MAX_ABELIAN_CHARACTER_TABLE_ORDER:
        raise OperationResourceAdmissionError(
            location=("group",),
            code="groups.characters.abelian.group_order_exceeds_envelope",
            message=f"complete Abelian character tables admit order at most {MAX_ABELIAN_CHARACTER_TABLE_ORDER}",
        )
    if exponent > MAX_ARITHMETIC_ORDER:
        raise OperationResourceAdmissionError(
            location=("group", "moduli"),
            code="groups.characters.abelian.cyclotomic_order_exceeds_envelope",
            message=f"exact cyclotomic character values admit exponent at most {MAX_ARITHMETIC_ORDER}",
        )
    cells = order * order
    coefficient_cells = cells * len(value_from_power(exponent, 0))
    if cells > MAX_ABELIAN_CHARACTER_TABLE_CELLS or coefficient_cells > 655_360:
        raise OperationResourceAdmissionError(
            location=("group",),
            code="groups.characters.abelian.table_output_exceeds_envelope",
            message="complete Abelian character table exceeds its exact output envelope",
        )

    moduli = group.moduli
    elements = tuple(product(*(range(modulus) for modulus in moduli)))
    if not moduli:
        elements = ((),)
    # Admission covers every n^2 output cell before any exact entry is made.
    power_values: dict[int, CyclotomicValue] = {}
    rows: list[FiniteAbelianCharacterRow] = []
    for frequency in elements:
        values: list[CyclotomicValue] = []
        for element in elements:
            numerator = sum(
                frequency[index] * element[index] * (exponent // modulus)
                for index, modulus in enumerate(moduli)
            )
            power = numerator % exponent
            value = power_values.get(power)
            if value is None:
                coefficients = value_from_power(exponent, power)
                value = CyclotomicValue._from_kernel(
                    order=exponent,
                    coefficients=tuple(
                        CanonicalRational.from_fraction(coefficient)
                        for coefficient in coefficients
                    ),
                )
                power_values[power] = value
            values.append(value)
        rows.append(
            FiniteAbelianCharacterRow(frequency=frequency, values=tuple(values))
        )
    return FiniteAbelianCharacterTableResult(
        group=group,
        elements=elements,
        rows=tuple(rows),
        cyclotomic_order=exponent,
    )


__all__ = ["finite_abelian_character_table"]
