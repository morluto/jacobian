from fractions import Fraction
from itertools import product

import pytest

from jacobian.canonical import encode_strict_json
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.groups.characters import (
    FiniteAbelianCharacterTableRequest,
    finite_abelian_character_table,
)
from jacobian.math.groups.characters._cyclotomic import (
    conjugate_value,
    multiply_values,
    value_from_power,
)
from jacobian.math.groups.characters._models import CyclotomicValue
from jacobian.math.groups.finite_abelian import FiniteAbelianProductGroup


def _fractions(value: CyclotomicValue) -> tuple[Fraction, ...]:
    return tuple(coefficient.as_fraction() for coefficient in value.coefficients)


@pytest.mark.parametrize("moduli", [(4,), (2, 2), (2, 4), ()])
def test_finite_abelian_character_table_is_exact_and_row_orthogonal(
    moduli: tuple[int, ...],
) -> None:
    group = FiniteAbelianProductGroup(moduli=moduli)
    result = finite_abelian_character_table(
        FiniteAbelianCharacterTableRequest(group=group)
    )
    expected_elements = tuple(product(*(range(modulus) for modulus in moduli)))
    assert result.elements == (expected_elements if moduli else ((),))
    exponent = result.cyclotomic_order

    # Independent direct evaluation of each dual pairing verifies every cell.
    for row in result.rows:
        for element, returned in zip(result.elements, row.values, strict=True):
            power = sum(
                row.frequency[index] * element[index] * (exponent // modulus)
                for index, modulus in enumerate(moduli)
            )
            assert _fractions(returned) == value_from_power(exponent, power)

    # Exact character orthogonality from the returned table: sum chi*conj(psi).
    for left_index, left in enumerate(result.rows):
        for right_index, right in enumerate(result.rows):
            total = (Fraction(0),) * len(left.values[0].coefficients)
            for left_value, right_value in zip(left.values, right.values, strict=True):
                product_value = multiply_values(
                    exponent,
                    _fractions(left_value),
                    conjugate_value(exponent, _fractions(right_value)),
                )
                total = tuple(a + b for a, b in zip(total, product_value, strict=True))
            target = (Fraction(group.order),) + (Fraction(0),) * (len(total) - 1)
            expected = (
                target if left_index == right_index else (Fraction(0),) * len(total)
            )
            assert total == expected


@pytest.mark.parametrize(
    ("moduli", "message"),
    [
        ((3, 3, 3, 3, 3, 3), "admit order"),
        ((61,), "exponent"),
    ],
)
def test_finite_abelian_character_table_admits_before_materializing(
    moduli: tuple[int, ...], message: str
) -> None:
    group = FiniteAbelianProductGroup(moduli=moduli)
    with pytest.raises(OperationResourceAdmissionError, match=message):
        finite_abelian_character_table(FiniteAbelianCharacterTableRequest(group=group))


def test_table_admits_seven_coordinates_when_output_fits() -> None:
    group = FiniteAbelianProductGroup(moduli=(2,) * 7)
    result = finite_abelian_character_table(
        FiniteAbelianCharacterTableRequest(group=group)
    )
    assert len(result.elements) == len(result.rows) == 128


def test_catalog_manifest_publishes_executable_character_table() -> None:
    operation_id = "finite_abelian_group.character_table.compute"
    tool = next(tool for tool in BUILTIN_TOOLS if tool.operation_id == operation_id)
    request = tool.request_type.model_validate_json(
        encode_strict_json(tool.examples[0].input), strict=True
    )
    result = tool.run(request)

    assert result.group.moduli == (2, 2)
    assert len(result.elements) == len(result.rows) == 4
