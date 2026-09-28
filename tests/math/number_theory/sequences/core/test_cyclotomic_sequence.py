"""Canonical finite sequences over one rational cyclotomic field."""

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.matrices.cyclic_linear._models import (
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.sequences.core import FiniteCyclotomicSequence


def _element(
    field: RationalCyclotomicField, *coordinates: int
) -> RationalCyclotomicElement:
    return RationalCyclotomicElement(
        field=field,
        coefficients_ascending=tuple(
            CanonicalRational(num=coordinate, den=1) for coordinate in coordinates
        ),
    )


def test_cyclotomic_sequence_preserves_parent_origin_and_empty_values() -> None:
    field = RationalCyclotomicField(order=3)
    source = FiniteCyclotomicSequence(
        index_origin=-2,
        field=field,
        values=(_element(field, 1, 0), _element(field, 0, 1)),
    )
    assert source.index_origin == -2
    assert source.values[1].field == field
    assert FiniteCyclotomicSequence(index_origin=0, field=field, values=()).values == ()
    assert type(source).model_validate_json(source.model_dump_json()) == source


def test_cyclotomic_sequence_rejects_coefficients_from_another_parent() -> None:
    field = RationalCyclotomicField(order=3)
    other_field = RationalCyclotomicField(order=4)
    with pytest.raises(ValidationError, match="cyclotomic_parent_mismatch"):
        FiniteCyclotomicSequence(
            index_origin=0,
            field=field,
            values=(_element(other_field, 1, 0),),
        )


def test_cyclotomic_sequence_requires_strict_integer_origin() -> None:
    field = RationalCyclotomicField(order=1)
    with pytest.raises(ValidationError):
        FiniteCyclotomicSequence.model_validate(
            {"index_origin": True, "field": field.model_dump(), "values": []}
        )
