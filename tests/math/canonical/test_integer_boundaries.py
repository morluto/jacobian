from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from jacobian._exact import NativeInteger
from jacobian.math.geometry.projective.values import PrimitiveProjectiveTriple
from jacobian.math.matrices.certified_snf.values import (
    SmithNormalFormCertificate,
)
from jacobian.math.matrices.values import IntegerMatrix


def test_smith_certificate_validates_large_canonical_invariant_factor() -> None:
    factor = "1" + ("0" * 5_000)
    source = IntegerMatrix(
        row_count=1,
        column_count=1,
        entries=((factor,),),
    )
    identity = IntegerMatrix(row_count=1, column_count=1, entries=(("1",),))

    certificate = SmithNormalFormCertificate(
        source=source,
        diagonal=source,
        left_transformation=identity,
        right_transformation=identity,
        rank=1,
        invariant_factors=(factor,),
        left_determinant="1",
        right_determinant="1",
    )

    assert certificate.invariant_factors == (factor,)


def test_primitive_projective_triple_accepts_large_canonical_coordinate() -> None:
    coordinate = "1" + ("0" * 5_000)

    triple = PrimitiveProjectiveTriple(coordinates=("1", coordinate, "0"))

    assert triple.coordinates == ("1", coordinate, "0")


@pytest.mark.parametrize("value", [True, 1.5, None, [], {}])
def test_native_integer_rejects_invalid_python_values_structurally(
    value: object,
) -> None:
    with pytest.raises(ValidationError) as error:
        TypeAdapter(NativeInteger).validate_python(value)

    assert error.value.errors()[0]["type"] == "canonical_integer.type"


def test_native_integer_round_trips_decimal_json_as_python_int() -> None:
    adapter = TypeAdapter(NativeInteger)

    assert adapter.validate_json('"-42"') == -42
    assert adapter.dump_json(-42) == b'"-42"'
