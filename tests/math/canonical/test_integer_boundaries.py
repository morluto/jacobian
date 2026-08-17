from __future__ import annotations

from jacobian.math.combinatorics._models import IntegerListRequest
from jacobian.math.geometry.projective.values import PrimitiveProjectiveTriple
from jacobian.math.matrices.certified_snf.values import (
    CertifiedIntegerMatrix,
    SmithNormalFormCertificate,
)


def test_nonnegative_integer_list_accepts_canonical_values_beyond_python_limit() -> (
    None
):
    value = "1" + ("0" * 5_000)

    request = IntegerListRequest(values=(value,))

    assert request.values == (value,)


def test_smith_certificate_validates_large_canonical_invariant_factor() -> None:
    factor = "1" + ("0" * 5_000)
    source = CertifiedIntegerMatrix(row_count=1, column_count=1, entries=(("0",),))
    diagonal = CertifiedIntegerMatrix(
        row_count=1,
        column_count=1,
        entries=((factor,),),
    )
    identity = CertifiedIntegerMatrix(row_count=1, column_count=1, entries=(("1",),))

    certificate = SmithNormalFormCertificate(
        source=source,
        diagonal=diagonal,
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
