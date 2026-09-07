from __future__ import annotations

from jacobian.math.geometry.projective.values import PrimitiveProjectiveTriple
from jacobian.math.matrices.certified_snf.values import (
    SmithNormalFormCertificate,
)
from jacobian.math.matrices.values import IntegerMatrix


def test_smith_certificate_validates_large_canonical_invariant_factor() -> None:
    factor = 10**5000
    source = IntegerMatrix(
        row_count=1,
        column_count=1,
        entries=((factor,),),
    )
    identity = IntegerMatrix(row_count=1, column_count=1, entries=((1,),))

    certificate = SmithNormalFormCertificate(
        source=source,
        diagonal=source,
        left_transformation=identity,
        right_transformation=identity,
        rank=1,
        invariant_factors=(factor,),
        left_determinant=1,
        right_determinant=1,
    )

    assert certificate.invariant_factors == (factor,)


def test_primitive_projective_triple_accepts_large_canonical_coordinate() -> None:
    coordinate = 10**5000

    triple = PrimitiveProjectiveTriple(coordinates=(1, coordinate, 0))

    assert triple.coordinates == (1, coordinate, 0)
