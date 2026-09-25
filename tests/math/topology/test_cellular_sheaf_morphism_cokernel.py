"""Exact pointwise quotient sheaves of cellular-sheaf morphisms."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.cellular_sheaves import (
    SheafField,
    SheafMorphismCokernelRequest,
    SheafMorphismCokernelResult,
    SheafMorphismResult,
    SheafStalk,
    cokernel_of_morphism,
    from_cover_maps,
    morphism,
)
from jacobian.math.topology.cellular_sheaves._models import CoverRestrictionMatrix


def _q(value: int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _constant_sheaf(rank: int, field: SheafField = SheafField.RATIONAL, prime=None):
    complex_ = canonical_complex(("a", "b"), (("a", "b"),))
    faces = tuple(face for group in complex_.faces_by_dimension for face in group.faces)
    covers = tuple(
        (face, coface)
        for coface in faces
        for face in faces
        if len(coface) == len(face) + 1 and set(face) < set(coface)
    )
    identity = tuple(
        tuple(
            (int(i == j) if field is SheafField.PRIME_FIELD else _q(int(i == j)))
            for i in range(rank)
        )
        for j in range(rank)
    )
    result = from_cover_maps(
        complex_,
        field,
        prime,
        tuple(
            SheafStalk(simplex=face, basis=tuple(f"x{i}" for i in range(rank)))
            for face in faces
        ),
        tuple(
            CoverRestrictionMatrix(source=face, target=coface, entries=identity)
            for face, coface in covers
        ),
    )
    assert result.sheaf is not None
    return result.sheaf


def _point_sheaf(rank: int, field: SheafField = SheafField.RATIONAL, prime=None):
    complex_ = canonical_complex(("a",), (("a",),))
    result = from_cover_maps(
        complex_,
        field,
        prime,
        (SheafStalk(simplex=("a",), basis=tuple(f"x{i}" for i in range(rank))),),
        (),
    )
    assert result.sheaf is not None
    return result.sheaf


def test_cokernel_projection_and_induced_restrictions_match_matrix_oracle():
    source = _constant_sheaf(1)
    target = _constant_sheaf(2)
    components = tuple(
        (cell, ((_q(1),), (_q(0),))) for cell in source.canonical_face_order
    )
    original = morphism(source, target, components)

    result = cokernel_of_morphism(original)

    assert tuple(len(stalk.basis) for stalk in result.cokernel.stalks) == (1, 1, 1)
    assert all(
        matrix == ((_q(0), _q(1)),)
        for _cell, matrix in result.projection.components
    )
    # Independent quotient check: q*phi=0, q(e_2)=1 and identity restrictions
    # induce the identity map on each one-dimensional quotient stalk.
    projection = dict(result.projection.components)
    lifts = {lift.simplex: lift.entries for lift in result.stalk_lifts}
    for cell, phi in original.components:
        q_matrix = projection[cell]
        lift = lifts[cell]
        assert tuple(
            sum(
                Fraction(q.num, q.den) * Fraction(lift[i][j].num, lift[i][j].den)
                for i, q in enumerate(q_matrix[0])
            )
            for j in range(1)
        ) == (Fraction(1),)
        assert tuple(
            sum(
                Fraction(q.num, q.den) * Fraction(phi[i][j].num, phi[i][j].den)
                for i, q in enumerate(q_matrix[0])
            )
            for j in range(1)
        ) == (Fraction(0),)
    assert all(
        restriction.entries == ((_q(1),),)
        for restriction in result.cokernel.cover_restrictions
    )
    assert SheafMorphismCokernelResult.model_validate_json(
        result.model_dump_json()
    ) == result
    tool = next(
        item
        for item in BUILTIN_TOOLS
        if item.operation_id == "cellular_sheaf.morphism.cokernel.compute"
    )
    assert tool.run(SheafMorphismCokernelRequest(morphism=original)) == result

    components = list(original.components)
    components[0] = (components[0][0], ((_q(0),), (_q(0),)))
    forged = SheafMorphismResult.model_construct(
        source=source,
        target=target,
        components=tuple(components),
        natural=True,
        obstruction=None,
    )
    with pytest.raises(OperationDomainValidationError, match="natural"):
        cokernel_of_morphism(forged)


def test_surjective_map_has_zero_quotient_axes_and_roundtrips():
    source = _point_sheaf(2)
    target = _point_sheaf(1)
    original = morphism(
        source,
        target,
        tuple((cell, ((_q(1), _q(0)),)) for cell in source.canonical_face_order),
    )
    result = cokernel_of_morphism(original)
    assert all(stalk.basis == () for stalk in result.cokernel.stalks)
    assert all(restriction.entries == () for restriction in result.cokernel.cover_restrictions)
    assert all(matrix == () for _, matrix in result.projection.components)
    assert SheafMorphismCokernelResult.model_validate_json(result.model_dump_json()) == result


def test_prime_field_cokernel_uses_the_declared_field():
    source = _point_sheaf(2, SheafField.PRIME_FIELD, 3)
    target = _point_sheaf(2, SheafField.PRIME_FIELD, 3)
    original = morphism(
        source,
        target,
        tuple((cell, ((1, 1), (2, 2))) for cell in source.canonical_face_order),
    )
    result = cokernel_of_morphism(original)
    assert result.cokernel.coefficient_field is SheafField.PRIME_FIELD
    assert result.cokernel.prime == 3
    assert all(len(stalk.basis) == 1 for stalk in result.cokernel.stalks)
