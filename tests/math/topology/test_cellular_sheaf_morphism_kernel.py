"""Exact kernel sheaves and their natural inclusions."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.cellular_sheaves import (
    SheafField,
    SheafMorphismKernelRequest,
    SheafMorphismKernelResult,
    SheafMorphismResult,
    SheafStalk,
    from_cover_maps,
    kernel_of_morphism,
    morphism,
)
from jacobian.math.topology.cellular_sheaves._models import CoverRestrictionMatrix
from jacobian.math.topology.cellular_sheaves.extensions import cochain_map


def _q(value: int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _sheaf(rank: int = 2, field: SheafField = SheafField.RATIONAL, prime=None):
    complex_ = canonical_complex(("a", "b"), (("a", "b"),))
    faces = tuple(face for group in complex_.faces_by_dimension for face in group.faces)
    covers = tuple(
        (face, coface)
        for coface in faces
        for face in faces
        if len(coface) == len(face) + 1 and set(face) < set(coface)
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
            CoverRestrictionMatrix(
                source=face,
                target=coface,
                entries=tuple(
                    tuple(
                        (1 if i == j else 0)
                        if field is SheafField.PRIME_FIELD
                        else _q(1 if i == j else 0)
                        for i in range(rank)
                    )
                    for j in range(rank)
                ),
            )
            for face, coface in covers
        ),
    )
    assert result.sheaf is not None
    return result.sheaf


def _triangle_sheaf():
    complex_ = canonical_complex(("a", "b", "c"), (("a", "b", "c"),))
    faces = tuple(face for group in complex_.faces_by_dimension for face in group.faces)
    covers = tuple(
        (face, coface)
        for coface in faces
        for face in faces
        if len(coface) == len(face) + 1 and set(face) < set(coface)
    )
    result = from_cover_maps(
        complex_,
        SheafField.RATIONAL,
        None,
        tuple(SheafStalk(simplex=face, basis=("x",)) for face in faces),
        tuple(
            CoverRestrictionMatrix(
                source=face,
                target=coface,
                entries=((_q(1),),),
            )
            for face, coface in covers
        ),
    )
    assert result.sheaf is not None
    return result.sheaf


def test_kernel_is_natural_inclusion_and_survives_json_roundtrip() -> None:
    source, target = _sheaf(), _sheaf(1)
    components = tuple(
        (cell, ((_q(1), _q(0)),)) for cell in source.canonical_face_order
    )
    f = morphism(source, target, components)
    result = kernel_of_morphism(f)
    assert tuple(len(stalk.basis) for stalk in result.kernel.stalks) == (1, 1, 1)
    assert result.inclusion.natural
    assert result.inclusion.target == source
    assert result.inclusion.components[0][1] == ((_q(0),), (_q(1),))
    assert all(item.entries == ((_q(1),),) for item in result.kernel.cover_restrictions)
    restored = SheafMorphismKernelResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert cochain_map(restored.inclusion).morphism == restored.inclusion
    operation = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "cellular_sheaf.morphism.kernel.compute"
    )
    assert operation.run(SheafMorphismKernelRequest(morphism=f)) == result


def test_empty_stalk_kernel_and_zero_restriction_axes_roundtrip() -> None:
    source, target = _sheaf(1), _sheaf(1)
    components = tuple((cell, ((_q(1),),)) for cell in source.canonical_face_order)
    result = kernel_of_morphism(morphism(source, target, components))
    assert all(stalk.basis == () for stalk in result.kernel.stalks)
    assert all(item.entries == () for item in result.kernel.cover_restrictions)
    restored = SheafMorphismKernelResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.kernel.canonical_face_order == source.canonical_face_order
    assert restored.kernel.coefficient_field is source.coefficient_field


def test_kernel_rechecks_a_forged_natural_flag() -> None:
    source, target = _sheaf(), _sheaf()
    components = tuple(
        (cell, ((_q(1), _q(0)), (_q(0), _q(1)))) for cell in source.canonical_face_order
    )
    natural = morphism(source, target, components)
    forged = natural.model_copy(update={"natural": False, "obstruction": "forged"})
    # Caller flags do not alter the mathematical morphism; the map is re-established.
    result = kernel_of_morphism(forged)
    assert result.morphism.natural


def test_kernel_uses_the_declared_prime_field() -> None:
    source, target = (
        _sheaf(field=SheafField.PRIME_FIELD, prime=3),
        _sheaf(1, field=SheafField.PRIME_FIELD, prime=3),
    )
    components = tuple((cell, ((1, 0),)) for cell in source.canonical_face_order)
    result = kernel_of_morphism(morphism(source, target, components))
    assert result.kernel.coefficient_field is SheafField.PRIME_FIELD
    assert result.kernel.prime == 3
    assert result.inclusion.components[0][1] == ((0,), (1,))


@pytest.mark.parametrize("forged_role", ("source", "target"))
def test_kernel_rechecks_parent_derived_restrictions(forged_role: str) -> None:
    source, target = _triangle_sheaf(), _triangle_sheaf()
    valid_source, valid_target = source, target
    forged_parent = source if forged_role == "source" else target
    derived = list(forged_parent.derived_restrictions)
    candidate_index = next(
        index
        for index, restriction in enumerate(derived)
        if len(restriction.target) - len(restriction.source) == 2
    )
    derived[candidate_index] = derived[candidate_index].model_copy(
        update={"entries": ((_q(0),),)}
    )
    forged_parent = forged_parent.model_copy(
        update={"derived_restrictions": tuple(derived)}
    )
    if forged_role == "source":
        source = forged_parent
    else:
        target = forged_parent

    components = tuple((cell, ((_q(1),),)) for cell in source.canonical_face_order)
    candidate = morphism(source, target, components)
    assert candidate.natural
    assert valid_source != source if forged_role == "source" else valid_target != target
    decoded = SheafMorphismResult.model_validate_json(candidate.model_dump_json())
    assert decoded.source == source
    assert decoded.target == target
    with pytest.raises(OperationDomainValidationError, match="functor diagram"):
        kernel_of_morphism(decoded)
