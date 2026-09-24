"""Exact induced maps on cellular-sheaf cohomology."""

import json
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.cellular_sheaves._models import (
    CoverRestrictionMatrix,
    SheafField,
    SheafStalk,
)
from jacobian.math.topology.cellular_sheaves.cohomology_maps import (
    SheafCohomologyMapResult,
    cohomology_map,
)
from jacobian.math.topology.cellular_sheaves.extensions import (
    SheafMorphismResult,
    morphism,
)
from jacobian.math.topology.cellular_sheaves.operations import from_cover_maps


def _q(value: int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _circle_sheaf(field: SheafField = SheafField.RATIONAL, prime: int | None = None):
    complex_ = canonical_complex(("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c")))
    faces = tuple(face for group in complex_.faces_by_dimension for face in group.faces)
    covers = tuple(
        (face, edge)
        for edge in faces
        for face in faces
        if len(edge) == len(face) + 1 and set(face) < set(edge)
    )
    result = from_cover_maps(
        complex_,
        field,
        prime,
        tuple(SheafStalk(simplex=face, basis=("x",)) for face in faces),
        tuple(
            CoverRestrictionMatrix(
                source=face,
                target=edge,
                entries=(((_q(1),),) if field is SheafField.RATIONAL else ((1,),)),
            )
            for face, edge in covers
        ),
    )
    assert result.sheaf is not None
    return result.sheaf


def test_induced_map_on_circle_constant_sheaf_cohomology() -> None:
    sheaf = _circle_sheaf()
    components = tuple((face, ((_q(3),),)) for face in sheaf.canonical_face_order)
    # This claim is deliberately false: the consumer recomputes naturality
    # and derives the induced map from the exact stalk components.
    claimed = SheafMorphismResult(
        source=sheaf,
        target=sheaf,
        components=components,
        natural=False,
        obstruction="caller supplied stale claim",
    )

    result = cohomology_map(claimed)

    assert tuple(group.betti_number for group in result.source_groups) == (1, 1)
    assert tuple(group.betti_number for group in result.target_groups) == (1, 1)
    assert result.components == (((_q(3),),), ((_q(3),),))
    assert (
        SheafCohomologyMapResult.model_validate_json(result.model_dump_json()) == result
    )
    forged = json.loads(result.model_dump_json())
    forged["morphism"]["natural"] = False
    forged["morphism"]["obstruction"] = "forged non-naturality"
    with pytest.raises(ValueError, match="natural morphism"):
        SheafCohomologyMapResult.model_validate_json(json.dumps(forged))


def test_induced_map_uses_exact_prime_field_coordinates() -> None:
    sheaf = _circle_sheaf(SheafField.PRIME_FIELD, 2)
    components = tuple((face, ((1,),)) for face in sheaf.canonical_face_order)
    result = cohomology_map(morphism(sheaf, sheaf, components))

    assert result.components == (((1,),), ((1,),))
    assert (
        SheafCohomologyMapResult.model_validate_json(result.model_dump_json()) == result
    )


def test_identity_on_four_disconnected_sections_retains_identity_matrix() -> None:
    complex_ = canonical_complex(("a", "b", "c", "d"), (("a",), ("b",), ("c",), ("d",)))
    sheaf_result = from_cover_maps(
        complex_,
        SheafField.RATIONAL,
        None,
        tuple(
            SheafStalk(simplex=face, basis=("x",))
            for group in complex_.faces_by_dimension
            for face in group.faces
        ),
        (),
    )
    assert sheaf_result.sheaf is not None
    sheaf = sheaf_result.sheaf
    components = tuple((face, ((_q(1),),)) for face in sheaf.canonical_face_order)

    result = cohomology_map(morphism(sheaf, sheaf, components))

    assert result.source_groups[0].betti_number == 4
    assert result.components == (
        (
            (_q(1), _q(0), _q(0), _q(0)),
            (_q(0), _q(1), _q(0), _q(0)),
            (_q(0), _q(0), _q(1), _q(0)),
            (_q(0), _q(0), _q(0), _q(1)),
        ),
    )


def test_map_into_zero_stalks_preserves_empty_cohomology_axes() -> None:
    source = _circle_sheaf()
    target_result = from_cover_maps(
        source.complex,
        SheafField.RATIONAL,
        None,
        tuple(
            SheafStalk(simplex=face, basis=()) for face in source.canonical_face_order
        ),
        tuple(
            CoverRestrictionMatrix(source=face, target=edge, entries=())
            for edge in source.canonical_face_order
            for face in source.canonical_face_order
            if len(edge) == len(face) + 1 and set(face) < set(edge)
        ),
    )
    assert target_result.sheaf is not None
    target = target_result.sheaf
    zero_components = tuple((face, ()) for face in source.canonical_face_order)

    result = cohomology_map(morphism(source, target, zero_components))

    assert tuple(group.betti_number for group in result.target_groups) == (0, 0)
    assert result.components == ((), ())


def test_catalog_exposes_the_induced_cohomology_map() -> None:
    assert "cellular_sheaf.morphism.cohomology_map.compute" in {
        tool.operation_id for tool in BUILTIN_TOOLS
    }


def test_nonnatural_morphism_has_no_induced_cohomology_map() -> None:
    sheaf = _circle_sheaf()
    axis = sheaf.canonical_face_order
    components = tuple((face, ((_q(2 if face == ("a",) else 1),),)) for face in axis)
    nonnatural = morphism(sheaf, sheaf, components)
    assert not nonnatural.natural
    with pytest.raises(OperationDomainValidationError, match="natural"):
        cohomology_map(nonnatural)
