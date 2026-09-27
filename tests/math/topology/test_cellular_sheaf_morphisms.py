"""Independent incidence-square checks and composition for sheaf maps."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.cellular_sheaves import (
    FiniteCellularSheaf,
    SheafCochainCoordinate,
    SheafCochainMapRequest,
    SheafCochainMapResult,
    SheafField,
    SheafMorphismResult,
    SheafStalk,
    from_cover_maps,
    morphism,
)
from jacobian.math.topology.cellular_sheaves._models import (
    CoverRestrictionMatrix,
    sheaf_scalar_json_bound,
)
from jacobian.math.topology.cellular_sheaves.extensions import (
    SheafMorphismComposeRequest,
    SheafMorphismRequest,
    cochain_map,
    compose_morphisms,
)


def _q(value: str | int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _triangle_sheaf(edge_scalar: CanonicalRational | None = None, rank: int = 1):
    if edge_scalar is None:
        edge_scalar = _q(1)
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
        tuple(
            SheafStalk(simplex=face, basis=tuple(f"x{index}" for index in range(rank)))
            for face in faces
        ),
        tuple(
            CoverRestrictionMatrix(
                source=face,
                target=coface,
                entries=(
                    tuple(
                        edge_scalar if len(coface) == 2 else _q("1")
                        for _ in range(rank)
                    ),
                )
                * rank,
            )
            for face, coface in covers
        ),
    )
    assert result.sheaf is not None
    return result.sheaf


def _independent_square_oracle(source, target, components) -> bool:
    """Check each incidence square with a separate Fraction implementation."""
    phi = {
        tuple(key): Fraction(matrix[0][0].num, matrix[0][0].den)
        for key, matrix in components
    }
    rho_f = {
        (item.source, item.target): Fraction(
            item.entries[0][0].num, item.entries[0][0].den
        )
        for item in source.cover_restrictions
    }
    rho_g = {
        (item.source, item.target): Fraction(
            item.entries[0][0].num, item.entries[0][0].den
        )
        for item in target.cover_restrictions
    }
    return all(
        rho_g[(face, coface)] * phi[face] == phi[coface] * rho_f[(face, coface)]
        for face, coface in rho_f
    )


def test_triangle_morphism_naturality_matches_independent_incidence_oracle() -> None:
    source = _triangle_sheaf()
    target = _triangle_sheaf()
    axis = source.canonical_face_order
    valid = tuple((face, ((_q("2"),),)) for face in axis)
    result = morphism(source, target, valid)
    assert result.natural
    assert _independent_square_oracle(source, target, result.components)

    corrupted = tuple(
        (face, ((_q("3") if face == ("a", "b") else _q("2"),),)) for face in axis
    )
    bad = morphism(source, target, corrupted)
    assert not bad.natural
    assert not _independent_square_oracle(source, target, bad.components)


def test_serialized_morphisms_compose_pointwise_and_remain_source_bound() -> None:
    f, g, h = _triangle_sheaf(), _triangle_sheaf(), _triangle_sheaf()
    axis = f.canonical_face_order
    first = morphism(f, g, tuple((face, ((_q("2"),),)) for face in axis))
    second = morphism(g, h, tuple((face, ((_q("3"),),)) for face in axis))
    first_copy = SheafMorphismResult.model_validate_json(first.model_dump_json())
    second_copy = SheafMorphismResult.model_validate_json(second.model_dump_json())
    composed = compose_morphisms(first_copy, second_copy)
    assert composed.source == f
    assert composed.target == h
    assert composed.natural
    assert tuple(matrix for _, matrix in composed.components) == (((_q("6"),),),) * len(
        axis
    )

    tool = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "cellular_sheaf.morphism.compose"
    )
    request = SheafMorphismComposeRequest(first=first_copy, second=second_copy)
    tool_result = tool.run(request)
    assert tool_result == composed


def test_catalog_morphism_example_remains_discoverable() -> None:
    assert any(
        tool.operation_id == "cellular_sheaf.morphism.compute" for tool in BUILTIN_TOOLS
    )


def test_composition_preserves_zero_dimensional_middle_stalks() -> None:
    f, g, h = _triangle_sheaf(), _triangle_sheaf(rank=0), _triangle_sheaf()
    axis = f.canonical_face_order
    zero_components = tuple((face, ()) for face in axis)
    into_zero = morphism(f, g, zero_components)
    out_of_zero = morphism(g, h, tuple((face, ((),)) for face in axis))
    composed = compose_morphisms(into_zero, out_of_zero)
    assert tuple(matrix for _, matrix in composed.components) == (((_q("0"),),),) * len(
        axis
    )
    assert SheafMorphismRequest is not None


def test_natural_morphism_induces_axis_bound_cochain_matrices() -> None:
    sheaf = _triangle_sheaf()
    components = tuple((face, ((_q("2"),),)) for face in sheaf.canonical_face_order)
    sheaf_map = morphism(sheaf, sheaf, components)
    result = cochain_map(sheaf_map)
    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored == result
    assert tuple(
        tuple(len(axis) for axis in axes)
        for axes in (result.source_bases, result.target_bases)
    ) == (
        (3, 3, 1),
        (3, 3, 1),
    )
    assert result.components == (
        (
            (_q("2"), _q("0"), _q("0")),
            (_q("0"), _q("2"), _q("0")),
            (_q("0"), _q("0"), _q("2")),
        ),
        (
            (_q("2"), _q("0"), _q("0")),
            (_q("0"), _q("2"), _q("0")),
            (_q("0"), _q("0"), _q("2")),
        ),
        ((_q("2"),),),
    )
    request = SheafCochainMapRequest(morphism=sheaf_map)
    tool = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "cellular_sheaf.morphism.cochain_map"
    )
    assert tool.run(request) == result


def test_cochain_map_rejects_malformed_native_morphism_before_dereference() -> None:
    with pytest.raises(OperationDomainValidationError):
        cochain_map(object())  # type: ignore[arg-type]
    malformed = SheafMorphismResult.model_construct()
    with pytest.raises(OperationDomainValidationError):
        cochain_map(malformed)


def test_rational_json_bound_accounts_for_both_decimal_components() -> None:
    # Rational wire scalars serialize numerator and denominator separately.
    assert sheaf_scalar_json_bound(1, 64) >= 2 * 64 + 20


def test_cochain_map_consumer_rechecks_serialized_naturality_claim() -> None:
    sheaf = _triangle_sheaf()
    components = tuple(
        (face, ((_q("3") if face == ("a", "b") else _q("2"),),))
        for face in sheaf.canonical_face_order
    )
    invalid_claim = SheafMorphismResult(
        source=sheaf,
        target=sheaf,
        components=components,
        natural=True,
    )
    try:
        cochain_map(invalid_claim)
    except OperationDomainValidationError as error:
        assert "natural" in str(error)
    else:
        raise AssertionError("a false serialized naturality claim was consumed")


def test_cochain_map_structural_axes_survive_serialization_without_replay() -> None:
    sheaf = _triangle_sheaf()
    components = tuple((face, ((_q("2"),),)) for face in sheaf.canonical_face_order)
    result = cochain_map(morphism(sheaf, sheaf, components))
    assert SheafCochainMapResult.model_validate_json(result.model_dump_json()) == result


def test_cochain_map_json_rejects_malformed_model_construct_axes_and_shapes() -> None:
    sheaf = _triangle_sheaf()
    components = tuple((face, ((_q("2"),),)) for face in sheaf.canonical_face_order)
    result = cochain_map(morphism(sheaf, sheaf, components))
    malformed_values = (
        SheafCochainMapResult.model_construct(
            morphism=result.morphism,
            source_bases=result.source_bases[:-1],
            target_bases=result.target_bases,
            components=result.components,
        ),
        SheafCochainMapResult.model_construct(
            morphism=result.morphism,
            source_bases=((), *result.source_bases[1:]),
            target_bases=result.target_bases,
            components=result.components,
        ),
        SheafCochainMapResult.model_construct(
            morphism=result.morphism,
            source_bases=result.source_bases,
            target_bases=result.target_bases,
            components=(((_q("2"),),), *result.components[1:]),
        ),
    )
    for malformed in malformed_values:
        with pytest.raises(ValidationError):
            SheafCochainMapResult.model_validate_json(malformed.model_dump_json())

    oversized = CanonicalRational(num=10**64, den=1)
    oversized_morphism = SheafMorphismResult.model_construct(
        source=sheaf,
        target=sheaf,
        components=tuple(
            (face, ((oversized,),)) for face in sheaf.canonical_face_order
        ),
        natural=True,
        obstruction=None,
    )
    oversized_result = SheafCochainMapResult.model_construct(
        morphism=oversized_morphism,
        source_bases=result.source_bases,
        target_bases=result.target_bases,
        components=result.components,
    )
    with pytest.raises(ValidationError):
        SheafCochainMapResult.model_validate_json(oversized_result.model_dump_json())

    payload = json.loads(result.model_dump_json())
    payload["components"].pop()
    with pytest.raises(ValidationError):
        SheafCochainMapResult.model_validate_json(json.dumps(payload))


def test_cochain_map_json_rejects_overrank_parent_before_axis_expansion() -> None:
    complex_ = canonical_complex(("a",), (("a",),))
    source_stalk = SheafStalk(
        simplex=("a",), basis=tuple(f"x{index}" for index in range(600))
    )
    target_stalk = SheafStalk(simplex=("a",), basis=())
    source = FiniteCellularSheaf(
        complex=complex_,
        coefficient_field=SheafField.RATIONAL,
        prime=None,
        stalks=(source_stalk,),
        cover_restrictions=(),
        derived_restrictions=(),
        diamonds=0,
        comparable_pairs=0,
    )
    target = FiniteCellularSheaf(
        complex=complex_,
        coefficient_field=SheafField.RATIONAL,
        prime=None,
        stalks=(target_stalk,),
        cover_restrictions=(),
        derived_restrictions=(),
        diamonds=0,
        comparable_pairs=0,
    )
    forged_morphism = SheafMorphismResult.model_construct(
        source=source,
        target=target,
        components=((("a",), ()),),
        natural=True,
        obstruction=None,
    )
    forged_result = SheafCochainMapResult.model_construct(
        morphism=forged_morphism,
        source_bases=(
            tuple(
                SheafCochainCoordinate(simplex=("a",), basis_label=f"x{index}")
                for index in range(600)
            ),
        ),
        target_bases=((),),
        components=((),),
    )
    with pytest.raises(ValidationError):
        SheafCochainMapResult.model_validate_json(forged_result.model_dump_json())


def test_component_scalar_digit_bound_precedes_scalar_parsing() -> None:
    sheaf = _triangle_sheaf()
    huge = CanonicalRational(num=10**64, den=1)
    components = tuple((face, ((huge,),)) for face in sheaf.canonical_face_order)
    try:
        morphism(sheaf, sheaf, components)
    except OperationResourceAdmissionError as error:
        assert "64 digits" in str(error)
    else:
        raise AssertionError("an over-bound component scalar was admitted")
