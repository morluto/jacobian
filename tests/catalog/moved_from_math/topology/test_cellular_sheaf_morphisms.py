"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/topology/test_cellular_sheaf_morphisms.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.cellular_sheaves import (
    SheafField,
    SheafMorphismResult,
    SheafStalk,
    from_cover_maps,
    morphism,
)
from jacobian.math.topology.cellular_sheaves._models import (
    CoverRestrictionMatrix,
)
from jacobian.math.topology.cellular_sheaves.extensions import (
    SheafCochainMapRequest,
    SheafMorphismComposeRequest,
    cochain_map,
    compose_morphisms,
)


def _q(value: str | int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _triangle_sheaf(
    edge_scalar: CanonicalRational | None = None,
    rank: int = 1,
    all_cover_scalar: bool = False,
):
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
                        edge_scalar if all_cover_scalar or len(coface) == 2 else _q("1")
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
