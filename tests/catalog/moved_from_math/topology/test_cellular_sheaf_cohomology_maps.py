"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/topology/test_cellular_sheaf_cohomology_maps.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.cellular_sheaves._models import (
    CoverRestrictionMatrix,
    FiniteCellularSheaf,
    SheafField,
    SheafStalk,
)
from jacobian.math.topology.cellular_sheaves.operations import from_cover_maps


def _q(value: int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _circle_sheaf(
    field: SheafField = SheafField.RATIONAL, prime: int | None = None
) -> FiniteCellularSheaf:
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


def _single_vertex_sheaf() -> FiniteCellularSheaf:
    complex_ = canonical_complex(("a",), (("a",),))
    result = from_cover_maps(
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
    assert result.sheaf is not None
    return result.sheaf


def test_catalog_exposes_the_induced_cohomology_map() -> None:
    assert "cellular_sheaf.morphism.cohomology_map.compute" in {
        tool.operation_id for tool in BUILTIN_TOOLS
    }
