"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/topology/test_cellular_sheaf_hodge.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.topology.cellular_sheaves import (
    SheafField,
    SheafStalk,
    from_cover_maps,
)
from jacobian.math.topology.cellular_sheaves._models import CoverRestrictionMatrix


def _q(value: str | int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _constant_sheaf(complex_, scalar=None, *, field=SheafField.RATIONAL, prime=None):
    if scalar is None:
        scalar = 1 if field is SheafField.PRIME_FIELD else _q(1)
    cells = tuple(face for group in complex_.faces_by_dimension for face in group.faces)
    covers = tuple(
        (face, coface)
        for coface in cells
        for face in cells
        if len(coface) == len(face) + 1 and set(face) < set(coface)
    )
    result = from_cover_maps(
        complex_,
        field,
        prime,
        tuple(SheafStalk(simplex=cell, basis=("e",)) for cell in cells),
        tuple(
            CoverRestrictionMatrix(source=a, target=b, entries=((scalar,),))
            for a, b in covers
        ),
    )
    assert result.sheaf is not None
    return result.sheaf


def _matrix(entries):
    return tuple(
        tuple(Fraction(value.num, value.den) for value in row) for row in entries
    )


def test_hodge_operation_is_published() -> None:
    assert "cellular_sheaf.hodge_laplacians.compute" in {
        tool.operation_id for tool in BUILTIN_TOOLS
    }
