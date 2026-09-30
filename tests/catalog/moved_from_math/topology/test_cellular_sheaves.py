"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/topology/test_cellular_sheaves.py``. The preamble is carried over verbatim so the moved tests
resolve every name they use.
"""

from __future__ import annotations

import json
from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.topology._models import FiniteSimplicialComplex, canonical_complex
from jacobian.math.topology.cellular_sheaves import (
    FromCoverMapsResult,
    SheafField,
    SheafOutcome,
    SheafStalk,
    from_cover_maps,
)
from jacobian.math.topology.cellular_sheaves._models import (
    CoverRestrictionMatrix,
    FromCoverMapsRequest,
)

OPERATION_ID = "cellular_sheaf.from_cover_maps.compute"


def _q(value: str | int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _cells(complex_: FiniteSimplicialComplex) -> list[tuple[str, ...]]:
    return [face for group in complex_.faces_by_dimension for face in group.faces]


def _native(request: FromCoverMapsRequest) -> FromCoverMapsResult:
    return from_cover_maps(
        request.complex,
        request.coefficient_field,
        request.prime,
        request.stalks,
        request.cover_maps,
    )


def _covers(
    complex_: FiniteSimplicialComplex,
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    known = set(_cells(complex_))
    covers = []
    for coface in known:
        for position in range(len(coface)):
            face = coface[:position] + coface[position + 1 :]
            if face in known:
                covers.append((face, coface))
    covers.sort()
    return covers


def _rank_one_request(
    complex_: FiniteSimplicialComplex,
    scalar=None,
    *,
    field: SheafField = SheafField.RATIONAL,
    prime: int | None = None,
) -> FromCoverMapsRequest:
    if scalar is None:
        scalar = 1 if field is SheafField.PRIME_FIELD else _q(1)
    return FromCoverMapsRequest(
        complex=complex_,
        coefficient_field=field,
        prime=prime,
        stalks=tuple(
            SheafStalk(simplex=cell, basis=("x",)) for cell in _cells(complex_)
        ),
        cover_maps=tuple(
            CoverRestrictionMatrix(source=source, target=target, entries=((scalar,),))
            for source, target in _covers(complex_)
        ),
    )


class TestNativeCatalogParity:
    def test_catalog_tool_runs_the_same_kernel(self) -> None:
        tool = next(tool for tool in BUILTIN_TOOLS if tool.operation_id == OPERATION_ID)
        request = _rank_one_request(_TRIANGLE)
        assert tool.run(request) == _native(request)

    def test_published_examples_execute(self) -> None:
        tool = next(tool for tool in BUILTIN_TOOLS if tool.operation_id == OPERATION_ID)
        outcomes = []
        for example in tool.examples:
            request = FromCoverMapsRequest.model_validate_json(
                json.dumps(example.input)
            )
            result = tool.run(request)
            outcomes.append(result.outcome)
        assert outcomes == [
            SheafOutcome.CELLULAR_SHEAF,
            SheafOutcome.NOT_A_SHEAF,
        ]


_INTERVAL = canonical_complex(("a", "b"), (("a", "b"),))
_TRIANGLE = canonical_complex(("a", "b", "c"), (("a", "b", "c"),))
